#include "threshold.hpp"

#include <png.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace {

int g_failures = 0;
volatile int g_sink = 0;

void expect(bool ok, const char* message) {
    if (ok) return;
    std::printf("FAIL %s\n", message);
    ++g_failures;
}

bool near(double a, double b) {
    const double d = a - b;
    return d < 1e-9 && d > -1e-9;
}

runLength run_at(int row, int cols, int cole) {
    runLength run{};
    run.row = static_cast<int16_t>(row);
    run.cols = static_cast<int16_t>(cols);
    run.cole = static_cast<int16_t>(cole);
    return run;
}

struct Built {
    std::vector<region> components;
    std::vector<runLength> runs;
    int count = -1;
    ThresholdStatus status = ThresholdStatus::InvalidArgument;
};

Built call_connection(const std::vector<runLength>& runs, Neighborhood neighborhood, int max_components = 0,
                      int threads = 1, int component_capacity = -1, int runs_capacity = -1) {
    Built out;
    const int n = static_cast<int>(runs.size());
    if (component_capacity < 0) component_capacity = std::max(n, 1);
    if (runs_capacity < 0) runs_capacity = std::max(n, 1);
    out.components.resize(static_cast<size_t>(std::max(component_capacity, 1)));
    out.runs.resize(static_cast<size_t>(std::max(runs_capacity, 1)));
    out.status = connection(runs.data(), n, neighborhood, out.components.data(), component_capacity, out.runs.data(),
                            runs_capacity, &out.count, max_components, threads);
    return out;
}

void expect_features(const region& reg, const char* name) {
    expect(reg.runLengthCount > 0 && reg.pRle != nullptr, name);
    if (reg.pRle == nullptr || reg.runLengthCount <= 0) return;
    int64_t col_sum2 = 0;
    int64_t row_sum = 0;
    uint32_t area = 0;
    int16_t row1 = reg.pRle[0].row;
    int16_t row2 = reg.pRle[0].row;
    int16_t col1 = reg.pRle[0].cols;
    int16_t col2 = reg.pRle[0].cole;
    for (int i = 0; i < reg.runLengthCount; ++i) {
        const runLength& run = reg.pRle[i];
        const int len = run.cole - run.cols + 1;
        expect(len > 0, name);
        col_sum2 += static_cast<int64_t>(run.cole + run.cols) * len;
        row_sum += static_cast<int64_t>(len) * run.row;
        area += static_cast<uint32_t>(len);
        if (run.row < row1) row1 = run.row;
        if (run.row > row2) row2 = run.row;
        if (run.cols < col1) col1 = run.cols;
        if (run.cole > col2) col2 = run.cole;
        if (i > 0) {
            const runLength& prev = reg.pRle[i - 1];
            const bool ordered = run.row > prev.row || (run.row == prev.row && run.cols >= prev.cols);
            expect(ordered, name);
        }
    }
    expect(reg.area == area, name);
    expect(near(reg.centerCol, (static_cast<double>(col_sum2) / 2.0) / static_cast<double>(area)), name);
    expect(near(reg.centerRow, static_cast<double>(row_sum) / static_cast<double>(area)), name);
    expect(reg.row1 == row1 && reg.col1 == col1 && reg.row2 == row2 && reg.col2 == col2, name);
    expect(reg.width == static_cast<int16_t>(col2 - col1 + 1), name);
    expect(reg.height == static_cast<int16_t>(row2 - row1 + 1), name);
    expect(near(reg.ratio, static_cast<double>(reg.height) / static_cast<double>(reg.width)), name);
}

struct Oracle {
    int components = 0;
    std::vector<int> order;
    std::vector<int> offset;
};

Oracle flood_runs(const std::vector<runLength>& runs, Neighborhood neighborhood) {
    Oracle oracle;
    if (runs.empty()) return oracle;
    int row1 = runs[0].row;
    int col1 = runs[0].cols;
    int row2 = runs[0].row;
    int col2 = runs[0].cole;
    for (const runLength& run : runs) {
        row1 = std::min(row1, static_cast<int>(run.row));
        col1 = std::min(col1, static_cast<int>(run.cols));
        row2 = std::max(row2, static_cast<int>(run.row));
        col2 = std::max(col2, static_cast<int>(run.cole));
    }
    const int height = row2 - row1 + 1;
    const int width = col2 - col1 + 1;
    std::vector<uint8_t> mask(static_cast<size_t>(height) * static_cast<size_t>(width), 0);
    std::vector<int> label(mask.size(), 0);
    for (const runLength& run : runs) {
        const int y = run.row - row1;
        uint8_t* dst = mask.data() + static_cast<size_t>(y) * static_cast<size_t>(width);
        for (int col = run.cols; col <= run.cole; ++col) dst[col - col1] = 1;
    }
    const int dirs = neighborhood == Neighborhood::Eight ? 8 : 4;
    const int dy[8] = {-1, -1, -1, 0, 0, 1, 1, 1};
    const int dx[8] = {-1, 0, 1, -1, 1, -1, 0, 1};
    const int dy4[4] = {-1, 1, 0, 0};
    const int dx4[4] = {0, 0, -1, 1};
    std::vector<int> stack;
    stack.reserve(1024);
    int components = 0;
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            const int start = y * width + x;
            if (mask[static_cast<size_t>(start)] == 0 || label[static_cast<size_t>(start)] != 0) continue;
            ++components;
            label[static_cast<size_t>(start)] = components;
            stack.push_back(start);
            while (!stack.empty()) {
                const int cur = stack.back();
                stack.pop_back();
                const int cy = cur / width;
                const int cx = cur - cy * width;
                for (int d = 0; d < dirs; ++d) {
                    const int ny = cy + (dirs == 8 ? dy[d] : dy4[d]);
                    const int nx = cx + (dirs == 8 ? dx[d] : dx4[d]);
                    if (ny < 0 || nx < 0 || ny >= height || nx >= width) continue;
                    const int next = ny * width + nx;
                    if (mask[static_cast<size_t>(next)] == 0 || label[static_cast<size_t>(next)] != 0) continue;
                    label[static_cast<size_t>(next)] = components;
                    stack.push_back(next);
                }
            }
        }
    }
    std::vector<int> run_component(runs.size());
    for (size_t i = 0; i < runs.size(); ++i) {
        const runLength& run = runs[i];
        const int y = run.row - row1;
        const int id = label[static_cast<size_t>(y * width + (run.cols - col1))];
        expect(id > 0, "flood pixel");
        for (int col = run.cols; col <= run.cole; ++col) {
            expect(label[static_cast<size_t>(y * width + (col - col1))] == id, "run crosses components");
        }
        run_component[i] = id - 1;
    }
    oracle.components = components;
    oracle.offset.assign(static_cast<size_t>(components + 1), 0);
    for (int id : run_component) ++oracle.offset[static_cast<size_t>(id + 1)];
    for (int i = 1; i <= components; ++i) oracle.offset[static_cast<size_t>(i)] += oracle.offset[static_cast<size_t>(i - 1)];
    oracle.order.resize(runs.size());
    std::vector<int> cursor = oracle.offset;
    for (size_t i = 0; i < runs.size(); ++i) {
        oracle.order[static_cast<size_t>(cursor[static_cast<size_t>(run_component[i])]++)] = static_cast<int>(i);
    }
    return oracle;
}

void expect_matches(const std::vector<runLength>& runs, const Built& got, const Oracle& oracle, const char* name) {
    expect(got.status == ThresholdStatus::Ok, name);
    expect(got.count == oracle.components, name);
    if (got.status != ThresholdStatus::Ok || got.count != oracle.components) {
        std::printf("  %s status=%d count=%d oracle=%d\n", name, static_cast<int>(got.status), got.count,
                    oracle.components);
        return;
    }
    int seen = 0;
    for (int c = 0; c < got.count; ++c) {
        const region& reg = got.components[static_cast<size_t>(c)];
        expect_features(reg, name);
        const int begin = oracle.offset[static_cast<size_t>(c)];
        const int end = oracle.offset[static_cast<size_t>(c + 1)];
        expect(reg.runLengthCount == end - begin, name);
        expect(reg.pRle == got.runs.data() + seen, name);
        if (reg.runLengthCount != end - begin) continue;
        for (int i = 0; i < reg.runLengthCount; ++i) {
            const runLength& expect_run = runs[static_cast<size_t>(oracle.order[static_cast<size_t>(begin + i)])];
            const runLength& actual = reg.pRle[i];
            expect(actual.row == expect_run.row && actual.cols == expect_run.cols && actual.cole == expect_run.cole, name);
        }
        seen += reg.runLengthCount;
    }
    expect(seen == static_cast<int>(runs.size()), name);
}

void expect_same_output(const Built& a, const Built& b, const char* name) {
    expect(a.status == b.status && a.count == b.count, name);
    if (a.status != ThresholdStatus::Ok || a.count != b.count) return;
    for (int c = 0; c < a.count; ++c) {
        const region& ra = a.components[static_cast<size_t>(c)];
        const region& rb = b.components[static_cast<size_t>(c)];
        expect(ra.area == rb.area && ra.runLengthCount == rb.runLengthCount, name);
        expect(ra.row1 == rb.row1 && ra.col1 == rb.col1 && ra.row2 == rb.row2 && ra.col2 == rb.col2, name);
        expect(near(ra.centerRow, rb.centerRow) && near(ra.centerCol, rb.centerCol), name);
        for (int i = 0; i < ra.runLengthCount; ++i) {
            expect(ra.pRle[i].row == rb.pRle[i].row && ra.pRle[i].cols == rb.pRle[i].cols &&
                       ra.pRle[i].cole == rb.pRle[i].cole,
                   name);
        }
    }
}

void check_case(const char* name, const std::vector<runLength>& runs, Neighborhood neighborhood) {
    const Oracle oracle = flood_runs(runs, neighborhood);
    const Built serial = call_connection(runs, neighborhood, 0, 1);
    expect_matches(runs, serial, oracle, name);
    if (runs.size() >= 8192) {
        const Built parallel = call_connection(runs, neighborhood, 0, 4);
        expect_matches(runs, parallel, oracle, name);
        expect_same_output(serial, parallel, name);
    }
}

std::vector<runLength> rect_runs(int row1, int col1, int row2, int col2) {
    std::vector<runLength> runs;
    for (int row = row1; row <= row2; ++row) runs.push_back(run_at(row, col1, col2));
    return runs;
}

long long count_links(const std::vector<runLength>& runs, int gap, bool naive) {
    std::vector<int> begin;
    begin.push_back(0);
    for (size_t i = 1; i < runs.size(); ++i) {
        if (runs[i].row != runs[i - 1].row) begin.push_back(static_cast<int>(i));
    }
    begin.push_back(static_cast<int>(runs.size()));
    long long links = 0;
    for (size_t s = 1; s < begin.size() - 1; ++s) {
        if (runs[static_cast<size_t>(begin[s])].row != runs[static_cast<size_t>(begin[s - 1])].row + 1) continue;
        const int a0 = begin[s - 1];
        const int a1 = begin[s];
        const int b0 = begin[s];
        const int b1 = begin[s + 1];
        if (naive) {
            for (int a = a0; a < a1; ++a) {
                for (int b = b0; b < b1; ++b) {
                    if (runs[static_cast<size_t>(a)].cols <= runs[static_cast<size_t>(b)].cole + gap &&
                        runs[static_cast<size_t>(b)].cols <= runs[static_cast<size_t>(a)].cole + gap) {
                        ++links;
                    }
                }
            }
        } else {
            int i = a0;
            for (int j = b0; j < b1; ++j) {
                const int bcols = runs[static_cast<size_t>(j)].cols;
                const int limit = runs[static_cast<size_t>(j)].cole + gap;
                while (i < a1 && runs[static_cast<size_t>(i)].cole + gap < bcols) ++i;
                for (int k = i; k < a1 && runs[static_cast<size_t>(k)].cols <= limit; ++k) {
                    if (runs[static_cast<size_t>(k)].cole + gap >= bcols) ++links;
                }
            }
        }
    }
    return links;
}

template <typename Fn>
double microseconds(Fn&& fn, int iters) {
    for (int i = 0; i < 3; ++i) fn();
    const auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < iters; ++i) fn();
    const auto t1 = std::chrono::steady_clock::now();
    return std::chrono::duration<double, std::micro>(t1 - t0).count() / iters;
}

struct Image {
    int width = 0;
    int height = 0;
    std::vector<uint8_t> pixels;
};

Image load_png(const std::string& path) {
    FILE* file = std::fopen(path.c_str(), "rb");
    if (file == nullptr) {
        std::perror(path.c_str());
        std::exit(1);
    }
    png_structp png = png_create_read_struct(PNG_LIBPNG_VER_STRING, nullptr, nullptr, nullptr);
    png_infop info = png_create_info_struct(png);
    if (setjmp(png_jmpbuf(png))) std::exit(1);
    png_init_io(png, file);
    png_read_info(png, info);
    Image image;
    image.width = static_cast<int>(png_get_image_width(png, info));
    image.height = static_cast<int>(png_get_image_height(png, info));
    image.pixels.resize(static_cast<size_t>(image.width) * static_cast<size_t>(image.height));
    std::vector<png_bytep> rows(static_cast<size_t>(image.height));
    for (int y = 0; y < image.height; ++y) {
        rows[static_cast<size_t>(y)] = image.pixels.data() + static_cast<size_t>(y) * static_cast<size_t>(image.width);
    }
    png_read_image(png, rows.data());
    png_destroy_read_struct(&png, &info, nullptr);
    std::fclose(file);
    return image;
}

std::vector<runLength> runs_from_image(const Image& image, uint8_t low, uint8_t high) {
    const int capacity = std::max(threshold_max_runs(image.width, image.height), 1);
    std::vector<runLength> runs(static_cast<size_t>(capacity));
    region region{};
    const ThresholdStatus status = threshold_avx2(image.pixels.data(), image.width, image.height, image.width, 0, 0,
                                                  image.width, image.height, low, high, &region, runs.data(), capacity, 1);
    expect(status == ThresholdStatus::Ok || status == ThresholdStatus::Empty, "threshold");
    runs.resize(static_cast<size_t>(std::max(region.runLengthCount, 0)));
    return runs;
}

}  // namespace

int main(int argc, char** argv) {
    const char* path = argc > 1 ? argv[1] : "img/bottle_crate_01.png";

    {
        const auto runs = rect_runs(5, 5, 10, 10);
        const Built got = call_connection(runs, Neighborhood::Eight);
        expect(got.status == ThresholdStatus::Ok && got.count == 1, "rectangle count");
        expect(got.components[0].area == 36, "rectangle area");
        expect(near(got.components[0].centerRow, 7.5) && near(got.components[0].centerCol, 7.5), "rectangle center");
        expect(got.components[0].row1 == 5 && got.components[0].col1 == 5 && got.components[0].row2 == 10 &&
                   got.components[0].col2 == 10,
               "rectangle bounds");
        expect(got.components[0].width == 6 && got.components[0].height == 6 && near(got.components[0].ratio, 1.0),
               "rectangle shape");
        expect(got.components[0].runLengthCount == 6, "rectangle runs");
        check_case("rectangle", runs, Neighborhood::Four);
        check_case("rectangle8", runs, Neighborhood::Eight);
    }

    {
        const std::vector<runLength> runs = {run_at(0, 0, 1)};
        const Built got = call_connection(runs, Neighborhood::Eight);
        expect(got.count == 1 && got.components[0].area == 2, "segment area");
        expect(near(got.components[0].centerCol, 0.5) && near(got.components[0].centerRow, 0.0), "segment center");
        expect(got.components[0].width == 2 && got.components[0].height == 1 && near(got.components[0].ratio, 0.5),
               "segment ratio");
    }

    {
        std::vector<runLength> runs = rect_runs(0, 0, 2, 2);
        auto right = rect_runs(0, 5, 2, 7);
        runs.insert(runs.end(), right.begin(), right.end());
        std::sort(runs.begin(), runs.end(), [](const runLength& a, const runLength& b) {
            return a.row < b.row || (a.row == b.row && a.cols < b.cols);
        });
        check_case("two blobs8", runs, Neighborhood::Eight);
        check_case("two blobs4", runs, Neighborhood::Four);
        const Built got = call_connection(runs, Neighborhood::Eight);
        expect(got.count == 2 && got.components[0].area == 9 && got.components[1].area == 9, "two blob areas");
    }

    {
        const std::vector<runLength> runs = {run_at(0, 0, 0), run_at(1, 1, 1), run_at(2, 2, 2), run_at(3, 3, 3)};
        check_case("diagonal8", runs, Neighborhood::Eight);
        check_case("diagonal4", runs, Neighborhood::Four);
        expect(call_connection(runs, Neighborhood::Eight).count == 1, "diagonal is one in 8");
        expect(call_connection(runs, Neighborhood::Four).count == 4, "diagonal is four in 4");
    }

    {
        const std::vector<runLength> gap = {run_at(0, 0, 0), run_at(0, 2, 2)};
        expect(call_connection(gap, Neighborhood::Eight).count == 2, "one pixel gap stays split");
        expect(call_connection(gap, Neighborhood::Four).count == 2, "one pixel gap stays split in 4");
        const std::vector<runLength> touch = {run_at(0, 0, 0), run_at(0, 1, 3)};
        expect(call_connection(touch, Neighborhood::Eight).count == 1, "touching runs merge");
        expect(call_connection(touch, Neighborhood::Four).count == 1, "touching runs merge in 4");
        const std::vector<runLength> skip = {run_at(0, 0, 4), run_at(2, 0, 4)};
        expect(call_connection(skip, Neighborhood::Eight).count == 2, "skipped row splits");
    }

    {
        const std::vector<runLength> slant = {run_at(0, 0, 10), run_at(1, 11, 20)};
        expect(call_connection(slant, Neighborhood::Eight).count == 1, "diagonal column touch is 8");
        expect(call_connection(slant, Neighborhood::Four).count == 2, "diagonal column touch is not 4");
        const std::vector<runLength> share = {run_at(0, 0, 10), run_at(1, 10, 20)};
        expect(call_connection(share, Neighborhood::Four).count == 1, "shared column is 4");
    }

    {
        const std::vector<runLength> bridge = {run_at(0, 0, 5), run_at(0, 4, 12), run_at(0, 10, 15)};
        const Built got = call_connection(bridge, Neighborhood::Four);
        expect(got.count == 1 && got.components[0].area == 21 && got.components[0].runLengthCount == 3,
               "overlapping runs unite");
        const std::vector<runLength> shape = {run_at(0, 0, 0), run_at(0, 2, 2), run_at(1, 0, 2),
                                              run_at(2, 0, 0), run_at(2, 2, 2)};
        check_case("u8", shape, Neighborhood::Eight);
        check_case("u4", shape, Neighborhood::Four);
        expect(call_connection(shape, Neighborhood::Four).count == 1, "U is one component");
    }

    {
        const std::vector<runLength> runs = {run_at(-2, -5, -5), run_at(-1, -4, -4)};
        expect(call_connection(runs, Neighborhood::Eight).count == 1, "negative diagonal 8");
        expect(call_connection(runs, Neighborhood::Four).count == 2, "negative diagonal 4");
        const Built one = call_connection({run_at(-2, -5, -5)}, Neighborhood::Eight);
        expect(near(one.components[0].centerRow, -2.0) && near(one.components[0].centerCol, -5.0), "negative center");
    }

    {
        const std::vector<runLength> runs = {run_at(0, 0, 0), run_at(0, 2, 2), run_at(3, 0, 0)};
        const Built kept = call_connection(runs, Neighborhood::Eight, 2);
        expect(kept.status == ThresholdStatus::Ok && kept.count == 2, "max_components keeps two");
        expect(kept.components[0].row1 == 0 && kept.components[0].col1 == 0, "max keeps first");
        expect(kept.components[1].row1 == 0 && kept.components[1].col1 == 2, "max keeps second");
        uint32_t area = 0;
        for (int i = 0; i < kept.count; ++i) area += kept.components[static_cast<size_t>(i)].area;
        expect(area == 2, "truncated area drops the third pixel");
        const Built all = call_connection(runs, Neighborhood::Eight, 0);
        expect(all.count == 3, "max 0 keeps all");
        const Built plenty = call_connection(runs, Neighborhood::Eight, 9);
        expect(plenty.count == 3, "max above the count keeps all");
    }

    {
        int count = 7;
        expect(connection(nullptr, 0, Neighborhood::Eight, nullptr, 0, nullptr, 0, &count) == ThresholdStatus::Empty &&
                   count == 0,
               "empty");
        const std::vector<runLength> runs = {run_at(0, 0, 0), run_at(0, 2, 2)};
        std::vector<region> components(2);
        std::vector<runLength> out_runs(2, run_at(9, 9, 9));
        components[0].area = 123;
        int reported = 0;
        const ThresholdStatus small =
            connection(runs.data(), 2, Neighborhood::Eight, components.data(), 1, out_runs.data(), 2, &reported);
        expect(small == ThresholdStatus::BufferTooSmall && reported == 2, "component buffer");
        expect(components[0].area == 123 && out_runs[0].row == 9, "no partial write");
        const ThresholdStatus short_runs =
            connection(runs.data(), 2, Neighborhood::Eight, components.data(), 2, out_runs.data(), 1, &reported);
        expect(short_runs == ThresholdStatus::BufferTooSmall && reported == 2 && out_runs[0].row == 9, "run buffer");
        runLength* alias = const_cast<runLength*>(runs.data());
        expect(connection(alias, 2, Neighborhood::Eight, components.data(), 2, alias, 2, &reported) ==
                   ThresholdStatus::InvalidArgument,
               "alias");
        const std::vector<runLength> bad_order = {run_at(1, 0, 0), run_at(0, 0, 0)};
        expect(call_connection(bad_order, Neighborhood::Eight).status == ThresholdStatus::InvalidArgument, "row order");
        const std::vector<runLength> bad_cols = {run_at(0, 3, 3), run_at(0, 1, 1)};
        expect(call_connection(bad_cols, Neighborhood::Eight).status == ThresholdStatus::InvalidArgument, "col order");
        const std::vector<runLength> bad_run = {run_at(0, 4, 2)};
        expect(call_connection(bad_run, Neighborhood::Eight).status == ThresholdStatus::InvalidArgument, "cole");
        expect(connection(runs.data(), 2, Neighborhood::Eight, components.data(), 2, out_runs.data(), 2, &reported, -1) ==
                   ThresholdStatus::InvalidArgument,
               "negative max");
        expect(connection(runs.data(), 2, Neighborhood::Eight, components.data(), 2, out_runs.data(), 2, &reported, 0, 0) ==
                   ThresholdStatus::InvalidArgument,
               "threads");
        expect(connection(runs.data(), 2, Neighborhood::Eight, components.data(), 2, out_runs.data(), 2, nullptr) ==
                   ThresholdStatus::InvalidArgument,
               "null count");
        region input{};
        input.pRle = const_cast<runLength*>(runs.data());
        input.runLengthCount = 2;
        reported = 0;
        expect(connection(input, Neighborhood::Eight, components.data(), 2, out_runs.data(), 2, &reported) ==
                       ThresholdStatus::Ok &&
                   reported == 2,
               "region overload");
    }

    {
        std::vector<runLength> chess;
        constexpr int kSide = 48;
        for (int row = 0; row < kSide; ++row) {
            for (int col = (row & 1); col < kSide; col += 2) chess.push_back(run_at(row, col, col));
        }
        check_case("chess48-8", chess, Neighborhood::Eight);
        check_case("chess48-4", chess, Neighborhood::Four);
        expect(call_connection(chess, Neighborhood::Eight).count == 1, "chess 8 is one");
        expect(call_connection(chess, Neighborhood::Four).count == kSide * kSide / 2, "chess 4 is pixels");
    }

    {
        std::vector<runLength> line;
        std::vector<runLength> diagonal;
        constexpr int kRows = 9000;
        line.reserve(kRows);
        diagonal.reserve(kRows);
        for (int row = 0; row < kRows; ++row) {
            line.push_back(run_at(row, 0, 0));
            diagonal.push_back(run_at(row, row, row));
        }
        for (Neighborhood neighborhood : {Neighborhood::Eight, Neighborhood::Four}) {
            const Built serial = call_connection(line, neighborhood, 0, 1);
            const Built parallel = call_connection(line, neighborhood, 0, 4);
            expect(serial.count == 1 && serial.components[0].area == static_cast<uint32_t>(kRows), "tall line");
            expect_same_output(serial, parallel, "tall line threads");
        }
        const Built diag8 = call_connection(diagonal, Neighborhood::Eight, 0, 4);
        const Built diag4 = call_connection(diagonal, Neighborhood::Four, 0, 4);
        expect(diag8.count == 1 && diag8.components[0].area == static_cast<uint32_t>(kRows), "tall diagonal 8");
        expect(diag4.count == kRows, "tall diagonal 4");
        expect_same_output(diag8, call_connection(diagonal, Neighborhood::Eight, 0, 1), "diagonal threads");
        expect_same_output(diag4, call_connection(diagonal, Neighborhood::Four, 0, 1), "diagonal4 threads");
        uint32_t diag_area = 0;
        for (int i = 0; i < diag4.count; ++i) diag_area += diag4.components[static_cast<size_t>(i)].area;
        expect(diag_area == static_cast<uint32_t>(kRows), "diagonal 4 area");
    }

    Image image = load_png(path);
    const std::vector<runLength> bottle = runs_from_image(image, 180, 255);
    check_case("bottle8", bottle, Neighborhood::Eight);
    check_case("bottle4", bottle, Neighborhood::Four);
    {
        uint32_t area8 = 0;
        uint32_t area4 = 0;
        const Built eight = call_connection(bottle, Neighborhood::Eight);
        const Built four = call_connection(bottle, Neighborhood::Four);
        for (int i = 0; i < eight.count; ++i) area8 += eight.components[static_cast<size_t>(i)].area;
        for (int i = 0; i < four.count; ++i) area4 += four.components[static_cast<size_t>(i)].area;
        uint32_t total = 0;
        for (const runLength& run : bottle) total += static_cast<uint32_t>(run.cole - run.cols + 1);
        expect(area8 == total && area4 == total, "bottle area conserved");
        std::printf("bottle runs=%zu area=%u components8=%d components4=%d\n", bottle.size(), total, eight.count,
                    four.count);
    }

    Image random = image;
    uint32_t state = 1;
    for (uint8_t& pixel : random.pixels) {
        state = state * 1664525u + 1013904223u;
        pixel = static_cast<uint8_t>((state >> 24) < 77 ? 255 : 0);
    }
    const std::vector<runLength> random_runs = runs_from_image(random, 255, 255);
    check_case("random8", random_runs, Neighborhood::Eight);
    check_case("random4", random_runs, Neighborhood::Four);
    expect_same_output(call_connection(random_runs, Neighborhood::Eight, 0, 1),
                       call_connection(random_runs, Neighborhood::Eight, 0, 4), "random threads");
    std::printf("random runs=%zu components8=%d\n", random_runs.size(),
                call_connection(random_runs, Neighborhood::Eight).count);

    std::vector<runLength> wide;
    constexpr int kWide = 2000;
    constexpr int kHigh = 40;
    wide.reserve(static_cast<size_t>(kWide * kHigh));
    for (int row = 0; row < kHigh; ++row) {
        for (int col = 0; col < kWide; ++col) wide.push_back(run_at(row, col, col));
    }
    const long long naive_links = count_links(wide, 1, true);
    const long long fast_links = count_links(wide, 1, false);
    expect(naive_links == fast_links && naive_links > 0, "two pointer links");
    check_case("wide8", wide, Neighborhood::Eight);
    const Built wide_got = call_connection(wide, Neighborhood::Eight, 0, 4);
    expect(wide_got.count == 1 && wide_got.components[0].area == static_cast<uint32_t>(kWide * kHigh), "wide is solid");

    std::vector<runLength> chess;
    chess.reserve(static_cast<size_t>(image.width * image.height / 2));
    for (int row = 0; row < image.height; ++row) {
        for (int col = (row & 1); col < image.width; col += 2) chess.push_back(run_at(row, col, col));
    }
    const Built chess8 = call_connection(chess, Neighborhood::Eight);
    const Built chess4 = call_connection(chess, Neighborhood::Four);
    expect(chess8.count == 1, "full chess 8");
    expect(chess4.count == static_cast<int>(chess.size()), "full chess 4");
    expect_same_output(chess8, call_connection(chess, Neighborhood::Eight, 0, 4), "full chess threads");

    const auto bench = [](const std::vector<runLength>& runs, Neighborhood neighborhood, int threads, int iters) {
        Built out = call_connection(runs, neighborhood, 0, threads);
        const double us = microseconds(
            [&] {
                out.status = connection(runs.data(), static_cast<int>(runs.size()), neighborhood, out.components.data(),
                                        static_cast<int>(out.components.size()), out.runs.data(),
                                        static_cast<int>(out.runs.size()), &out.count, 0, threads);
                g_sink += out.count;
            },
            iters);
        return us;
    };
    const double bottle8_us = bench(bottle, Neighborhood::Eight, 1, 200);
    const double bottle4_us = bench(bottle, Neighborhood::Four, 1, 200);
    const double random1_us = bench(random_runs, Neighborhood::Eight, 1, 20);
    const double random4_us = bench(random_runs, Neighborhood::Eight, 4, 20);
    const double chess8_us = bench(chess, Neighborhood::Eight, 1, 20);
    const double chess8p_us = bench(chess, Neighborhood::Eight, 4, 20);
    const double chess4_us = bench(chess, Neighborhood::Four, 1, 10);
    const double wide1_us = bench(wide, Neighborhood::Eight, 1, 20);
    const double wide4_us = bench(wide, Neighborhood::Eight, 4, 20);
    const double naive_us = microseconds([&] { g_sink += static_cast<int>(count_links(wide, 1, true)); }, 1);
    const double fast_us = microseconds([&] { g_sink += static_cast<int>(count_links(wide, 1, false)); }, 20);

    std::printf("bottle us/call   8-connect %.1f | 4-connect %.1f\n", bottle8_us, bottle4_us);
    std::printf("random us/call   serial %.1f | 4 threads %.1f\n", random1_us, random4_us);
    std::printf("chess  us/call   8-connect %.1f | 8-connect x4 %.1f | 4-connect %.1f\n", chess8_us, chess8p_us,
                chess4_us);
    std::printf("wide   us/call   connection %.1f | x4 %.1f | adjacent scan %.1f | all-pairs %.1f | links=%lld\n",
                wide1_us, wide4_us, fast_us, naive_us, naive_links);
    std::printf("%s\n", g_failures == 0 ? "PASS" : "FAIL");
    return g_failures == 0 ? 0 : 1;
}
