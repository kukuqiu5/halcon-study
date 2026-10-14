#include "threshold.hpp"

#include <png.h>

#include <immintrin.h>

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

void expect(bool ok, const char* message) {
    if (ok) return;
    std::printf("FAIL %s\n", message);
    ++g_failures;
}

struct Scalar {
    region reg{};
    std::vector<runLength> runs;
    ThresholdStatus status = ThresholdStatus::Empty;
};

Scalar scalar_threshold(const uint8_t* image, int step, int x, int y, int roi_width, int roi_height,
                        uint8_t low, uint8_t high) {
    Scalar out;
    int64_t col_sum2 = 0;
    int64_t row_sum = 0;
    bool any = false;
    int16_t row1 = 0;
    int16_t row2 = 0;
    int16_t col1 = 0;
    int16_t col2 = 0;
    for (int row = y; row < y + roi_height; ++row) {
        const uint8_t* src = image + static_cast<long>(row) * step + x;
        bool open = false;
        int start = 0;
        const size_t row_begin = out.runs.size();
        for (int col = 0; col < roi_width; ++col) {
            const bool foreground = src[col] >= low && src[col] <= high;
            if (foreground && !open) {
                start = x + col;
                open = true;
            } else if (!foreground && open) {
                runLength run{};
                run.row = static_cast<int16_t>(row);
                run.cols = static_cast<int16_t>(start);
                run.cole = static_cast<int16_t>(x + col - 1);
                const int len = run.cole - run.cols + 1;
                col_sum2 += static_cast<int64_t>(run.cole + run.cols) * len;
                row_sum += static_cast<int64_t>(len) * row;
                out.runs.push_back(run);
                open = false;
            }
        }
        if (open) {
            runLength run{};
            run.row = static_cast<int16_t>(row);
            run.cols = static_cast<int16_t>(start);
            run.cole = static_cast<int16_t>(x + roi_width - 1);
            const int len = run.cole - run.cols + 1;
            col_sum2 += static_cast<int64_t>(run.cole + run.cols) * len;
            row_sum += static_cast<int64_t>(len) * row;
            out.runs.push_back(run);
        }
        if (out.runs.size() > row_begin) {
            if (!any) {
                row1 = static_cast<int16_t>(row);
                col1 = out.runs[row_begin].cols;
                any = true;
            }
            row2 = static_cast<int16_t>(row);
            if (out.runs[row_begin].cols < col1) col1 = out.runs[row_begin].cols;
            if (out.runs.back().cole > col2 || out.runs.size() == row_begin + (any ? out.runs.size() : 0)) {
                col2 = out.runs.back().cole;
            }
            col2 = std::max(col2, out.runs.back().cole);
        }
    }
    if (!any) {
        out.status = ThresholdStatus::Empty;
        return out;
    }
    out.status = ThresholdStatus::Ok;
    out.reg.area = static_cast<uint32_t>(out.runs.size() == 0 ? 0 : 0);
    uint32_t area = 0;
    for (const runLength& run : out.runs) area += static_cast<uint32_t>(run.cole - run.cols + 1);
    out.reg.area = area;
    out.reg.centerCol = (static_cast<double>(col_sum2) / 2.0) / static_cast<double>(area);
    out.reg.centerRow = static_cast<double>(row_sum) / static_cast<double>(area);
    out.reg.row1 = row1;
    out.reg.col1 = col1;
    out.reg.row2 = row2;
    out.reg.col2 = col2;
    out.reg.width = static_cast<int16_t>(col2 - col1 + 1);
    out.reg.height = static_cast<int16_t>(row2 - row1 + 1);
    out.reg.ratio = static_cast<double>(out.reg.height) / static_cast<double>(out.reg.width);
    out.reg.runLengthCount = static_cast<int>(out.runs.size());
    out.reg.pRle = out.runs.data();
    return out;
}

bool near(double a, double b) {
    const double d = a - b;
    return d < 1e-9 && d > -1e-9;
}

bool same_runs(const region& got, const runLength* runs, const Scalar& ref) {
    if (got.area != ref.reg.area || got.runLengthCount != ref.reg.runLengthCount) return false;
    if (got.row1 != ref.reg.row1 || got.col1 != ref.reg.col1 || got.row2 != ref.reg.row2 || got.col2 != ref.reg.col2) return false;
    if (got.width != ref.reg.width || got.height != ref.reg.height || !near(got.ratio, ref.reg.ratio)) return false;
    if (!near(got.centerRow, ref.reg.centerRow) || !near(got.centerCol, ref.reg.centerCol)) return false;
    for (int i = 0; i < got.runLengthCount; ++i) {
        if (runs[i].row != ref.runs[static_cast<size_t>(i)].row ||
            runs[i].cols != ref.runs[static_cast<size_t>(i)].cols ||
            runs[i].cole != ref.runs[static_cast<size_t>(i)].cole) {
            return false;
        }
    }
    return true;
}

void check_image(const char* name, const uint8_t* image, int width, int height, int step, int x, int y,
                 int roi_width, int roi_height, uint8_t low, uint8_t high) {
    const Scalar ref = scalar_threshold(image, step, x, y, roi_width, roi_height, low, high);
    const int capacity = std::max(threshold_max_runs(roi_width, roi_height), 1);
    std::vector<runLength> avx_runs(static_cast<size_t>(capacity));
    std::vector<runLength> sse_runs(static_cast<size_t>(capacity));
    std::vector<runLength> par_runs(static_cast<size_t>(capacity));
    region avx{};
    region sse{};
    region parallel{};
    const ThresholdStatus avx_status = threshold_avx2(image, width, height, step, x, y, roi_width, roi_height, low, high,
                                                      &avx, avx_runs.data(), capacity, 1);
    const ThresholdStatus sse_status = threshold_sse(image, width, height, step, x, y, roi_width, roi_height, low, high,
                                                     &sse, sse_runs.data(), capacity, 1);
    const ThresholdStatus par_status = threshold_avx2(image, width, height, step, x, y, roi_width, roi_height, low, high,
                                                      &parallel, par_runs.data(), capacity, 4);
    expect(avx_status == ref.status, name);
    expect(sse_status == ref.status, name);
    expect(par_status == ref.status, name);
    if (ref.status == ThresholdStatus::Ok) {
        expect(same_runs(avx, avx_runs.data(), ref), name);
        expect(same_runs(sse, sse_runs.data(), ref), name);
        expect(same_runs(parallel, par_runs.data(), ref), name);
        if (!same_runs(avx, avx_runs.data(), ref)) {
            std::printf("  avx area=%u runs=%d center=(%.6f,%.6f) rect=(%d,%d)-(%d,%d)\n", avx.area, avx.runLengthCount,
                        avx.centerRow, avx.centerCol, avx.row1, avx.col1, avx.row2, avx.col2);
            std::printf("  ref area=%u runs=%d center=(%.6f,%.6f) rect=(%d,%d)-(%d,%d)\n", ref.reg.area,
                        ref.reg.runLengthCount, ref.reg.centerRow, ref.reg.centerCol, ref.reg.row1, ref.reg.col1,
                        ref.reg.row2, ref.reg.col2);
        }
    }
}

#if !defined(_MSC_VER)
unsigned char _BitScanForward(unsigned long* index, unsigned long mask) {
    if (mask == 0) return 0;
    *index = static_cast<unsigned long>(__builtin_ctzl(mask));
    return 1;
}
int __popcnt(unsigned int value) { return __builtin_popcount(value); }
#define _mm256_loadu_epi8(p) _mm256_loadu_si256(reinterpret_cast<const __m256i*>(p))
#endif

void threshold_original(uint8_t* pBuf, int step, int xs, int ys, int rc_width, int rc_height, uint8_t lowVal,
                        uint8_t highVal, region* pRegion, runLength* pRLEData) {
    int region_width = ((rc_width + 1) % 32 == 0) ? rc_width + 1 : rc_width + 33 - (rc_width + 1) % 32;
    pRegion->col1 = 32000;
    pRegion->col2 = 0;
    pRegion->row1 = 0;
    pRegion->row2 = 0;
    pRegion->runLengthCount = 0;
    pRegion->area = 0;
    uint8_t* line = new uint8_t[region_width];
    const __m256i low = _mm256_set1_epi8(static_cast<char>(lowVal));
    const __m256i high = _mm256_set1_epi8(static_cast<char>(highVal));
    double col_sum = 0;
    double row_sum = 0;
    bool found = false;
    runLength* cursor = pRLEData;
    for (int j = ys; j < ys + rc_height; ++j) {
        std::memset(line + rc_width, 0, static_cast<size_t>(region_width - rc_width));
        std::memcpy(line, pBuf + static_cast<long>(j) * step + xs, static_cast<size_t>(rc_width));
        __m128i left = _mm_setzero_si128();
        int row_runs = 0;
        int count_index = 0;
        int start_col = xs;
        runLength* row_first = cursor;
        for (int i = 0; i < region_width; i += 32) {
            const __m256i src = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(line + i));
            const __m256i in_range = _mm256_and_si256(
                _mm256_cmpeq_epi8(_mm256_max_epu8(src, low), src),
                _mm256_cmpeq_epi8(_mm256_min_epu8(src, high), src));
            const __m128i low_half = _mm256_extractf128_si256(in_range, 0);
            const __m128i high_half = _mm256_extractf128_si256(in_range, 1);
            const __m256i shifted = _mm256_setr_m128i(_mm_alignr_epi8(low_half, left, 15),
                                                      _mm_alignr_epi8(high_half, low_half, 15));
            left = high_half;
            pRegion->area += static_cast<uint32_t>(__popcnt(static_cast<unsigned>(_mm256_movemask_epi8(in_range))));
            unsigned edge = static_cast<unsigned>(_mm256_movemask_epi8(_mm256_xor_si256(in_range, shifted)));
            while (edge != 0) {
                unsigned long shift = 0;
                _BitScanForward(&shift, edge);
                if (count_index % 2 == 1) {
                    cursor->cole = static_cast<int16_t>(start_col + static_cast<int>(shift) - 1);
                    const int len = cursor->cole - cursor->cols + 1;
                    col_sum += (cursor->cole + cursor->cols) * len / 2.0;
                    row_sum += static_cast<double>(len) * j;
                    ++cursor;
                    ++pRegion->runLengthCount;
                    ++row_runs;
                } else {
                    cursor->row = static_cast<int16_t>(j);
                    cursor->cols = static_cast<int16_t>(start_col + static_cast<int>(shift));
                }
                edge &= ~(1u << shift);
                ++count_index;
            }
            start_col += 32;
        }
        if (row_runs > 0) {
            if (!found) {
                pRegion->row1 = static_cast<int16_t>(j);
                found = true;
            }
            pRegion->row2 = static_cast<int16_t>(j);
            if (row_first->cols < pRegion->col1) pRegion->col1 = row_first->cols;
            if ((row_first + row_runs - 1)->cole > pRegion->col2) pRegion->col2 = (row_first + row_runs - 1)->cole;
        }
    }
    if (pRegion->area == 0) {
        pRegion->centerRow = 0;
        pRegion->centerCol = 0;
    } else {
        pRegion->centerCol = col_sum / pRegion->area;
        pRegion->centerRow = row_sum / pRegion->area;
        pRegion->width = static_cast<int16_t>(pRegion->col2 - pRegion->col1 + 1);
        pRegion->height = static_cast<int16_t>(pRegion->row2 - pRegion->row1 + 1);
        pRegion->ratio = static_cast<double>(pRegion->height) / static_cast<double>(pRegion->width);
        pRegion->pRle = pRLEData;
    }
    delete[] line;
}

template <typename Fn>
double microseconds(Fn&& fn, int iters) {
    for (int i = 0; i < 5; ++i) fn();
    const auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < iters; ++i) fn();
    const auto t1 = std::chrono::steady_clock::now();
    return std::chrono::duration<double, std::micro>(t1 - t0).count() / iters;
}

struct Image {
    int width = 0;
    int height = 0;
    int step = 0;
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
    image.step = image.width;
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

}  // namespace

int main(int argc, char** argv) {
    const char* path = argc > 1 ? argv[1] : "img/bottle_crate_01.png";

    std::vector<uint8_t> rect(32 * 16, 0);
    for (int row = 5; row <= 10; ++row) {
        for (int col = 5; col <= 10; ++col) rect[static_cast<size_t>(row * 32 + col)] = 255;
    }
    check_image("rectangle", rect.data(), 32, 16, 32, 0, 0, 32, 16, 180, 255);
    {
        const int capacity = threshold_max_runs(32, 16);
        std::vector<runLength> runs(static_cast<size_t>(capacity));
        region got{};
        expect(threshold_avx2(rect.data(), 32, 16, 32, 0, 0, 32, 16, 180, 255, &got, runs.data(), capacity, 1) ==
                   ThresholdStatus::Ok,
               "rectangle status");
        expect(got.area == 36 && got.width == 6 && got.height == 6, "rectangle shape");
        expect(near(got.centerRow, 7.5) && near(got.centerCol, 7.5) && near(got.ratio, 1.0), "rectangle center");
        expect(got.row1 == 5 && got.col1 == 5 && got.row2 == 10 && got.col2 == 10 && got.runLengthCount == 6,
               "rectangle bounds");
    }

    std::vector<uint8_t> bit31(64, 0);
    bit31[31] = 255;
    check_image("bit31", bit31.data(), 64, 1, 64, 0, 0, 64, 1, 180, 255);
    std::vector<uint8_t> through31(64, 0);
    std::memset(through31.data(), 255, 32);
    check_image("run-through-31", through31.data(), 64, 1, 64, 0, 0, 64, 1, 180, 255);

    for (int width : {1, 15, 16, 17, 31, 32, 33, 63, 64, 100}) {
        std::vector<uint8_t> image(static_cast<size_t>(width) * 9, 0);
        uint32_t state = 1;
        for (uint8_t& pixel : image) {
            state = state * 1664525u + 1013904223u;
            pixel = static_cast<uint8_t>(state >> 24);
        }
        char label[32];
        std::snprintf(label, sizeof(label), "random-w%d", width);
        check_image(label, image.data(), width, 9, width, 0, 0, width, 9, 180, 255);
    }

    std::vector<uint8_t> padded(13 * 40, 0);
    for (int row = 0; row < 13; ++row) {
        for (int col = 0; col < 30; ++col) {
            padded[static_cast<size_t>(row * 40 + col)] = static_cast<uint8_t>((row * 3 + col * 5) & 255);
        }
    }
    check_image("padded-roi", padded.data(), 30, 13, 40, 2, 1, 20, 8, 40, 200);

    std::vector<uint8_t> blank(16 * 4, 0);
    check_image("blank", blank.data(), 16, 4, 16, 0, 0, 16, 4, 180, 255);
    check_image("inverted-range", rect.data(), 32, 16, 32, 0, 0, 32, 16, 200, 100);

    {
        region out{};
        runLength one{};
        expect(threshold_avx2(nullptr, 8, 8, 8, 0, 0, 8, 8, 0, 255, &out, &one, 1, 1) == ThresholdStatus::InvalidArgument,
               "null image");
        expect(threshold_avx2(blank.data(), 16, 4, 16, -1, 0, 4, 4, 0, 255, &out, &one, 1, 1) ==
                   ThresholdStatus::InvalidArgument,
               "negative x");
        expect(threshold_avx2(blank.data(), 16, 4, 15, 0, 0, 4, 4, 0, 255, &out, &one, 1, 1) ==
                   ThresholdStatus::InvalidArgument,
               "short step");
        expect(threshold_avx2(blank.data(), 16, 4, 16, 0, 0, 4, 4, 0, 255, &out, &one, 1, 0) ==
                   ThresholdStatus::InvalidArgument,
               "zero threads");
        expect(threshold_avx2(rect.data(), 32, 16, 32, 0, 0, 32, 16, 180, 255, &out, &one, 1, 1) ==
                   ThresholdStatus::BufferTooSmall,
               "tiny buffer");
    }

    Image bottle = load_png(path);
    expect(bottle.width == 640 && bottle.height == 512, "bottle size");
    check_image("bottle", bottle.pixels.data(), bottle.width, bottle.height, bottle.step, 0, 0, bottle.width,
                bottle.height, 180, 255);
    check_image("bottle-roi", bottle.pixels.data(), bottle.width, bottle.height, bottle.step, 100, 80, 200, 120, 180, 255);
    check_image("bottle-sse-threads", bottle.pixels.data(), bottle.width, bottle.height, bottle.step, 0, 0, bottle.width,
                bottle.height, 128, 255);

    std::vector<uint8_t> solid(static_cast<size_t>(bottle.width) * bottle.height, 255);
    std::vector<uint8_t> chess(static_cast<size_t>(bottle.width) * bottle.height);
    for (int row = 0; row < bottle.height; ++row) {
        for (int col = 0; col < bottle.width; ++col) {
            chess[static_cast<size_t>(row) * bottle.width + col] = (col & 1) ? 0 : 255;
        }
    }
    check_image("solid", solid.data(), bottle.width, bottle.height, bottle.width, 0, 0, bottle.width, bottle.height, 180,
                255);
    check_image("chess", chess.data(), bottle.width, bottle.height, bottle.width, 0, 0, bottle.width, bottle.height, 180,
                255);

    const int capacity = threshold_max_runs(bottle.width, bottle.height);
    std::vector<runLength> original_runs(static_cast<size_t>(capacity));
    std::vector<runLength> fast_runs(static_cast<size_t>(capacity));
    std::vector<runLength> parallel_runs(static_cast<size_t>(capacity));
    region original{};
    region fast{};
    region parallel{};
    const double us_original = microseconds(
        [&] {
            threshold_original(bottle.pixels.data(), bottle.step, 0, 0, bottle.width, bottle.height, 180, 255, &original,
                               original_runs.data());
        },
        1000);
    const double us_fast = microseconds(
        [&] {
            threshold_avx2(bottle.pixels.data(), bottle.width, bottle.height, bottle.step, 0, 0, bottle.width,
                           bottle.height, 180, 255, &fast, fast_runs.data(), capacity, 1);
        },
        1000);
    const double us_parallel = microseconds(
        [&] {
            threshold_avx2(bottle.pixels.data(), bottle.width, bottle.height, bottle.step, 0, 0, bottle.width,
                           bottle.height, 180, 255, &parallel, parallel_runs.data(), capacity, 4);
        },
        1000);
    const Scalar bottle_ref = scalar_threshold(bottle.pixels.data(), bottle.step, 0, 0, bottle.width, bottle.height, 180, 255);
    expect(same_runs(fast, fast_runs.data(), bottle_ref), "bench serial bottle");
    expect(same_runs(parallel, parallel_runs.data(), bottle_ref), "bench parallel bottle");
    expect(original.area == bottle_ref.reg.area && original.runLengthCount == bottle_ref.reg.runLengthCount,
           "original bottle area");

    std::vector<runLength> chess_runs(static_cast<size_t>(capacity));
    region chess_region{};
    const double us_chess_original = microseconds(
        [&] {
            threshold_original(chess.data(), bottle.width, 0, 0, bottle.width, bottle.height, 180, 255, &chess_region,
                               original_runs.data());
        },
        50);
    const double us_chess_fast = microseconds(
        [&] {
            threshold_avx2(chess.data(), bottle.width, bottle.height, bottle.width, 0, 0, bottle.width, bottle.height, 180,
                           255, &chess_region, chess_runs.data(), capacity, 1);
        },
        50);
    const double us_chess_parallel = microseconds(
        [&] {
            threshold_avx2(chess.data(), bottle.width, bottle.height, bottle.width, 0, 0, bottle.width, bottle.height, 180,
                           255, &chess_region, chess_runs.data(), capacity, 4);
        },
        50);
    const Scalar chess_ref = scalar_threshold(chess.data(), bottle.width, 0, 0, bottle.width, bottle.height, 180, 255);
    expect(same_runs(chess_region, chess_runs.data(), chess_ref), "bench parallel chess");
    expect(original.area == bottle_ref.reg.area && original.runLengthCount == bottle_ref.reg.runLengthCount,
           "original bottle area");

    std::printf("bottle us/call  original %.1f | optimized %.1f (%.2fx) | 4 threads %.1f (%.2fx)\n", us_original, us_fast,
                us_original / us_fast, us_parallel, us_original / us_parallel);
    std::printf("chess  us/call  original %.1f | optimized %.1f (%.2fx) | 4 threads %.1f (%.2fx)\n", us_chess_original,
                us_chess_fast, us_chess_original / us_chess_fast, us_chess_parallel,
                us_chess_original / us_chess_parallel);
    std::printf("bottle area=%u runs=%d center=(%.6f,%.6f)\n", fast.area, fast.runLengthCount, fast.centerRow,
                fast.centerCol);
    if (g_failures != 0) {
        std::printf("FAILED %d\n", g_failures);
        return 1;
    }
    std::printf("PASS\n");
    return 0;
}
