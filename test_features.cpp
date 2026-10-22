#include "features.hpp"

#include <png.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <string>
#include <utility>
#include <vector>

namespace {

int g_failures = 0;
volatile int g_sink = 0;
const double kPi = std::acos(-1.0);

void expect(bool ok, const char* message) {
    if (ok) return;
    std::printf("FAIL %s\n", message);
    ++g_failures;
}

void expect_status(ThresholdStatus got, ThresholdStatus want, const char* name) {
    if (got == want) return;
    std::printf("FAIL %s status=%d want=%d\n", name, static_cast<int>(got), static_cast<int>(want));
    ++g_failures;
}

bool near(double a, double b, double tol = 1e-8) {
    const double d = std::abs(a - b);
    return d <= tol || d <= tol * std::max(std::abs(a), std::abs(b));
}

void expect_near(double got, double want, const char* name, double tol = 1e-8) {
    if (near(got, want, tol)) return;
    std::printf("FAIL %s got=%.12g want=%.12g\n", name, got, want);
    ++g_failures;
}

runLength run_at(int row, int cols, int cole) {
    runLength run{};
    run.row = static_cast<int16_t>(row);
    run.cols = static_cast<int16_t>(cols);
    run.cole = static_cast<int16_t>(cole);
    return run;
}

region from_runs(std::vector<runLength>& runs) {
    region reg{};
    reg.centerRow = 999;
    reg.centerCol = -999;
    reg.area = 1;
    reg.row1 = 7;
    reg.col1 = 7;
    reg.row2 = 9;
    reg.col2 = 9;
    reg.width = 3;
    reg.height = 3;
    reg.ratio = 9;
    reg.runLengthCount = static_cast<int>(runs.size());
    reg.pRle = runs.empty() ? nullptr : runs.data();
    return reg;
}

std::vector<runLength> rectangle(int row1, int col1, int row2, int col2) {
    std::vector<runLength> runs;
    runs.reserve(static_cast<size_t>(row2 - row1 + 1));
    for (int row = row1; row <= row2; ++row) runs.push_back(run_at(row, col1, col2));
    return runs;
}

std::vector<runLength> pixels_to_runs(std::vector<std::pair<int, int>> pixels) {
    std::sort(pixels.begin(), pixels.end());
    pixels.erase(std::unique(pixels.begin(), pixels.end()), pixels.end());
    std::vector<runLength> runs;
    for (const auto& pixel : pixels) {
        if (!runs.empty() && runs.back().row == pixel.first && runs.back().cole + 1 == pixel.second) {
            runs.back().cole = static_cast<int16_t>(pixel.second);
        } else {
            runs.push_back(run_at(pixel.first, pixel.second, pixel.second));
        }
    }
    return runs;
}

struct Moments {
    int64_t area = 0;
    double row = 0;
    double column = 0;
    double m11 = 0;
    double m20 = 0;
    double m02 = 0;
    double max_dist = 0;
};

Moments pixel_moments(const std::vector<runLength>& runs) {
    Moments out;
    int64_t sum_r = 0;
    int64_t sum_c = 0;
    for (const runLength& run : runs) {
        for (int col = run.cols; col <= run.cole; ++col) {
            ++out.area;
            sum_r += run.row;
            sum_c += col;
        }
    }
    if (out.area == 0) return out;
    out.row = static_cast<double>(sum_r) / static_cast<double>(out.area);
    out.column = static_cast<double>(sum_c) / static_cast<double>(out.area);
    double s11 = 0;
    double s20 = 0;
    double s02 = 0;
    for (const runLength& run : runs) {
        const double dr = run.row - out.row;
        for (int col = run.cols; col <= run.cole; ++col) {
            const double dc = col - out.column;
            s20 += dr * dr;
            s02 += dc * dc;
            s11 += dr * dc;
            out.max_dist = std::max(out.max_dist, std::hypot(dr, dc));
        }
    }
    const double area = static_cast<double>(out.area);
    out.m11 = s11 / area;
    out.m20 = s20 / area;
    out.m02 = s02 / area;
    return out;
}

void expect_moments(const region& reg, const char* name) {
    double m11 = -1;
    double m20 = -1;
    double m02 = -1;
    double ia = -1;
    double ib = -1;
    const ThresholdStatus status = moments_region_2nd(reg, &m11, &m20, &m02, &ia, &ib);
    const bool empty = reg.runLengthCount == 0;
    expect_status(status, empty ? ThresholdStatus::Empty : ThresholdStatus::Ok, name);
    if (empty) {
        expect(m11 == 0 && m20 == 0 && m02 == 0 && ia == 0 && ib == 0, name);
        return;
    }
    const Moments oracle = pixel_moments(std::vector<runLength>(reg.pRle, reg.pRle + reg.runLengthCount));
    expect_near(m11, oracle.m11, name);
    expect_near(m20, oracle.m20, name);
    expect_near(m02, oracle.m02, name);
    const double diff = oracle.m20 - oracle.m02;
    const double root = std::sqrt(diff * diff + 4.0 * oracle.m11 * oracle.m11);
    expect_near(ia, oracle.m20 + oracle.m02 + root, name);
    expect_near(ib, std::max(0.0, oracle.m20 + oracle.m02 - root), name);

    double ra = -1;
    double rb = -1;
    double phi = 99;
    expect_status(elliptic_axis(reg, &ra, &rb, &phi), ThresholdStatus::Ok, name);
    expect_near(ra, std::sqrt(std::max(0.0, 2.0 * ia)), name);
    expect_near(rb, std::sqrt(std::max(0.0, 2.0 * ib)), name);
    expect(phi > -kPi / 2.0 - 1e-12 && phi <= kPi / 2.0 + 1e-12, name);

    double circ = -1;
    expect_status(circularity(reg, &circ), ThresholdStatus::Ok, name);
    double want_circ = 1;
    if (oracle.max_dist > 1e-12) {
        want_circ = std::min(1.0, static_cast<double>(oracle.area) / (kPi * oracle.max_dist * oracle.max_dist));
    }
    expect_near(circ, want_circ, name);
}

struct TripleCircle {
    double row = 0;
    double column = 0;
    double radius = 0;
};

TripleCircle circle_from_two(std::pair<int, int> a, std::pair<int, int> b) {
    TripleCircle out;
    out.row = (a.first + b.first) / 2.0;
    out.column = (a.second + b.second) / 2.0;
    out.radius = std::hypot(a.first - b.first, a.second - b.second) / 2.0;
    return out;
}

TripleCircle circle_from_three(std::pair<int, int> a, std::pair<int, int> b, std::pair<int, int> c) {
    const double d = 2.0 * (a.second * (b.first - c.first) + b.second * (c.first - a.first) + c.second * (a.first - b.first));
    TripleCircle ab = circle_from_two(a, b);
    TripleCircle bc = circle_from_two(b, c);
    TripleCircle ac = circle_from_two(a, c);
    if (std::abs(d) < 1e-12) {
        if (ab.radius >= bc.radius && ab.radius >= ac.radius) return ab;
        return bc.radius >= ac.radius ? bc : ac;
    }
    const double a2 = static_cast<double>(a.second) * a.second + static_cast<double>(a.first) * a.first;
    const double b2 = static_cast<double>(b.second) * b.second + static_cast<double>(b.first) * b.first;
    const double c2 = static_cast<double>(c.second) * c.second + static_cast<double>(c.first) * c.first;
    TripleCircle out;
    out.column = (a2 * (b.first - c.first) + b2 * (c.first - a.first) + c2 * (a.first - b.first)) / d;
    out.row = (a2 * (c.second - b.second) + b2 * (a.second - c.second) + c2 * (b.second - a.second)) / d;
    out.radius = std::hypot(out.row - a.first, out.column - a.second);
    return out;
}

bool circle_covers(TripleCircle circle, const std::vector<std::pair<int, int>>& points) {
    for (const auto& point : points) {
        if (std::hypot(point.first - circle.row, point.second - circle.column) > circle.radius + 1e-7) return false;
    }
    return true;
}

TripleCircle brute_circle(const std::vector<std::pair<int, int>>& points) {
    TripleCircle best;
    best.radius = 1e300;
    if (points.size() == 1) {
        best.row = points[0].first;
        best.column = points[0].second;
        best.radius = 0;
        return best;
    }
    for (size_t i = 0; i < points.size(); ++i) {
        for (size_t j = i + 1; j < points.size(); ++j) {
            const TripleCircle circle = circle_from_two(points[i], points[j]);
            if (circle.radius < best.radius && circle_covers(circle, points)) best = circle;
        }
    }
    for (size_t i = 0; i < points.size(); ++i) {
        for (size_t j = i + 1; j < points.size(); ++j) {
            for (size_t k = j + 1; k < points.size(); ++k) {
                const TripleCircle circle = circle_from_three(points[i], points[j], points[k]);
                if (circle.radius < best.radius && circle_covers(circle, points)) best = circle;
            }
        }
    }
    return best;
}

void expect_circle(const std::vector<std::pair<int, int>>& points, const char* name) {
    std::vector<runLength> runs = pixels_to_runs(points);
    region reg = from_runs(runs);
    double row = -1;
    double column = -1;
    double radius = -1;
    expect_status(smallest_circle(reg, &row, &column, &radius), ThresholdStatus::Ok, name);
    const TripleCircle want = brute_circle(points);
    expect_near(radius, want.radius, name, 1e-6);
    expect_near(row, want.row, name, 1e-6);
    expect_near(column, want.column, name, 1e-6);
    for (const auto& point : points) {
        expect(std::hypot(point.first - row, point.second - column) <= radius + 1e-6, name);
    }
}

const char* kNames[] = {
    "area",        "row",          "column",       "row1",         "column1",      "row2",
    "column2",     "width",        "height",       "ratio",        "circularity",  "compactness",
    "contlength",  "convexity",    "rectangularity", "ra",         "rb",           "phi",
    "orientation", "anisometry",   "bulkiness",    "struct_factor", "outer_radius", "max_diameter",
    "rect2_phi",   "rect2_len1",   "rect2_len2",   "moments_m11",  "moments_m20",  "moments_m02",
    "moments_ia",  "moments_ib",
};

double feature_at(const double* values, const char* name) {
    for (size_t i = 0; i < sizeof(kNames) / sizeof(kNames[0]); ++i) {
        if (std::strcmp(kNames[i], name) == 0) return values[i];
    }
    expect(false, name);
    return 0;
}

void expect_catalog(const region& reg, const char* name) {
    double values[sizeof(kNames) / sizeof(kNames[0])];
    const int count = static_cast<int>(sizeof(kNames) / sizeof(kNames[0]));
    const ThresholdStatus status = region_features(reg, kNames, count, values);
    const bool empty = reg.runLengthCount == 0;
    expect_status(status, empty ? ThresholdStatus::Empty : ThresholdStatus::Ok, name);
    int64_t area = -1;
    double row = -1;
    double column = -1;
    expect_status(area_center(reg, &area, &row, &column), status, name);
    expect_near(feature_at(values, "area"), static_cast<double>(area), name);
    expect_near(feature_at(values, "row"), row, name);
    expect_near(feature_at(values, "column"), column, name);
    int row1 = -1;
    int col1 = -1;
    int row2 = -1;
    int col2 = -1;
    expect_status(smallest_rectangle1(reg, &row1, &col1, &row2, &col2), status, name);
    expect_near(feature_at(values, "row1"), row1, name);
    expect_near(feature_at(values, "column1"), col1, name);
    expect_near(feature_at(values, "row2"), row2, name);
    expect_near(feature_at(values, "column2"), col2, name);
    if (!empty) {
        expect_near(feature_at(values, "width"), col2 - col1 + 1, name);
        expect_near(feature_at(values, "height"), row2 - row1 + 1, name);
        expect_near(feature_at(values, "ratio"), static_cast<double>(row2 - row1 + 1) / (col2 - col1 + 1), name);
    }
    double length = -1;
    double circ = -1;
    double compact = -1;
    double conv = -1;
    double rect = -1;
    expect_status(contlength(reg, &length), status, name);
    expect_status(circularity(reg, &circ), status, name);
    expect_status(compactness(reg, &compact), status, name);
    expect_status(convexity(reg, &conv), status, name);
    expect_status(rectangularity(reg, &rect), status, name);
    expect_near(feature_at(values, "contlength"), length, name);
    expect_near(feature_at(values, "circularity"), circ, name);
    expect_near(feature_at(values, "compactness"), compact, name);
    expect_near(feature_at(values, "convexity"), conv, name);
    expect_near(feature_at(values, "rectangularity"), rect, name);
    double ra = -1;
    double rb = -1;
    double phi = -1;
    expect_status(elliptic_axis(reg, &ra, &rb, &phi), status, name);
    expect_near(feature_at(values, "ra"), ra, name);
    expect_near(feature_at(values, "rb"), rb, name);
    expect_near(feature_at(values, "phi"), phi, name);
    expect_near(feature_at(values, "orientation"), phi, name);
    double anisometry = -1;
    double bulkiness = -1;
    double structure = -1;
    expect_status(eccentricity(reg, &anisometry, &bulkiness, &structure), status, name);
    expect_near(feature_at(values, "anisometry"), anisometry, name);
    expect_near(feature_at(values, "bulkiness"), bulkiness, name);
    expect_near(feature_at(values, "struct_factor"), structure, name);
    double crow = -1;
    double ccol = -1;
    double radius = -1;
    expect_status(smallest_circle(reg, &crow, &ccol, &radius), status, name);
    expect_near(feature_at(values, "outer_radius"), radius, name);
    int drow1 = -1;
    int dcol1 = -1;
    int drow2 = -1;
    int dcol2 = -1;
    double diameter = -1;
    expect_status(diameter_region(reg, &drow1, &dcol1, &drow2, &dcol2, &diameter), status, name);
    expect_near(feature_at(values, "max_diameter"), diameter, name);
    double rrow = -1;
    double rcol = -1;
    double rphi = -1;
    double len1 = -1;
    double len2 = -1;
    expect_status(smallest_rectangle2(reg, &rrow, &rcol, &rphi, &len1, &len2), status, name);
    expect_near(feature_at(values, "rect2_phi"), rphi, name);
    expect_near(feature_at(values, "rect2_len1"), len1, name);
    expect_near(feature_at(values, "rect2_len2"), len2, name);
    expect(len1 + 1e-9 >= len2, name);
    double m11 = -1;
    double m20 = -1;
    double m02 = -1;
    double ia = -1;
    double ib = -1;
    expect_status(moments_region_2nd(reg, &m11, &m20, &m02, &ia, &ib), status, name);
    expect_near(feature_at(values, "moments_m11"), m11, name);
    expect_near(feature_at(values, "moments_m20"), m20, name);
    expect_near(feature_at(values, "moments_m02"), m02, name);
    expect_near(feature_at(values, "moments_ia"), ia, name);
    expect_near(feature_at(values, "moments_ib"), ib, name);
}

void check_rectangle_family() {
    for (int height = 1; height <= 30; ++height) {
        for (int width = 1; width <= 30; ++width) {
            std::vector<runLength> runs = rectangle(0, 0, height - 1, width - 1);
            region reg = from_runs(runs);
            double rectangular = -1;
            double conv = -1;
            double length = -1;
            expect_status(rectangularity(reg, &rectangular), ThresholdStatus::Ok, "rect rectangularity");
            expect_status(convexity(reg, &conv), ThresholdStatus::Ok, "rect convexity");
            expect_status(contlength(reg, &length), ThresholdStatus::Ok, "rect contlength");
            if (!near(rectangular, 1.0) || !near(conv, 1.0) || !near(length, 2.0 * (width + height - 2))) {
                std::printf("FAIL rect %dx%d rectangularity=%.12g convexity=%.12g contlength=%.12g\n", width, height,
                            rectangular, conv, length);
                ++g_failures;
                return;
            }
        }
    }
    for (int length = 1; length <= 20; ++length) {
        std::vector<runLength> horizontal = rectangle(4, -3, 4, -3 + length - 1);
        region hreg = from_runs(horizontal);
        double phi = 99;
        double ra = -1;
        double rb = -1;
        expect_status(elliptic_axis(hreg, &ra, &rb, &phi), ThresholdStatus::Ok, "hbar phi");
        expect_near(phi, 0.0, "hbar phi");
        expect_near(rb, 0.0, "hbar rb");
        double anisometry = 99;
        double bulkiness = 99;
        double structure = 99;
        expect_status(eccentricity(hreg, &anisometry, &bulkiness, &structure), ThresholdStatus::Ok, "hbar ecc");
        expect(anisometry == 0 && bulkiness == 0 && structure == 0, "hbar ecc zero");
        double row = -1;
        double column = -1;
        double rphi = 99;
        double len1 = -1;
        double len2 = -1;
        expect_status(smallest_rectangle2(hreg, &row, &column, &rphi, &len1, &len2), ThresholdStatus::Ok, "hbar rect2");
        expect_near(row, 4, "hbar rect2 row");
        expect_near(column, -3 + (length - 1) / 2.0, "hbar rect2 col");
        expect_near(rphi, 0, "hbar rect2 phi");
        expect_near(len1, (length - 1) / 2.0, "hbar len1");
        expect_near(len2, 0, "hbar len2");
        double radius = -1;
        expect_status(smallest_circle(hreg, &row, &column, &radius), ThresholdStatus::Ok, "hbar circle");
        expect_near(radius, (length - 1) / 2.0, "hbar radius");
        int a = 0;
        int b = 0;
        int c = 0;
        int d = 0;
        double diameter = -1;
        expect_status(diameter_region(hreg, &a, &b, &c, &d, &diameter), ThresholdStatus::Ok, "hbar diameter");
        expect_near(diameter, length - 1, "hbar diameter");

        std::vector<runLength> vertical = rectangle(-2, 6, -2 + length - 1, 6);
        region vreg = from_runs(vertical);
        expect_status(elliptic_axis(vreg, &ra, &rb, &phi), ThresholdStatus::Ok, "vbar phi");
        expect_near(phi, length == 1 ? 0.0 : kPi / 2.0, "vbar phi");
        expect_near(rb, 0, "vbar rb");
        expect_status(smallest_rectangle2(vreg, &row, &column, &rphi, &len1, &len2), ThresholdStatus::Ok, "vbar rect2");
        expect_near(rphi, length == 1 ? 0.0 : kPi / 2.0, "vbar rect2 phi");
        expect_near(len1, (length - 1) / 2.0, "vbar len1");
        expect_near(len2, 0, "vbar len2");
    }
}

void check_closed_forms() {
    std::vector<runLength> square = rectangle(5, 5, 10, 10);
    region reg = from_runs(square);
    int64_t area = -1;
    double row = -1;
    double column = -1;
    expect_status(area_center(reg, &area, &row, &column), ThresholdStatus::Ok, "square area");
    expect(area == 36, "square area");
    expect_near(row, 7.5, "square row");
    expect_near(column, 7.5, "square column");
    int row1 = 0;
    int col1 = 0;
    int row2 = 0;
    int col2 = 0;
    expect_status(smallest_rectangle1(reg, &row1, &col1, &row2, &col2), ThresholdStatus::Ok, "square rect1");
    expect(row1 == 5 && col1 == 5 && row2 == 10 && col2 == 10, "square rect1");
    double circ = -1;
    expect_status(circularity(reg, &circ), ThresholdStatus::Ok, "square circ");
    expect_near(circ, 36.0 / (kPi * 12.5), "square circ");
    double length = -1;
    expect_status(contlength(reg, &length), ThresholdStatus::Ok, "square length");
    expect_near(length, 20, "square length");
    double compact = -1;
    expect_status(compactness(reg, &compact), ThresholdStatus::Ok, "square compact");
    expect_near(compact, 1, "square compact");
    double conv = -1;
    expect_status(convexity(reg, &conv), ThresholdStatus::Ok, "square conv");
    expect_near(conv, 1, "square conv");
    double rectangular = -1;
    expect_status(rectangularity(reg, &rectangular), ThresholdStatus::Ok, "square rectangular");
    expect_near(rectangular, 1, "square rectangular");
    double ra = -1;
    double rb = -1;
    double phi = 99;
    expect_status(elliptic_axis(reg, &ra, &rb, &phi), ThresholdStatus::Ok, "square axis");
    expect_near(ra, std::sqrt(35.0 / 3.0), "square ra");
    expect_near(rb, ra, "square rb");
    expect_near(phi, 0, "square phi");
    double anisometry = -1;
    double bulkiness = -1;
    double structure = -1;
    expect_status(eccentricity(reg, &anisometry, &bulkiness, &structure), ThresholdStatus::Ok, "square ecc");
    expect_near(anisometry, 1, "square anisometry");
    expect_near(bulkiness, kPi * ra * rb / 36.0, "square bulkiness");
    expect_near(structure, anisometry * bulkiness - 1.0, "square struct");
    double crow = -1;
    double ccol = -1;
    double radius = -1;
    expect_status(smallest_circle(reg, &crow, &ccol, &radius), ThresholdStatus::Ok, "square circle");
    expect_near(crow, 7.5, "square circle row");
    expect_near(ccol, 7.5, "square circle col");
    expect_near(radius, std::hypot(2.5, 2.5), "square radius");
    double rrow = -1;
    double rcol = -1;
    double rphi = 99;
    double len1 = -1;
    double len2 = -1;
    expect_status(smallest_rectangle2(reg, &rrow, &rcol, &rphi, &len1, &len2), ThresholdStatus::Ok, "square rect2");
    expect_near(rrow, 7.5, "square rect2 row");
    expect_near(rcol, 7.5, "square rect2 col");
    expect_near(len1, 2.5, "square len1");
    expect_near(len2, 2.5, "square len2");
    expect(near(rphi, 0.0) || near(std::abs(rphi), kPi / 2.0), "square rect2 phi");
    int d1 = 0;
    int d2 = 0;
    int d3 = 0;
    int d4 = 0;
    double diameter = -1;
    expect_status(diameter_region(reg, &d1, &d2, &d3, &d4, &diameter), ThresholdStatus::Ok, "square diameter");
    expect_near(diameter, std::hypot(5, 5), "square diameter");

    std::vector<runLength> wide = rectangle(0, 0, 1, 4);
    region wreg = from_runs(wide);
    double m11 = -1;
    double m20 = -1;
    double m02 = -1;
    double ia = -1;
    double ib = -1;
    expect_status(moments_region_2nd(wreg, &m11, &m20, &m02, &ia, &ib), ThresholdStatus::Ok, "2x5 moments");
    expect_near(m11, 0, "2x5 m11");
    expect_near(m20, 0.25, "2x5 m20");
    expect_near(m02, 2, "2x5 m02");
    expect_near(ia, 4, "2x5 ia");
    expect_near(ib, 0.5, "2x5 ib");
    expect_status(elliptic_axis(wreg, &ra, &rb, &phi), ThresholdStatus::Ok, "2x5 axis");
    expect_near(ra, std::sqrt(8.0), "2x5 ra");
    expect_near(rb, 1, "2x5 rb");
    expect_near(phi, 0, "2x5 phi");

    std::vector<runLength> bar = rectangle(0, 0, 0, 4);
    region breg = from_runs(bar);
    expect_status(moments_region_2nd(breg, &m11, &m20, &m02, &ia, &ib), ThresholdStatus::Ok, "1x5 moments");
    expect_near(m20, 0, "1x5 m20");
    expect_near(m02, 2, "1x5 m02");
    expect_status(elliptic_axis(breg, &ra, &rb, &phi), ThresholdStatus::Ok, "1x5 axis");
    expect_near(ra, std::sqrt(8.0), "1x5 ra");
    expect_near(rb, 0, "1x5 rb");
    expect_status(contlength(breg, &length), ThresholdStatus::Ok, "1x5 length");
    expect_near(length, 8, "1x5 length");
    expect_status(circularity(breg, &circ), ThresholdStatus::Ok, "1x5 circ");
    expect_near(circ, 5.0 / (kPi * 4.0), "1x5 circ");

    std::vector<runLength> big = rectangle(0, 0, 99, 99);
    region big_reg = from_runs(big);
    expect_status(contlength(big_reg, &length), ThresholdStatus::Ok, "100 length");
    expect_near(length, 2.0 * (100 + 100 - 2), "100 length");
    expect_status(compactness(big_reg, &compact), ThresholdStatus::Ok, "100 compact");
    const double raw = length * length / (4.0 * kPi * 10000.0);
    expect_near(compact, std::max(1.0, raw), "100 compact");
    expect(compact > 1.2 && compact < 1.3, "100 compact range");

    std::vector<runLength> shifted = rectangle(-3, -5, -1, -2);
    region sreg = from_runs(shifted);
    expect_status(area_center(sreg, &area, &row, &column), ThresholdStatus::Ok, "neg center");
    expect(area == 12, "neg area");
    expect_near(row, -2, "neg row");
    expect_near(column, -3.5, "neg col");
    expect_moments(sreg, "neg moments");
    expect_status(rectangularity(sreg, &rectangular), ThresholdStatus::Ok, "neg rectangular");
    expect_near(rectangular, 1, "neg rectangular");
}

void check_shapes() {
    std::vector<runLength> pixel = pixels_to_runs({{5, 7}});
    region preg = from_runs(pixel);
    int64_t area = -1;
    double row = -1;
    double column = -1;
    expect_status(area_center(preg, &area, &row, &column), ThresholdStatus::Ok, "pixel area");
    expect(area == 1 && row == 5 && column == 7, "pixel center");
    double length = -1;
    double circ = -1;
    double compact = -1;
    double conv = -1;
    double rectangular = -1;
    double ra = -1;
    double rb = -1;
    double phi = 99;
    expect_status(contlength(preg, &length), ThresholdStatus::Ok, "pixel length");
    expect_status(circularity(preg, &circ), ThresholdStatus::Ok, "pixel circ");
    expect_status(compactness(preg, &compact), ThresholdStatus::Ok, "pixel compact");
    expect_status(convexity(preg, &conv), ThresholdStatus::Ok, "pixel conv");
    expect_status(rectangularity(preg, &rectangular), ThresholdStatus::Ok, "pixel rectangular");
    expect_status(elliptic_axis(preg, &ra, &rb, &phi), ThresholdStatus::Ok, "pixel axis");
    expect(length == 0 && circ == 1 && compact == 1 && conv == 1 && rectangular == 1, "pixel features");
    expect(ra == 0 && rb == 0 && phi == 0, "pixel axis values");
    double radius = -1;
    expect_status(smallest_circle(preg, &row, &column, &radius), ThresholdStatus::Ok, "pixel circle");
    expect(radius == 0 && row == 5 && column == 7, "pixel circle values");

    std::vector<runLength> diag2 = pixels_to_runs({{0, 0}, {1, 1}});
    region dreg = from_runs(diag2);
    expect_status(contlength(dreg, &length), ThresholdStatus::Ok, "diag2 length");
    expect_near(length, 2.0 * std::sqrt(2.0), "diag2 length");
    expect_status(convexity(dreg, &conv), ThresholdStatus::Ok, "diag2 conv");
    expect_near(conv, 1, "diag2 conv");

    std::vector<runLength> diag3 = pixels_to_runs({{0, 0}, {1, 1}, {2, 2}});
    region diag3_reg = from_runs(diag3);
    expect_status(contlength(diag3_reg, &length), ThresholdStatus::Ok, "diag3 length");
    expect_near(length, 4.0 * std::sqrt(2.0), "diag3 length");

    std::vector<runLength> gap = pixels_to_runs({{0, 0}, {0, 2}});
    region greg = from_runs(gap);
    expect_status(area_center(greg, &area, &row, &column), ThresholdStatus::Ok, "gap area");
    expect(area == 2, "gap area");
    expect_status(contlength(greg, &length), ThresholdStatus::Ok, "gap length");
    expect_near(length, 0, "gap length");
    expect_status(convexity(greg, &conv), ThresholdStatus::Ok, "gap conv");
    expect_near(conv, 2.0 / 3.0, "gap conv");

    std::vector<runLength> skew = pixels_to_runs({{0, 0}, {2, 1}});
    region skreg = from_runs(skew);
    expect_status(convexity(skreg, &conv), ThresholdStatus::Ok, "skew conv");
    expect_near(conv, 1, "skew conv");
    expect_status(contlength(skreg, &length), ThresholdStatus::Ok, "skew length");
    expect_near(length, 0, "skew length");

    std::vector<runLength> lshape = pixels_to_runs({{0, 0}, {0, 1}, {0, 2}, {1, 0}, {2, 0}});
    region lreg = from_runs(lshape);
    expect_status(area_center(lreg, &area, &row, &column), ThresholdStatus::Ok, "L area");
    expect(area == 5, "L area");
    expect_status(convexity(lreg, &conv), ThresholdStatus::Ok, "L conv");
    expect_near(conv, 5.0 / 6.0, "L conv");
    expect_moments(lreg, "L moments");
    expect_catalog(lreg, "L catalog");

    std::vector<runLength> parts = rectangle(0, 0, 1, 1);
    std::vector<runLength> other = rectangle(5, 5, 6, 6);
    parts.insert(parts.end(), other.begin(), other.end());
    region preg2 = from_runs(parts);
    expect_status(contlength(preg2, &length), ThresholdStatus::Ok, "two squares");
    expect_near(length, 8, "two squares");

    auto frame_of = [](int outer, int hole1, int hole2) {
        std::vector<runLength> runs;
        for (int row = 0; row < outer; ++row) {
            if (row < hole1 || row > hole2) {
                runs.push_back(run_at(row, 0, outer - 1));
            } else {
                runs.push_back(run_at(row, 0, hole1 - 1));
                runs.push_back(run_at(row, hole2 + 1, outer - 1));
            }
        }
        return runs;
    };
    std::vector<runLength> frame = frame_of(10, 3, 6);
    region freg = from_runs(frame);
    expect_status(area_center(freg, &area, &row, &column), ThresholdStatus::Ok, "frame area");
    expect(area == 84, "frame area");
    expect_status(convexity(freg, &conv), ThresholdStatus::Ok, "frame conv");
    expect_near(conv, 0.84, "frame conv");
    expect_status(contlength(freg, &length), ThresholdStatus::Ok, "frame length");
    expect_near(length, 36, "frame length");
    int d1 = 0;
    int d2 = 0;
    int d3 = 0;
    int d4 = 0;
    double diameter = -1;
    expect_status(diameter_region(freg, &d1, &d2, &d3, &d4, &diameter), ThresholdStatus::Ok, "frame diameter");
    expect_near(diameter, std::hypot(9, 9), "frame diameter");

    std::vector<runLength> thin = frame_of(4, 1, 2);
    region treg = from_runs(thin);
    expect_status(area_center(treg, &area, &row, &column), ThresholdStatus::Ok, "thin area");
    expect(area == 12, "thin area");
    expect_status(contlength(treg, &length), ThresholdStatus::Ok, "thin length");
    expect_near(length, 12, "thin length");
    expect_status(convexity(treg, &conv), ThresholdStatus::Ok, "thin conv");
    expect_near(conv, 0.75, "thin conv");

    std::vector<runLength> wall = frame_of(8, 1, 6);
    region wreg = from_runs(wall);
    expect_status(area_center(wreg, &area, &row, &column), ThresholdStatus::Ok, "wall area");
    expect(area == 28, "wall area");
    expect_status(contlength(wreg, &length), ThresholdStatus::Ok, "wall length");
    expect_near(length, 28, "wall length");
    expect_status(convexity(wreg, &conv), ThresholdStatus::Ok, "wall conv");
    expect_near(conv, 28.0 / 64.0, "wall conv");

    std::vector<runLength> overlap;
    overlap.push_back(run_at(0, 0, 5));
    overlap.push_back(run_at(0, 3, 8));
    region oreg = from_runs(overlap);
    expect_status(area_center(oreg, &area, &row, &column), ThresholdStatus::Ok, "overlap");
    expect(area == 9, "overlap area");
    expect_near(column, 4, "overlap column");

    std::vector<runLength> disk;
    for (int r = -15; r <= 15; ++r) {
        int left = 16;
        int right = -16;
        for (int c = -15; c <= 15; ++c) {
            if (r * r + c * c <= 15 * 15) {
                left = std::min(left, c);
                right = std::max(right, c);
            }
        }
        if (left <= right) disk.push_back(run_at(r, left, right));
    }
    region disk_reg = from_runs(disk);
    double anisometry = -1;
    double bulkiness = -1;
    double structure = -1;
    expect_status(circularity(disk_reg, &circ), ThresholdStatus::Ok, "disk circ");
    expect_status(eccentricity(disk_reg, &anisometry, &bulkiness, &structure), ThresholdStatus::Ok, "disk ecc");
    expect_status(convexity(disk_reg, &conv), ThresholdStatus::Ok, "disk conv");
    expect(circ > 0.85, "disk circ range");
    expect(anisometry > 0.99 && anisometry < 1.15, "disk anisometry");
    expect(conv > 0.9, "disk conv range");
    std::printf("disk circularity=%.6f anisometry=%.6f convexity=%.6f\n", circ, anisometry, conv);

    expect_circle({{0, 0}}, "circle point");
    expect_circle({{0, 0}, {0, 4}}, "circle bar");
    expect_circle({{0, 0}, {0, 2}, {2, 0}}, "circle right");
    expect_circle({{0, 0}, {4, 0}, {1, 1}}, "circle obtuse");
    expect_circle({{0, 0}, {0, 3}, {3, 0}, {3, 3}}, "circle square");
    expect_circle({{0, 0}, {0, 1}, {0, 4}}, "circle collinear");
    std::vector<std::pair<int, int>> cloud = {{0, 0}, {1, 4}, {3, 1}, {4, 5}, {6, 2}, {2, 7}, {5, 6}};
    expect_circle(cloud, "circle cloud");
}

void check_invalid_and_select() {
    std::vector<runLength> empty;
    region ereg = from_runs(empty);
    int64_t area = 42;
    double row = 3;
    double column = 4;
    expect_status(area_center(ereg, &area, &row, &column), ThresholdStatus::Empty, "empty");
    expect(area == 0 && row == 0 && column == 0, "empty zeros");
    expect_catalog(ereg, "empty catalog");
    double values[4] = {-123, -123, -123, -123};
    const char* names[] = {"area", "row", "circularity", "contlength"};
    expect_status(region_features(ereg, names, 4, values), ThresholdStatus::Empty, "empty features");
    expect(values[0] == 0 && values[1] == 0 && values[2] == 0 && values[3] == 0, "empty feature zeros");

    area = 42;
    expect_status(area_center(ereg, &area, nullptr, &column), ThresholdStatus::InvalidArgument, "null out");
    expect(area == 42, "null out keeps value");
    expect_status(area_center(ereg, nullptr, &row, &column), ThresholdStatus::InvalidArgument, "null area");

    region bad{};
    bad.runLengthCount = -1;
    expect_status(area_center(bad, &area, &row, &column), ThresholdStatus::InvalidArgument, "negative count");
    bad.runLengthCount = 1;
    bad.pRle = nullptr;
    expect_status(area_center(bad, &area, &row, &column), ThresholdStatus::InvalidArgument, "null runs");

    std::vector<runLength> unsorted = {run_at(1, 0, 1), run_at(0, 0, 1)};
    region ureg = from_runs(unsorted);
    values[0] = -123;
    expect_status(region_features(ureg, names, 1, values), ThresholdStatus::InvalidArgument, "unsorted row");
    expect(values[0] == 0, "invalid geometry writes zero");
    std::vector<runLength> bad_cols = {run_at(0, 2, 1)};
    region creg = from_runs(bad_cols);
    expect_status(contlength(creg, &row), ThresholdStatus::InvalidArgument, "cole < cols");
    std::vector<runLength> bad_order = {run_at(0, 3, 4), run_at(0, 0, 1)};
    region oreg = from_runs(bad_order);
    expect_status(convexity(oreg, &row), ThresholdStatus::InvalidArgument, "unsorted cols");

    const char* unknown[] = {"area", "no_such_feature"};
    values[0] = -123;
    values[1] = -123;
    std::vector<runLength> pixel = pixels_to_runs({{1, 1}});
    region preg = from_runs(pixel);
    expect_status(region_features(preg, unknown, 2, values), ThresholdStatus::InvalidArgument, "unknown name");
    expect(values[0] == -123 && values[1] == -123, "unknown name writes nothing");

    std::vector<runLength> square = rectangle(5, 5, 10, 10);
    std::vector<runLength> bar = rectangle(0, 0, 0, 4);
    region regions[3] = {from_runs(pixel), from_runs(square), from_runs(bar)};
    const char* select_names[] = {"area", "circularity"};
    double mins[] = {1, 0.9};
    double maxs[] = {10, 1};
    int indices[4] = {7, 7, 7, 7};
    int selected = 9;
    expect_status(select_shape(regions, 3, select_names, 2, ShapeLogic::And, mins, maxs, indices, 4, &selected),
                  ThresholdStatus::Ok, "select and");
    expect(selected == 1 && indices[0] == 0 && indices[1] == 7, "select and indices");

    mins[0] = 30;
    mins[1] = 0.3;
    maxs[0] = 40;
    maxs[1] = 0.5;
    indices[0] = 7;
    indices[1] = 7;
    expect_status(select_shape(regions, 3, select_names, 2, ShapeLogic::Or, mins, maxs, indices, 4, &selected),
                  ThresholdStatus::Ok, "select or");
    expect(selected == 2 && indices[0] == 1 && indices[1] == 2, "select or indices");

    indices[0] = 7;
    indices[1] = 7;
    expect_status(select_shape(regions, 3, select_names, 2, ShapeLogic::Or, mins, maxs, indices, 1, &selected),
                  ThresholdStatus::BufferTooSmall, "select capacity");
    expect(selected == 2 && indices[0] == 7 && indices[1] == 7, "select capacity keeps indices");

    mins[0] = 1000;
    maxs[0] = 2000;
    expect_status(select_shape(regions, 3, select_names, 2, ShapeLogic::And, mins, maxs, indices, 4, &selected),
                  ThresholdStatus::Empty, "select none");
    expect(selected == 0 && indices[0] == 7, "select none count");

    selected = 9;
    expect_status(select_shape(regions, 3, select_names, 2, ShapeLogic::Or, mins, maxs, nullptr, 4, &selected),
                  ThresholdStatus::InvalidArgument, "select null indices");
    expect(selected == 1, "select null indices count");
    selected = 9;
    expect_status(select_shape(nullptr, -1, select_names, 2, ShapeLogic::And, mins, maxs, indices, 4, &selected),
                  ThresholdStatus::InvalidArgument, "select negative count");
    expect(selected == 9, "select negative keeps count");
    selected = 9;
    expect_status(select_shape(nullptr, 0, select_names, 2, ShapeLogic::And, mins, maxs, indices, 4, &selected),
                  ThresholdStatus::Empty, "select zero regions");
    expect(selected == 0, "select zero count");

    const char* bad_name[] = {"Area"};
    double one_min[] = {0};
    double one_max[] = {10};
    selected = 9;
    expect_status(select_shape(regions, 3, bad_name, 1, ShapeLogic::And, one_min, one_max, indices, 4, &selected),
                  ThresholdStatus::InvalidArgument, "select bad name");
    expect(selected == 0, "select bad name count");
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

template <typename Fn>
double microseconds(Fn&& fn, int iters) {
    for (int i = 0; i < 2; ++i) fn();
    const auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < iters; ++i) fn();
    const auto t1 = std::chrono::steady_clock::now();
    return std::chrono::duration<double, std::micro>(t1 - t0).count() / iters;
}

void check_bottle_and_speed(const char* path) {
    const Image image = load_png(path);
    expect(image.width == 640 && image.height == 512, "bottle size");
    const int capacity = threshold_max_runs(image.width, image.height);
    std::vector<runLength> runs(static_cast<size_t>(capacity));
    region reg{};
    expect_status(threshold_avx2(image.pixels.data(), image.width, image.height, image.width, 0, 0, image.width,
                                 image.height, 180, 255, &reg, runs.data(), capacity, 1),
                  ThresholdStatus::Ok, "bottle threshold");
    expect(reg.area == 6092, "bottle area");
    int64_t area = -1;
    double row = -1;
    double column = -1;
    expect_status(area_center(reg, &area, &row, &column), ThresholdStatus::Ok, "bottle center");
    expect(area == 6092, "bottle feature area");
    expect_near(row, reg.centerRow, "bottle row");
    expect_near(column, reg.centerCol, "bottle column");

    const int count = static_cast<int>(sizeof(kNames) / sizeof(kNames[0]));
    std::vector<double> values(static_cast<size_t>(count));
    expect_status(region_features(reg, kNames, count, values.data()), ThresholdStatus::Ok, "bottle features");
    for (int i = 0; i < count; ++i) expect(std::isfinite(values[static_cast<size_t>(i)]), kNames[i]);
    expect(feature_at(values.data(), "circularity") >= 0 && feature_at(values.data(), "circularity") <= 1, "bottle circ");
    expect(feature_at(values.data(), "compactness") >= 1, "bottle compact");
    expect(feature_at(values.data(), "convexity") > 0 && feature_at(values.data(), "convexity") <= 1, "bottle conv");
    expect(feature_at(values.data(), "rectangularity") >= 0 && feature_at(values.data(), "rectangularity") <= 1,
           "bottle rectangular");
    expect(feature_at(values.data(), "rect2_len1") + 1e-9 >= feature_at(values.data(), "rect2_len2"), "bottle len");
    std::printf(
        "bottle area=%.0f center=(%.6f,%.6f) circularity=%.6f compactness=%.6f convexity=%.6f rectangularity=%.6f\n",
        feature_at(values.data(), "area"), feature_at(values.data(), "row"), feature_at(values.data(), "column"),
        feature_at(values.data(), "circularity"), feature_at(values.data(), "compactness"),
        feature_at(values.data(), "convexity"), feature_at(values.data(), "rectangularity"));
    std::printf("bottle ra=%.6f rb=%.6f phi=%.6f contlength=%.6f outer_radius=%.6f diameter=%.6f\n",
                feature_at(values.data(), "ra"), feature_at(values.data(), "rb"), feature_at(values.data(), "phi"),
                feature_at(values.data(), "contlength"), feature_at(values.data(), "outer_radius"),
                feature_at(values.data(), "max_diameter"));

    const double bottle_us = microseconds([&] { g_sink += static_cast<int>(region_features(reg, kNames, count, values.data())); }, 30);
    std::printf("bottle all-features us/call %.1f\n", bottle_us);

    const int components_cap = reg.runLengthCount;
    std::vector<region> components(static_cast<size_t>(components_cap));
    std::vector<runLength> comp_runs(static_cast<size_t>(components_cap));
    int component_count = 0;
    expect_status(connection(reg, Neighborhood::Eight, components.data(), components_cap, comp_runs.data(), components_cap,
                             &component_count, 0, 1),
                  ThresholdStatus::Ok, "bottle connection");
    const char* area_name[] = {"area"};
    double min_area[] = {100};
    double max_area[] = {1e12};
    std::vector<int> indices(static_cast<size_t>(component_count), -1);
    int selected = -1;
    expect_status(select_shape(components.data(), component_count, area_name, 1, ShapeLogic::And, min_area, max_area,
                               indices.data(), component_count, &selected),
                  ThresholdStatus::Ok, "bottle select");
    std::vector<char> keep(static_cast<size_t>(component_count), 0);
    for (int i = 0; i < selected; ++i) keep[static_cast<size_t>(indices[static_cast<size_t>(i)])] = 1;
    int64_t largest_area = -1;
    int largest = 0;
    for (int i = 0; i < component_count; ++i) {
        int64_t comp_area = -1;
        double comp_row = 0;
        double comp_col = 0;
        expect_status(area_center(components[static_cast<size_t>(i)], &comp_area, &comp_row, &comp_col),
                      ThresholdStatus::Ok, "bottle component");
        expect((comp_area >= 100) == (keep[static_cast<size_t>(i)] != 0), "bottle select predicate");
        if (comp_area > largest_area) {
            largest_area = comp_area;
            largest = i;
        }
    }
    expect_status(region_features(components[static_cast<size_t>(largest)], kNames, count, values.data()), ThresholdStatus::Ok,
                  "largest component");
    std::printf("bottle components=%d selected area>=100: %d\n", component_count, selected);
    std::printf(
        "largest component area=%.0f %gx%g circularity=%.6f compactness=%.6f convexity=%.6f rectangularity=%.6f contlength=%.3f\n",
        feature_at(values.data(), "area"), feature_at(values.data(), "width"), feature_at(values.data(), "height"),
        feature_at(values.data(), "circularity"), feature_at(values.data(), "compactness"),
        feature_at(values.data(), "convexity"), feature_at(values.data(), "rectangularity"),
        feature_at(values.data(), "contlength"));

    std::vector<runLength> huge = rectangle(0, 0, 799, 1999);
    region hreg = from_runs(huge);
    expect_status(area_center(hreg, &area, &row, &column), ThresholdStatus::Ok, "huge area");
    expect(area == 1600000, "huge area");
    expect_near(row, 399.5, "huge row");
    expect_near(column, 999.5, "huge column");
    double rectangular = -1;
    double conv = -1;
    double length = -1;
    double rphi = 99;
    double len1 = -1;
    double len2 = -1;
    double rrow = 0;
    double rcol = 0;
    expect_status(rectangularity(hreg, &rectangular), ThresholdStatus::Ok, "huge rectangular");
    expect_status(convexity(hreg, &conv), ThresholdStatus::Ok, "huge conv");
    expect_status(contlength(hreg, &length), ThresholdStatus::Ok, "huge length");
    expect_status(smallest_rectangle2(hreg, &rrow, &rcol, &rphi, &len1, &len2), ThresholdStatus::Ok, "huge rect2");
    expect_near(rectangular, 1, "huge rectangular");
    expect_near(conv, 1, "huge conv");
    expect_near(length, 2.0 * (2000 + 800 - 2), "huge length");
    expect_near(rphi, 0, "huge phi");
    expect_near(len1, 1999 / 2.0, "huge len1");
    expect_near(len2, 799 / 2.0, "huge len2");
    const double huge_us = microseconds([&] { g_sink += static_cast<int>(region_features(hreg, kNames, count, values.data())); }, 10);
    std::printf("rect 2000x800 all-features us/call %.1f contlength=%.1f\n", huge_us, length);
}

}  // namespace

int main(int argc, char** argv) {
    const char* path = argc > 1 ? argv[1] : "img/bottle_crate_01.png";
    check_closed_forms();
    check_shapes();
    check_rectangle_family();
    check_invalid_and_select();
    std::vector<runLength> square = rectangle(5, 5, 10, 10);
    region reg = from_runs(square);
    expect_moments(reg, "square oracle");
    expect_catalog(reg, "square catalog");
    std::vector<runLength> pixel = pixels_to_runs({{5, 7}});
    region preg = from_runs(pixel);
    expect_catalog(preg, "pixel catalog");
    check_bottle_and_speed(path);
    std::printf("%s\n", g_failures == 0 ? "PASS" : "FAIL");
    return g_failures == 0 ? 0 : 1;
}
