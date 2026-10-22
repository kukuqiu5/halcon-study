#include "features.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <numeric>
#include <vector>

namespace {

constexpr double kPi = 3.14159265358979323846;
constexpr double kSqrt2 = 1.4142135623730950488;
constexpr double kEps = 1e-9;

struct Pt {
    int r = 0;
    int c = 0;
};

struct Scratch {
    std::vector<runLength> canon;
    std::vector<runLength> comp_runs;
    std::vector<region> comps;
    std::vector<Pt> points;
    std::vector<Pt> hull;
    std::vector<int> span_row;
    std::vector<int> span_begin;
};

Scratch& scratch() {
    thread_local Scratch local;
    return local;
}

double norm_phi(double phi) {
    while (phi <= -kPi / 2.0) phi += kPi;
    while (phi > kPi / 2.0) phi -= kPi;
    return phi;
}

int64_t sum_squares_range(int64_t a, int64_t b) {
    auto prefix = [](int64_t n) -> int64_t {
        if (n <= 0) return 0;
        return n * (n + 1) * (2 * n + 1) / 6;
    };
    if (a > b) return 0;
    if (a >= 0) return prefix(b) - prefix(a - 1);
    if (b < 0) return prefix(-a) - prefix(-b - 1);
    return prefix(-a) + prefix(b);
}

ThresholdStatus canonicalize(const region& reg, std::vector<runLength>& out) {
    out.clear();
    if (reg.runLengthCount < 0) return ThresholdStatus::InvalidArgument;
    if (reg.runLengthCount == 0) return ThresholdStatus::Empty;
    if (reg.pRle == nullptr) return ThresholdStatus::InvalidArgument;
    const runLength* runs = reg.pRle;
    const int n = reg.runLengthCount;
    for (int i = 0; i < n; ++i) {
        const runLength run = runs[i];
        if (run.cole < run.cols) return ThresholdStatus::InvalidArgument;
        if (i > 0) {
            const runLength prev = runs[i - 1];
            if (run.row < prev.row || (run.row == prev.row && run.cols < prev.cols)) {
                return ThresholdStatus::InvalidArgument;
            }
        }
        if (!out.empty() && out.back().row == run.row && run.cols <= out.back().cole + 1) {
            if (run.cole > out.back().cole) out.back().cole = run.cole;
        } else {
            out.push_back(run);
        }
    }
    if (out.empty()) return ThresholdStatus::Empty;
    return ThresholdStatus::Ok;
}

struct Basic {
    int64_t area = 0;
    double row = 0;
    double column = 0;
    int row1 = 0;
    int col1 = 0;
    int row2 = 0;
    int col2 = 0;
    double m11 = 0;
    double m20 = 0;
    double m02 = 0;
    double ia = 0;
    double ib = 0;
    double ra = 0;
    double rb = 0;
    double phi = 0;
    double max_dist = 0;
};

Basic measure_basic(const std::vector<runLength>& runs) {
    Basic out;
    int64_t sum_r = 0;
    int64_t sum_c = 0;
    int64_t sum_r2 = 0;
    int64_t sum_c2 = 0;
    int64_t sum_rc = 0;
    out.row1 = runs.front().row;
    out.row2 = runs.back().row;
    out.col1 = runs.front().cols;
    out.col2 = runs.front().cole;
    for (const runLength& run : runs) {
        const int64_t cb = run.cols;
        const int64_t ce = run.cole;
        const int64_t len = ce - cb + 1;
        const int64_t row = run.row;
        const int64_t sum_cols = len * (cb + ce) / 2;
        out.area += len;
        sum_r += len * row;
        sum_c += sum_cols;
        sum_r2 += len * row * row;
        sum_c2 += sum_squares_range(cb, ce);
        sum_rc += row * sum_cols;
        if (run.cols < out.col1) out.col1 = run.cols;
        if (run.cole > out.col2) out.col2 = run.cole;
    }
    const double area = static_cast<double>(out.area);
    out.row = static_cast<double>(sum_r) / area;
    out.column = static_cast<double>(sum_c) / area;
    const double mu20 = static_cast<double>(sum_r2) - static_cast<double>(sum_r) * out.row;
    const double mu02 = static_cast<double>(sum_c2) - static_cast<double>(sum_c) * out.column;
    const double mu11 = static_cast<double>(sum_rc) - static_cast<double>(sum_r) * out.column;
    out.m20 = mu20 / area;
    out.m02 = mu02 / area;
    out.m11 = mu11 / area;
    const double diff = out.m20 - out.m02;
    const double root = std::sqrt(diff * diff + 4.0 * out.m11 * out.m11);
    out.ia = out.m20 + out.m02 + root;
    out.ib = out.m20 + out.m02 - root;
    if (out.ib < 0 && out.ib > -1e-12) out.ib = 0;
    out.ra = std::sqrt(std::max(0.0, 2.0 * out.ia));
    out.rb = std::sqrt(std::max(0.0, 2.0 * out.ib));
    out.phi = norm_phi(0.5 * std::atan2(2.0 * out.m11, out.m02 - out.m20));
    for (const runLength& run : runs) {
        const double dr0 = run.row - out.row;
        const double left = run.cols - out.column;
        const double right = run.cole - out.column;
        out.max_dist = std::max(out.max_dist, std::max(std::hypot(dr0, left), std::hypot(dr0, right)));
    }
    return out;
}

long long cross(Pt o, Pt a, Pt b) {
    return static_cast<long long>(a.c - o.c) * (b.r - o.r) - static_cast<long long>(a.r - o.r) * (b.c - o.c);
}

void build_hull(const std::vector<runLength>& runs, std::vector<Pt>& points, std::vector<Pt>& hull) {
    points.clear();
    points.reserve(static_cast<size_t>(runs.size()) * 2);
    for (const runLength& run : runs) {
        points.push_back(Pt{run.row, run.cols});
        if (run.cole != run.cols) points.push_back(Pt{run.row, run.cole});
    }
    std::sort(points.begin(), points.end(), [](Pt a, Pt b) { return a.c < b.c || (a.c == b.c && a.r < b.r); });
    points.erase(std::unique(points.begin(), points.end(), [](Pt a, Pt b) { return a.r == b.r && a.c == b.c; }),
                 points.end());
    hull.clear();
    if (points.size() <= 1) {
        hull = points;
        return;
    }
    hull.reserve(points.size() + 2);
    for (Pt p : points) {
        while (hull.size() >= 2 && cross(hull[hull.size() - 2], hull[hull.size() - 1], p) <= 0) hull.pop_back();
        hull.push_back(p);
    }
    const size_t lower = hull.size();
    for (int i = static_cast<int>(points.size()) - 2; i >= 0; --i) {
        Pt p = points[static_cast<size_t>(i)];
        while (hull.size() > lower && cross(hull[hull.size() - 2], hull[hull.size() - 1], p) <= 0) hull.pop_back();
        hull.push_back(p);
    }
    if (!hull.empty()) hull.pop_back();
}

int64_t hull_pixel_area(const std::vector<Pt>& hull) {
    if (hull.empty()) return 0;
    if (hull.size() == 1) return 1;
    if (hull.size() == 2) {
        const int dr = std::abs(hull[1].r - hull[0].r);
        const int dc = std::abs(hull[1].c - hull[0].c);
        return static_cast<int64_t>(std::gcd(dr, dc)) + 1;
    }
    int row1 = hull[0].r;
    int row2 = hull[0].r;
    for (Pt p : hull) {
        row1 = std::min(row1, p.r);
        row2 = std::max(row2, p.r);
    }
    int64_t count = 0;
    const int n = static_cast<int>(hull.size());
    for (int row = row1; row <= row2; ++row) {
        double left = 1e300;
        double right = -1e300;
        for (int i = 0; i < n; ++i) {
            const Pt a = hull[static_cast<size_t>(i)];
            const Pt b = hull[static_cast<size_t>((i + 1) % n)];
            if (a.r == b.r) {
                if (a.r == row) {
                    left = std::min(left, static_cast<double>(std::min(a.c, b.c)));
                    right = std::max(right, static_cast<double>(std::max(a.c, b.c)));
                }
                continue;
            }
            const int lo = std::min(a.r, b.r);
            const int hi = std::max(a.r, b.r);
            if (row < lo || row > hi) continue;
            const double t = static_cast<double>(row - a.r) / static_cast<double>(b.r - a.r);
            const double x = a.c + t * (b.c - a.c);
            left = std::min(left, x);
            right = std::max(right, x);
        }
        if (left > right) continue;
        const int c0 = static_cast<int>(std::ceil(left - 1e-8));
        const int c1 = static_cast<int>(std::floor(right + 1e-8));
        if (c0 <= c1) count += static_cast<int64_t>(c1 - c0 + 1);
    }
    return count;
}

struct Rect2 {
    double row = 0;
    double column = 0;
    double phi = 0;
    double len1 = 0;
    double len2 = 0;
};

Rect2 min_area_rect(const std::vector<Pt>& hull) {
    Rect2 best;
    if (hull.empty()) return best;
    if (hull.size() == 1) {
        best.row = hull[0].r;
        best.column = hull[0].c;
        return best;
    }
    double best_area = 1e300;
    const int n = static_cast<int>(hull.size());
    for (int i = 0; i < n; ++i) {
        const Pt a = hull[static_cast<size_t>(i)];
        const Pt b = hull[static_cast<size_t>((i + 1) % n)];
        const double dx = b.c - a.c;
        const double dy = b.r - a.r;
        const double elen = std::hypot(dx, dy);
        if (elen <= kEps) continue;
        const double ux = dx / elen;
        const double uy = dy / elen;
        const double vx = -uy;
        const double vy = ux;
        double min_u = 1e300;
        double max_u = -1e300;
        double min_v = 1e300;
        double max_v = -1e300;
        for (Pt p : hull) {
            const double u = p.c * ux + p.r * uy;
            const double v = p.c * vx + p.r * vy;
            min_u = std::min(min_u, u);
            max_u = std::max(max_u, u);
            min_v = std::min(min_v, v);
            max_v = std::max(max_v, v);
        }
        const double width = max_u - min_u;
        const double height = max_v - min_v;
        const double area = width * height;
        if (area + 1e-9 >= best_area) continue;
        best_area = area;
        const double cu = (min_u + max_u) / 2.0;
        const double cv = (min_v + max_v) / 2.0;
        best.column = cu * ux + cv * vx;
        best.row = cu * uy + cv * vy;
        if (height > width) {
            best.phi = norm_phi(std::atan2(vy, vx));
            best.len1 = height / 2.0;
            best.len2 = width / 2.0;
        } else {
            best.phi = norm_phi(std::atan2(uy, ux));
            best.len1 = width / 2.0;
            best.len2 = height / 2.0;
        }
    }
    return best;
}

struct Circle {
    double row = 0;
    double column = 0;
    double radius = 0;
};

Circle circle_two(Pt a, Pt b) {
    Circle out;
    out.row = (a.r + b.r) / 2.0;
    out.column = (a.c + b.c) / 2.0;
    out.radius = std::hypot(a.r - b.r, a.c - b.c) / 2.0;
    return out;
}

Circle circle_three(Pt a, Pt b, Pt c) {
    const double d = 2.0 * (a.c * (b.r - c.r) + b.c * (c.r - a.r) + c.c * (a.r - b.r));
    Circle ab = circle_two(a, b);
    Circle bc = circle_two(b, c);
    Circle ac = circle_two(a, c);
    if (std::abs(d) < 1e-12) {
        if (ab.radius >= bc.radius && ab.radius >= ac.radius) return ab;
        return bc.radius >= ac.radius ? bc : ac;
    }
    const double a2 = static_cast<double>(a.c) * a.c + static_cast<double>(a.r) * a.r;
    const double b2 = static_cast<double>(b.c) * b.c + static_cast<double>(b.r) * b.r;
    const double c2 = static_cast<double>(c.c) * c.c + static_cast<double>(c.r) * c.r;
    Circle out;
    out.column = (a2 * (b.r - c.r) + b2 * (c.r - a.r) + c2 * (a.r - b.r)) / d;
    out.row = (a2 * (c.c - b.c) + b2 * (a.c - c.c) + c2 * (b.c - a.c)) / d;
    out.radius = std::hypot(out.row - a.r, out.column - a.c);
    return out;
}

bool circle_contains(Circle circle, Pt p) {
    return std::hypot(p.r - circle.row, p.c - circle.column) <= circle.radius + 1e-8;
}

Circle enclosing_circle(std::vector<Pt> points) {
    Circle circle;
    if (points.empty()) return circle;
    uint32_t state = 1;
    for (int i = static_cast<int>(points.size()) - 1; i > 0; --i) {
        state = state * 1664525u + 1013904223u;
        const int j = static_cast<int>(state % static_cast<uint32_t>(i + 1));
        std::swap(points[static_cast<size_t>(i)], points[static_cast<size_t>(j)]);
    }
    circle.row = points[0].r;
    circle.column = points[0].c;
    for (size_t i = 0; i < points.size(); ++i) {
        if (circle_contains(circle, points[i])) continue;
        circle = Circle{static_cast<double>(points[i].r), static_cast<double>(points[i].c), 0};
        for (size_t j = 0; j < i; ++j) {
            if (circle_contains(circle, points[j])) continue;
            circle = circle_two(points[i], points[j]);
            for (size_t k = 0; k < j; ++k) {
                if (circle_contains(circle, points[k])) continue;
                const Circle next = circle_three(points[i], points[j], points[k]);
                if (circle_contains(next, points[i]) && circle_contains(next, points[j])) circle = next;
            }
        }
    }
    return circle;
}

struct Diameter {
    Pt a;
    Pt b;
    double distance = 0;
};

Diameter max_diameter(const std::vector<Pt>& hull) {
    Diameter best;
    for (size_t i = 0; i < hull.size(); ++i) {
        for (size_t j = i + 1; j < hull.size(); ++j) {
            const double dist = std::hypot(hull[i].r - hull[j].r, hull[i].c - hull[j].c);
            if (dist > best.distance) {
                best.distance = dist;
                best.a = hull[i];
                best.b = hull[j];
            }
        }
    }
    if (hull.size() == 1) {
        best.a = hull[0];
        best.b = hull[0];
    }
    return best;
}

struct Ival {
    int lo = 0;
    int hi = -1;
    bool any = false;
};

Ival solve_abs(double scale, double bias, double limit) {
    Ival out;
    if (limit < -kEps) return out;
    if (std::abs(scale) < 1e-15) {
        if (std::abs(bias) <= limit + 1e-8) out.any = true;
        return out;
    }
    double lo = (-limit - bias) / scale;
    double hi = (limit - bias) / scale;
    if (lo > hi) std::swap(lo, hi);
    out.lo = static_cast<int>(std::ceil(lo - 1e-8));
    out.hi = static_cast<int>(std::floor(hi + 1e-8));
    return out;
}

Ival intersect_ival(Ival a, Ival b) {
    if (a.any) return b;
    if (b.any) return a;
    Ival out;
    out.lo = std::max(a.lo, b.lo);
    out.hi = std::min(a.hi, b.hi);
    return out;
}

Ival columns_in_rect(int row, double center_row, double center_col, double phi, double half_major, double half_minor) {
    double cs = std::cos(phi);
    double sn = std::sin(phi);
    // phi 取到 π/2 时，库函数给出的 cos 不是精确 0。轴对齐的约束按 0 处理。
    if (std::abs(cs) < 1e-12) cs = 0;
    if (std::abs(sn) < 1e-12) sn = 0;
    const double dr = row - center_row;
    const double bias_x = -center_col * cs + dr * sn;
    const double bias_y = center_col * sn + dr * cs;
    return intersect_ival(solve_abs(cs, bias_x, half_major), solve_abs(-sn, bias_y, half_minor));
}

double measure_rectangularity(const std::vector<runLength>& runs, const Basic& basic) {
    const double half_major = std::sqrt(3.0) * basic.ra / 2.0;
    const double half_minor = std::sqrt(3.0) * basic.rb / 2.0;
    const double cs = std::cos(basic.phi);
    const double sn = std::sin(basic.phi);
    double row_min = 1e300;
    double row_max = -1e300;
    for (int s1 = -1; s1 <= 1; s1 += 2) {
        for (int s2 = -1; s2 <= 1; s2 += 2) {
            const double row = basic.row + s1 * half_major * sn + s2 * half_minor * cs;
            row_min = std::min(row_min, row);
            row_max = std::max(row_max, row);
        }
    }
    int64_t rect_pixels = 0;
    const int r0 = static_cast<int>(std::floor(row_min - 1.0));
    const int r1 = static_cast<int>(std::ceil(row_max + 1.0));
    for (int row = r0; row <= r1; ++row) {
        const Ival span = columns_in_rect(row, basic.row, basic.column, basic.phi, half_major, half_minor);
        if (!span.any && span.lo <= span.hi) rect_pixels += static_cast<int64_t>(span.hi - span.lo + 1);
    }
    int64_t inside = 0;
    for (const runLength& run : runs) {
        const Ival span = columns_in_rect(run.row, basic.row, basic.column, basic.phi, half_major, half_minor);
        if (span.any) {
            inside += run.cole - run.cols + 1;
            continue;
        }
        const int lo = std::max(span.lo, static_cast<int>(run.cols));
        const int hi = std::min(span.hi, static_cast<int>(run.cole));
        if (lo <= hi) inside += static_cast<int64_t>(hi - lo + 1);
    }
    if (rect_pixels <= 0) return 0;
    const double value = (2.0 * static_cast<double>(inside) - static_cast<double>(basic.area)) / static_cast<double>(rect_pixels);
    if (value < 0) return 0;
    if (value > 1) return 1;
    return value;
}

void build_spans(const runLength* runs, int count, std::vector<int>& rows, std::vector<int>& begin) {
    rows.clear();
    begin.clear();
    for (int i = 0; i < count; ++i) {
        if (i == 0 || runs[i].row != runs[i - 1].row) {
            rows.push_back(runs[i].row);
            begin.push_back(i);
        }
    }
    begin.push_back(count);
}

bool contains_pixel(const runLength* runs, const std::vector<int>& rows, const std::vector<int>& begin, int row, int col) {
    int lo = 0;
    int hi = static_cast<int>(rows.size());
    while (lo < hi) {
        const int mid = (lo + hi) / 2;
        if (rows[static_cast<size_t>(mid)] < row) lo = mid + 1;
        else hi = mid;
    }
    if (lo == static_cast<int>(rows.size()) || rows[static_cast<size_t>(lo)] != row) return false;
    int a = begin[static_cast<size_t>(lo)];
    int b = begin[static_cast<size_t>(lo + 1)];
    while (a < b) {
        const int mid = (a + b) / 2;
        if (runs[mid].cole < col) a = mid + 1;
        else b = mid;
    }
    return a < begin[static_cast<size_t>(lo + 1)] && runs[a].cols <= col && col <= runs[a].cole;
}

double trace_outer(const runLength* runs, int count, int64_t area) {
    if (count <= 0 || area <= 0) return 0;
    Scratch& work = scratch();
    build_spans(runs, count, work.span_row, work.span_begin);
    const int sr = runs[0].row;
    const int sc = runs[0].cols;
    static const int dr[8] = {0, 1, 1, 1, 0, -1, -1, -1};
    static const int dc[8] = {1, 1, 0, -1, -1, -1, 0, 1};
    int r = sr;
    int c = sc;
    int back = 4;
    int first_r = 0;
    int first_c = 0;
    bool has_first = false;
    double length = 0;
    const int limit = static_cast<int>(std::min<int64_t>(area * 8 + 16, 2000000000));
    for (int step = 0; step < limit; ++step) {
        bool found = false;
        int nr = r;
        int nc = c;
        int nd = 0;
        for (int k = 1; k <= 8; ++k) {
            nd = (back + k) % 8;
            nr = r + dr[nd];
            nc = c + dc[nd];
            if (contains_pixel(runs, work.span_row, work.span_begin, nr, nc)) {
                found = true;
                break;
            }
        }
        if (!found) return 0;
        if (has_first && r == sr && c == sc && nr == first_r && nc == first_c) break;
        length += (dr[nd] == 0 || dc[nd] == 0) ? 1.0 : kSqrt2;
        if (!has_first) {
            first_r = nr;
            first_c = nc;
            has_first = true;
        }
        back = (nd + 4) % 8;
        r = nr;
        c = nc;
    }
    return length;
}

double measure_contlength(const std::vector<runLength>& runs) {
    const int n = static_cast<int>(runs.size());
    if (n == 0) return 0;
    Scratch& work = scratch();
    work.comps.resize(static_cast<size_t>(n));
    work.comp_runs.resize(static_cast<size_t>(n));
    int count = 0;
    const ThresholdStatus status = connection(runs.data(), n, Neighborhood::Eight, work.comps.data(), n, work.comp_runs.data(),
                                              n, &count, 0, 1);
    if (status != ThresholdStatus::Ok || count <= 0) return 0;
    double length = 0;
    for (int i = 0; i < count; ++i) {
        const region& comp = work.comps[static_cast<size_t>(i)];
        int64_t comp_area = 0;
        for (int k = 0; k < comp.runLengthCount; ++k) {
            comp_area += static_cast<int64_t>(comp.pRle[k].cole) - comp.pRle[k].cols + 1;
        }
        length += trace_outer(comp.pRle, comp.runLengthCount, comp_area);
    }
    return length;
}

enum class Feat : int {
    Area,
    Row,
    Column,
    Row1,
    Column1,
    Row2,
    Column2,
    Width,
    Height,
    Ratio,
    Circularity,
    Compactness,
    ContLength,
    Convexity,
    Rectangularity,
    Ra,
    Rb,
    Phi,
    Orientation,
    Anisometry,
    Bulkiness,
    StructFactor,
    OuterRadius,
    MaxDiameter,
    Rect2Phi,
    Rect2Len1,
    Rect2Len2,
    M11,
    M20,
    M02,
    Ia,
    Ib,
    Count
};

struct Shape {
    Basic basic;
    double contlength = 0;
    double circularity = 0;
    double compactness = 0;
    double convexity = 0;
    double rectangularity = 0;
    double anisometry = 0;
    double bulkiness = 0;
    double struct_factor = 0;
    Rect2 rect2;
    Circle circle;
    Diameter diameter;
    bool empty = false;
};

Feat parse_feature(const char* name) {
    if (name == nullptr) return Feat::Count;
    struct Item {
        const char* name;
        Feat id;
    };
    static const Item items[] = {
        {"area", Feat::Area},
        {"row", Feat::Row},
        {"column", Feat::Column},
        {"row1", Feat::Row1},
        {"column1", Feat::Column1},
        {"row2", Feat::Row2},
        {"column2", Feat::Column2},
        {"width", Feat::Width},
        {"height", Feat::Height},
        {"ratio", Feat::Ratio},
        {"circularity", Feat::Circularity},
        {"compactness", Feat::Compactness},
        {"contlength", Feat::ContLength},
        {"convexity", Feat::Convexity},
        {"rectangularity", Feat::Rectangularity},
        {"ra", Feat::Ra},
        {"rb", Feat::Rb},
        {"phi", Feat::Phi},
        {"orientation", Feat::Orientation},
        {"anisometry", Feat::Anisometry},
        {"bulkiness", Feat::Bulkiness},
        {"struct_factor", Feat::StructFactor},
        {"outer_radius", Feat::OuterRadius},
        {"max_diameter", Feat::MaxDiameter},
        {"rect2_phi", Feat::Rect2Phi},
        {"rect2_len1", Feat::Rect2Len1},
        {"rect2_len2", Feat::Rect2Len2},
        {"moments_m11", Feat::M11},
        {"moments_m20", Feat::M20},
        {"moments_m02", Feat::M02},
        {"moments_ia", Feat::Ia},
        {"moments_ib", Feat::Ib},
    };
    for (const Item& item : items) {
        if (std::strcmp(item.name, name) == 0) return item.id;
    }
    return Feat::Count;
}

struct Need {
    bool contour = false;
    bool hull = false;
    bool rectangularity = false;
    bool circle = false;
};

Need need_of(Feat feat) {
    Need need;
    if (feat == Feat::ContLength || feat == Feat::Compactness) need.contour = true;
    if (feat == Feat::Convexity || feat == Feat::Rect2Phi || feat == Feat::Rect2Len1 || feat == Feat::Rect2Len2 ||
        feat == Feat::MaxDiameter) {
        need.hull = true;
    }
    if (feat == Feat::OuterRadius) need.circle = true;
    if (feat == Feat::Rectangularity) need.rectangularity = true;
    return need;
}

void fill_derived(Shape& shape) {
    const Basic& basic = shape.basic;
    if (basic.max_dist <= kEps) {
        shape.circularity = basic.area > 0 ? 1.0 : 0.0;
    } else {
        const double raw = static_cast<double>(basic.area) / (kPi * basic.max_dist * basic.max_dist);
        shape.circularity = std::min(1.0, raw);
    }
    if (basic.area <= 0) {
        shape.compactness = 0;
    } else {
        const double raw = shape.contlength * shape.contlength / (4.0 * kPi * static_cast<double>(basic.area));
        shape.compactness = std::max(1.0, raw);
    }
    if (basic.rb <= kEps) {
        shape.anisometry = 0;
        shape.bulkiness = 0;
        shape.struct_factor = 0;
    } else {
        shape.anisometry = basic.ra / basic.rb;
        shape.bulkiness = kPi * basic.ra * basic.rb / static_cast<double>(basic.area);
        shape.struct_factor = shape.anisometry * shape.bulkiness - 1.0;
    }
}

Shape analyze(const std::vector<runLength>& runs, Need need) {
    Shape shape;
    if (runs.empty()) {
        shape.empty = true;
        return shape;
    }
    shape.basic = measure_basic(runs);
    if (need.contour) shape.contlength = measure_contlength(runs);
    fill_derived(shape);
    if (need.hull || need.circle) {
        Scratch& work = scratch();
        build_hull(runs, work.points, work.hull);
        if (need.hull) {
            const int64_t hull_area = std::max<int64_t>(hull_pixel_area(work.hull), 1);
            shape.convexity = std::min(1.0, static_cast<double>(shape.basic.area) / static_cast<double>(hull_area));
            shape.rect2 = min_area_rect(work.hull);
            shape.diameter = max_diameter(work.hull);
        }
        if (need.circle) shape.circle = enclosing_circle(work.hull);
    }
    if (need.rectangularity) shape.rectangularity = measure_rectangularity(runs, shape.basic);
    return shape;
}

double feature_value(const Shape& shape, Feat feat) {
    const Basic& basic = shape.basic;
    const double width = basic.area == 0 ? 0 : static_cast<double>(basic.col2 - basic.col1 + 1);
    const double height = basic.area == 0 ? 0 : static_cast<double>(basic.row2 - basic.row1 + 1);
    switch (feat) {
        case Feat::Area: return static_cast<double>(basic.area);
        case Feat::Row: return basic.row;
        case Feat::Column: return basic.column;
        case Feat::Row1: return basic.row1;
        case Feat::Column1: return basic.col1;
        case Feat::Row2: return basic.row2;
        case Feat::Column2: return basic.col2;
        case Feat::Width: return width;
        case Feat::Height: return height;
        case Feat::Ratio: return width == 0 ? 0 : height / width;
        case Feat::Circularity: return shape.circularity;
        case Feat::Compactness: return shape.compactness;
        case Feat::ContLength: return shape.contlength;
        case Feat::Convexity: return shape.convexity;
        case Feat::Rectangularity: return shape.rectangularity;
        case Feat::Ra: return basic.ra;
        case Feat::Rb: return basic.rb;
        case Feat::Phi: return basic.phi;
        case Feat::Orientation: return basic.phi;
        case Feat::Anisometry: return shape.anisometry;
        case Feat::Bulkiness: return shape.bulkiness;
        case Feat::StructFactor: return shape.struct_factor;
        case Feat::OuterRadius: return shape.circle.radius;
        case Feat::MaxDiameter: return shape.diameter.distance;
        case Feat::Rect2Phi: return shape.rect2.phi;
        case Feat::Rect2Len1: return shape.rect2.len1;
        case Feat::Rect2Len2: return shape.rect2.len2;
        case Feat::M11: return basic.m11;
        case Feat::M20: return basic.m20;
        case Feat::M02: return basic.m02;
        case Feat::Ia: return basic.ia;
        case Feat::Ib: return basic.ib;
        case Feat::Count: break;
    }
    return 0;
}

ThresholdStatus load_shape(const region& reg, Need need, Shape* shape) {
    Scratch& work = scratch();
    const ThresholdStatus status = canonicalize(reg, work.canon);
    if (status != ThresholdStatus::Ok) {
        *shape = Shape{};
        shape->empty = true;
        return status;
    }
    *shape = analyze(work.canon, need);
    return ThresholdStatus::Ok;
}

template <typename Fn>
ThresholdStatus with_shape(const region& reg, Need need, Fn&& fn) {
    Shape shape;
    const ThresholdStatus status = load_shape(reg, need, &shape);
    if (status != ThresholdStatus::Ok) {
        fn(shape);
        return status;
    }
    fn(shape);
    return ThresholdStatus::Ok;
}

}  // namespace

ThresholdStatus area_center(const region& reg, int64_t* area, double* row, double* column) {
    if (area == nullptr || row == nullptr || column == nullptr) return ThresholdStatus::InvalidArgument;
    return with_shape(reg, Need{}, [&](const Shape& shape) {
        *area = shape.basic.area;
        *row = shape.basic.row;
        *column = shape.basic.column;
    });
}

ThresholdStatus smallest_rectangle1(const region& reg, int* row1, int* column1, int* row2, int* column2) {
    if (row1 == nullptr || column1 == nullptr || row2 == nullptr || column2 == nullptr) return ThresholdStatus::InvalidArgument;
    return with_shape(reg, Need{}, [&](const Shape& shape) {
        *row1 = shape.basic.row1;
        *column1 = shape.basic.col1;
        *row2 = shape.basic.row2;
        *column2 = shape.basic.col2;
    });
}

ThresholdStatus smallest_rectangle2(const region& reg, double* row, double* column, double* phi, double* length1,
                                    double* length2) {
    if (row == nullptr || column == nullptr || phi == nullptr || length1 == nullptr || length2 == nullptr) {
        return ThresholdStatus::InvalidArgument;
    }
    Need need;
    need.hull = true;
    return with_shape(reg, need, [&](const Shape& shape) {
        *row = shape.rect2.row;
        *column = shape.rect2.column;
        *phi = shape.rect2.phi;
        *length1 = shape.rect2.len1;
        *length2 = shape.rect2.len2;
    });
}

ThresholdStatus smallest_circle(const region& reg, double* row, double* column, double* radius) {
    if (row == nullptr || column == nullptr || radius == nullptr) return ThresholdStatus::InvalidArgument;
    Need need;
    need.circle = true;
    return with_shape(reg, need, [&](const Shape& shape) {
        *row = shape.circle.row;
        *column = shape.circle.column;
        *radius = shape.circle.radius;
    });
}

ThresholdStatus moments_region_2nd(const region& reg, double* m11, double* m20, double* m02, double* ia, double* ib) {
    if (m11 == nullptr || m20 == nullptr || m02 == nullptr || ia == nullptr || ib == nullptr) {
        return ThresholdStatus::InvalidArgument;
    }
    return with_shape(reg, Need{}, [&](const Shape& shape) {
        *m11 = shape.basic.m11;
        *m20 = shape.basic.m20;
        *m02 = shape.basic.m02;
        *ia = shape.basic.ia;
        *ib = shape.basic.ib;
    });
}

ThresholdStatus elliptic_axis(const region& reg, double* ra, double* rb, double* phi) {
    if (ra == nullptr || rb == nullptr || phi == nullptr) return ThresholdStatus::InvalidArgument;
    return with_shape(reg, Need{}, [&](const Shape& shape) {
        *ra = shape.basic.ra;
        *rb = shape.basic.rb;
        *phi = shape.basic.phi;
    });
}

ThresholdStatus eccentricity(const region& reg, double* anisometry, double* bulkiness, double* structure_factor) {
    if (anisometry == nullptr || bulkiness == nullptr || structure_factor == nullptr) return ThresholdStatus::InvalidArgument;
    return with_shape(reg, Need{}, [&](const Shape& shape) {
        *anisometry = shape.anisometry;
        *bulkiness = shape.bulkiness;
        *structure_factor = shape.struct_factor;
    });
}

ThresholdStatus contlength(const region& reg, double* length) {
    if (length == nullptr) return ThresholdStatus::InvalidArgument;
    Need need;
    need.contour = true;
    return with_shape(reg, need, [&](const Shape& shape) { *length = shape.contlength; });
}

ThresholdStatus circularity(const region& reg, double* value) {
    if (value == nullptr) return ThresholdStatus::InvalidArgument;
    return with_shape(reg, Need{}, [&](const Shape& shape) { *value = shape.circularity; });
}

ThresholdStatus compactness(const region& reg, double* value) {
    if (value == nullptr) return ThresholdStatus::InvalidArgument;
    Need need;
    need.contour = true;
    return with_shape(reg, need, [&](const Shape& shape) { *value = shape.compactness; });
}

ThresholdStatus convexity(const region& reg, double* value) {
    if (value == nullptr) return ThresholdStatus::InvalidArgument;
    Need need;
    need.hull = true;
    return with_shape(reg, need, [&](const Shape& shape) { *value = shape.convexity; });
}

ThresholdStatus rectangularity(const region& reg, double* value) {
    if (value == nullptr) return ThresholdStatus::InvalidArgument;
    Need need;
    need.rectangularity = true;
    return with_shape(reg, need, [&](const Shape& shape) { *value = shape.rectangularity; });
}

ThresholdStatus diameter_region(const region& reg, int* row1, int* column1, int* row2, int* column2, double* diameter) {
    if (row1 == nullptr || column1 == nullptr || row2 == nullptr || column2 == nullptr || diameter == nullptr) {
        return ThresholdStatus::InvalidArgument;
    }
    Need need;
    need.hull = true;
    return with_shape(reg, need, [&](const Shape& shape) {
        *row1 = shape.diameter.a.r;
        *column1 = shape.diameter.a.c;
        *row2 = shape.diameter.b.r;
        *column2 = shape.diameter.b.c;
        *diameter = shape.diameter.distance;
    });
}

ThresholdStatus region_features(const region& reg, const char* const* features, int feature_count, double* values) {
    if (features == nullptr || feature_count <= 0 || values == nullptr) return ThresholdStatus::InvalidArgument;
    Need need;
    std::vector<Feat> ids(static_cast<size_t>(feature_count));
    for (int i = 0; i < feature_count; ++i) {
        ids[static_cast<size_t>(i)] = parse_feature(features[i]);
        if (ids[static_cast<size_t>(i)] == Feat::Count) return ThresholdStatus::InvalidArgument;
        const Need part = need_of(ids[static_cast<size_t>(i)]);
        need.contour = need.contour || part.contour;
        need.hull = need.hull || part.hull;
        need.rectangularity = need.rectangularity || part.rectangularity;
        need.circle = need.circle || part.circle;
    }
    Shape shape;
    const ThresholdStatus status = load_shape(reg, need, &shape);
    for (int i = 0; i < feature_count; ++i) values[i] = feature_value(shape, ids[static_cast<size_t>(i)]);
    return status;
}

ThresholdStatus select_shape(const region* regions, int region_count, const char* const* features, int feature_count,
                             ShapeLogic logic, const double* min_value, const double* max_value, int* indices,
                             int index_capacity, int* selected_count) {
    if (selected_count == nullptr || region_count < 0 || feature_count <= 0 || index_capacity < 0) {
        return ThresholdStatus::InvalidArgument;
    }
    *selected_count = 0;
    if (region_count == 0) return ThresholdStatus::Empty;
    if (regions == nullptr || features == nullptr || min_value == nullptr || max_value == nullptr) {
        return ThresholdStatus::InvalidArgument;
    }
    std::vector<int> keep;
    keep.reserve(static_cast<size_t>(region_count));
    std::vector<double> values(static_cast<size_t>(feature_count));
    for (int i = 0; i < region_count; ++i) {
        const ThresholdStatus status = region_features(regions[i], features, feature_count, values.data());
        if (status == ThresholdStatus::InvalidArgument) return status;
        int hits = 0;
        for (int f = 0; f < feature_count; ++f) {
            if (values[static_cast<size_t>(f)] >= min_value[f] && values[static_cast<size_t>(f)] <= max_value[f]) ++hits;
        }
        const bool pass = logic == ShapeLogic::And ? hits == feature_count : hits > 0;
        if (pass) keep.push_back(i);
    }
    *selected_count = static_cast<int>(keep.size());
    if (static_cast<int>(keep.size()) > index_capacity) return ThresholdStatus::BufferTooSmall;
    if (indices == nullptr && !keep.empty()) return ThresholdStatus::InvalidArgument;
    for (int i = 0; i < static_cast<int>(keep.size()); ++i) indices[i] = keep[static_cast<size_t>(i)];
    return keep.empty() ? ThresholdStatus::Empty : ThresholdStatus::Ok;
}
