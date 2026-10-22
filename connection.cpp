#include "threshold.hpp"

#include <immintrin.h>

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <cstring>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

namespace {

// 并查集按大小合并，根的 parent 为负的集合大小。路径压缩用 path halving。
inline int find_root(int* parent, int x) {
    int p = parent[x];
    if (p < 0) return x;
    while (true) {
        const int gp = parent[p];
        if (gp < 0) return p;
        parent[x] = gp;
        x = p;
        p = gp;
    }
}

inline int unite_roots(int* parent, int a, int b) {
    if (a == b) return a;
    if (parent[a] > parent[b]) std::swap(a, b);
    parent[a] += parent[b];
    parent[b] = a;
    return a;
}

inline void unite(int* parent, int a, int b) {
    a = find_root(parent, a);
    b = find_root(parent, b);
    unite_roots(parent, a, b);
}

// 同一行区间已按起点排序。相接（间隔 0）或重叠并到同一集合。
// 普通 threshold 输出同一行至少隔一个背景像素，内层循环不会进来。
void unite_row(const runLength* runs, int* parent, int begin, int end) {
    int live = begin;
    for (int i = begin + 1; i < end; ++i) {
        const int cols = runs[i].cols;
        while (live < i && runs[live].cole + 1 < cols) ++live;
        for (int j = live; j < i; ++j) {
            if (runs[j].cole + 1 >= cols) unite(parent, j, i);
        }
    }
}

// 上一行 [a0, a1) 与当前行 [b0, b1)。Gap=1 为 8 连通，Gap=0 为 4 连通。
// 两行都按列起点排序时，i 只增不减。一段重叠里的根先合成一个，再和当前游程合并一次。
template <int Gap, bool Overlap>
void link_rows(const runLength* runs, int* parent, int a0, int a1, int b0, int b1) {
    int i = a0;
    for (int j = b0; j < b1; ++j) {
        const int bcols = runs[j].cols;
        const int limit = runs[j].cole + Gap;
        while (i < a1 && runs[i].cole + Gap < bcols) ++i;
        int linked = -1;
        for (int k = i; k < a1 && runs[k].cols <= limit; ++k) {
            if (Overlap && runs[k].cole + Gap < bcols) continue;
            const int root = find_root(parent, k);
            if (root != linked) linked = linked < 0 ? root : unite_roots(parent, linked, root);
        }
        if (linked >= 0) {
            const int root = find_root(parent, j);
            if (root != linked) unite_roots(parent, linked, root);
        }
    }
}

void link_rows(const runLength* runs, int* parent, int a0, int a1, int b0, int b1, int gap, bool overlap) {
    if (gap == 1) {
        if (overlap) link_rows<1, true>(runs, parent, a0, a1, b0, b1);
        else link_rows<1, false>(runs, parent, a0, a1, b0, b1);
    } else if (overlap) {
        link_rows<0, true>(runs, parent, a0, a1, b0, b1);
    } else {
        link_rows<0, false>(runs, parent, a0, a1, b0, b1);
    }
}

struct Feat {
    uint32_t area = 0;
    int64_t col_sum2 = 0;
    int64_t row_sum = 0;
    int runs = 0;
    int16_t row1 = 0;
    int16_t col1 = 0;
    int16_t row2 = 0;
    int16_t col2 = 0;
    bool any = false;
};

void accumulate(Feat& feat, const runLength& run) {
    const int len = run.cole - run.cols + 1;
    feat.col_sum2 += static_cast<int64_t>(run.cole + run.cols) * len;
    feat.row_sum += static_cast<int64_t>(len) * run.row;
    feat.area += static_cast<uint32_t>(len);
    if (!feat.any) {
        feat.row1 = run.row;
        feat.row2 = run.row;
        feat.col1 = run.cols;
        feat.col2 = run.cole;
        feat.any = true;
    } else {
        if (run.row < feat.row1) feat.row1 = run.row;
        if (run.row > feat.row2) feat.row2 = run.row;
        if (run.cols < feat.col1) feat.col1 = run.cols;
        if (run.cole > feat.col2) feat.col2 = run.cole;
    }
    ++feat.runs;
}

void publish_run(region& out, runLength* run) {
    const int len = run->cole - run->cols + 1;
    out.area = static_cast<uint32_t>(len);
    out.centerRow = run->row;
    out.centerCol = static_cast<double>(run->cols + run->cole) * 0.5;
    out.row1 = run->row;
    out.row2 = run->row;
    out.col1 = run->cols;
    out.col2 = run->cole;
    out.width = static_cast<int16_t>(len);
    out.height = 1;
    out.ratio = 1.0 / static_cast<double>(len);
    out.runLengthCount = 1;
    out.pRle = run;
}

void publish(region& out, const Feat& feat, runLength* runs) {
    out.area = feat.area;
    out.centerCol = (static_cast<double>(feat.col_sum2) / 2.0) / static_cast<double>(feat.area);
    out.centerRow = static_cast<double>(feat.row_sum) / static_cast<double>(feat.area);
    out.row1 = feat.row1;
    out.col1 = feat.col1;
    out.row2 = feat.row2;
    out.col2 = feat.col2;
    out.width = static_cast<int16_t>(feat.col2 - feat.col1 + 1);
    out.height = static_cast<int16_t>(feat.row2 - feat.row1 + 1);
    out.ratio = static_cast<double>(out.height) / static_cast<double>(out.width);
    out.runLengthCount = feat.runs;
    out.pRle = runs;
}

struct Scratch {
    std::vector<int> parent;
    std::vector<int> label;
    std::vector<int> row_begin;
    std::vector<int> counts;
    std::vector<int> cursor;
    std::vector<Feat> feats;

    void ensure(int n) {
        if (static_cast<int>(parent.size()) < n) {
            parent.resize(static_cast<size_t>(n));
            label.resize(static_cast<size_t>(n));
            counts.resize(static_cast<size_t>(n));
            cursor.resize(static_cast<size_t>(n));
            feats.resize(static_cast<size_t>(n));
        }
        if (static_cast<int>(row_begin.size()) < n + 1) row_begin.resize(static_cast<size_t>(n + 1));
    }
};

Scratch& scratch() {
    thread_local Scratch local;
    return local;
}

struct Worker {
    Worker() : thread_([this] { loop(); }) {}
    ~Worker() {
        stop_.store(true, std::memory_order_release);
        {
            std::lock_guard<std::mutex> lock(mutex_);
            cv_.notify_one();
        }
        thread_.join();
    }

    void post(void (*fn)(void*), void* ctx) {
        fn_ = fn;
        ctx_ = ctx;
        pending_ = ++ticket_;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            posted_.store(pending_, std::memory_order_release);
            cv_.notify_one();
        }
    }

    void wait() {
        while (finished_.load(std::memory_order_acquire) != pending_) _mm_pause();
    }

    Worker(const Worker&) = delete;
    Worker& operator=(const Worker&) = delete;

private:
    void loop() {
        int seen = 0;
        while (true) {
            void (*fn)(void*) = nullptr;
            void* ctx = nullptr;
            int ticket = 0;
            {
                std::unique_lock<std::mutex> lock(mutex_);
                cv_.wait(lock, [&] {
                    return stop_.load(std::memory_order_acquire) ||
                           posted_.load(std::memory_order_acquire) != seen;
                });
                if (stop_.load(std::memory_order_acquire)) return;
                ticket = posted_.load(std::memory_order_acquire);
                fn = fn_;
                ctx = ctx_;
            }
            seen = ticket;
            fn(ctx);
            finished_.store(ticket, std::memory_order_release);
        }
    }

    std::thread thread_;
    std::mutex mutex_;
    std::condition_variable cv_;
    std::atomic<bool> stop_{false};
    std::atomic<int> posted_{0};
    std::atomic<int> finished_{0};
    int ticket_ = 0;
    int pending_ = 0;
    void (*fn_)(void*) = nullptr;
    void* ctx_ = nullptr;
};

struct Pool {
    std::mutex mutex;
    std::vector<std::unique_ptr<Worker>> workers;
};

Pool& pool() {
    static Pool instance;
    return instance;
}

struct Band {
    const runLength* runs = nullptr;
    int* parent = nullptr;
    const int* row_begin = nullptr;
    int span0 = 0;
    int span1 = 0;
    int gap = 1;
    bool horizontal = false;
    bool overlap = false;
};

template <int Gap, bool Overlap>
void run_rows(const Band& band) {
    for (int s = band.span0; s < band.span1; ++s) {
        const int begin = band.row_begin[s];
        const int end = band.row_begin[s + 1];
        if (band.horizontal) unite_row(band.runs, band.parent, begin, end);
        if (s == band.span0) continue;
        const int prev = band.row_begin[s - 1];
        if (band.runs[begin].row == band.runs[prev].row + 1) {
            link_rows<Gap, Overlap>(band.runs, band.parent, prev, begin, begin, end);
        }
    }
}

void run_band(void* ctx) {
    const Band& band = *static_cast<Band*>(ctx);
    if (band.gap == 1) {
        if (band.overlap) run_rows<1, true>(band);
        else run_rows<1, false>(band);
    } else if (band.overlap) {
        run_rows<0, true>(band);
    } else {
        run_rows<0, false>(band);
    }
}

void parallel_bands(Band* bands, int threads) {
    Pool& workers = pool();
    std::lock_guard<std::mutex> lock(workers.mutex);
    while (static_cast<int>(workers.workers.size()) < threads - 1) {
        workers.workers.push_back(std::make_unique<Worker>());
    }
    for (int i = 1; i < threads; ++i) workers.workers[static_cast<size_t>(i - 1)]->post(run_band, &bands[i]);
    run_band(&bands[0]);
    for (int i = 1; i < threads; ++i) workers.workers[static_cast<size_t>(i - 1)]->wait();
}

bool ranges_overlap(const runLength* a, int na, const runLength* b, int nb) {
    if (a == nullptr || b == nullptr || na <= 0 || nb <= 0) return false;
    const auto* a_end = a + na;
    const auto* b_end = b + nb;
    return a < b_end && b < a_end;
}

constexpr int kParallelMinRuns = 8192;

}  // namespace

ThresholdStatus connection(const runLength* runs, int run_count, Neighborhood neighborhood, region* components,
                           int component_capacity, runLength* runs_out, int runs_capacity, int* component_count,
                           int max_components, int threads) {
    if (component_count == nullptr) return ThresholdStatus::InvalidArgument;
    *component_count = 0;
    if (run_count < 0 || threads <= 0 || max_components < 0 || component_capacity < 0 || runs_capacity < 0) {
        return ThresholdStatus::InvalidArgument;
    }
    if (run_count == 0) return ThresholdStatus::Empty;
    if (runs == nullptr || components == nullptr || runs_out == nullptr) return ThresholdStatus::InvalidArgument;
    if (ranges_overlap(runs, run_count, runs_out, runs_capacity)) return ThresholdStatus::InvalidArgument;

    const int gap = neighborhood == Neighborhood::Eight ? 1 : 0;
    Scratch& work = scratch();
    work.ensure(run_count);
    int* row_begin = work.row_begin.data();
    int spans = 0;
    bool horizontal = false;
    bool overlap = false;
    row_begin[0] = 0;
    for (int i = 0; i < run_count; ++i) {
        const runLength& run = runs[i];
        if (run.cole < run.cols) return ThresholdStatus::InvalidArgument;
        if (i == 0) continue;
        const runLength& prev = runs[i - 1];
        if (run.row < prev.row || (run.row == prev.row && run.cols < prev.cols)) {
            return ThresholdStatus::InvalidArgument;
        }
        if (run.row != prev.row) {
            row_begin[++spans] = i;
        } else if (run.cols <= prev.cole) {
            overlap = true;
            horizontal = true;
        } else if (run.cols <= prev.cole + 1) {
            horizontal = true;
        }
    }
    row_begin[++spans] = run_count;

    int* parent = work.parent.data();
    std::fill(parent, parent + run_count, -1);

    int use_threads = threads;
    if (run_count < kParallelMinRuns || spans < 2) use_threads = 1;
    use_threads = std::min(use_threads, spans);

    if (use_threads <= 1) {
        Band band;
        band.runs = runs;
        band.parent = parent;
        band.row_begin = row_begin;
        band.span0 = 0;
        band.span1 = spans;
        band.gap = gap;
        band.horizontal = horizontal;
        band.overlap = overlap;
        run_band(&band);
    } else {
        std::vector<Band> bands(static_cast<size_t>(use_threads));
        int span_cursor = 0;
        for (int t = 0; t < use_threads; ++t) {
            const int remaining = use_threads - t;
            const int spans_left = spans - span_cursor;
            int take = spans_left / remaining;
            if (t + 1 < use_threads) {
                const int target = (run_count * (t + 1)) / use_threads;
                int count = 1;
                while (count < spans_left - (remaining - 1) && row_begin[span_cursor + count] < target) ++count;
                take = count;
            }
            bands[static_cast<size_t>(t)].runs = runs;
            bands[static_cast<size_t>(t)].parent = parent;
            bands[static_cast<size_t>(t)].row_begin = row_begin;
            bands[static_cast<size_t>(t)].span0 = span_cursor;
            bands[static_cast<size_t>(t)].span1 = span_cursor + take;
            bands[static_cast<size_t>(t)].gap = gap;
            bands[static_cast<size_t>(t)].horizontal = horizontal;
            bands[static_cast<size_t>(t)].overlap = overlap;
            span_cursor += take;
        }
        parallel_bands(bands.data(), use_threads);
        for (int t = 1; t < use_threads; ++t) {
            const int span = bands[static_cast<size_t>(t)].span0;
            const int begin = row_begin[span];
            const int prev = row_begin[span - 1];
            if (runs[begin].row == runs[prev].row + 1) {
                link_rows(runs, parent, prev, begin, begin, row_begin[span + 1], gap, overlap);
            }
        }
    }

    const int whole = find_root(parent, 0);
    if (parent[whole] == -run_count) {
        *component_count = 1;
        if (component_capacity < 1 || runs_capacity < run_count) return ThresholdStatus::BufferTooSmall;
        std::memcpy(runs_out, runs, static_cast<size_t>(run_count) * sizeof(runLength));
        Feat feat{};
        for (int i = 0; i < run_count; ++i) accumulate(feat, runs_out[i]);
        publish(components[0], feat, runs_out);
        return ThresholdStatus::Ok;
    }

    bool singletons = true;
    for (int i = 0; i < run_count; ++i) {
        if (parent[i] != -1) {
            singletons = false;
            break;
        }
    }
    if (singletons) {
        const int keep = max_components > 0 ? std::min(max_components, run_count) : run_count;
        *component_count = keep;
        if (component_capacity < keep || runs_capacity < keep) return ThresholdStatus::BufferTooSmall;
        std::memcpy(runs_out, runs, static_cast<size_t>(keep) * sizeof(runLength));
        for (int i = 0; i < keep; ++i) publish_run(components[i], runs_out + i);
        return ThresholdStatus::Ok;
    }

    int* label = work.label.data();
    std::fill(label, label + run_count, 0);
    int found = 0;
    const int limit = max_components > 0 ? max_components : run_count;
    for (int i = 0; i < run_count; ++i) {
        const int root = find_root(parent, i);
        int id = label[root];
        if (id == 0) {
            if (found >= limit) {
                id = -1;
            } else {
                id = ++found;
            }
            label[root] = id;
        }
        label[i] = id;
    }

    *component_count = found;
    if (found == 0) return ThresholdStatus::Empty;
    if (component_capacity < found) return ThresholdStatus::BufferTooSmall;

    int* counts = work.counts.data();
    std::fill(counts, counts + found, 0);
    for (int i = 0; i < run_count; ++i) {
        if (label[i] > 0) ++counts[label[i] - 1];
    }
    int* cursor = work.cursor.data();
    int placed = 0;
    for (int c = 0; c < found; ++c) {
        cursor[c] = placed;
        placed += counts[c];
    }
    if (runs_capacity < placed) return ThresholdStatus::BufferTooSmall;

    Feat* feats = work.feats.data();
    for (int c = 0; c < found; ++c) feats[c] = Feat{};
    for (int i = 0; i < run_count; ++i) {
        const int id = label[i];
        if (id <= 0) continue;
        const int slot = id - 1;
        const int pos = cursor[slot]++;
        runs_out[pos] = runs[i];
        accumulate(feats[slot], runs[i]);
    }

    int offset = 0;
    for (int c = 0; c < found; ++c) {
        publish(components[c], feats[c], runs_out + offset);
        offset += counts[c];
    }
    return ThresholdStatus::Ok;
}
