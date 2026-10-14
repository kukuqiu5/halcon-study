#include "threshold.hpp"

#include <immintrin.h>

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <cstring>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

namespace {

struct Partial {
    uint32_t area = 0;
    int64_t col_sum2 = 0;
    int64_t row_sum = 0;
    int runs = 0;
    int16_t row1 = 0;
    int16_t col1 = 0;
    int16_t row2 = 0;
    int16_t col2 = 0;
    bool any = false;
    bool overflow = false;
};

struct Scan {
    runLength* cursor = nullptr;
    runLength* end = nullptr;
    int count_index = 0;
    int row_runs = 0;
    int start_col = 0;
    Partial partial;
    __m128i left = _mm_setzero_si128();
};

struct Lut {
    uint8_t count[256]{};
    uint8_t pos[256][8]{};
    Lut() {
        for (int mask = 0; mask < 256; ++mask) {
            int n = 0;
            for (int bit = 0; bit < 8; ++bit) {
                if (mask & (1 << bit)) pos[mask][n++] = static_cast<uint8_t>(bit);
            }
            count[mask] = static_cast<uint8_t>(n);
        }
    }
};

const Lut& edge_lut() {
    static const Lut table;
    return table;
}

inline void emit_edge(Scan& scan, unsigned shift, int row) {
    if (scan.partial.overflow) return;
    if ((scan.count_index & 1) == 0) {
        if (scan.cursor >= scan.end) {
            scan.partial.overflow = true;
            return;
        }
        scan.cursor->row = static_cast<int16_t>(row);
        scan.cursor->cols = static_cast<int16_t>(scan.start_col + static_cast<int>(shift));
    } else {
        scan.cursor->cole = static_cast<int16_t>(scan.start_col + static_cast<int>(shift) - 1);
        const int len = scan.cursor->cole - scan.cursor->cols + 1;
        scan.partial.col_sum2 += static_cast<int64_t>(scan.cursor->cole + scan.cursor->cols) * len;
        scan.partial.row_sum += static_cast<int64_t>(len) * row;
        scan.partial.area += static_cast<uint32_t>(len);
        ++scan.cursor;
        ++scan.partial.runs;
        ++scan.row_runs;
    }
    ++scan.count_index;
}

// 稀疏边缘用 tzcnt/blsr；稠密边缘用查表，避开逐位依赖链。
inline void consume_edges(Scan& scan, unsigned edge_mask, int row) {
    if (edge_mask == 0 || scan.partial.overflow) return;
    if (__builtin_popcount(edge_mask) <= 4) {
        while (edge_mask != 0) {
            const unsigned shift = static_cast<unsigned>(__builtin_ctz(edge_mask));
            emit_edge(scan, shift, row);
            edge_mask &= edge_mask - 1;
        }
        return;
    }
    const Lut& lut = edge_lut();
    for (int byte = 0; byte < 4; ++byte) {
        const unsigned piece = (edge_mask >> (byte * 8)) & 255u;
        const int n = lut.count[piece];
        const uint8_t* pos = lut.pos[piece];
        for (int i = 0; i < n; ++i) {
            emit_edge(scan, static_cast<unsigned>(pos[i]) + static_cast<unsigned>(byte * 8), row);
        }
    }
}

inline void finish_row(Scan& scan, int row) {
    if (scan.row_runs <= 0) return;
    const runLength* first = scan.cursor - scan.row_runs;
    const runLength* last = scan.cursor - 1;
    if (!scan.partial.any) {
        scan.partial.row1 = static_cast<int16_t>(row);
        scan.partial.col1 = first->cols;
        scan.partial.col2 = last->cole;
        scan.partial.any = true;
    } else {
        if (first->cols < scan.partial.col1) scan.partial.col1 = first->cols;
        if (last->cole > scan.partial.col2) scan.partial.col2 = last->cole;
    }
    scan.partial.row2 = static_cast<int16_t>(row);
}

inline void consume_avx_chunk(Scan& scan, __m256i src, __m256i low, __m256i high, int row) {
    const __m256i in_range = _mm256_and_si256(
        _mm256_cmpeq_epi8(_mm256_max_epu8(src, low), src),
        _mm256_cmpeq_epi8(_mm256_min_epu8(src, high), src));
    const __m128i low_half = _mm256_extractf128_si256(in_range, 0);
    const __m128i high_half = _mm256_extractf128_si256(in_range, 1);
    const __m256i shifted = _mm256_setr_m128i(
        _mm_alignr_epi8(low_half, scan.left, 15),
        _mm_alignr_epi8(high_half, low_half, 15));
    scan.left = high_half;
    const auto edge_mask = static_cast<unsigned>(_mm256_movemask_epi8(_mm256_xor_si256(in_range, shifted)));
    consume_edges(scan, edge_mask, row);
    scan.start_col += 32;
}

inline void consume_sse_chunk(Scan& scan, __m128i src, __m128i low, __m128i high, int row) {
    const __m128i in_range = _mm_and_si128(
        _mm_cmpeq_epi8(_mm_max_epu8(src, low), src),
        _mm_cmpeq_epi8(_mm_min_epu8(src, high), src));
    const __m128i shifted = _mm_alignr_epi8(in_range, scan.left, 15);
    scan.left = in_range;
    const auto edge_mask = static_cast<unsigned>(_mm_movemask_epi8(_mm_xor_si128(in_range, shifted)));
    consume_edges(scan, edge_mask, row);
    scan.start_col += 16;
}

Partial scan_avx(const uint8_t* image, int step, int x, int y, int roi_width, int roi_height,
                 uint8_t low_value, uint8_t high_value, runLength* runs, int capacity) {
    Scan scan;
    scan.cursor = runs;
    scan.end = runs + capacity;
    const __m256i low = _mm256_set1_epi8(static_cast<char>(low_value));
    const __m256i high = _mm256_set1_epi8(static_cast<char>(high_value));
    const int full = roi_width & ~31;
    for (int row = y; row < y + roi_height; ++row) {
        const uint8_t* src = image + static_cast<long>(row) * step + x;
        scan.left = _mm_setzero_si128();
        scan.row_runs = 0;
        scan.count_index = 0;
        scan.start_col = x;
        for (int col = 0; col < full; col += 32) {
            consume_avx_chunk(scan, _mm256_loadu_si256(reinterpret_cast<const __m256i*>(src + col)), low, high, row);
            if (scan.partial.overflow) return scan.partial;
        }
        alignas(32) uint8_t tail[32] = {};
        const int remain = roi_width - full;
        if (remain > 0) std::memcpy(tail, src + full, static_cast<size_t>(remain));
        consume_avx_chunk(scan, _mm256_load_si256(reinterpret_cast<const __m256i*>(tail)), low, high, row);
        if (scan.partial.overflow) return scan.partial;
        finish_row(scan, row);
    }
    return scan.partial;
}

Partial scan_sse(const uint8_t* image, int step, int x, int y, int roi_width, int roi_height,
                 uint8_t low_value, uint8_t high_value, runLength* runs, int capacity) {
    Scan scan;
    scan.cursor = runs;
    scan.end = runs + capacity;
    const __m128i low = _mm_set1_epi8(static_cast<char>(low_value));
    const __m128i high = _mm_set1_epi8(static_cast<char>(high_value));
    const int full = roi_width & ~15;
    for (int row = y; row < y + roi_height; ++row) {
        const uint8_t* src = image + static_cast<long>(row) * step + x;
        scan.left = _mm_setzero_si128();
        scan.row_runs = 0;
        scan.count_index = 0;
        scan.start_col = x;
        for (int col = 0; col < full; col += 16) {
            consume_sse_chunk(scan, _mm_loadu_si128(reinterpret_cast<const __m128i*>(src + col)), low, high, row);
            if (scan.partial.overflow) return scan.partial;
        }
        alignas(16) uint8_t tail[16] = {};
        const int remain = roi_width - full;
        if (remain > 0) std::memcpy(tail, src + full, static_cast<size_t>(remain));
        consume_sse_chunk(scan, _mm_load_si128(reinterpret_cast<const __m128i*>(tail)), low, high, row);
        if (scan.partial.overflow) return scan.partial;
        finish_row(scan, row);
    }
    return scan.partial;
}

void publish(region* out, const Partial& partial, runLength* runs) {
    if (!partial.any || partial.area == 0) {
        *out = region{};
        out->pRle = nullptr;
        return;
    }
    out->area = partial.area;
    out->centerCol = (static_cast<double>(partial.col_sum2) / 2.0) / static_cast<double>(partial.area);
    out->centerRow = static_cast<double>(partial.row_sum) / static_cast<double>(partial.area);
    out->row1 = partial.row1;
    out->col1 = partial.col1;
    out->row2 = partial.row2;
    out->col2 = partial.col2;
    out->width = static_cast<int16_t>(partial.col2 - partial.col1 + 1);
    out->height = static_cast<int16_t>(partial.row2 - partial.row1 + 1);
    out->ratio = static_cast<double>(out->height) / static_cast<double>(out->width);
    out->runLengthCount = partial.runs;
    out->pRle = runs;
}

bool arguments_ok(const uint8_t* image, int width, int height, int step, int x, int y,
                  int roi_width, int roi_height, region* out, runLength* runs, int run_capacity,
                  int threads) {
    if (image == nullptr || out == nullptr || runs == nullptr) return false;
    if (width <= 0 || height <= 0 || step < width || run_capacity <= 0 || threads <= 0) return false;
    if (x < 0 || y < 0 || roi_width <= 0 || roi_height <= 0) return false;
    if (x > width - roi_width || y > height - roi_height) return false;
    if (y > 32767 || x > 32767) return false;
    if (roi_height > 32767 - y || roi_width > 32767 - x) return false;
    return true;
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
                    return stop_.load(std::memory_order_acquire) || posted_.load(std::memory_order_acquire) != seen;
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

struct BandArg {
    const uint8_t* image = nullptr;
    int step = 0;
    int x = 0;
    int y = 0;
    int width = 0;
    int height = 0;
    uint8_t low = 0;
    uint8_t high = 0;
    bool sse = false;
    runLength* runs = nullptr;
    int capacity = 0;
    Partial partial;
};

void run_band(void* ctx) {
    auto* band = static_cast<BandArg*>(ctx);
    band->partial = band->sse
                        ? scan_sse(band->image, band->step, band->x, band->y, band->width, band->height,
                                   band->low, band->high, band->runs, band->capacity)
                        : scan_avx(band->image, band->step, band->x, band->y, band->width, band->height,
                                   band->low, band->high, band->runs, band->capacity);
}

Partial merge_bands(BandArg* bands, int count, runLength* runs) {
    Partial merged;
    int dest = 0;
    int offset = 0;
    for (int i = 0; i < count; ++i) {
        const int capacity = bands[i].capacity;
        if (bands[i].partial.overflow) {
            merged.overflow = true;
            return merged;
        }
        if (bands[i].partial.runs > 0 && dest != offset) {
            std::memmove(runs + dest, runs + offset, static_cast<size_t>(bands[i].partial.runs) * sizeof(runLength));
        }
        if (bands[i].partial.any) {
            if (!merged.any) {
                merged.row1 = bands[i].partial.row1;
                merged.col1 = bands[i].partial.col1;
                merged.col2 = bands[i].partial.col2;
                merged.any = true;
            } else {
                if (bands[i].partial.col1 < merged.col1) merged.col1 = bands[i].partial.col1;
                if (bands[i].partial.col2 > merged.col2) merged.col2 = bands[i].partial.col2;
            }
            merged.row2 = bands[i].partial.row2;
            merged.area += bands[i].partial.area;
            merged.col_sum2 += bands[i].partial.col_sum2;
            merged.row_sum += bands[i].partial.row_sum;
        }
        dest += bands[i].partial.runs;
        offset += capacity;
    }
    merged.runs = dest;
    return merged;
}

void parallel_bands(BandArg* bands, int threads) {
    Pool& workers = pool();
    std::lock_guard<std::mutex> lock(workers.mutex);
    while (static_cast<int>(workers.workers.size()) < threads - 1) {
        workers.workers.push_back(std::make_unique<Worker>());
    }
    for (int i = 1; i < threads; ++i) {
        workers.workers[static_cast<size_t>(i - 1)]->post(run_band, &bands[i]);
    }
    run_band(&bands[0]);
    for (int i = 1; i < threads; ++i) {
        workers.workers[static_cast<size_t>(i - 1)]->wait();
    }
}

ThresholdStatus finish(region* out, const Partial& partial, runLength* runs) {
    if (partial.overflow) {
        *out = region{};
        return ThresholdStatus::BufferTooSmall;
    }
    if (!partial.any) {
        publish(out, partial, runs);
        return ThresholdStatus::Empty;
    }
    publish(out, partial, runs);
    return ThresholdStatus::Ok;
}

template <bool kSse>
ThresholdStatus threshold_impl(const uint8_t* image, int width, int height, int step, int x, int y,
                               int roi_width, int roi_height, uint8_t low, uint8_t high, region* out,
                               runLength* runs, int run_capacity, int threads) {
    if (!arguments_ok(image, width, height, step, x, y, roi_width, roi_height, out, runs, run_capacity, threads)) {
        if (out != nullptr) *out = region{};
        return ThresholdStatus::InvalidArgument;
    }
    threads = std::min(threads, roi_height);
    if (threads > 1 && run_capacity < threshold_max_runs(roi_width, roi_height)) threads = 1;
    if (threads <= 1) {
        const Partial partial = kSse
                                    ? scan_sse(image, step, x, y, roi_width, roi_height, low, high, runs, run_capacity)
                                    : scan_avx(image, step, x, y, roi_width, roi_height, low, high, runs, run_capacity);
        return finish(out, partial, runs);
    }

    std::vector<BandArg> bands(static_cast<size_t>(threads));
    int row_cursor = y;
    int slot = 0;
    for (int i = 0; i < threads; ++i) {
        const int remaining_bands = threads - i;
        const int remaining_rows = y + roi_height - row_cursor;
        const int rows = remaining_rows / remaining_bands;
        bands[static_cast<size_t>(i)].image = image;
        bands[static_cast<size_t>(i)].step = step;
        bands[static_cast<size_t>(i)].x = x;
        bands[static_cast<size_t>(i)].y = row_cursor;
        bands[static_cast<size_t>(i)].width = roi_width;
        bands[static_cast<size_t>(i)].height = rows;
        bands[static_cast<size_t>(i)].low = low;
        bands[static_cast<size_t>(i)].high = high;
        bands[static_cast<size_t>(i)].sse = kSse;
        bands[static_cast<size_t>(i)].capacity = threshold_max_runs(roi_width, rows);
        bands[static_cast<size_t>(i)].runs = runs + slot;
        slot += bands[static_cast<size_t>(i)].capacity;
        row_cursor += rows;
    }
    parallel_bands(bands.data(), threads);
    const Partial merged = merge_bands(bands.data(), threads, runs);
    return finish(out, merged, runs);
}

}  // namespace

int threshold_max_runs(int roi_width, int roi_height) {
    if (roi_width <= 0 || roi_height <= 0) return 0;
    return ((roi_width + 1) / 2) * roi_height;
}

ThresholdStatus threshold_avx2(const uint8_t* image, int width, int height, int step, int x, int y,
                               int roi_width, int roi_height, uint8_t low, uint8_t high, region* out,
                               runLength* runs, int run_capacity, int threads) {
    return threshold_impl<false>(image, width, height, step, x, y, roi_width, roi_height, low, high, out, runs,
                                 run_capacity, threads);
}

ThresholdStatus threshold_sse(const uint8_t* image, int width, int height, int step, int x, int y,
                              int roi_width, int roi_height, uint8_t low, uint8_t high, region* out,
                              runLength* runs, int run_capacity, int threads) {
    return threshold_impl<true>(image, width, height, step, x, y, roi_width, roi_height, low, high, out, runs,
                                run_capacity, threads);
}
