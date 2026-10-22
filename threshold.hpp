#pragma once

#include <cstdint>

// 与笔记中的 Region / 游程布局一致：每段游程是 (row, 起始列, 结束列)。
struct runLength {
    int16_t row;
    int16_t cols;
    int16_t cole;
};

struct region {
    double centerRow;
    double centerCol;
    uint32_t area;
    int16_t row1;
    int16_t col1;
    int16_t row2;
    int16_t col2;
    int16_t width;
    int16_t height;
    double ratio;
    int runLengthCount;
    runLength* pRle;
};

enum class ThresholdStatus {
    Ok = 0,
    Empty = 1,
    BufferTooSmall = 2,
    InvalidArgument = 3,
};

// 一个宽 roi_width、高 roi_height 的区域最多有多少段游程。
int threshold_max_runs(int roi_width, int roi_height);

// 灰度图二值化，输出 Halcon 风格的游程区域。
// threads 为 1 时单线程；大于 1 时按行分带，线程池在进程内复用。
// 坐标使用 int16，ROI 右下角不得超过 32767。
ThresholdStatus threshold_avx2(const uint8_t* image, int width, int height, int step,
                               int x, int y, int roi_width, int roi_height,
                               uint8_t low, uint8_t high, region* out,
                               runLength* runs, int run_capacity, int threads = 1);

ThresholdStatus threshold_sse(const uint8_t* image, int width, int height, int step,
                              int x, int y, int roi_width, int roi_height,
                              uint8_t low, uint8_t high, region* out,
                              runLength* runs, int run_capacity, int threads = 1);

// Halcon set_system('neighborhood', 4|8)。前景默认 8 连通。
enum class Neighborhood { Four = 4, Eight = 8 };

// 把已按 (row, cols) 排好序的游程区域拆成连通域，对应 Halcon connection。
// 同一行上相接或重叠的游程会并到一起；只在行号恰好相差 1 时做纵向比较。
// 8 连通的列间隔是 1，4 连通必须列区间重叠。输出按扫描时第一次遇到的顺序，
// 游程本身按行优先时，这与 sort_region(..., 'first_point', 'true', 'row') 一致。
// max_components 对应 max_connection：0 返回全部，大于 0 只保留最先遇到的若干个。
// runs_out 不能与 runs 重叠。每个连通域的 pRle 指向 runs_out 里连续的一段。
// 调用方至少要为 run_count 段游程准备空间；连通域个数最多也是 run_count。
// 容量不够时返回 BufferTooSmall，*component_count 写成需要的连通域个数，不写部分结果。
// threads 大于 1 且游程足够多时按整行分带并行，结果与单线程相同。
ThresholdStatus connection(const runLength* runs, int run_count, Neighborhood neighborhood,
                           region* components, int component_capacity, runLength* runs_out, int runs_capacity,
                           int* component_count, int max_components = 0, int threads = 1);

inline ThresholdStatus connection(const region& input, Neighborhood neighborhood, region* components,
                                  int component_capacity, runLength* runs_out, int runs_capacity,
                                  int* component_count, int max_components = 0, int threads = 1) {
    return connection(input.pRle, input.runLengthCount, neighborhood, components, component_capacity, runs_out,
                      runs_capacity, component_count, max_components, threads);
}
