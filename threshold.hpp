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
