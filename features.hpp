#pragma once

#include "threshold.hpp"

// Halcon 区域特征。像素按中心点参与矩、圆度和外接几何，和 elliptic_axis 的约定一致。
// 因此单像素的外接圆半径是 0，高度为 1 的区域 rect2_len2 是 0。
// 空区域各特征写 0，并返回 Empty。游程必须按 (row, cols) 排好序。

ThresholdStatus area_center(const region& reg, int64_t* area, double* row, double* column);

ThresholdStatus smallest_rectangle1(const region& reg, int* row1, int* column1, int* row2, int* column2);

// Row、Column 是中心。Length1、Length2 是半边长，Length1 >= Length2。Phi 属于 (-pi/2, pi/2]。
ThresholdStatus smallest_rectangle2(const region& reg, double* row, double* column, double* phi, double* length1,
                                    double* length2);

ThresholdStatus smallest_circle(const region& reg, double* row, double* column, double* radius);

// 二阶矩已按面积归一。M20 随行变化，M02 随列变化。
// Ia、Ib 是主轴上的二阶矩。Ra = sqrt(2*Ia)，Rb = sqrt(2*Ib)。
ThresholdStatus moments_region_2nd(const region& reg, double* m11, double* m20, double* m02, double* ia, double* ib);

ThresholdStatus elliptic_axis(const region& reg, double* ra, double* rb, double* phi);

// Rb 为 0 时三个量都写 0，避免除以短轴。
ThresholdStatus eccentricity(const region& reg, double* anisometry, double* bulkiness, double* structure_factor);

// 外轮廓链长。坐标轴方向一步记 1，对角线记 sqrt(2)。孔的轮廓不计入。单像素为 0。
ThresholdStatus contlength(const region& reg, double* length);

// min(1, area / (pi * max^2))，max 是中心到轮廓像素的最大距离。
ThresholdStatus circularity(const region& reg, double* value);

// max(1, length^2 / (4 * pi * area))。
ThresholdStatus compactness(const region& reg, double* value);

// 区域面积 / 凸包内的像素数。
ThresholdStatus convexity(const region& reg, double* value);

// 与区域一阶、二阶矩相同的矩形，用对称差面积对矩形面积归一。矩形为 1。
ThresholdStatus rectangularity(const region& reg, double* value);

// 像素中心之间的最大距离，以及取到该距离的两个端点。
ThresholdStatus diameter_region(const region& reg, int* row1, int* column1, int* row2, int* column2, double* diameter);

// 名字与 Halcon region_features 相同。一次调用里相同的几何量只算一遍。
// 支持 area,row,column,row1,column1,row2,column2,width,height,ratio,
// circularity,compactness,contlength,convexity,rectangularity,
// ra,rb,phi,orientation,anisometry,bulkiness,struct_factor,
// outer_radius,max_diameter,rect2_phi,rect2_len1,rect2_len2,
// moments_m11,moments_m20,moments_m02,moments_ia,moments_ib。
ThresholdStatus region_features(const region& reg, const char* const* features, int feature_count, double* values);

enum class ShapeLogic { And, Or };

// 按特征区间筛选。And 要求每个特征都落在自己的 [min, max] 里，Or 表示至少一个。
// 容量不够时返回 BufferTooSmall，*selected_count 写成需要的个数，不写下标。
// indices 为空且有命中时同样先写上个数，再返回 InvalidArgument。
ThresholdStatus select_shape(const region* regions, int region_count, const char* const* features, int feature_count,
                             ShapeLogic logic, const double* min_value, const double* max_value, int* indices,
                             int index_capacity, int* selected_count);
