#pragma once
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include <cmath>

namespace hydra {

inline float l2_norm_sq(const pcl::PointXYZ& a, const pcl::PointXYZ& b) {
  return std::pow(a.x - b.x, 2) + std::pow(a.y - b.y, 2) + std::pow(a.z - b.z, 2);
}

float chamferDistance(pcl::PointCloud<pcl::PointXYZ>::ConstPtr cloud_a_ptr,
                       pcl::PointCloud<pcl::PointXYZ>::ConstPtr cloud_b_ptr);
float chamferDistance(pcl::PointCloud<pcl::PointXYZ>::ConstPtr cloud_a_ptr,
                       pcl::PointCloud<pcl::PointXYZ>::ConstPtr cloud_b_ptr,
                       int k);
}  // namespace hydra
