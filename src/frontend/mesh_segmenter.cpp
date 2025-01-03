/* -----------------------------------------------------------------------------
 * Copyright 2022 Massachusetts Institute of Technology.
 * All Rights Reserved
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 *  1. Redistributions of source code must retain the above copyright notice,
 *     this list of conditions and the following disclaimer.
 *
 *  2. Redistributions in binary form must reproduce the above copyright notice,
 *     this list of conditions and the following disclaimer in the documentation
 *     and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 * Research was sponsored by the United States Air Force Research Laboratory and
 * the United States Air Force Artificial Intelligence Accelerator and was
 * accomplished under Cooperative Agreement Number FA8750-19-2-1000. The views
 * and conclusions contained in this document are those of the authors and should
 * not be interpreted as representing the official policies, either expressed or
 * implied, of the United States Air Force or the U.S. Government. The U.S.
 * Government is authorized to reproduce and distribute reprints for Government
 * purposes notwithstanding any copyright notation herein.
 * -------------------------------------------------------------------------- */
#include "hydra/frontend/mesh_segmenter.h"

#include <glog/logging.h>
#include <kimera_pgmo/mesh_delta.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/io/pcd_io.h>
#include <spark_dsg/bounding_box.h>
#include <spark_dsg/instance_views.h>
#include <spark_dsg/mesh.h>
#include <spark_dsg/node_attributes.h>
#include <spark_dsg/scene_graph_node.h>

#include <boost/iostreams/categories.hpp>
#include <boost/mpl/size.hpp>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "hydra/input/input_data.h"
#include "hydra/reconstruction/reconstruction_output.h"
#define PCL_NO_PRECOMPILE
#include <pcl/segmentation/extract_clusters.h>
#undef PCL_NO_PRECOMPILE
#include <config_utilities/config.h>
#include <config_utilities/types/conversions.h>
#include <config_utilities/types/enum.h>
#include <config_utilities/validation.h>
#include <hydra/utils/cloud_distance.h>
#include <spark_dsg/bounding_box_extraction.h>

#include "hydra/common/global_info.h"
#include "hydra/common/semantic_color_map.h"
#include "hydra/utils/mesh_utilities.h"
#include "hydra/utils/timing_utilities.h"

namespace hydra {

using Clusters = MeshSegmenter::Clusters;
using LabelClusters = MeshSegmenter::LabelClusters;
using KdTreeT = pcl::search::KdTree<pcl::PointXYZRGBA>;
using timing::ScopedTimer;
using CloudPoint = pcl::PointXYZRGBA;
using MeshCloud = pcl::PointCloud<CloudPoint>;
using InstanceData = std::pair<MaskData, MeshCloud::Ptr>;
using ClassToInstance = std::unordered_map<int64, std::vector<InstanceData>>;

void declare_config(MeshSegmenter::Config& config) {
  using namespace config;
  name("MeshSegmenterConfig");
  field<CharConversion>(config.prefix, "prefix");
  // TODO(nathan) string to number conversion
  field(config.layer_id, "layer_id");
  field(config.active_index_horizon_m, "active_index_horizon_m");
  field(config.cluster_tolerance, "cluster_tolerance");
  field(config.min_cluster_size, "min_cluster_size");
  field(config.max_cluster_size, "max_cluster_size");
  enum_field(config.bounding_box_type,
             "bounding_box_type",
             {{spark_dsg::BoundingBox::Type::INVALID, "INVALID"},
              {spark_dsg::BoundingBox::Type::AABB, "AABB"},
              {spark_dsg::BoundingBox::Type::OBB, "OBB"},
              {spark_dsg::BoundingBox::Type::RAABB, "RAABB"}});
  config.labels = GlobalInfo::instance().getLabelSpaceConfig().object_labels;
  field(config.timer_namespace, "timer_namespace");
  field(config.sinks, "sinks");
}

template <typename LList, typename RList>
void mergeList(LList& lhs, const RList& rhs) {
  std::unordered_set<size_t> seen(lhs.begin(), lhs.end());
  for (const auto idx : rhs) {
    if (seen.count(idx)) {
      continue;
    }

    lhs.push_back(idx);
    seen.insert(idx);
  }
}

template <typename T>
std::string printLabels(const std::set<T>& labels) {
  std::stringstream ss;
  ss << "[";
  auto iter = labels.begin();
  while (iter != labels.end()) {
    ss << static_cast<uint64_t>(*iter);
    ++iter;
    if (iter != labels.end()) {
      ss << ", ";
    }
  }
  ss << "]";
  return ss.str();
}

inline bool nodesMatch(const SceneGraphNode& lhs_node, const SceneGraphNode& rhs_node) {
  // TODO: (phuoc) Fix this
  //! BUG: need to fix this merging of nodes:
  // - Curently too many instances, need to merge only when iou is high, or one box is
  // inside another (essentially doing NMS)
  auto& lhs_node_attr = lhs_node.attributes<ObjectNodeAttributes>();
  auto& rhs_node_attr = rhs_node.attributes<ObjectNodeAttributes>();
  // if (lhs_node_attr.bounding_box.type == spark_dsg::BoundingBox::Type::AABB) {
  //   float bbox_iou_thresh = 0.8;
  //   return lhs_node_attr.bounding_box.computeIoU(rhs_node_attr.bounding_box) >
  //          bbox_iou_thresh;
  // }
  return lhs_node_attr.bounding_box.contains(rhs_node_attr.position);
}

inline bool nodesMatch(const Cluster& cluster, const SceneGraphNode& node) {
  return node.attributes<SemanticNodeAttributes>().bounding_box.contains(
      cluster.centroid);
}

std::vector<size_t> getActiveIndices(const kimera_pgmo::MeshDelta& delta,
                                     const std::optional<Eigen::Vector3d>& pos,
                                     double horizon_m) {
  const auto indices = delta.getActiveIndices();

  std::vector<size_t> active;
  active.reserve(indices->size());
  if (!pos) {
    for (const auto& idx : *indices) {
      active.push_back(idx - delta.vertex_start);
    }

    return active;
  }

  const Eigen::Vector3d root_pos = *pos;
  for (const size_t idx : *indices) {
    const auto delta_idx = delta.getLocalIndex(idx);
    const auto& p = delta.vertex_updates->at(delta_idx);
    const Eigen::Vector3d vertex_pos(p.x, p.y, p.z);
    if ((vertex_pos - root_pos).norm() < horizon_m) {
      active.push_back(delta_idx);
    }
  }

  VLOG(2) << "[Mesh Segmenter] Active indices: " << indices->size()
          << " (used: " << active.size() << ")";
  return active;
}

LabelIndices getLabelIndices(const MeshSegmenter::Config& config,
                             const kimera_pgmo::MeshDelta& delta,
                             const std::vector<size_t>& indices) {
  CHECK(delta.hasSemantics());
  const auto& labels = delta.semantic_updates;

  LabelIndices label_indices;
  std::set<uint32_t> seen_labels;
  for (const auto idx : indices) {
    if (static_cast<size_t>(idx) >= labels.size()) {
      LOG(ERROR) << "bad index " << idx << "(of " << labels.size() << ")";
      continue;
    }

    const auto label = labels[idx];
    seen_labels.insert(label);
    if (!config.labels.count(label)) {
      continue;
    }

    auto iter = label_indices.find(label);
    if (iter == label_indices.end()) {
      iter = label_indices.emplace(label, std::vector<size_t>()).first;
    }

    iter->second.push_back(idx);
  }

  VLOG(2) << "[Mesh Segmenter] Seen labels: " << printLabels(seen_labels);
  return label_indices;
}

// TEST: get instances cluster, then use extractIndices to find inliers within mesh
// updates

MeshCloud::Ptr downsampleCloud(MeshCloud::Ptr cloud, float vox_size) {
  // Downsample
  MeshCloud::Ptr cloud_downsampled(new MeshCloud);
  pcl::VoxelGrid<CloudPoint> vg;
  vg.setInputCloud(cloud);
  vg.setLeafSize(vox_size, vox_size, vox_size);
  vg.filter(*cloud_downsampled);
  return cloud_downsampled;
}

MeshCloud::Ptr cloudStatisticalOutlierRemoval(MeshCloud::Ptr cloud,
                                              int mean_k,
                                              float stddev_mul_thresh) {
  // Create the filtering object
  MeshCloud::Ptr cloud_filtered(new MeshCloud);
  pcl::StatisticalOutlierRemoval<CloudPoint> sor(true);
  sor.setInputCloud(cloud);
  sor.setMeanK(mean_k);
  sor.setStddevMulThresh(stddev_mul_thresh);
  sor.filter(*cloud_filtered);
  return cloud_filtered;
}

ClassToInstance computeInstancesClouds(const ReconstructionOutput& input,
                                       MeshSegmenter::Config config) {
  //! TODO: (phuoc) Assign the mask here
  const cv::Mat& vertex_map = input.sensor_data->vertex_map;
  const int& rows = vertex_map.size().height;
  const int& cols = vertex_map.size().width;
  std::vector<MaskData> instance_masks_data = input.sensor_data->instance_masks;

  ClassToInstance cls_to_instance;
  for (const auto& mask_data : instance_masks_data) {
    MeshCloud::Ptr instance_mesh_ptr(new MeshCloud);
    for (int r = 0; r < rows; r++) {
      for (int c = 0; c < cols; c++) {
        if (mask_data.mask.at<uint8_t>(r, c) != 0) {
          auto point = vertex_map.at<cv::Vec3f>(r, c);
          CloudPoint cloud_point;
          cloud_point.x = point[0];
          cloud_point.y = point[1];
          cloud_point.z = point[2];
          instance_mesh_ptr->push_back(cloud_point);
        }
      }
    }
    // // TODO: (phuoc) Use config file for the parameters
    // Downsample
    float vox_grid_size = 0.01f;
    MeshCloud::Ptr cloud_downsampled =
        downsampleCloud(instance_mesh_ptr, vox_grid_size);
    // remove outlier
    MeshCloud::Ptr cloud_filtered =
        cloudStatisticalOutlierRemoval(cloud_downsampled, 50, 0.5);

    // TEST: not cluster and push everything after filtering
    MeshCloud::Ptr instance_cloud(new MeshCloud);
    for (const auto& point : cloud_filtered->points) {
      instance_cloud->push_back(point);
    }
    int64 class_id = mask_data.class_id;
    InstanceData instance_data = std::make_pair(mask_data, instance_cloud);
    if (cls_to_instance.find(class_id) == cls_to_instance.end()) {
      std::vector<std::pair<MaskData, MeshCloud::Ptr>> mesh_vec;
      mesh_vec.push_back(instance_data);
      cls_to_instance.insert(std::make_pair(class_id, mesh_vec));
    } else {
      cls_to_instance.at(class_id).push_back(instance_data);
    }
    if (class_id == 60) {
      pcl::io::savePCDFileASCII("/home/ros/debug/CloudFiltered" + std::to_string(std::rand()) + ".png", *cloud_filtered);
    }
    //
    // //! TEST: Find Cluster (Try to fix splattering issue)
    // KdTreeT::Ptr tree(new KdTreeT());
    // tree->setInputCloud(cloud_filtered);
    // pcl::EuclideanClusterExtraction<CloudPoint> estimator;
    // estimator.setClusterTolerance(config.cluster_tolerance);
    // estimator.setMinClusterSize(config.min_cluster_size);
    // estimator.setMaxClusterSize(config.max_cluster_size);
    // estimator.setSearchMethod(tree);
    // estimator.setInputCloud(cloud_filtered);
    // std::vector<pcl::PointIndices> cluster_indices;
    // estimator.extract(cluster_indices);
    //
    // size_t max_cluster_sz = 0;
    // int largest_cluster_id = -1;
    // int cluster_sum = 0;
    // // TODO: (phuoc) fix this
    // // BUG: Still separates some objects into pieces, consider re-merging strategy
    // for (size_t i = 0; i < cluster_indices.size(); i++) {
    //   size_t cluster_sz = cluster_indices[i].indices.size();
    //   cluster_sum = cluster_sum + cluster_sz;
    //   if (cluster_sz > max_cluster_sz) {
    //     max_cluster_sz = cluster_sz;
    //     largest_cluster_id = i;
    //   }
    // }
    //! BUG: Missing some items here, check why
    // if (largest_cluster_id > -1) {
    //   MeshCloud::Ptr instance_cloud(new MeshCloud);
    //   for (const auto& id : cluster_indices[largest_cluster_id].indices) {
    //     instance_cloud->push_back(cloud_filtered->points[id]);
    //   }
    //   int64 class_id = mask_data.class_id;
    //   InstanceData instance_data = std::make_pair(mask_data, instance_cloud);
    //   if (cls_to_instance.find(class_id) == cls_to_instance.end()) {
    //     std::vector<std::pair<MaskData, MeshCloud::Ptr>> mesh_vec;
    //     mesh_vec.push_back(instance_data);
    //     cls_to_instance.insert(std::make_pair(class_id, mesh_vec));
    //   } else {
    //     cls_to_instance.at(class_id).push_back(instance_data);
    //   }
    // }
  }
  //! BUG: Debugging by viewing cloud (Somehow instance views only have 1
  // point????)
  // for (const auto& cls_mesh : cls_to_mesh) {
  //   const int64 class_id = cls_mesh.first;
  //   const auto& meshes = cls_mesh.second;
  //   int count = 0;
  //   for (const auto& mesh : meshes) {
  //     std::string filename = "/home/ros/debug/" + std::to_string(class_id) + "_" +
  //                            std::to_string(count++) + ".pcd";
  //     pcl::io::savePCDFileASCII(filename, *mesh);
  //   }
  // }
  return cls_to_instance;
}

float l2_norm_sq(const CloudPoint& a, const CloudPoint& b) {
  return std::pow(a.x - b.x, 2) + std::pow(a.y - b.y, 2) + std::pow(a.z - b.z, 2);
}

//! TODO: (phuoc) Re-Implement function to get the inlier vertices in the mesh with
//! comparison to the instance clusters (using k-nearest neighbor search instead of all
//! pairs for speed if possible)
Clusters findInstanceClusters(const MeshSegmenter::Config& config,
                              const kimera_pgmo::MeshDelta& delta,
                              const std::vector<size_t>& indices,
                              const int64& class_id,
                              const ClassToInstance& class_to_instance,
                              std::unordered_set<size_t>& registered_indices) {
  Clusters clusters;

  float threshold = 0.025;

  if (class_to_instance.find(class_id) != class_to_instance.end()) {
    for (const auto& instance_data : class_to_instance.at(class_id)) {
      const MaskData& instance_mask = instance_data.first;
      MeshCloud::Ptr instance_mesh = instance_data.second;
      Cluster instance_cluster;
      //! NOTE: Go over all points in vertex updates, go check if it is close to a point
      //! of instance mesh
      pcl::KdTreeFLANN<CloudPoint> kdtree;
      for (const auto& point_id : indices) {
        if (registered_indices.find(point_id) == registered_indices.end()) {
          const auto& point_D = delta.vertex_updates->at(point_id);
          kdtree.setInputCloud(instance_mesh);
          int k = 10;
          // holds the resultant indices of the neighboring points
          std::vector<int> pointIdxKNNSearch(k);
          // holds the resultant squared distances to nearby points
          std::vector<float> pointKNNSqrNorm(k);
          if (kdtree.nearestKSearch(point_D, k, pointIdxKNNSearch, pointKNNSqrNorm) >
              0) {
            for (std::size_t i = 0; i < pointIdxKNNSearch.size(); ++i) {
              int idx = pointIdxKNNSearch[i];
              CloudPoint instance_point;
              instance_point.x = instance_mesh->points[idx].x;
              instance_point.y = instance_mesh->points[idx].y;
              instance_point.z = instance_mesh->points[idx].z;
              // compute l2 distance squared
              float dist = l2_norm_sq(point_D, instance_point);
              if (dist < threshold) {
                //! NOTE: Registering index
                registered_indices.insert(point_id);
                instance_cluster.indices.push_back(point_id);
                const Eigen::Vector3d pos(point_D.x, point_D.y, point_D.z);
                instance_cluster.centroid += pos;
              }
            }
          }
        }
      }
      // LOG(INFO) << "Got instance cluster of size " <<
      // instance_cluster.indices.size();
      if (instance_cluster.indices.size() >= config.min_cluster_size) {
        instance_cluster.centroid /= instance_cluster.indices.size();
        instance_cluster.mask = instance_mask;
        clusters.push_back(instance_cluster);
      }
    }
  }
  return clusters;
}

Clusters findClusters(const MeshSegmenter::Config& config,
                      const kimera_pgmo::MeshDelta& delta,
                      const std::vector<size_t>& indices) {
  pcl::IndicesPtr pcl_indices(new pcl::Indices(indices.begin(), indices.end()));

  KdTreeT::Ptr tree(new KdTreeT());
  tree->setInputCloud(delta.vertex_updates, pcl_indices);

  pcl::EuclideanClusterExtraction<pcl::PointXYZRGBA> estimator;
  estimator.setClusterTolerance(config.cluster_tolerance);
  estimator.setMinClusterSize(config.min_cluster_size);
  estimator.setMaxClusterSize(config.max_cluster_size);
  estimator.setSearchMethod(tree);
  estimator.setInputCloud(delta.vertex_updates);
  estimator.setIndices(pcl_indices);

  std::vector<pcl::PointIndices> cluster_indices;
  estimator.extract(cluster_indices);

  Clusters clusters;
  clusters.resize(cluster_indices.size());
  for (size_t k = 0; k < clusters.size(); ++k) {
    auto& cluster = clusters.at(k);
    const auto& curr_indices = cluster_indices.at(k).indices;
    for (const auto local_idx : curr_indices) {
      cluster.indices.push_back(delta.getGlobalIndex(local_idx));

      const auto& p = delta.vertex_updates->at(local_idx);
      const Eigen::Vector3d pos(p.x, p.y, p.z);
      cluster.centroid += pos;
    }

    if (curr_indices.size()) {
      cluster.centroid /= curr_indices.size();
    }
  }

  return clusters;
}

MeshSegmenter::MeshSegmenter(const Config& config)
    : config(config::checkValid(config)),
      next_node_id_(config.prefix, 0),
      sinks_(Sink::instantiate(config.sinks)) {
  VLOG(2) << "[Mesh Segmenter] using labels: " << printLabels(config.labels);
  for (const auto& label : config.labels) {
    active_nodes_[label] = std::set<NodeId>();
  }
}

LabelClusters MeshSegmenter::detect(const ReconstructionOutput& input,
                                    uint64_t timestamp_ns,
                                    const kimera_pgmo::MeshDelta& delta,
                                    const std::optional<Eigen::Vector3d>& pos) {
  const auto timer_name = config.timer_namespace + "_detection";
  ScopedTimer timer(timer_name, timestamp_ns, true, 1, false);

  const auto indices = getActiveIndices(delta, pos, config.active_index_horizon_m);

  LabelClusters label_clusters;
  if (indices.empty()) {
    VLOG(2) << "[Mesh Segmenter] No active indices in mesh";
    return label_clusters;
  }

  const auto label_indices = getLabelIndices(config, delta, indices);
  if (label_indices.empty()) {
    VLOG(2) << "[Mesh Segmenter] No vertices found matching desired labels";
    Sink::callAll(sinks_, timestamp_ns, delta, indices, label_indices);
    return label_clusters;
  }

  std::unordered_set<size_t> registered_indices;
  for (const auto label : config.labels) {
    if (!label_indices.count(label)) {
      continue;
    }

    if (label_indices.at(label).size() < config.min_cluster_size) {
      continue;
    }

    // const auto clusters = findClusters(config, delta, label_indices.at(label));
    const auto class_to_mesh = computeInstancesClouds(input, config);
    const auto clusters = findInstanceClusters(config,
                                               delta,
                                               label_indices.at(label),
                                               label,
                                               class_to_mesh,
                                               registered_indices);

    VLOG(2) << "[Mesh Segmenter]  - Found " << clusters.size()
            << " cluster(s) of label " << static_cast<int>(label);
    label_clusters.insert({label, clusters});
  }

  Sink::callAll(sinks_, timestamp_ns, delta, indices, label_indices);
  return label_clusters;
}

void MeshSegmenter::archiveOldNodes(const DynamicSceneGraph& graph,
                                    size_t num_archived_vertices) {
  std::set<NodeId> archived;
  for (const auto& label : config.labels) {
    std::list<NodeId> removed_nodes;
    for (const auto& node_id : active_nodes_.at(label)) {
      if (!graph.hasNode(node_id)) {
        removed_nodes.push_back(node_id);
        continue;
      }

      auto& attrs = graph.getNode(node_id).attributes<ObjectNodeAttributes>();
      bool is_active = false;
      for (const auto index : attrs.mesh_connections) {
        if (index >= num_archived_vertices) {
          is_active = true;
          break;
        }
      }

      attrs.is_active = is_active;
      if (!attrs.is_active) {
        removed_nodes.push_back(node_id);
      }
    }

    for (const auto& node_id : removed_nodes) {
      active_nodes_[label].erase(node_id);
    }
  }
}

void MeshSegmenter::updateGraph(uint64_t timestamp_ns,
                                const LabelClusters& clusters,
                                size_t num_archived_vertices,
                                DynamicSceneGraph& graph) {
  ScopedTimer timer(config.timer_namespace + "_graph_update", timestamp_ns);
  archiveOldNodes(graph, num_archived_vertices);

  for (auto&& [label, clusters_for_label] : clusters) {
    for (const auto& cluster : clusters_for_label) {
      bool matches_prev_node = false;
      std::vector<NodeId> nodes_not_in_graph;
      for (const auto& prev_node_id : active_nodes_.at(label)) {
        const auto& prev_node = graph.getNode(prev_node_id);
        //! NOTE: Compare with multiple prev nodes, and then merge with the best node to
        //! avoid merging 2 objects that are close
        if (nodesMatch(cluster, prev_node)) {
          updateNodeInGraph(graph, cluster, prev_node, timestamp_ns);
          matches_prev_node = true;
          break;
        }
      }

      if (!matches_prev_node) {
        addNodeToGraph(graph, cluster, label, timestamp_ns);
      }

      mergeActiveNodes(graph, label);
    }
  }
}

void MeshSegmenter::mergeActiveNodes(DynamicSceneGraph& graph, uint32_t label) {
  std::set<NodeId> merged_nodes;

  auto& curr_active = active_nodes_.at(label);
  for (const auto& node_id : curr_active) {
    if (merged_nodes.count(node_id)) {
      continue;
    }
    const auto& node = graph.getNode(node_id);

    std::list<NodeId> to_merge;
    for (const auto& other_id : curr_active) {
      if (node_id == other_id) {
        continue;
      }

      if (merged_nodes.count(other_id)) {
        continue;
      }

      const auto& other = graph.getNode(other_id);
      if (nodesMatch(node, other) || nodesMatch(other, node)) {
        to_merge.push_back(other_id);
      }
    }

    auto& attrs = node.attributes<ObjectNodeAttributes>();
    for (const auto& other_id : to_merge) {
      const auto& other = graph.getNode(other_id);
      auto& other_attrs = other.attributes<ObjectNodeAttributes>();
      // TODO: (phuoc) merge masks list
      mergeList(attrs.mesh_connections, other_attrs.mesh_connections);
      attrs.instance_views.merge_views(other_attrs.instance_views);
      graph.removeNode(other_id);
      merged_nodes.insert(other_id);
    }

    if (!to_merge.empty()) {
      updateObjectGeometry(*graph.mesh(), attrs);
    }
  }

  for (const auto& node_id : merged_nodes) {
    curr_active.erase(node_id);
  }
}

std::unordered_set<NodeId> MeshSegmenter::getActiveNodes() const {
  std::unordered_set<NodeId> active_nodes;
  for (const auto& label_nodes_pair : active_nodes_) {
    active_nodes.insert(label_nodes_pair.second.begin(), label_nodes_pair.second.end());
  }
  return active_nodes;
}

void MeshSegmenter::updateNodeInGraph(DynamicSceneGraph& graph,
                                      const Cluster& cluster,
                                      const SceneGraphNode& node,
                                      uint64_t timestamp) {
  auto& attrs = node.attributes<ObjectNodeAttributes>();
  attrs.last_update_time_ns = timestamp;
  attrs.is_active = true;

  std::shared_ptr<cv::Mat> mask_to_assign;
  mask_to_assign = std::make_shared<cv::Mat>(cluster.mask.mask);
  attrs.instance_views.add_view(graph.mapViewCount(), *mask_to_assign);

  mergeList(attrs.mesh_connections, cluster.indices);
  updateObjectGeometry(*graph.mesh(), attrs);
}

void MeshSegmenter::addNodeToGraph(DynamicSceneGraph& graph,
                                   const Cluster& cluster,
                                   uint32_t label,
                                   uint64_t timestamp) {
  if (cluster.indices.empty()) {
    LOG(ERROR) << "Encountered empty cluster with label" << static_cast<int>(label)
               << " @ " << timestamp << "[ns]";
    return;
  }

  auto attrs = std::make_unique<ObjectNodeAttributes>();
  attrs->last_update_time_ns = timestamp;
  attrs->is_active = true;
  attrs->semantic_label = label;
  attrs->name = NodeSymbol(next_node_id_).getLabel();
  const auto& label_to_name = GlobalInfo::instance().getLabelToNameMap();
  auto iter = label_to_name.find(label);
  if (iter != label_to_name.end()) {
    attrs->name = iter->second;
  } else {
    VLOG(2) << "Missing semantic label from map: " << std::to_string(label);
  }

  attrs->mesh_connections.insert(
      attrs->mesh_connections.begin(), cluster.indices.begin(), cluster.indices.end());

  // NOTE: add mask here
  std::shared_ptr<cv::Mat> mask_to_assign;
  mask_to_assign = std::make_shared<cv::Mat>(cluster.mask.mask);
  attrs->instance_views.add_view(graph.mapViewCount(), *mask_to_assign);

  auto label_map = GlobalInfo::instance().getSemanticColorMap();
  if (!label_map || !label_map->isValid()) {
    label_map = GlobalInfo::instance().setRandomColormap();
    CHECK(label_map != nullptr);
  }

  attrs->color = label_map->getColorFromLabel(label);

  updateObjectGeometry(*graph.mesh(), *attrs, nullptr, config.bounding_box_type);

  graph.emplaceNode(config.layer_id, next_node_id_, std::move(attrs));
  active_nodes_.at(label).insert(next_node_id_);
  ++next_node_id_;
}

}  // namespace hydra
