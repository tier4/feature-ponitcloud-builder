/*
 * Software License Agreement (BSD License)
 *
 *  Point Cloud Library (PCL) - www.pointclouds.org
 *  Copyright (c) 2010-2011, Willow Garage, Inc.
 *
 *  All rights reserved.
 *
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions
 *  are met:
 *
 *   * Redistributions of source code must retain the above copyright
 *     notice, this list of conditions and the following disclaimer.
 *   * Redistributions in binary form must reproduce the above
 *     copyright notice, this list of conditions and the following
 *     disclaimer in the documentation and/or other materials provided
 *     with the distribution.
 *   * Neither the name of the copyright holder(s) nor the names of its
 *     contributors may be used to endorse or promote products derived
 *     from this software without specific prior written permission.
 *
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 *  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 *  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
 *  FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
 *  COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
 *  INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 *  BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 *  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 *  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 *  LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
 *  ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 *  POSSIBILITY OF SUCH DAMAGE.
 *
 */

#ifndef INCLUDE_MAPPING_FILTER_HPP_
#define INCLUDE_MAPPING_FILTER_HPP_

#include "mapping/filter.hpp"

#include <pcl/common/point_tests.h>
#include <pcl/common/common.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/point_types.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/filters/voxel_grid_covariance.h>

#include <Eigen/Eigenvalues>

#include <limits>
#include <map>
#include <tuple>
#include <vector>

template<typename PointT>
std::tuple<Eigen::Vector3f, Eigen::Vector3f> getMinMax3D(
    const typename pcl::PointCloud<PointT> & input) {
  Eigen::Vector4f min_p, max_p;
  pcl::getMinMax3D<PointT>(input, min_p, max_p);
  return std::make_tuple(min_p.head(3), max_p.head(3));
}

std::tuple<Eigen::Vector3i, Eigen::Vector3i> getMinMaxBoundingBox(
  const Eigen::Vector3f & min_p,
  const Eigen::Vector3f & max_p,
  const Eigen::Vector3f & inverse_leaf_size) {
  Eigen::Vector3i min_b, max_b;
  // Compute the minimum and maximum bounding box values
  min_b(0) = static_cast<int>(std::floor(min_p(0) * inverse_leaf_size(0)));
  max_b(0) = static_cast<int>(std::floor(max_p(0) * inverse_leaf_size(0)));
  min_b(1) = static_cast<int>(std::floor(min_p(1) * inverse_leaf_size(1)));
  max_b(1) = static_cast<int>(std::floor(max_p(1) * inverse_leaf_size(1)));
  min_b(2) = static_cast<int>(std::floor(min_p(2) * inverse_leaf_size(2)));
  max_b(2) = static_cast<int>(std::floor(max_p(2) * inverse_leaf_size(2)));
  return std::make_tuple(min_b, max_b);
}

template<typename PointT>
class Filter {
 public:
using Leaf = typename pcl::VoxelGridCovariance<PointT>::Leaf;
using LeafConstPtr = typename pcl::VoxelGridCovariance<PointT>::LeafConstPtr;

LeafConstPtr getLeaf(const PointT & p) const {
  // Generate index associated with p
  const double cx = p.x * inverse_leaf_size_(0);
  const double cy = p.y * inverse_leaf_size_(1);
  const double cz = p.z * inverse_leaf_size_(2);

  const int ijk0 = static_cast<int>(std::floor(cx) - min_b_(0));
  const int ijk1 = static_cast<int>(std::floor(cy) - min_b_(1));
  const int ijk2 = static_cast<int>(std::floor(cz) - min_b_(2));

  // Compute the centroid leaf index
  const int idx =
    ijk0 * divb_mul_(0) +
    ijk1 * divb_mul_(1) +
    ijk2 * divb_mul_(2);

  // Find leaf associated with index
  const auto leaf_iter = leaves_.find(idx);
  if (leaf_iter != leaves_.end()) {
    // If such a leaf exists return the pointer to the leaf structure
    return LeafConstPtr(&(leaf_iter->second));
  }
  return nullptr;
}

Filter(
  const typename pcl::PointCloud<PointT>::ConstPtr input_,
  const Eigen::Vector3f leaf_size,
  const int min_points_per_voxel_)
  : inverse_leaf_size_(
      Eigen::Array4f(
        1. / leaf_size(0),
        1. / leaf_size(1),
        1. / leaf_size(2),
        1.)
) {
  std::vector<int> voxel_centroids_leaf_indices_;

  // Has the input dataset been set already?
  assert(!input_ && "Input point cloud is empty!");

  const auto [min_p, max_p] = getMinMax3D<PointT>(*input_);

  // Check that the leaf size is not too small, given the size of the data

  const double ax = (max_p[0] - min_p[0]) * inverse_leaf_size_[0];
  const double ay = (max_p[1] - min_p[1]) * inverse_leaf_size_[1];
  const double az = (max_p[2] - min_p[2]) * inverse_leaf_size_[2];

  const std::int64_t dx = static_cast<std::int64_t>(ax) + 1;
  const std::int64_t dy = static_cast<std::int64_t>(ay) + 1;
  const std::int64_t dz = static_cast<std::int64_t>(az) + 1;

  // Copy the header (and thus the frame_id) + allocate enough space for points
  typename pcl::PointCloud<PointT> output;
  output.height = 1;        // downsampling breaks the organized structure
  output.is_dense = true;   // we filter out invalid points

  if ((dx*dy*dz) > std::numeric_limits<std::int32_t>::max()) {
    std::cerr
      << "Leaf size is too small for the input dataset. "
      << "Integer indices would overflow."
      << std::endl;
    output.clear();
    return;
  }

  Eigen::Vector4i max_b_;
  Eigen::Vector4i div_b_;

  const auto [min_b, max_b] = getMinMaxBoundingBox(
    min_p, max_p, inverse_leaf_size_.head(3));
  min_b_.head(3) = min_b;
  max_b_.head(3) = max_b;

  // Compute the number of divisions needed along all axis
  div_b_ = max_b_ - min_b_ + Eigen::Vector4i::Ones();
  div_b_[3] = 0;

  // Set up the division multiplier
  divb_mul_ = Eigen::Vector4i(1, div_b_[0], div_b_[0] * div_b_[1], 0);

  int centroid_size = 4;

  // ---[ RGB special case
  std::vector<pcl::PCLPointField> fields;
  int rgba_index = -1;
  rgba_index = pcl::getFieldIndex<PointT>("rgb", fields);
  if (rgba_index == -1) {
    rgba_index = pcl::getFieldIndex<PointT>("rgba", fields);
  }

  if (rgba_index >= 0) {
    rgba_index = fields[rgba_index].offset;
    centroid_size += 4;
  }

  // First pass: go over all points and insert them into the right leaf
  for (const auto & point : *input_) {
    if (!input_->is_dense) {
      // Check if the point is invalid
      if (!pcl::isXYZFinite(point)) {
        continue;
      }
    }

    // Compute the centroid leaf index
    const Eigen::Vector4i ijk =
        Eigen::floor(point.getArray4fMap() * inverse_leaf_size_.array())
            .template cast<int>();
    // divb_mul_[3] = 0 by assignment
    int idx = (ijk - min_b_).dot(divb_mul_);

    Leaf& leaf = leaves_[idx];
    if (leaf.nr_points == 0) {
      leaf.centroid.resize(centroid_size);
      leaf.centroid.setZero();
    }

    const Eigen::Vector3d pt3d = point.getVector3fMap().template cast<double>();
    // Accumulate point sum for centroid calculation
    leaf.mean_ += pt3d;
    // Accumulate x*xT for single pass covariance calculation
    leaf.cov_ += pt3d * pt3d.transpose();

    // Do we need to process all the fields?
    leaf.centroid.template head<3>() += point.getVector3fMap();
    ++leaf.nr_points;
  }

  // Second pass:
  // go over all leaves and compute centroids and covariance matrices
  output.reserve(leaves_.size());
  voxel_centroids_leaf_indices_.reserve(leaves_.size());

  // Eigen values and vectors calculated to prevent near singluar matrices
  Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> eigensolver;

  // Eigen values less than a threshold of max eigen value are inflated
  // to a set fraction of the max eigen value.

  for (auto it = leaves_.begin(); it != leaves_.end(); ++it) {
    // Normalize the centroid
    Leaf & leaf = it->second;

    // Normalize the centroid
    leaf.centroid /= static_cast<float>(leaf.nr_points);
    // Point sum used for single pass covariance calculation
    const Eigen::Vector3d pt_sum = leaf.mean_;
    // Normalize mean
    leaf.mean_ /= leaf.nr_points;

    // If the voxel contains sufficient points, its covariance is calculated and
    // is added to the voxel centroids and output clouds.
    // Points with less than the minimum points will have a can not be
    // accurately approximated using a normal distribution.
    if (leaf.nr_points < min_points_per_voxel_) {
      continue;
    }

    output.push_back(PointT());

    output.back().x = leaf.centroid[0];
    output.back().y = leaf.centroid[1];
    output.back().z = leaf.centroid[2];

    // Stores the voxel indice for fast access searching
    voxel_centroids_leaf_indices_.push_back(static_cast<int>(it->first));

    // Single pass covariance calculation
    leaf.cov_ =
      (leaf.cov_ - pt_sum * leaf.mean_.transpose()) / (leaf.nr_points - 1.0);

    // Normalize Eigen Val such that max no more than 100x min.
    eigensolver.compute(leaf.cov_);
    Eigen::Matrix3d eigen_val = eigensolver.eigenvalues().asDiagonal();
    leaf.evecs_ = eigensolver.eigenvectors();

    if (eigen_val(0, 0) < -Eigen::NumTraits<double>::dummy_precision() ||
        eigen_val(1, 1) < -Eigen::NumTraits<double>::dummy_precision() ||
        eigen_val(2, 2) <= 0) {
      std::cerr << "Invalid eigen value! "
                << "(" << eigen_val(0, 0)
                << "," << eigen_val(1, 1)
                << "," << eigen_val(2, 2)
                << ")" << std::endl;
      assert(false);
      leaf.nr_points = -1;
      continue;
    }

    // Avoids matrices near singularities (eq 6.11)[Magnusson 2009]

    const double min_cov_eigvalue_mult_ = 0.01;
    const double min_cov_eigvalue = min_cov_eigvalue_mult_ * eigen_val(2, 2);
    if (eigen_val(0, 0) < min_cov_eigvalue) {
      eigen_val(0, 0) = min_cov_eigvalue;

      if (eigen_val(1, 1) < min_cov_eigvalue) {
        eigen_val(1, 1) = min_cov_eigvalue;
      }

      leaf.cov_ = leaf.evecs_ * eigen_val * leaf.evecs_.inverse();
    }
    leaf.evals_ = eigen_val.diagonal();

    leaf.icov_ = leaf.cov_.inverse();
    if (leaf.icov_.maxCoeff() == std::numeric_limits<float>::infinity() ||
        leaf.icov_.minCoeff() == -std::numeric_limits<float>::infinity()) {
      assert(false);
      leaf.nr_points = -1;
    }
  }

  output.width = output.size();
}

 private:
  const Eigen::Array4f inverse_leaf_size_;
  std::map<std::size_t, Leaf> leaves_;
  Eigen::Vector4i min_b_;
  Eigen::Vector4i divb_mul_;
};

#endif  // INCLUDE_MAPPING_FILTER_HPP_
