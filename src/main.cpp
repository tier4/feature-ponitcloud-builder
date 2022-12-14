/*
 * Software License Agreement (BSD License)
 *
 *  Copyright (c) 2022, Takeshi Ishita
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

#include <pcl/filters/voxel_grid_covariance.h>
#include <pcl/io/pcd_io.h>
#include <pcl/common/io.h>
#include <pcl/point_types.h>

#include <Eigen/Core>

#include <cstdlib>
#include <iostream>
#include <string>

#include "mapping/filter.hpp"
#include "mapping/leaf.hpp"


template<typename PointT>
typename pcl::PointCloud<PointT>::Ptr read_pcd(const std::string & filepath) {
  typename pcl::PointCloud<PointT>::Ptr cloud(new pcl::PointCloud<PointT>);
  const int success = pcl::io::loadPCDFile<PointT>(filepath, *cloud);
  if (success == -1) {
    throw std::invalid_argument("Couldn't read file" + filepath);
  }
  return cloud;
}

template<typename PointT>
Eigen::Vector3d xyz_as_vector3d(const PointT & p) {
  return Eigen::Vector3d(p.x, p.y, p.z);
}

template<typename PointT>
bool has_edge(
  const Eigen::Vector3d & eigenvalues,
  const double eigenvalue_ratio) {
  return
    eigenvalues(2) > eigenvalues(0) * eigenvalue_ratio &&
    eigenvalues(2) > eigenvalues(1) * eigenvalue_ratio;
}

template<typename PointT>
bool has_surface(
  const Eigen::Vector3d & eigenvalues,
  const double eigenvalue_ratio) {
  return
    eigenvalues(0) < eigenvalues(2) * eigenvalue_ratio &&
    eigenvalues(0) < eigenvalues(1) * eigenvalue_ratio;
}

inline double mahalanobis(
  const Eigen::Matrix3d & inverse_covariance,
  const Eigen::Vector3d & x1,
  const Eigen::Vector3d & x2) {
  return x1.transpose() * inverse_covariance * x2;
}

inline Eigen::Vector3d principal(const Eigen::Matrix3d & eigenvectors) {
  return eigenvectors.col(2);
}

inline double point_plane_distance(
  const Eigen::VectorXd & w,
  const Eigen::VectorXd & x) {
  assert(w.size() == x.size());
  return std::fabs(w.dot(x)) / w.norm();
}

template<typename PointT>
bool has_valid_covariance(const LeafConstPtr & leaf) {
  return leaf->nr_points >= 0;
}

Eigen::Vector3d surface_normal(const Eigen::Matrix3d & eigenvectors) {
  // eigen vector corresponding to the smallest eigen value
  return eigenvectors.col(0);
}

template<typename PointT>
double point_plane_distance(
  const LeafConstPtr & leaf,
  const pcl::PointXYZ & p) {
  const Eigen::Vector3d d = xyz_as_vector3d(p) - leaf->mean_;
  return point_plane_distance(surface_normal(leaf->evecs_), d);
}

template<typename PointT>
double edge_point_distance(
  const LeafConstPtr & leaf,
  const pcl::PointXYZ & p) {
  const Eigen::Vector3d diff = xyz_as_vector3d(p) - leaf->mean_;
  return mahalanobis(leaf->icov_, principal(leaf->evecs_), diff);
}

template<typename PointT>
bool is_valid(const LeafConstPtr & leaf) {
  return leaf != nullptr && has_valid_covariance<PointT>(leaf);
}

template<typename PointT>
typename pcl::PointCloud<PointT>::Ptr ExtractEdge(
  const Filter<PointT> & voxels,
  const typename pcl::PointCloud<PointT>::Ptr & input_cloud,
  const double edge_eigenvalue_ratio,
  const double edge_neighbor_threshold) {
  typename pcl::PointCloud<PointT>::Ptr edge(new pcl::PointCloud<PointT>);
  for (auto p : *input_cloud) {
    const auto leaf = voxels.getLeaf(p);

    if (!is_valid<PointT>(leaf)) {
      continue;
    }

    if (has_edge<PointT>(leaf->evals_, edge_eigenvalue_ratio) &&
        edge_point_distance<PointT>(leaf, p) < edge_neighbor_threshold) {
      edge->push_back(p);
    }
  }
  return edge;
}

template<typename PointT>
typename pcl::PointCloud<PointT>::Ptr ExtractSurface(
  const Filter<PointT> & voxels,
  const typename pcl::PointCloud<PointT>::Ptr & input_cloud,
  const double surface_eigenvalue_ratio,
  const double surface_neighbor_threshold) {
  typename pcl::PointCloud<PointT>::Ptr surface(new pcl::PointCloud<PointT>);
  for (auto p : *input_cloud) {
    const auto leaf = voxels.getLeaf(p);

    if (!is_valid<PointT>(leaf)) {
      continue;
    }

    if (has_surface<PointT>(leaf->evals_, surface_eigenvalue_ratio) &&
        point_plane_distance<PointT>(leaf, p) < surface_neighbor_threshold) {
      surface->push_back(p);
    }
  }
  return surface;
}

constexpr int min_points_per_voxel = 10;
constexpr int min_points_per_voxel_ = 0;

int main(int argc, char * argv[]) {
  if (argc < 2) {
    std::cout << "Usage: ./mapping <pcd file>" << std::endl;
    return 1;
  }

  using PointT = pcl::PointXYZ;

  const auto pcd_cloud = read_pcd<PointT>(argv[1]);

  const Eigen::Vector3d voxel_sizes1(2., 2., 2.);

  const Filter<PointT> filter1(pcd_cloud, voxel_sizes1, min_points_per_voxel_);
  const auto edge1 = ExtractEdge(filter1, pcd_cloud, 5.0, 0.2);
  const auto surface1 = ExtractSurface(filter1, pcd_cloud, 0.1, 1.0);

  const Eigen::Vector3d voxel_sizes2(0.5, 0.5, 0.5);
  const Filter<PointT> filter2(pcd_cloud, voxel_sizes2, min_points_per_voxel_);
  const auto edge2 = ExtractEdge(filter2, edge1, 5.0, 0.1);
  const auto surface2 = ExtractSurface(filter2, surface1, 0.1, 1.0);

  // std::cout << "edge1->size() == " << edge1->size() << std::endl;
  // std::cout << "surface1->size() == " << surface1->size() << std::endl;
  pcl::io::savePCDFile<PointT>(std::string{"edge.pcd"}, *edge1, true);
  pcl::io::savePCDFile<PointT>(std::string{"surface.pcd"}, *surface1, true);

  return 0;
}
