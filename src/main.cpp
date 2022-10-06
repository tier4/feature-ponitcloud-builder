#include <Eigen/Core>

#include <iostream>

#include <pcl/filters/voxel_grid_covariance.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>

template<typename PointT>
typename pcl::PointCloud<PointT>::Ptr read_pcd(const std::string & filepath)
{
  typename pcl::PointCloud<PointT>::Ptr cloud(new pcl::PointCloud<PointT>);
  const int success = pcl::io::loadPCDFile<PointT>(filepath, *cloud);
  if (success == -1)
  {
    throw std::invalid_argument("Couldn't read file" + filepath);
  }
  return cloud;
}

template<typename PointT>
pcl::VoxelGridCovariance<PointT> crop_into_voxels(
  const typename pcl::PointCloud<PointT>::ConstPtr & cloud,
  const Eigen::Vector3d & voxel_size)
{
  pcl::VoxelGridCovariance<PointT> cells;
  cells.setLeafSize(voxel_size(0), voxel_size(1), voxel_size(2));
  cells.setInputCloud(cloud);
  cells.filter(true);
  return cells;
}

template<typename PointT>
Eigen::Vector3d xyz_as_vector3d(const PointT & p)
{
  return Eigen::Vector3d(p.x, p.y, p.z);
}

template<typename PointT>
bool is_valid(const typename pcl::VoxelGridCovariance<PointT>::LeafConstPtr & leaf)
{
  return
    leaf->nr_points > 10 &&
    leaf->evals_(2) > leaf->evals_(0) * 10.0 &&
    leaf->evals_(2) > leaf->evals_(1) * 5.0;
}

inline double mahalanobis(
  const Eigen::Matrix3d & inverse_covariance,
  const Eigen::Vector3d & x1,
  const Eigen::Vector3d & x2)
{
  return x1.transpose() * inverse_covariance * x2;
}

inline Eigen::Vector3d principal(const Eigen::Matrix3d & eigenvectors)
{
  return eigenvectors.col(2);
}

constexpr double max_mahalanobis = 1.5;

int main(int argc, char * argv[])
{
  if (argc < 2) {
    std::cout << "Usage: ./mapping <pcd file>" << std::endl;
    return 1;
  }

  using PointT = pcl::PointXYZ;

  const double voxel_size = 2.0;
  const Eigen::Vector3d voxel_sizes(voxel_size, voxel_size, voxel_size);
  const auto pcd_cloud = read_pcd<PointT>(argv[1]);
  auto voxels = crop_into_voxels<PointT>(pcd_cloud, voxel_sizes);

  pcl::PointCloud<PointT>::Ptr filtered(new pcl::PointCloud<PointT>);
  for (auto p : *pcd_cloud) {
    const auto leaf = voxels.getLeaf(p);
    if (!is_valid<PointT>(leaf)) {
     continue;
    }

    const Eigen::Vector3d diff = xyz_as_vector3d(p) - leaf->mean_;
    if (mahalanobis(leaf->icov_, principal(leaf->evecs_), diff) < max_mahalanobis) {
      filtered->push_back(p);
    }
  }

  pcl::io::savePCDFile<PointT>(std::string{"filtered.pcd"}, *filtered, true);

  return 0;
}
