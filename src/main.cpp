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


int main(int argc, char * argv[])
{
  if (argc < 2) {
    std::cout << "Usage: ./mapping <pcd file>" << std::endl;
    return 1;
  }

  using PointT = pcl::PointXYZ;

  const auto cloud = read_pcd<PointT>(argv[1]);
  crop_into_voxels<PointT>(cloud, Eigen::Vector3d(1., 1., 1.));

  return 0;
}
