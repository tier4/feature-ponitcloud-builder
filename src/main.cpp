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
  const Eigen::Vector3d & voxel_size,
  const int min_points_per_voxel)
{
  pcl::VoxelGridCovariance<PointT> cells;
  cells.setLeafSize(voxel_size(0), voxel_size(1), voxel_size(2));
  cells.setInputCloud(cloud);
  cells.filter(true);
  cells.setMinPointPerVoxel(min_points_per_voxel);
  return cells;
}

template<typename PointT>
Eigen::Vector3d xyz_as_vector3d(const PointT & p)
{
  return Eigen::Vector3d(p.x, p.y, p.z);
}

template<typename PointT>
bool has_edge(const Eigen::Vector3d & eigenvalues, const double eigenvalue_ratio)
{
  return
    eigenvalues(2) > eigenvalues(0) * eigenvalue_ratio &&
    eigenvalues(2) > eigenvalues(1) * eigenvalue_ratio;
}

template<typename PointT>
bool has_surface(const Eigen::Vector3d & eigenvalues, const double eigenvalue_ratio)
{
  return
    eigenvalues(0) < eigenvalues(2) * eigenvalue_ratio &&
    eigenvalues(0) < eigenvalues(1) * eigenvalue_ratio;
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

inline double point_plane_distance(const Eigen::VectorXd & w, const Eigen::VectorXd & x)
{
  assert(w.size() == x.size());
  return std::fabs(w.dot(x)) / w.norm();
}

template<typename PointT>
bool has_valid_covariance(
  const typename pcl::VoxelGridCovariance<PointT>::LeafConstPtr & leaf)
{
  return leaf->nr_points >= 0;
}

Eigen::Vector3d surface_normal(const Eigen::Matrix3d & eigenvectors)
{
  // eigen vector corresponding to the smallest eigen value
  return eigenvectors.col(0);
}

template<typename PointT>
double point_plane_distance(
  const typename pcl::VoxelGridCovariance<PointT>::LeafConstPtr & leaf,
  const pcl::PointXYZ & p)
{
  const Eigen::Vector3d d = xyz_as_vector3d(p) - leaf->mean_;
  return point_plane_distance(surface_normal(leaf->evecs_), d);
}

template<typename PointT>
double edge_point_distance(
  const typename pcl::VoxelGridCovariance<PointT>::LeafConstPtr & leaf, const pcl::PointXYZ & p)
{
  const Eigen::Vector3d diff = xyz_as_vector3d(p) - leaf->mean_;
  return mahalanobis(leaf->icov_, principal(leaf->evecs_), diff);
}

constexpr double edge_neighbor_threshold = 1.0;
constexpr double surface_neighbor_threshold = 1.0;
constexpr double max_mahalanobis = 1.0;
constexpr double voxel_size = 1.0;
constexpr int min_points_per_voxel = 10;
constexpr double edge_eigenvalue_ratio = 8.0;
constexpr double surface_eigenvalue_ratio = 0.1;

int main(int argc, char * argv[])
{
  if (argc < 2) {
    std::cout << "Usage: ./mapping <pcd file>" << std::endl;
    return 1;
  }

  using PointT = pcl::PointXYZ;

  const Eigen::Vector3d voxel_sizes(voxel_size, voxel_size, voxel_size);

  const auto pcd_cloud = read_pcd<PointT>(argv[1]);
  auto voxels = crop_into_voxels<PointT>(pcd_cloud, voxel_sizes, min_points_per_voxel);

  pcl::PointCloud<PointT>::Ptr edge(new pcl::PointCloud<PointT>);
  pcl::PointCloud<PointT>::Ptr surface(new pcl::PointCloud<PointT>);
  for (auto p : *pcd_cloud) {
    const pcl::VoxelGridCovariance<PointT>::LeafConstPtr leaf = voxels.getLeaf(p);

    if (leaf == nullptr) {
      continue;
    }

    if (!has_valid_covariance<PointT>(leaf)) {
      continue;
    }

    if (has_edge<PointT>(leaf->evals_, edge_eigenvalue_ratio) &&
        edge_point_distance<PointT>(leaf, p) < edge_neighbor_threshold) {
      edge->push_back(p);
    }

    if (has_surface<PointT>(leaf->evals_, surface_eigenvalue_ratio) &&
        point_plane_distance<PointT>(leaf, p) < surface_neighbor_threshold) {
      surface->push_back(p);
    }
  }

  pcl::io::savePCDFile<PointT>(std::string{"edge.pcd"}, *edge, true);
  pcl::io::savePCDFile<PointT>(std::string{"surface.pcd"}, *surface, true);

  return 0;
}
