#include <Eigen/Core>

#include <chrono>
#include <cstdlib>
#include <iostream>
#include <string>

#include <pcl/filters/voxel_grid_covariance.h>
#include <pcl/io/pcd_io.h>
#include <pcl/common/io.h>
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

template<typename PointT>
bool is_valid(const typename pcl::VoxelGridCovariance<PointT>::LeafConstPtr & leaf) {
  return leaf != nullptr && has_valid_covariance<PointT>(leaf);
}

template<typename PointT>
typename pcl::PointCloud<PointT>::Ptr ExtractEdge(
  typename pcl::VoxelGridCovariance<PointT> & voxels,
  const typename pcl::PointCloud<PointT>::Ptr & input_cloud,
  const double edge_eigenvalue_ratio,
  const double edge_neighbor_threshold)
{
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
  typename pcl::VoxelGridCovariance<PointT> & voxels,
  const typename pcl::PointCloud<PointT>::Ptr & input_cloud,
  const double surface_eigenvalue_ratio,
  const double surface_neighbor_threshold)
{
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

constexpr double voxel_size = 1.0;
constexpr int min_points_per_voxel = 10;
constexpr double edge_neighbor_threshold = 0.1;
constexpr double surface_neighbor_threshold = 1.0;
constexpr double edge_eigenvalue_ratio = 10.0;
constexpr double surface_eigenvalue_ratio = 0.1;

int main(int argc, char * argv[])
{
  if (argc < 2) {
    std::cout << "Usage: ./mapping <pcd file>" << std::endl;
    return 1;
  }

  using std::chrono::duration_cast;
  using std::chrono::high_resolution_clock;
  using std::chrono::milliseconds;
  using PointT = pcl::PointXYZ;

  const Eigen::Vector3d voxel_sizes(voxel_size, voxel_size, voxel_size);

  const auto pcd_cloud = read_pcd<PointT>(argv[1]);

  const auto start_crop = high_resolution_clock::now();

  auto voxels = crop_into_voxels<PointT>(pcd_cloud, voxel_sizes, min_points_per_voxel);

  const auto stop_crop = high_resolution_clock::now();
  const auto start_mapping = high_resolution_clock::now();

  const auto edge = ExtractEdge(
    voxels, pcd_cloud,
    edge_eigenvalue_ratio,
    edge_neighbor_threshold);
  const auto surface = ExtractSurface(
    voxels, pcd_cloud,
    surface_eigenvalue_ratio,
    surface_neighbor_threshold);

  const auto stop_mapping = high_resolution_clock::now();
  const auto duration_crop = duration_cast<milliseconds>(stop_crop - start_crop);
  const auto duration_mapping = duration_cast<milliseconds>(stop_mapping - start_mapping);

  std::cout << "voxel creation : " << duration_crop.count() << " milliseconds" << std::endl;
  std::cout << "map creation   : " << duration_mapping.count() << " milliseconds" << std::endl;

  pcl::io::savePCDFile<PointT>(std::string{"edge.pcd"}, *edge, true);
  pcl::io::savePCDFile<PointT>(std::string{"surface.pcd"}, *surface, true);

  return 0;
}
