/*
 * Unit tests for pclomp::GeneralizedIterativeClosestPoint.
 *
 * Laid out like PCL's test/registration tests: gtest, one executable per file, the data file
 * passed as a command-line argument (see test/CMakeLists.txt).
 */
#include <gtest/gtest.h>

#include <pcl/common/transforms.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pclomp/gicp_omp.h>
#include <gicp_omp_impl.hpp>

#include <cmath>
#include <random>

using PointT = pcl::PointXYZ;
using CloudT = pcl::PointCloud<PointT>;
using GICP = pclomp::GeneralizedIterativeClosestPoint<PointT, PointT>;
using Vector6d = GICP::Vector6d;
using Matrix6d = GICP::Matrix6d;

namespace {

CloudT::Ptr cloud_target;  // downsampled data file, loaded in main()

// Exposes the protected optimization functor and sets up the state it reads (point pairs,
// Mahalanobis matrices) directly, without running a registration.
class GICPProbe : public GICP {
public:
  using GICP::OptimizationFunctorWithIndices;

  explicit GICPProbe(int seed, int n = 300) {
    std::mt19937 rng(seed);
    std::normal_distribution<float> nd(0.f, 1.f);
    for (int i = 0; i < n; ++i) {
      const PointT p(5 * nd(rng), 5 * nd(rng), 2 * nd(rng));
      src_.push_back(p);
      tgt_.push_back(PointT(p.x + 0.3f + 0.05f * nd(rng), p.y - 0.2f + 0.05f * nd(rng),
                            p.z + 0.1f + 0.05f * nd(rng)));
      idx_.push_back(i);
      Eigen::Matrix3f a;
      for (int r = 0; r < 3; ++r)
        for (int c = 0; c < 3; ++c) a(r, c) = nd(rng);
      mahalanobis_.push_back(Eigen::Matrix4f::Zero());
      mahalanobis_.back().topLeftCorner<3, 3>() =
          a * a.transpose() + 0.1f * Eigen::Matrix3f::Identity();
    }
    tmp_src_ = &src_;
    tmp_tgt_ = &tgt_;
    tmp_idx_src_ = &idx_;
    tmp_idx_tgt_ = &idx_;
    // computeTransformation() always optimizes relative to an identity base. The rotation terms
    // of df/dfddf use base_transformation_ * p including the base's translation, which
    // applyState() does not rotate, so they only hold for a base without translation.
    base_transformation_ = Eigen::Matrix4f::Identity();
  }

  void setGradientTolerances(double translation, double rotation) {
    translation_gradient_tolerance_ = translation;
    rotation_gradient_tolerance_ = rotation;
  }

private:
  CloudT src_, tgt_;
  std::vector<int> idx_;
};

// Parameter vectors (x, y, z, roll, pitch, yaw) to evaluate the derivatives at.
const std::vector<Vector6d>& testStates() {
  static const std::vector<Vector6d> states = [] {
    std::vector<Vector6d> s(4);
    s[0] << 0, 0, 0, 0, 0, 0;
    s[1] << 0.1, -0.2, 0.05, 0.3, -0.4, 0.7;
    s[2] << -0.3, 0.1, 0.2, -1.0, 1.2, -2.5;
    s[3] << 1.0, 2.0, -0.5, 2.0, -0.2, 3.0;
    return s;
  }();
  return states;
}

double relativeError(const Eigen::MatrixXd& actual, const Eigen::MatrixXd& expected) {
  return (actual - expected).norm() / expected.norm();
}

// Central differences; the cost is evaluated in float, so a large step keeps the rounding noise
// (~1e-7 relative / h) well below the tolerances used below.
constexpr double kStep = 1e-3;

}  // namespace

TEST(PCLOMP_GICP, GradientMatchesFiniteDifferences) {
  for (int seed = 1; seed <= 3; ++seed) {
    GICPProbe probe(seed);
    GICPProbe::OptimizationFunctorWithIndices functor(&probe);
    for (const Vector6d& x : testStates()) {
      Vector6d g_fd;
      for (int k = 0; k < 6; ++k) {
        const Vector6d e = kStep * Vector6d::Unit(k);
        g_fd[k] = (functor(x + e) - functor(x - e)) / (2 * kStep);
      }
      Vector6d g_df, g_fdf, g_newton;
      Matrix6d hessian;
      double f_fdf;
      functor.df(x, g_df);
      functor.fdf(x, f_fdf, g_fdf);
      functor.dfddf(x, g_newton, hessian);
      EXPECT_LT(relativeError(g_df, g_fd), 1e-3) << "seed " << seed << ", x " << x.transpose();
      EXPECT_LT(relativeError(g_newton, g_fd), 1e-3) << "seed " << seed << ", x " << x.transpose();
      EXPECT_NEAR(f_fdf, functor(x), 1e-6 * std::abs(f_fdf));
      EXPECT_LT(relativeError(g_fdf, g_df), 1e-6);
    }
  }
}

TEST(PCLOMP_GICP, HessianMatchesFiniteDifferences) {
  for (int seed = 1; seed <= 3; ++seed) {
    GICPProbe probe(seed);
    GICPProbe::OptimizationFunctorWithIndices functor(&probe);
    for (const Vector6d& x : testStates()) {
      Matrix6d h_fd;
      for (int k = 0; k < 6; ++k) {
        const Vector6d e = kStep * Vector6d::Unit(k);
        Vector6d g_plus, g_minus;
        functor.df(x + e, g_plus);
        functor.df(x - e, g_minus);
        h_fd.col(k) = (g_plus - g_minus) / (2 * kStep);
      }
      Vector6d gradient;
      Matrix6d hessian;
      functor.dfddf(x, gradient, hessian);
      EXPECT_LT(relativeError(hessian, h_fd), 2e-3) << "seed " << seed << ", x " << x.transpose()
                                                    << "\nanalytic:\n" << hessian
                                                    << "\nnumeric:\n" << h_fd;
      EXPECT_EQ(hessian, hessian.transpose());
    }
  }
}

TEST(PCLOMP_GICP, CheckGradient) {
  GICPProbe probe(1);
  GICPProbe::OptimizationFunctorWithIndices functor(&probe);
  probe.setGradientTolerances(1e-2, 1e-3);
  Vector6d g;
  g << 5e-3, -5e-3, 0, 5e-4, 0, -5e-4;
  EXPECT_EQ(functor.checkGradient(g), BFGSSpace::Success);
  g << 2e-2, 0, 0, 0, 0, 0;  // translation part above its tolerance
  EXPECT_EQ(functor.checkGradient(g), BFGSSpace::Running);
  g << 0, 0, 0, 0, 2e-3, 0;  // rotation part above its tolerance
  EXPECT_EQ(functor.checkGradient(g), BFGSSpace::Running);
  probe.setGradientTolerances(-1, 1e-3);
  EXPECT_EQ(functor.checkGradient(g), BFGSSpace::NegativeGradientEpsilon);
}

TEST(PCLOMP_GICP, RejectsTooFewCorrespondences) {
  GICP gicp;
  CloudT cloud;
  for (int i = 0; i < 2; ++i) cloud.push_back(PointT(i, 2 * i, 0));
  const std::vector<int> indices{0, 1};
  Eigen::Matrix4f t = Eigen::Matrix4f::Identity();
  EXPECT_THROW(gicp.estimateRigidTransformationNewton(cloud, indices, cloud, indices, t),
               pcl::NotEnoughPointsException);
  EXPECT_THROW(gicp.estimateRigidTransformationBFGS(cloud, indices, cloud, indices, t),
               pcl::NotEnoughPointsException);
}

// Aligns a transformed copy of the data cloud back onto it; the result must recover the known
// offset. Runs the default Newton optimizer and BFGS.
class PCLOMP_GICP_Align : public ::testing::TestWithParam<bool> {};

TEST_P(PCLOMP_GICP_Align, RecoversKnownOffset) {
  const bool use_bfgs = GetParam();
  const Eigen::Affine3f offset =
      Eigen::Translation3f(0.3f, -0.2f, 0.1f) *
      Eigen::AngleAxisf(0.05f, Eigen::Vector3f(0.2f, 0.3f, 1.f).normalized());
  CloudT::Ptr source(new CloudT);
  pcl::transformPointCloud(*cloud_target, *source, offset.inverse());

  GICP gicp;
  if (use_bfgs) gicp.useBFGS();
  gicp.setMaxCorrespondenceDistance(1.0);
  gicp.setInputSource(source);
  gicp.setInputTarget(cloud_target);
  CloudT aligned;
  gicp.align(aligned);

  ASSERT_TRUE(gicp.hasConverged());
  const Eigen::Matrix4f error = gicp.getFinalTransformation() * offset.inverse().matrix();
  const float translation_error = error.block<3, 1>(0, 3).norm();
  const float angle_error =
      Eigen::AngleAxisf(Eigen::Matrix3f(error.topLeftCorner<3, 3>())).angle();
  // Measured: Newton < 1 um / 1e-8 rad, BFGS ~0.1 mm / 4e-6 rad.
  EXPECT_LT(translation_error, 1e-3) << "final transformation:\n" << gicp.getFinalTransformation();
  EXPECT_LT(angle_error, 1e-4) << "final transformation:\n" << gicp.getFinalTransformation();
}

INSTANTIATE_TEST_SUITE_P(Optimizers, PCLOMP_GICP_Align, ::testing::Values(false, true),
                         [](const ::testing::TestParamInfo<bool>& info) {
                           return info.param ? "BFGS" : "Newton";
                         });

int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
  if (argc < 2) {
    std::cerr << "No test file given. Please pass the path to data/251370668.pcd." << std::endl;
    return -1;
  }
  CloudT::Ptr raw(new CloudT);
  if (pcl::io::loadPCDFile(argv[1], *raw) < 0) {
    std::cerr << "Failed to read test file " << argv[1] << std::endl;
    return -1;
  }
  cloud_target.reset(new CloudT);
  pcl::VoxelGrid<PointT> voxel_grid;
  voxel_grid.setLeafSize(0.3f, 0.3f, 0.3f);
  voxel_grid.setInputCloud(raw);
  voxel_grid.filter(*cloud_target);
  return RUN_ALL_TESTS();
}
