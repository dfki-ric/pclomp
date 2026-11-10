/*
 * Unit tests for pclomp::NormalDistributionsTransform.
 *
 * Laid out like PCL's test/registration tests: gtest, one executable per file, the data file
 * passed as a command-line argument (see test/CMakeLists.txt).
 */
#include <gtest/gtest.h>

#include <pcl/common/transforms.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pclomp/ndt_omp.h>
#include <ndt_omp_impl.hpp>

#include <cmath>
#include <random>

using PointT = pcl::PointXYZ;
using CloudT = pcl::PointCloud<PointT>;
using NDT = pclomp::NormalDistributionsTransform<PointT, PointT>;
using Vector6d = Eigen::Matrix<double, 6, 1>;
using Matrix6d = Eigen::Matrix<double, 6, 6>;

namespace {

CloudT::Ptr cloud_target;  // downsampled data file, loaded in main()

// Exposes the protected derivative helpers.
class NDTProbe : public NDT {
public:
  using NDT::computeAngleDerivatives;
  using NDT::computePointDerivatives;
  using NDT::updateDerivatives;
  using NDT::gauss_d1_;
  using NDT::gauss_d2_;
};

// T(p) x as NDT parametrizes it: translation, then roll about X, pitch about Y, yaw about Z
// (x' = Rx(p3) Ry(p4) Rz(p5) x + t).
Eigen::Vector3d transformPoint(const Vector6d& p, const Eigen::Vector3d& x) {
  const Eigen::Affine3d t = Eigen::Translation3d(p.head<3>()) *
                            Eigen::AngleAxisd(p[3], Eigen::Vector3d::UnitX()) *
                            Eigen::AngleAxisd(p[4], Eigen::Vector3d::UnitY()) *
                            Eigen::AngleAxisd(p[5], Eigen::Vector3d::UnitZ());
  return t * x;
}

// Parameter vectors to evaluate the derivatives at. computeAngleDerivatives() treats angles
// below 1e-4 as exactly zero, so the angles stay clear of that.
const std::vector<Vector6d>& testStates() {
  static const std::vector<Vector6d> states = [] {
    std::vector<Vector6d> s(4);
    s[0] << 0.1, -0.2, 0.05, 0.3, -0.4, 0.7;
    s[1] << -0.3, 0.1, 0.2, -1.0, 1.2, -2.5;
    s[2] << 1.0, 2.0, -0.5, 2.0, -0.2, 3.0;
    s[3] << 0, 0, 0, 0.01, 0.02, -0.01;
    return s;
  }();
  return states;
}

const std::vector<Eigen::Vector3d>& testPoints() {
  static const std::vector<Eigen::Vector3d> points{
      {1.0, 2.0, 3.0}, {-4.0, 0.5, 1.5}, {10.0, -7.0, -2.0}};
  return points;
}

// Second derivatives of T(p) x by central differences: column j of block i is
// d^2 T(p) x / dp_i dp_j.
Eigen::Matrix<double, 18, 6> numericPointHessian(const Vector6d& p, const Eigen::Vector3d& x) {
  const double h = 1e-4;
  Eigen::Matrix<double, 18, 6> hessian;
  for (int i = 0; i < 6; ++i) {
    for (int j = 0; j < 6; ++j) {
      const Vector6d ei = h * Vector6d::Unit(i), ej = h * Vector6d::Unit(j);
      hessian.block<3, 1>(3 * i, j) =
          (transformPoint(p + ei + ej, x) - transformPoint(p + ei - ej, x) -
           transformPoint(p - ei + ej, x) + transformPoint(p - ei - ej, x)) /
          (4 * h * h);
    }
  }
  return hessian;
}

Eigen::Matrix<double, 3, 6> numericPointGradient(const Vector6d& p, const Eigen::Vector3d& x) {
  const double h = 1e-6;
  Eigen::Matrix<double, 3, 6> gradient;
  for (int i = 0; i < 6; ++i) {
    const Vector6d e = h * Vector6d::Unit(i);
    gradient.col(i) = (transformPoint(p + e, x) - transformPoint(p - e, x)) / (2 * h);
  }
  return gradient;
}

}  // namespace

// The double-precision point derivatives (Magnusson 2009, eq. 6.18-6.21).
TEST(PCLOMP_NDT, PointDerivativesMatchFiniteDifferences) {
  NDTProbe ndt;
  for (Vector6d p : testStates()) {
    ndt.computeAngleDerivatives(p, true);
    for (Eigen::Vector3d x : testPoints()) {
      Eigen::Matrix<double, 3, 6> gradient = Eigen::Matrix<double, 3, 6>::Zero();
      gradient.block<3, 3>(0, 0).setIdentity();
      Eigen::Matrix<double, 18, 6> hessian = Eigen::Matrix<double, 18, 6>::Zero();
      ndt.computePointDerivatives(x, gradient, hessian, true);
      const double tol = 1e-5 * x.norm();
      EXPECT_TRUE(gradient.isApprox(numericPointGradient(p, x), 1e-6))
          << "p " << p.transpose() << ", x " << x.transpose() << "\nanalytic:\n"
          << gradient << "\nnumeric:\n" << numericPointGradient(p, x);
      EXPECT_LT((hessian - numericPointHessian(p, x)).cwiseAbs().maxCoeff(), tol)
          << "p " << p.transpose() << ", x " << x.transpose() << "\nanalytic:\n"
          << hessian << "\nnumeric:\n" << numericPointHessian(p, x);
    }
  }
}

// The single-precision variant that computeDerivatives() actually uses (4-row blocks). Before the
// h_ang.row(6) sign fix, its d^2/dpitch^2 z component had the wrong sign.
TEST(PCLOMP_NDT, FloatPointDerivativesMatchFiniteDifferences) {
  NDTProbe ndt;
  for (Vector6d p : testStates()) {
    ndt.computeAngleDerivatives(p, true);
    for (Eigen::Vector3d x : testPoints()) {
      Eigen::Matrix<float, 4, 6> gradient4 = Eigen::Matrix<float, 4, 6>::Zero();
      gradient4.block<3, 3>(0, 0).setIdentity();
      Eigen::Matrix<float, 24, 6> hessian4 = Eigen::Matrix<float, 24, 6>::Zero();
      ndt.computePointDerivatives(x, gradient4, hessian4, true);

      const Eigen::Matrix<double, 3, 6> gradient_fd = numericPointGradient(p, x);
      const Eigen::Matrix<double, 18, 6> hessian_fd = numericPointHessian(p, x);
      const double tol = 1e-5 * x.norm();
      EXPECT_LT((gradient4.topRows<3>().cast<double>() - gradient_fd).cwiseAbs().maxCoeff(), tol)
          << "p " << p.transpose() << ", x " << x.transpose();
      EXPECT_EQ(gradient4.row(3).norm(), 0.f);
      for (int i = 0; i < 6; ++i) {
        const Eigen::Matrix<double, 3, 6> block = hessian4.block<3, 6>(4 * i, 0).cast<double>();
        EXPECT_LT((block - hessian_fd.block<3, 6>(3 * i, 0)).cwiseAbs().maxCoeff(), tol)
            << "p " << p.transpose() << ", x " << x.transpose() << ", block " << i
            << "\nanalytic:\n" << block << "\nnumeric:\n" << hessian_fd.block<3, 6>(3 * i, 0);
        EXPECT_EQ(hessian4.row(4 * i + 3).norm(), 0.f);
      }
    }
  }
}

// Score, gradient and Hessian of one point against one voxel (eq. 6.9, 6.12, 6.13) against the
// score function evaluated directly.
TEST(PCLOMP_NDT, ScoreDerivativesMatchFiniteDifferences) {
  NDTProbe ndt;
  const Eigen::Vector3d x(1.0, -0.5, 0.8);
  const Eigen::Vector3d mean(1.3, -0.4, 0.6);
  Eigen::Matrix3d a;
  a << 0.4, 0.1, 0.0, -0.2, 0.3, 0.1, 0.1, 0.0, 0.2;
  const Eigen::Matrix3d c_inv = (a * a.transpose() + 0.05 * Eigen::Matrix3d::Identity()).inverse();

  const auto score = [&](const Vector6d& p) {
    const Eigen::Vector3d q = transformPoint(p, x) - mean;
    return -ndt.gauss_d1_ * std::exp(-0.5 * ndt.gauss_d2_ * q.dot(c_inv * q));
  };
  const auto analytic = [&](Vector6d p, Vector6d& gradient, Matrix6d& hessian) {
    ndt.computeAngleDerivatives(p, true);
    Eigen::Matrix<float, 4, 6> gradient4 = Eigen::Matrix<float, 4, 6>::Zero();
    gradient4.block<3, 3>(0, 0).setIdentity();
    Eigen::Matrix<float, 24, 6> hessian4 = Eigen::Matrix<float, 24, 6>::Zero();
    Eigen::Vector3d x_copy = x;
    ndt.computePointDerivatives(x_copy, gradient4, hessian4, true);
    gradient.setZero();
    hessian.setZero();
    return ndt.updateDerivatives(gradient, hessian, gradient4, hessian4,
                                 transformPoint(p, x) - mean, c_inv, true);
  };

  for (Vector6d p : testStates()) {
    p.head<3>() *= 0.1;  // keep the point within the voxel's Gaussian
    Vector6d gradient;
    Matrix6d hessian;
    const double s = analytic(p, gradient, hessian);
    ASSERT_NE(s, 0.0) << "point rejected as an outlier at p " << p.transpose();
    EXPECT_NEAR(s, score(p), 1e-5 * std::abs(score(p)));

    // The score function is double; the analytic gradient is computed in float, so its
    // differences need the larger step.
    Vector6d gradient_fd;
    Matrix6d hessian_fd;
    for (int i = 0; i < 6; ++i) {
      const Vector6d e = 1e-4 * Vector6d::Unit(i), e_hessian = 1e-2 * Vector6d::Unit(i);
      gradient_fd[i] = (score(p + e) - score(p - e)) / 2e-4;
      Vector6d g_plus, g_minus;
      Matrix6d unused;
      analytic(p + e_hessian, g_plus, unused);
      analytic(p - e_hessian, g_minus, unused);
      hessian_fd.col(i) = (g_plus - g_minus) / 2e-2;
    }
    EXPECT_LT((gradient - gradient_fd).norm(), 1e-4 * gradient_fd.norm() + 1e-6)
        << "p " << p.transpose() << "\nanalytic: " << gradient.transpose()
        << "\nnumeric:  " << gradient_fd.transpose();
    EXPECT_LT((hessian - hessian_fd).norm(), 1e-3 * hessian_fd.norm() + 1e-5)
        << "p " << p.transpose() << "\nanalytic:\n" << hessian << "\nnumeric:\n" << hessian_fd;
  }
}

// Aligns a transformed copy of the data cloud back onto it with each neighbor search method; the
// result must recover the known offset.
class PCLOMP_NDT_Align : public ::testing::TestWithParam<pclomp::NeighborSearchMethod> {};

TEST_P(PCLOMP_NDT_Align, RecoversKnownOffset) {
  const Eigen::Affine3f offset =
      Eigen::Translation3f(0.3f, -0.2f, 0.1f) *
      Eigen::AngleAxisf(0.05f, Eigen::Vector3f(0.2f, 0.3f, 1.f).normalized());
  CloudT::Ptr source(new CloudT);
  pcl::transformPointCloud(*cloud_target, *source, offset.inverse());

  NDT ndt;
  ndt.setNeighborhoodSearchMethod(GetParam());
  ndt.setResolution(1.0);
  ndt.setTransformationEpsilon(1e-4);
  ndt.setInputSource(source);
  ndt.setInputTarget(cloud_target);
  CloudT aligned;
  ndt.align(aligned);

  ASSERT_TRUE(ndt.hasConverged());
  const Eigen::Matrix4f error = ndt.getFinalTransformation() * offset.inverse().matrix();
  const float translation_error = error.block<3, 1>(0, 3).norm();
  const float angle_error =
      Eigen::AngleAxisf(Eigen::Matrix3f(error.topLeftCorner<3, 3>())).angle();
  // Measured: ~1.7 mm / ~0.1 mrad, except DIRECT26 with ~15 mm / ~2.1 mrad: its neighborhood
  // (pcl::getAllNeighborCellIndices(), same in PCL 1.14) is the 26 surrounding voxels WITHOUT the
  // one the point lies in.
  const bool direct26 = GetParam() == pclomp::DIRECT26;
  EXPECT_LT(translation_error, direct26 ? 3e-2 : 5e-3)
      << "final transformation:\n" << ndt.getFinalTransformation();
  EXPECT_LT(angle_error, direct26 ? 4e-3 : 5e-4)
      << "final transformation:\n" << ndt.getFinalTransformation();
}

INSTANTIATE_TEST_SUITE_P(SearchMethods, PCLOMP_NDT_Align,
                         ::testing::Values(pclomp::KDTREE, pclomp::DIRECT26, pclomp::DIRECT7,
                                           pclomp::DIRECT1),
                         [](const ::testing::TestParamInfo<pclomp::NeighborSearchMethod>& info) {
                           switch (info.param) {
                             case pclomp::KDTREE: return "KDTREE";
                             case pclomp::DIRECT26: return "DIRECT26";
                             case pclomp::DIRECT7: return "DIRECT7";
                             default: return "DIRECT1";
                           }
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
