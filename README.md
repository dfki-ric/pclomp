# ndt_omp
This package provides an OpenMP-boosted Normal Distributions Transform (and GICP) algorithm derived from pcl. The NDT algorithm is modified to be SSE-friendly and multi-threaded. It can run up to 10 times faster than its original version in pcl.

[![Build](https://github.com/koide3/ndt_omp/actions/workflows/build.yml/badge.svg)](https://github.com/koide3/ndt_omp/actions/workflows/build.yml)

### Benchmark (on Core i7-6700K)
```
$ roscd ndt_omp/data
$ rosrun ndt_omp align 251370668.pcd 251371071.pcd
--- pcl::NDT ---
single : 282.222[msec]
10times: 2921.92[msec]
fitness: 0.213937

--- pclomp::NDT (KDTREE, 1 threads) ---
single : 207.697[msec]
10times: 2059.19[msec]
fitness: 0.213937

--- pclomp::NDT (DIRECT7, 1 threads) ---
single : 139.433[msec]
10times: 1356.79[msec]
fitness: 0.214205

--- pclomp::NDT (DIRECT1, 1 threads) ---
single : 34.6418[msec]
10times: 317.03[msec]
fitness: 0.208511

--- pclomp::NDT (KDTREE, 8 threads) ---
single : 54.9903[msec]
10times: 500.51[msec]
fitness: 0.213937

--- pclomp::NDT (DIRECT7, 8 threads) ---
single : 63.1442[msec]
10times: 343.336[msec]
fitness: 0.214205

--- pclomp::NDT (DIRECT1, 8 threads) ---
single : 17.2353[msec]
10times: 100.025[msec]
fitness: 0.208511
```

Several methods for neighbor voxel search are implemented. If you select pclomp::KDTREE, results will be completely same as the original pcl::NDT. We recommend to use pclomp::DIRECT7 which is faster and stable. If you need extremely fast registration, choose pclomp::DIRECT1, but it might be a bit unstable.

<img src="data/screenshot.png" height="400pix" /><br>
Red: target, Green: source, Blue: aligned

## Unit tests
gtest-based, laid out like PCL's own tests (`test/`):
```
cmake -S . -B build -DBUILD_TESTING=ON && cmake --build build && (cd build && ctest --output-on-failure)
```
They check the GICP and NDT analytic derivatives against finite differences, GICP's gradient
convergence test, and that both recover a known offset on `data/251370668.pcd` (GICP with the
Newton and BFGS optimizers, NDT with each neighbor search method).
CI (`.github/workflows/build.yml`) builds with colcon and runs them on Ubuntu 22.04 and 24.04
(26.04 as optional jobs) with gcc and clang, via `docker/Dockerfile`. The same image runs them
locally on another Ubuntu release (`ROS_DISTRO` humble, jazzy or lyrical; `COMPILER` gcc or
clang):
```
docker build -f docker/Dockerfile --build-arg ROS_DISTRO=humble --build-arg COMPILER=clang .
```

## Related packages
- [ndt_omp](https://github.com/koide3/ndt_omp)
- [fast_gicp](https://github.com/SMRT-AIST/fast_gicp)
