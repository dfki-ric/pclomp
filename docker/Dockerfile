# Builds pclomp with colcon and runs its unit tests on a ROS 2 base image, one distribution per
# Ubuntu release: humble (22.04), jazzy (24.04), lyrical (26.04). ROS itself is not needed; the
# images just bring colcon and rosdep. Used by .github/workflows/build.yml; locally:
#   docker build -f docker/Dockerfile --build-arg ROS_DISTRO=jazzy --build-arg COMPILER=clang .
ARG ROS_DISTRO=jazzy
FROM ros:${ROS_DISTRO}

# gcc or clang (clang needs libomp for OpenMP)
ARG COMPILER=gcc

# dependencies from package.xml first, so this layer is cached across source changes
COPY package.xml /colcon_ws/src/pclomp/package.xml
# clang uses the newest GCC installation it finds for libstdc++, which need not be the default
# g++'s (Ubuntu 26.04: clang pulls in GCC 16's runtime, g++ is 15), so add the matching
# libstdc++-<version>-dev
RUN apt-get update \
  && if [ "$COMPILER" = clang ]; then \
    apt-get install -y --no-install-recommends clang libomp-dev \
    && gcc_version=$(clang++ -v 2>&1 | sed -n 's|^Selected GCC installation: .*/||p') \
    && apt-get install -y --no-install-recommends "libstdc++-${gcc_version}-dev"; \
  fi \
  && rosdep update --rosdistro "$ROS_DISTRO" \
  && rosdep install --from-paths /colcon_ws/src --ignore-src -y \
  && rm -rf /var/lib/apt/lists/*

COPY . /colcon_ws/src/pclomp/

WORKDIR /colcon_ws
# colcon test exits 0 even when tests fail; colcon test-result fails the build then
RUN if [ "$COMPILER" = clang ]; then export CC=clang CXX=clang++; fi \
  && colcon build --event-handlers console_direct+ \
    --cmake-args -DBUILD_TESTING=ON -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  && colcon test --event-handlers console_direct+ \
  && colcon test-result --verbose
