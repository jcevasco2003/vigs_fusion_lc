# VIGS-Fusion LC

## Tested Environment

This project has been tested with:

* ROS2 Rolling
* CUDA 12.3
* OpenCV 4.8.0 (compiled with CUDA support)
* SuiteSparse 5.10.1
* Eigen3
* glog
* Ceres Solver 2.0.0
* Open3D (compiled with CUDA support)

---

## External Dependencies

### cuda_depth_register

This project depends on the `cuda_depth_register` ROS2 node, which performs registration of depth images into the RGB camera frame.

Repository:

https://github.com/jcevasco2003/ros2_cuda_depth_register

The package must be available in the workspace before compilation.

---

## OpenCV 4.8.0

Compiled using GCC/G++ 10.

```bash
cd ~

git clone https://github.com/opencv/opencv.git
git clone https://github.com/opencv/opencv_contrib.git

cd opencv
git checkout 4.8.0

cd ../opencv_contrib
git checkout 4.8.0

cd ~/opencv
mkdir build && cd build

CC=gcc-10 CXX=g++-10 cmake \
-D CMAKE_BUILD_TYPE=RELEASE \
-D CMAKE_INSTALL_PREFIX=/usr/local \
-D WITH_CUDA=ON \
-D WITH_CUDNN=ON \
-D WITH_CUBLAS=ON \
-D WITH_TBB=ON \
-D WITH_GTK=ON \
-D WITH_QT=OFF \
-D OPENCV_ENABLE_NONFREE=ON \
-D CUDA_ARCH_BIN=8.6 \
-D OPENCV_EXTRA_MODULES_PATH=$HOME/opencv_contrib/modules \
-D BUILD_EXAMPLES=OFF \
-D BUILD_TESTS=OFF \
-D BUILD_PERF_TESTS=OFF \
-D OPENCV_GENERATE_PKGCONFIG=ON \
-D PYTHON3_PACKAGES_PATH=/usr/lib/python3.10/dist-packages \
-D BUILD_opencv_python3=ON \
-D BUILD_opencv_cudaarithm=ON \
-D BUILD_opencv_cudaimgproc=ON \
-D BUILD_opencv_cudawarping=ON \
-D BUILD_opencv_cudafilters=ON \
-D BUILD_opencv_java=OFF \
..

make -j$(nproc)
sudo make install
sudo ldconfig
```

Additionally, apply the fix described in:

https://github.com/opencv/opencv/issues/23893

---

## SuiteSparse 5.10.1

Compiled using GCC/G++ 12.

```bash
cd ~

git clone https://github.com/DrTimothyAldenDavis/SuiteSparse.git

cd SuiteSparse
git checkout v5.10.1

make -j$(nproc)
sudo make install
sudo ldconfig
```

---

## Eigen

```bash
sudo apt install libeigen3-dev
```

---

## glog

```bash
sudo apt install libgoogle-glog-dev
```

---

## Ceres Solver 2.0.0

```bash
cd ~

git clone https://github.com/ceres-solver/ceres-solver.git

cd ceres-solver
git checkout 2.0.0

mkdir build && cd build

cmake .. \
-DBUILD_TESTING=OFF \
-DBUILD_EXAMPLES=OFF \
-DCMAKE_BUILD_TYPE=Release

make -j$(nproc)

sudo make install
sudo ldconfig
```

---

## Open3D

Open3D was compiled with CUDA support using GCC/G++ 11.

```bash
cd ~/Open3D

rm -rf build
mkdir build && cd build

export CC=/usr/bin/gcc-11
export CXX=/usr/bin/g++-11
export CUDAHOSTCXX=/usr/bin/g++-11

cmake -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER=/usr/bin/gcc-11 \
  -DCMAKE_CXX_COMPILER=/usr/bin/g++-11 \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.3/bin/nvcc \
  -DCMAKE_CUDA_HOST_COMPILER=/usr/bin/g++-11 \
  -DCMAKE_CUDA_FLAGS="--compiler-bindir=/usr/bin/g++-11" \
  -DBUILD_CUDA_MODULE=ON \
  -DBUILD_PYTHON_MODULE=OFF \
  -DBUILD_GUI=OFF \
  -DBUILD_WEBRTC=OFF \
  -DBUILD_EXAMPLES=OFF \
  -DUSE_SYSTEM_EIGEN3=ON \
  -DCMAKE_CUDA_ARCHITECTURES=86 \
  -DCMAKE_INSTALL_PREFIX=$HOME/open3d_install \
  ..

ninja
ninja install
```

---

## Datasets

Datasets used by VIGS-Fusion are available at:

https://entrepot.recherche.data.gouv.fr/dataset.xhtml?persistentId=doi:10.57745/CI0K9G

---

## Build

```bash
colcon build --symlink-install
```

---

## Usage

```bash
source install/setup.bash

ros2 launch f_vigs_slam gs_slam_node.py dataset:=vigs
```

---

## Notes

* CUDA 12.3 was used during development.
* OpenCV is required with CUDA support enabled.
* Open3D is required with CUDA support enabled.
* The `cuda_depth_register` node is mandatory for registering depth images into rgb frame.
