cmake_minimum_required(VERSION 3.8)
set(CMAKE_CUDA_HOST_COMPILER /usr/bin/gcc-12)
set(CMAKE_CUDA_ARCHITECTURES 86)

project(f_vigs_slam)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

set(CMAKE_CUDA_COMPILER "/usr/local/cuda/bin/nvcc")
enable_language("CUDA")

set(CMAKE_CUDA17_STANDARD_COMPILE_OPTION "-std=c++17")
set(CMAKE_CUDA17_EXTENSION_COMPILE_OPTION "-std=c++17")
#set(CUDA_UTILS_DIR ${PROJECT_SOURCE_DIR}/include/cuda_utils/build)

list(APPEND CMAKE_CUDA_FLAGS "-w")

# 1. Buscar dependencias
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
#find_package(cuda_utils REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(image_transport REQUIRED)
find_package(image_geometry REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_eigen REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)
#find_package(CUDA REQUIRED )
find_package(OpenCV 4 REQUIRED COMPONENTS core imgproc cudaimgproc cudaarithm highgui)
find_package(CUDAToolkit REQUIRED)
find_package(Eigen3 REQUIRED)
if(NOT Open3D_DIR)
  set(Open3D_DIR "$ENV{HOME}/open3d_install/lib/cmake/Open3D" CACHE PATH "Path to Open3DConfig.cmake")
  message(STATUS "Open3D_DIR not set, defaulting to ${Open3D_DIR}")
endif()
find_package(Open3D CONFIG REQUIRED)
find_package(Ceres REQUIRED)
find_package(message_filters REQUIRED)

include_directories(
  include
  ${CMAKE_CURRENT_SOURCE_DIR}
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
  ${OpenCV_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIR}
  ${CERES_INCLUDE_DIRS}
  ${CMAKE_CURRENT_SOURCE_DIR}/test/reference
  #${CMAKE_CURRENT_SOURCE_DIR}/../VIGS-Fusion/include # VIGS
  #${CMAKE_CURRENT_SOURCE_DIR}/../VIGS-Fusion/include/cuda_utils/include
  #${PROJECT_SOURCE_DIR}/include/cuda_utils
)

install(
  DIRECTORY config
  DESTINATION share/${PROJECT_NAME}
)


add_library(GSSlam SHARED
            src/GSSlam.cu
            src/LoopClosure.cu
            src/LoopClosureModule.cu
            src/GSCudaKernels.cu
            src/FrequencyMask.cu
            src/RgbdPoseCost.cu
            src/Preintegration.cpp
            src/ImuCostFunction.cpp
            src/PoseLocalParameterization.cpp
            src/MarginalizationFactor.cpp
            src/KeyframeSelector.cpp
            src/NetVLADWrapper.cu
            src/LoopClosureOpen3D.cu
            src/LoopClosureModule_GPU_Kernels.cu
#            src/GaussianSplattingSlam.cu
#            src/GaussianSplattingSlamKernels.cu
#            src/GaussianSplattingSlamTypes.cu
#            src/GaussianSplattingKeyframe.cu
#            src/Utility.cpp
#            src/PoseLocalParameterization.cpp
#            src/GSPoseCost.cu
#            src/ImuCostFunction.cpp
#            src/MarginalizationFactor.cpp
#            src/BetaBinomialGenerator.cpp
#            src/GaussianSplattingViewer.cpp
)

#ament_target_dependencies( GaussianSplattingSlam
#  cuda_utils
#)

target_include_directories(GSSlam
  PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
  $<INSTALL_INTERFACE:include>
  ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
#  include/cuda_utils/include
)

message(STATUS "OpenCV_LIBS: ${OpenCV_LIBS}")
message(STATUS "OpenCV_INCLUDE_DIRS: ${OpenCV_INCLUDE_DIRS}")

target_link_libraries(GSSlam
  ${OpenCV_LIBRARIES}
  ${OpenCV_CUDAIMGPROC_LIBS}
  ${CUDA_LIBRARIES}
  CUDA::cublas
  CUDA::cufft
  ${CERES_LIBRARIES}
  Open3D::Open3D
  #${CUDA_UTILS_DIR}/libcuda_utils.so
)
target_compile_definitions(GSSlam
  PRIVATE
  F_VIGS_SLAM_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}"
)
set_target_properties(GSSlam
  PROPERTIES
  CUDA_STANDARD 17
  CUDA_STANDARD_REQUIRED ON
)

add_library(GSSlamNode SHARED
            src/GSSlamNode.cpp
            src/GaussianSplattingViewer.cpp
)

set_source_files_properties(
  src/GSSlamNode.cpp
  PROPERTIES
  LANGUAGE CUDA
)

set_source_files_properties(
  src/GaussianSplattingViewer.cpp
  PROPERTIES
  LANGUAGE CUDA
)

set_source_files_properties(
  src/LoopClosureOpen3D.cu
  PROPERTIES
  LANGUAGE CUDA
)

set_target_properties(GSSlamNode
  PROPERTIES
  CUDA_STANDARD 17
  CUDA_STANDARD_REQUIRED ON
  CUDA_SEPARABLE_COMPILATION ON
)

ament_target_dependencies(GSSlamNode
  rclcpp
  #cuda_utils
  nav_msgs
  image_transport
  tf2
  tf2_ros
  tf2_eigen
  tf2_geometry_msgs
  image_geometry
  message_filters
  sensor_msgs
  cv_bridge
)

#target_include_directories(gs_slam_node
#  PUBLIC
#  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
#)

target_include_directories(GSSlamNode
  PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
  $<INSTALL_INTERFACE:include>
)

target_link_libraries(GSSlamNode
  GSSlam
  ${OpenCV_LIBRARIES}
  ${CERES_LIBRARIES}
  Open3D::Open3D
  #${CUDA_UTILS_DIR}/libcuda_utils.so
)


#add_executable(f_vigs_slam
#              src/gs_slam
#)

#target_link_libraries(f_vigs_slam
#  GSSlam
#  GSSlamNode
#  ${OpenCV_LIBRARIES}
#  ${CUDA_LIBRARIES}
#  ${CERES_LIBRARIES}
#)

#ament_target_dependencies(f_vigs_slam
#  rclcpp
#)



add_executable(gs_slam_node
  src/gs_slam_node.cpp
)

target_link_libraries(gs_slam_node
  GSSlamNode
)

ament_target_dependencies(gs_slam_node
  rclcpp
)

install(TARGETS 
  GSSlam
  GSSlamNode
  #f_vigs_slam
  gs_slam_node
  RUNTIME DESTINATION lib/${PROJECT_NAME}
  LIBRARY DESTINATION lib
)

install(DIRECTORY
  launch
  DESTINATION share/${PROJECT_NAME}
)

install(DIRECTORY
  scripts
  DESTINATION share/${PROJECT_NAME}
)

#if(BUILD_TESTING)
#  find_package(ament_lint_auto REQUIRED)
#  set(ament_cmake_copyright_FOUND TRUE)
#  set(ament_cmake_cpplint_FOUND TRUE)
#  ament_lint_auto_find_test_dependencies()
#endif()

#if(BUILD_TESTING)
#  find_package(ament_lint_auto REQUIRED)

#  add_executable(test_imu_residual
#    test/test_imu_residual.cpp
#    test/reference/Preintegration_original.cpp
#    test/reference/ImuCostFunction_original.cpp
#  )

#  target_link_libraries(test_imu_residual
#    GSSlam
#    ${CERES_LIBRARIES}
#    Eigen3::Eigen
#  )

#  add_test(NAME test_imu_residual COMMAND test_imu_residual)

#  set(ament_cmake_uncrustify_FOUND TRUE)
#  set(ament_cmake_cpplint_FOUND TRUE)
#  set(ament_cmake_flake8_FOUND TRUE)
#  set(ament_cmake_lint_cmake_FOUND TRUE)
#  set(ament_cmake_copyright_FOUND TRUE)
#  set(ament_cmake_cpplint_FOUND TRUE)
#  ament_lint_auto_find_test_dependencies()
#endif()

set(AMENT_ENABLE_LINTING FALSE)
if(BUILD_TESTING)

  find_package(ament_cmake_gtest REQUIRED)
  #find_package(CUDA REQUIRED)

  # ----------- Test para IMU residuals -----------
  ament_add_gtest(test_imu_residual
    test/test_imu_residual.cpp
    test/reference/Preintegration_original.cpp
    test/reference/ImuCostFunction_original.cpp
  )

  target_link_libraries(test_imu_residual
    GSSlam
    ${CERES_LIBRARIES}
    Eigen3::Eigen
  )

  # ----------- A/B test: backend RGB-D pose Jacobians -----------
  ament_add_gtest(test_rgbd_pose_backend_ab
    test/test_rgbd_pose_backend_ab.cu
    test/reference/GaussianSplattingSlamPoseKernels_original.cu
  )

  target_link_libraries(test_rgbd_pose_backend_ab
    GSSlam
    ${OpenCV_LIBRARIES}
    ${OpenCV_CUDAIMGPROC_LIBS}
    ${CUDA_LIBRARIES}
    CUDA::cudart
    Eigen3::Eigen
  )

  target_include_directories(test_rgbd_pose_backend_ab PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/test/reference
    ${CMAKE_CURRENT_SOURCE_DIR}/../VIGS-Fusion/include
    ${CMAKE_CURRENT_SOURCE_DIR}/../VIGS-Fusion/include/cuda_utils/include
    ${OpenCV_INCLUDE_DIRS}
    ${EIGEN3_INCLUDE_DIR}
  )

  set_target_properties(test_rgbd_pose_backend_ab PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON
  )

  # ----------- A/B test: kernel de normales vs original -----------
 # ament_add_gtest(test_normals_kernel_ab
 #test/test_normals_kernel_ab.cu
 #   test/reference/NormalsKernel_original.cu
 # )

 # target_link_libraries(test_normals_kernel_ab
 #   GSSlam
 #   ${OpenCV_LIBRARIES}
 #   Eigen3::Eigen
 # )

 #target_include_directories(test_normals_kernel_ab PUBLIC
 #   ${CMAKE_CURRENT_SOURCE_DIR}/include
 #   ${CMAKE_CURRENT_SOURCE_DIR}/test/reference
 #   ${CMAKE_CURRENT_SOURCE_DIR}/../VIGS-Fusion/include # VIGS
 #   ${CMAKE_CURRENT_SOURCE_DIR}/../VIGS-Fusion/include/cuda_utils/include
 #   ${OpenCV_INCLUDE_DIRS}
 # )

#  set_target_properties(test_normals_kernel_ab PROPERTIES
#    CUDA_SEPARABLE_COMPILATION ON
#    POSITION_INDEPENDENT_CODE ON
#  )

  # ----------- Test para GSCudaKernels -----------
  #ament_add_gtest(test_gs_kernels
  #  test/test_gs_kernels.cu
  #)

  #target_link_libraries(test_gs_kernels
  #  GSSlam
  #  cuda 
  #  cudart
  #  #${CUDA_LIBRARIES}
  #  ${OpenCV_LIBRARIES}
  #)

  #set_target_properties(test_gs_kernels PROPERTIES
  #  CUDA_SEPARABLE_COMPILATION ON
  #  POSITION_INDEPENDENT_CODE ON
  #)

  #target_include_directories(test_gs_kernels PUBLIC
  #  ${CMAKE_CURRENT_SOURCE_DIR}/include
  #  ${OpenCV_INCLUDE_DIRS}
  #)

  set(ament_cmake_uncrustify_FOUND TRUE)
  set(ament_cmake_cpplint_FOUND TRUE)
  set(ament_cmake_flake8_FOUND TRUE)
  set(ament_cmake_lint_cmake_FOUND TRUE)
  set(ament_cmake_copyright_FOUND TRUE)
  set(ament_cmake_cpplint_FOUND TRUE)
  #ament_lint_auto_find_test_dependencies()

  # ---------------- Test GSSlam Rasterization ----------------
  #ament_add_gtest(test_rasterization
  #  test/test_rasterization.cpp
  #)

  # Linkear librerías necesarias
  #target_link_libraries(test_rasterization
  #  GSSlam                 # Tu librería principal
  #  ${CERES_LIBRARIES}     # Si tu GSSlam usa Ceres
  #  Eigen3::Eigen
  #  ${CUDA_LIBRARIES}      # Para kernels CUDA
  #)

  # Incluir paths de headers
  #target_include_directories(test_rasterization PUBLIC
  #  ${CMAKE_CURRENT_SOURCE_DIR}/include
  #  ${CMAKE_CURRENT_SOURCE_DIR}/src
  #  ${CMAKE_CURRENT_SOURCE_DIR}/test/reference  # si necesitás headers de test reference
  #)

  # Habilitar compilación CUDA si los tests usan kernels
  #set_target_properties(test_rasterization PROPERTIES
  #  CUDA_SEPARABLE_COMPILATION ON
  #  POSITION_INDEPENDENT_CODE ON
  #)

  # ----------- Test para MarginalizationFactor -----------
  #ament_add_gtest(test_marginalization_factor
  #  test/test_marginalization_factor.cpp
  #  test/reference/MarginalizationFactor_original.cpp
  #  test/reference/MarginalizationFactor_original.hpp
  #)

  #target_link_libraries(test_marginalization_factor
  #  GSSlam
  #  ${CERES_LIBRARIES}
  #  Eigen3::Eigen
  #)

endif()

ament_package()