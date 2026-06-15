#pragma once

#include <cuda_runtime.h>
#include "GaussianSplattingSlamTypes.hpp"

namespace gaussian_splatting_slam_original
{
__global__ void computeNormalsFromDepth_kernel(
    float4 *normalsData,
    cudaTextureObject_t texDepth,
    const gaussian_splatting_slam::CameraParameters cameraParams,
    uint32_t width,
    uint32_t height,
    uint32_t normalsStep);
}
