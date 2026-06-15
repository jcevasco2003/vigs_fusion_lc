#pragma once

#include "gaussian_splatting_slam/GaussianSplattingSlamTypes.hpp"

#define GSS_BLOCK_X 16
#define GSS_BLOCK_Y 16
#define GSS_BLOCK_SIZE (GSS_BLOCK_X * GSS_BLOCK_Y)

namespace gaussian_splatting_slam
{
    __global__ void computeNormalsFromDepth_kernel(
            float4 *normalsData,
            cudaTextureObject_t texDepth,
            const CameraParameters cameraParams,
            uint32_t width,
            uint32_t height,
            uint32_t normalsStep);
}