#pragma once

#include <cuda_runtime.h>

#include "GaussianSplattingSlamTypes.hpp"

#ifndef GSS_BLOCK_X
#define GSS_BLOCK_X 16
#endif

#ifndef GSS_BLOCK_Y
#define GSS_BLOCK_Y 16
#endif

namespace gaussian_splatting_slam_original
{
    using gaussian_splatting_slam::CameraParameters;
    using gaussian_splatting_slam::MotionTrackingData;
    using gaussian_splatting_slam::Pose3D;

    __global__ void optimizePoseGN3_fast_kernel(
        MotionTrackingData *__restrict__ mtd,
        const uint2 *__restrict__ ranges,
        const uint32_t *__restrict__ indices,
        const float3 *__restrict__ imgPositions,
        const float3 *__restrict__ imgInvSigmas,
        const float2 *__restrict__ pHats,
        const float3 *__restrict__ colors,
        const float *__restrict__ alphas,
        cudaTextureObject_t texRGBA,
        cudaTextureObject_t texDepth,
        cudaTextureObject_t texDx,
        cudaTextureObject_t texDy,
        const Pose3D cameraPose,
        const CameraParameters cameraParams,
        float3 bgColor,
        float alphaThresh,
        float colorThresh,
        float depthThresh,
        uint2 numTiles,
        uint32_t width,
        uint32_t height);
}