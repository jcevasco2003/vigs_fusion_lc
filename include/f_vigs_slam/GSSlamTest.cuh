#pragma once

#include "f_vigs_slam/GSSlam.cuh"

namespace f_vigs_slam
{
    class GSSlamTest : public GSSlam
    {
    public:
        GSSlamTest() = default;

        // Ejemplo de acceso directo a miembros protegidos
        Gaussians& gaussians() { return gaussians_; }
        uint32_t& nGaussians() { return n_Gaussians; }
        cv::cuda::GpuMat& rgbGpu() { return rgb_gpu_; }
        cv::cuda::GpuMat& depthGpu() { return depth_gpu_; }

        // Acceso a parámetros de optimización
        int& poseIterations() { return pose_iterations_; }
        int& gaussianIterations() { return gaussian_iterations_; }
        float& etaPose() { return eta_pose_; }
        float& etaGaussian() { return eta_gaussian_; }

        // Acceso a pirámides
        std::vector<cv::cuda::GpuMat>& pyrColor() { return pyr_color_; }
        std::vector<cv::cuda::GpuMat>& pyrDepth() { return pyr_depth_; }

        // Acceso a estado IMU
        bool& imuInitialized() { return imu_initialized_; }
        Preintegration*& preintegration() { return preint_; }
        double* Pcur() { return P_cur_; }
        double* VBcur() { return VB_cur_; }

        // Keyframes y covisibilidad
        std::vector<KeyframeData>& keyframes() { return keyframes_; }
        float& covisibilityThreshold() { return covisibility_threshold_; }

        // Exponer tile_size_ y max_Gaussians
        uint2& tileSize() { return tile_size_; }
        uint32_t& maxGaussians() { return max_Gaussians; }

        // Exponer render targets
        cv::cuda::GpuMat& renderedRgbGpu() { return rendered_rgb_gpu_; }
        cv::cuda::GpuMat& renderedDepthGpu() { return rendered_depth_gpu_; }
    };

}