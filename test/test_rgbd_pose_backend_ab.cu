#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/core/cuda.hpp>

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include "f_vigs_slam/GSCudaKernels.cuh"
#include "f_vigs_slam/RepresentationClasses.hpp"
#include "reference/GaussianSplattingSlamPoseKernels_original.hpp"

namespace
{
struct BackendResult
{
    Eigen::Matrix<double, 6, 6> JtJ = Eigen::Matrix<double, 6, 6>::Zero();
    Eigen::Matrix<double, 6, 1> Jtr = Eigen::Matrix<double, 6, 1>::Zero();
};

struct Scene
{
    struct Config
    {
        int width = 16;
        int height = 16;
        int num_gaussians = 8;
        float depth_base = 1.0f;
        float depth_tilt_x = 0.001f;
        float depth_tilt_y = 0.0005f;
        float grad_x_base = 0.03f;
        float grad_y_base = 0.04f;
        float color_scale = 1.0f;
        float alpha_base = 0.95f;
        float alpha_step = -0.01f;
        float center_x = 8.0f;
        float center_y = 8.0f;
        float spread = 2.0f;
        float alpha_thresh = 0.99f;
        float color_thresh = 0.2f;
        float depth_thresh = 0.4f;
        float invalid_depth_ratio = 0.0f;
        Eigen::Vector3d t_imu_cam = Eigen::Vector3d(0.03, -0.02, 0.01);
        Eigen::Quaterniond q_imu_cam = Eigen::Quaterniond(Eigen::AngleAxisd(0.04, Eigen::Vector3d::UnitY()));
        Eigen::Vector3d p_imu = Eigen::Vector3d(0.15, -0.08, 0.25);
        Eigen::Quaterniond q_imu = Eigen::Quaterniond(Eigen::AngleAxisd(0.08, Eigen::Vector3d::UnitZ()));
    };

    int width = 16;
    int height = 16;
    int num_tiles_x = 1;
    int num_tiles_y = 1;

    cv::cuda::GpuMat rgb_gpu;
    cv::cuda::GpuMat depth_gpu;
    cv::cuda::GpuMat grad_x_gpu;
    cv::cuda::GpuMat grad_y_gpu;

    cudaTextureObject_t tex_rgb = 0;
    cudaTextureObject_t tex_depth = 0;
    cudaTextureObject_t tex_grad_x = 0;
    cudaTextureObject_t tex_grad_y = 0;

    thrust::device_vector<uint2> ranges;
    thrust::device_vector<uint32_t> indices;
    thrust::device_vector<float4> positions_2d;
    thrust::device_vector<float4> inv_covariances_2d;
    thrust::device_vector<float3> positions_2d_reference;
    thrust::device_vector<float3> inv_covariances_2d_reference;
    thrust::device_vector<float2> p_hats;
    thrust::device_vector<float4> colors_current;
    thrust::device_vector<float3> colors_reference;
    thrust::device_vector<float> alphas;

    f_vigs_slam::Pose pose_current;
    f_vigs_slam::IntrinsicParameters intrinsics_current;
    gaussian_splatting_slam::Pose3D pose_reference;
    gaussian_splatting_slam::CameraParameters intrinsics_reference;

    Eigen::Vector3d t_imu_cam = Eigen::Vector3d(0.03, -0.02, 0.01);
    Eigen::Quaterniond q_imu_cam = Eigen::Quaterniond(Eigen::AngleAxisd(0.04, Eigen::Vector3d::UnitY()));
    Eigen::Vector3d p_imu = Eigen::Vector3d(0.15, -0.08, 0.25);
    Eigen::Quaterniond q_imu = Eigen::Quaterniond(Eigen::AngleAxisd(0.08, Eigen::Vector3d::UnitZ()));

    float3 bg_color = make_float3(0.03f, 0.04f, 0.05f);
    float alpha_thresh = 0.1f;
    float color_thresh = 0.2f;
    float depth_thresh = 0.4f;

    void build(const Config &cfg)
    {
        width = cfg.width;
        height = cfg.height;
        num_tiles_x = (width + 15) / 16;
        num_tiles_y = (height + 15) / 16;
        t_imu_cam = cfg.t_imu_cam;
        q_imu_cam = cfg.q_imu_cam;
        p_imu = cfg.p_imu;
        q_imu = cfg.q_imu;
        alpha_thresh = cfg.alpha_thresh;
        color_thresh = cfg.color_thresh;
        depth_thresh = cfg.depth_thresh;

        intrinsics_current.f = make_float2(120.0f, 118.0f);
        intrinsics_current.c = make_float2(width * 0.5f, height * 0.5f);
        intrinsics_reference.f = intrinsics_current.f;
        intrinsics_reference.c = intrinsics_current.c;

        const Eigen::Vector3d p_cam = p_imu + q_imu * t_imu_cam;
        const Eigen::Quaterniond q_cam = q_imu * q_imu_cam;

        pose_current.position = make_float3(static_cast<float>(p_cam.x()),
                                            static_cast<float>(p_cam.y()),
                                            static_cast<float>(p_cam.z()));
        pose_current.orientation = make_float4(static_cast<float>(q_cam.x()),
                                               static_cast<float>(q_cam.y()),
                                               static_cast<float>(q_cam.z()),
                                               static_cast<float>(q_cam.w()));

        pose_reference.position = pose_current.position;
        pose_reference.orientation = pose_current.orientation;

        rgb_gpu.create(height, width, CV_8UC4);
        depth_gpu.create(height, width, CV_32F);
        grad_x_gpu.create(height, width, CV_32FC4);
        grad_y_gpu.create(height, width, CV_32FC4);

        cv::Mat rgb(height, width, CV_8UC4);
        cv::Mat depth(height, width, CV_32F);
        cv::Mat grad_x(height, width, CV_32FC4);
        cv::Mat grad_y(height, width, CV_32FC4);

        for (int y = 0; y < height; ++y)
        {
            auto *rgb_row = rgb.ptr<cv::Vec4b>(y);
            auto *depth_row = depth.ptr<float>(y);
            auto *gx_row = grad_x.ptr<cv::Vec4f>(y);
            auto *gy_row = grad_y.ptr<cv::Vec4f>(y);
            for (int x = 0; x < width; ++x)
            {
                const float xf = static_cast<float>(x);
                const float yf = static_cast<float>(y);
                rgb_row[x] = cv::Vec4b(
                    static_cast<uint8_t>(std::min(255.0f, (20.0f + 2.0f * xf) * cfg.color_scale)),
                    static_cast<uint8_t>(std::min(255.0f, (15.0f + 3.0f * yf) * cfg.color_scale)),
                    static_cast<uint8_t>(std::min(255.0f, (40.0f + xf + yf) * cfg.color_scale)),
                    255);
                depth_row[x] = cfg.depth_base + cfg.depth_tilt_x * xf + cfg.depth_tilt_y * yf;
                gx_row[x] = cv::Vec4f(cfg.grad_x_base, 0.5f * cfg.grad_x_base, 0.7f * cfg.grad_x_base, 0.0f);
                gy_row[x] = cv::Vec4f(0.25f * cfg.grad_y_base, cfg.grad_y_base, 0.6f * cfg.grad_y_base, 0.0f);

                if (cfg.invalid_depth_ratio > 0.0f)
                {
                    const float pattern = std::fmod(xf + 3.0f * yf, 10.0f) / 10.0f;
                    if (pattern < cfg.invalid_depth_ratio)
                    {
                        depth_row[x] = 0.0f;
                    }
                }
            }
        }

        rgb_gpu.upload(rgb);
        depth_gpu.upload(depth);
        grad_x_gpu.upload(grad_x);
        grad_y_gpu.upload(grad_y);

        tex_rgb = f_vigs_slam::createTextureObject<uchar4>(rgb_gpu);
        tex_depth = f_vigs_slam::createTextureObject<float>(depth_gpu);
        tex_grad_x = f_vigs_slam::createTextureObject<float4>(grad_x_gpu);
        tex_grad_y = f_vigs_slam::createTextureObject<float4>(grad_y_gpu);

        const int num_gaussians = cfg.num_gaussians;
        ranges.resize(static_cast<size_t>(num_tiles_x * num_tiles_y));
        for (int tile = 0; tile < num_tiles_x * num_tiles_y; ++tile)
        {
            ranges[static_cast<size_t>(tile)] = make_uint2(0, num_gaussians);
        }

        indices.resize(num_gaussians);
        positions_2d.resize(num_gaussians);
        inv_covariances_2d.resize(num_gaussians);
        positions_2d_reference.resize(num_gaussians);
        inv_covariances_2d_reference.resize(num_gaussians);
        p_hats.resize(num_gaussians);
        colors_current.resize(num_gaussians);
        colors_reference.resize(num_gaussians);
        alphas.resize(num_gaussians);

        for (int i = 0; i < num_gaussians; ++i)
        {
            const float t = static_cast<float>(i) / static_cast<float>(std::max(1, num_gaussians - 1));
            const float angle = 6.2831853f * t;
            const float cx = cfg.center_x + cfg.spread * std::cos(angle);
            const float cy = cfg.center_y + cfg.spread * std::sin(angle);

            // Wider projected splats (smaller inverse covariance) to ensure
            // synthetic pixels actually receive non-trivial contributions.
            const float cov_xx = 0.008f + 0.004f * (1.0f - t);
            const float cov_xy = -0.001f + 0.002f * t;
            const float cov_yy = 0.009f + 0.004f * t;

            const float phx = -0.015f + 0.03f * t;
            const float phy = 0.02f - 0.03f * t;

            const float cr = 0.15f + 0.75f * t;
            const float cg = 0.9f - 0.7f * t;
            const float cb = 0.25f + 0.5f * (1.0f - t);
            const float alpha = std::max(0.05f, std::min(0.99f, cfg.alpha_base + cfg.alpha_step * static_cast<float>(i)));

            indices[i] = static_cast<uint32_t>(i);
            positions_2d[i] = make_float4(cx, cy, cfg.depth_base + 0.1f + 0.07f * t, 0.0f);
            inv_covariances_2d[i] = make_float4(cov_xx, cov_xy, cov_yy, 0.0f);
            positions_2d_reference[i] = make_float3(cx, cy, cfg.depth_base + 0.1f + 0.07f * t);
            inv_covariances_2d_reference[i] = make_float3(cov_xx, cov_xy, cov_yy);
            p_hats[i] = make_float2(phx, phy);
            colors_current[i] = make_float4(cr, cg, cb, 1.0f);
            colors_reference[i] = make_float3(cr, cg, cb);
            alphas[i] = alpha;
        }
    }

    ~Scene()
    {
        if (tex_rgb) cudaDestroyTextureObject(tex_rgb);
        if (tex_depth) cudaDestroyTextureObject(tex_depth);
        if (tex_grad_x) cudaDestroyTextureObject(tex_grad_x);
        if (tex_grad_y) cudaDestroyTextureObject(tex_grad_y);
    }
};

inline Eigen::Matrix3d skewSymmetric(const Eigen::Vector3d &v)
{
    Eigen::Matrix3d m;
    m << 0.0, -v.z(), v.y(),
         v.z(), 0.0, -v.x(),
        -v.y(), v.x(), 0.0;
    return m;
}

template <typename OutputT>
BackendResult unpackAndTransform(
    const OutputT &out,
    const Eigen::Vector3d &t_imu_cam,
    const Eigen::Quaterniond &q_imu_cam,
    const Eigen::Quaterniond &q_imu)
{
    BackendResult result;

    int idx = 0;
    for (int i = 0; i < 6; ++i)
    {
        for (int j = i; j < 6; ++j)
        {
            const double value = static_cast<double>(out.JtJ[idx++]);
            result.JtJ(i, j) = value;
            result.JtJ(j, i) = value;
        }
        result.Jtr(i) = -static_cast<double>(out.Jtr[i]);
    }

    const Eigen::Matrix3d R_imu_cam = q_imu_cam.toRotationMatrix();
    const Eigen::Matrix3d R_imu = q_imu.toRotationMatrix();
    const Eigen::Matrix3d P_imu_cam_skew = skewSymmetric(t_imu_cam);

    Eigen::Matrix<double, 6, 6> J_cam_imu = Eigen::Matrix<double, 6, 6>::Zero();
    J_cam_imu.block<3, 3>(0, 0) = Eigen::Matrix3d::Identity();
    J_cam_imu.block<3, 3>(0, 3) = -R_imu * P_imu_cam_skew;
    J_cam_imu.block<3, 3>(3, 3) = R_imu_cam.transpose();

    result.JtJ = J_cam_imu.transpose() * result.JtJ * J_cam_imu;
    result.Jtr = J_cam_imu.transpose() * result.Jtr;

    return result;
}

struct DifferenceMetrics
{
    double max_abs_jtj = 0.0;
    double max_abs_jtr = 0.0;
    double l2_jtj = 0.0;
    double l2_jtr = 0.0;
};

struct AbsoluteMetrics
{
    double max_abs_jtj = 0.0;
    double max_abs_jtr = 0.0;
    double l2_jtj = 0.0;
    double l2_jtr = 0.0;
    int nonzero_jtj = 0;
    int nonzero_jtr = 0;
};

DifferenceMetrics compare(const BackendResult &current, const BackendResult &reference)
{
    DifferenceMetrics metrics;
    for (int i = 0; i < 6; ++i)
    {
        for (int j = 0; j < 6; ++j)
        {
            const double diff = std::abs(current.JtJ(i, j) - reference.JtJ(i, j));
            metrics.max_abs_jtj = std::max(metrics.max_abs_jtj, diff);
            metrics.l2_jtj += diff * diff;
        }
        const double diff_jtr = std::abs(current.Jtr(i) - reference.Jtr(i));
        metrics.max_abs_jtr = std::max(metrics.max_abs_jtr, diff_jtr);
        metrics.l2_jtr += diff_jtr * diff_jtr;
    }

    metrics.l2_jtj = std::sqrt(metrics.l2_jtj);
    metrics.l2_jtr = std::sqrt(metrics.l2_jtr);
    return metrics;
}

AbsoluteMetrics summarizeAbsolute(const BackendResult &result)
{
    AbsoluteMetrics metrics;
    constexpr double eps = 1e-12;

    for (int i = 0; i < 6; ++i)
    {
        for (int j = 0; j < 6; ++j)
        {
            const double v = std::abs(result.JtJ(i, j));
            metrics.max_abs_jtj = std::max(metrics.max_abs_jtj, v);
            metrics.l2_jtj += v * v;
            if (v > eps)
            {
                ++metrics.nonzero_jtj;
            }
        }

        const double r = std::abs(result.Jtr(i));
        metrics.max_abs_jtr = std::max(metrics.max_abs_jtr, r);
        metrics.l2_jtr += r * r;
        if (r > eps)
        {
            ++metrics.nonzero_jtr;
        }
    }

    metrics.l2_jtj = std::sqrt(metrics.l2_jtj);
    metrics.l2_jtr = std::sqrt(metrics.l2_jtr);
    return metrics;
}

BackendResult runCurrentKernel(const Scene &scene);
BackendResult runReferenceKernel(const Scene &scene);

void runScenarioAndCheckEquivalent(const Scene::Config &cfg, double tol)
{
    Scene scene;
    scene.build(cfg);

    const BackendResult current = runCurrentKernel(scene);
    const BackendResult reference = runReferenceKernel(scene);
    const DifferenceMetrics metrics = compare(current, reference);
    const AbsoluteMetrics current_abs = summarizeAbsolute(current);
    const AbsoluteMetrics reference_abs = summarizeAbsolute(reference);

    std::cout << "[RgbdPoseBackendAB] scenario: "
              << "w=" << cfg.width
              << " h=" << cfg.height
              << " ng=" << cfg.num_gaussians
              << " invalid_depth_ratio=" << cfg.invalid_depth_ratio
              << " max_abs_jtj=" << metrics.max_abs_jtj
              << " max_abs_jtr=" << metrics.max_abs_jtr
              << " l2_jtj=" << metrics.l2_jtj
              << " l2_jtr=" << metrics.l2_jtr
              << " | current_abs(max_jtj=" << current_abs.max_abs_jtj
              << ", max_jtr=" << current_abs.max_abs_jtr
              << ", l2_jtj=" << current_abs.l2_jtj
              << ", l2_jtr=" << current_abs.l2_jtr
              << ", nnz_jtj=" << current_abs.nonzero_jtj
              << ", nnz_jtr=" << current_abs.nonzero_jtr
              << ")"
              << " | reference_abs(max_jtj=" << reference_abs.max_abs_jtj
              << ", max_jtr=" << reference_abs.max_abs_jtr
              << ", l2_jtj=" << reference_abs.l2_jtj
              << ", l2_jtr=" << reference_abs.l2_jtr
              << ", nnz_jtj=" << reference_abs.nonzero_jtj
              << ", nnz_jtr=" << reference_abs.nonzero_jtr
              << ")" << std::endl;

    EXPECT_GT(current_abs.nonzero_jtj + current_abs.nonzero_jtr, 0);
    EXPECT_GT(reference_abs.nonzero_jtj + reference_abs.nonzero_jtr, 0);

    EXPECT_TRUE(std::isfinite(metrics.max_abs_jtj));
    EXPECT_TRUE(std::isfinite(metrics.max_abs_jtr));
    EXPECT_TRUE(std::isfinite(metrics.l2_jtj));
    EXPECT_TRUE(std::isfinite(metrics.l2_jtr));
    EXPECT_LE(metrics.max_abs_jtj, tol);
    EXPECT_LE(metrics.max_abs_jtr, tol);
}

BackendResult runCurrentKernel(const Scene &scene)
{
    thrust::device_vector<f_vigs_slam::PoseOptimizationRgbdData> out(1);
    cudaMemset(thrust::raw_pointer_cast(out.data()), 0, sizeof(f_vigs_slam::PoseOptimizationRgbdData));

    dim3 block(16, 16);
    dim3 grid(static_cast<unsigned int>(scene.num_tiles_x),
              static_cast<unsigned int>(scene.num_tiles_y));

    f_vigs_slam::getRgbdPoseJacobians_fast<<<grid, block>>>(
        thrust::raw_pointer_cast(out.data()),
        thrust::raw_pointer_cast(scene.ranges.data()),
        thrust::raw_pointer_cast(scene.indices.data()),
        thrust::raw_pointer_cast(scene.positions_2d.data()),
        thrust::raw_pointer_cast(scene.inv_covariances_2d.data()),
        thrust::raw_pointer_cast(scene.p_hats.data()),
        thrust::raw_pointer_cast(scene.colors_current.data()),
        thrust::raw_pointer_cast(scene.alphas.data()),
        scene.tex_rgb,
        scene.tex_depth,
        scene.tex_grad_x,
        scene.tex_grad_y,
        scene.pose_current,
        scene.intrinsics_current,
        scene.bg_color,
        scene.alpha_thresh,
        scene.color_thresh,
        scene.depth_thresh,
        scene.width,
        scene.height,
        scene.num_tiles_x,
        scene.num_tiles_y);

    EXPECT_EQ(cudaGetLastError(), cudaSuccess);
    EXPECT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    f_vigs_slam::PoseOptimizationRgbdData host_out{};
    cudaMemcpy(&host_out, thrust::raw_pointer_cast(out.data()), sizeof(host_out), cudaMemcpyDeviceToHost);

    return unpackAndTransform(host_out, scene.t_imu_cam, scene.q_imu_cam, scene.q_imu);
}

BackendResult runReferenceKernel(const Scene &scene)
{
    thrust::device_vector<gaussian_splatting_slam::MotionTrackingData> out(1);
    cudaMemset(thrust::raw_pointer_cast(out.data()), 0, sizeof(gaussian_splatting_slam::MotionTrackingData));

    dim3 block(16, 16);
    dim3 grid(static_cast<unsigned int>(scene.num_tiles_x),
              static_cast<unsigned int>(scene.num_tiles_y));

    gaussian_splatting_slam_original::optimizePoseGN3_fast_kernel<<<grid, block>>>(
        thrust::raw_pointer_cast(out.data()),
        thrust::raw_pointer_cast(scene.ranges.data()),
        thrust::raw_pointer_cast(scene.indices.data()),
        thrust::raw_pointer_cast(scene.positions_2d_reference.data()),
        thrust::raw_pointer_cast(scene.inv_covariances_2d_reference.data()),
        thrust::raw_pointer_cast(scene.p_hats.data()),
        thrust::raw_pointer_cast(scene.colors_reference.data()),
        thrust::raw_pointer_cast(scene.alphas.data()),
        scene.tex_rgb,
        scene.tex_depth,
        scene.tex_grad_x,
        scene.tex_grad_y,
        scene.pose_reference,
        scene.intrinsics_reference,
        scene.bg_color,
        scene.alpha_thresh,
        scene.color_thresh,
        scene.depth_thresh,
        make_uint2(static_cast<uint32_t>(scene.num_tiles_x), static_cast<uint32_t>(scene.num_tiles_y)),
        static_cast<uint32_t>(scene.width),
        static_cast<uint32_t>(scene.height));

    EXPECT_EQ(cudaGetLastError(), cudaSuccess);
    EXPECT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    gaussian_splatting_slam::MotionTrackingData host_out{};
    cudaMemcpy(&host_out, thrust::raw_pointer_cast(out.data()), sizeof(host_out), cudaMemcpyDeviceToHost);

    return unpackAndTransform(host_out, scene.t_imu_cam, scene.q_imu_cam, scene.q_imu);
}
} // namespace

TEST(RgbdPoseBackendAB, FastKernelMatchesReferenceOnSyntheticBaseline)
{
    Scene::Config cfg;
    runScenarioAndCheckEquivalent(cfg, 1e-6);
}

TEST(RgbdPoseBackendAB, FastKernelMatchesReferenceOnWideImageAndManyGaussians)
{
    Scene::Config cfg;
    cfg.width = 64;
    cfg.height = 48;
    cfg.num_gaussians = 12;
    cfg.center_x = 30.0f;
    cfg.center_y = 22.0f;
    cfg.spread = 3.8f;
    cfg.depth_base = 1.8f;
    cfg.depth_tilt_x = -0.0012f;
    cfg.depth_tilt_y = 0.0009f;
    runScenarioAndCheckEquivalent(cfg, 1e-6);
}

TEST(RgbdPoseBackendAB, FastKernelMatchesReferenceWithSparseDepth)
{
    Scene::Config cfg;
    cfg.width = 40;
    cfg.height = 30;
    cfg.num_gaussians = 10;
    cfg.invalid_depth_ratio = 0.35f;
    cfg.alpha_thresh = 0.2f;
    cfg.color_thresh = 0.25f;
    cfg.depth_thresh = 0.55f;
    runScenarioAndCheckEquivalent(cfg, 1e-6);
}

TEST(RgbdPoseBackendAB, FastKernelMatchesReferenceWithLowAlphaAndStrongGradients)
{
    Scene::Config cfg;
    cfg.width = 32;
    cfg.height = 32;
    cfg.num_gaussians = 8;
    cfg.alpha_base = 0.25f;
    cfg.alpha_step = -0.01f;
    cfg.grad_x_base = 0.10f;
    cfg.grad_y_base = 0.09f;
    cfg.color_scale = 0.7f;
    runScenarioAndCheckEquivalent(cfg, 1e-6);
}

TEST(RgbdPoseBackendAB, FastKernelMatchesReferenceWithDifferentPoseAndExtrinsics)
{
    Scene::Config cfg;
    cfg.width = 48;
    cfg.height = 36;
    cfg.num_gaussians = 9;
    cfg.t_imu_cam = Eigen::Vector3d(-0.04, 0.03, 0.015);
    cfg.q_imu_cam = Eigen::Quaterniond(Eigen::AngleAxisd(0.12, Eigen::Vector3d::UnitX()));
    cfg.p_imu = Eigen::Vector3d(-0.25, 0.06, 0.45);
    cfg.q_imu = Eigen::Quaterniond(Eigen::AngleAxisd(-0.2, Eigen::Vector3d::UnitY()));
    cfg.depth_base = 2.2f;
    cfg.spread = 2.5f;
    cfg.center_x = 20.0f;
    cfg.center_y = 17.0f;
    runScenarioAndCheckEquivalent(cfg, 1e-6);
}