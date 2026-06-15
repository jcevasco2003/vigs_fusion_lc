#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <limits>

#include <opencv2/core.hpp>
#include <opencv2/core/cuda.hpp>

#include "f_vigs_slam/GSCudaKernels.cuh"
#include "f_vigs_slam/RepresentationClasses.hpp"
#include "reference/NormalsKernel_original.cuh"

namespace
{
struct ErrorMetrics
{
    float max_abs = 0.0f;
    double mean_abs = 0.0;
    int valid_count = 0;
};

inline gaussian_splatting_slam::CameraParameters toOriginalCamera(
    const f_vigs_slam::IntrinsicParameters &k)
{
    gaussian_splatting_slam::CameraParameters out;
    out.f = k.f;
    out.c = k.c;
    return out;
}

ErrorMetrics compareNormals(const cv::Mat &a, const cv::Mat &b)
{
    ErrorMetrics m;
    if (a.rows != b.rows || a.cols != b.cols || a.type() != CV_32FC4 || b.type() != CV_32FC4)
    {
        return m;
    }

    double sum = 0.0;

    for (int y = 0; y < a.rows; ++y)
    {
        const auto *ra = a.ptr<cv::Vec4f>(y);
        const auto *rb = b.ptr<cv::Vec4f>(y);
        for (int x = 0; x < a.cols; ++x)
        {
            for (int c = 0; c < 4; ++c)
            {
                const float da = ra[x][c];
                const float db = rb[x][c];
                if (!std::isfinite(da) || !std::isfinite(db))
                {
                    continue;
                }
                const float err = std::abs(da - db);
                m.max_abs = std::max(m.max_abs, err);
                sum += err;
                ++m.valid_count;
            }
        }
    }

    if (m.valid_count > 0)
    {
        m.mean_abs = sum / static_cast<double>(m.valid_count);
    }
    return m;
}

cv::Mat makeDepthPlane(int width, int height, float base, float tilt_x, float tilt_y)
{
    cv::Mat depth(height, width, CV_32F);
    for (int y = 0; y < height; ++y)
    {
        float *row = depth.ptr<float>(y);
        for (int x = 0; x < width; ++x)
        {
            row[x] = base + tilt_x * static_cast<float>(x) + tilt_y * static_cast<float>(y);
        }
    }
    return depth;
}

void launchBothKernels(
    const cv::cuda::GpuMat &depth_gpu,
    const f_vigs_slam::IntrinsicParameters &k,
    cv::cuda::GpuMat &normals_fvigs,
    cv::cuda::GpuMat &normals_original)
{
    dim3 block(16, 16);
    dim3 grid(
        (depth_gpu.cols + block.x - 1) / block.x,
        (depth_gpu.rows + block.y - 1) / block.y);

    auto tex_depth = f_vigs_slam::createTextureObject<float>(depth_gpu);

    f_vigs_slam::computeNormalsFromDepth_kernel<<<grid, block>>>(
        tex_depth,
        reinterpret_cast<float4 *>(normals_fvigs.ptr<float4>()),
        normals_fvigs.step,
        depth_gpu.cols,
        depth_gpu.rows,
        k);

    gaussian_splatting_slam_original::computeNormalsFromDepth_kernel<<<grid, block>>>(
        reinterpret_cast<float4 *>(normals_original.ptr<float4>()),
        tex_depth,
        toOriginalCamera(k),
        static_cast<uint32_t>(depth_gpu.cols),
        static_cast<uint32_t>(depth_gpu.rows),
        static_cast<uint32_t>(normals_original.step));

    ASSERT_EQ(cudaGetLastError(), cudaSuccess);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    ASSERT_EQ(cudaDestroyTextureObject(tex_depth), cudaSuccess);
}
}

TEST(NormalsKernelAB, PlaneInputMatchesOriginalWithinPracticalTolerance)
{
    const int width = 320;
    const int height = 240;

    f_vigs_slam::IntrinsicParameters k;
    k.f = make_float2(525.0f, 525.0f);
    k.c = make_float2(width * 0.5f, height * 0.5f);

    cv::Mat depth = makeDepthPlane(width, height, 1.0f, 1e-4f, -8e-5f);
    cv::cuda::GpuMat depth_gpu;
    depth_gpu.upload(depth);

    cv::cuda::GpuMat normals_fvigs(height, width, CV_32FC4);
    cv::cuda::GpuMat normals_original(height, width, CV_32FC4);

    launchBothKernels(depth_gpu, k, normals_fvigs, normals_original);

    cv::Mat h_fvigs;
    cv::Mat h_original;
    normals_fvigs.download(h_fvigs);
    normals_original.download(h_original);

    ASSERT_EQ(h_fvigs.rows, h_original.rows);
    ASSERT_EQ(h_fvigs.cols, h_original.cols);
    ASSERT_EQ(h_fvigs.type(), CV_32FC4);
    ASSERT_EQ(h_original.type(), CV_32FC4);

    const ErrorMetrics metrics = compareNormals(h_fvigs, h_original);

    ASSERT_GT(metrics.valid_count, 0);
    EXPECT_LE(metrics.max_abs, 2.0e-2f);
    EXPECT_LE(metrics.mean_abs, 2.0e-3);
}

TEST(NormalsKernelAB, HandlesDepthWithInvalidPixelsLikeOriginal)
{
    const int width = 128;
    const int height = 96;

    f_vigs_slam::IntrinsicParameters k;
    k.f = make_float2(400.0f, 400.0f);
    k.c = make_float2(width * 0.5f, height * 0.5f);

    cv::Mat depth = makeDepthPlane(width, height, 1.2f, 0.0f, 0.0f);
    for (int y = 10; y < 25; ++y)
    {
        float *row = depth.ptr<float>(y);
        for (int x = 15; x < 30; ++x)
        {
            row[x] = 0.0f;
        }
    }

    cv::cuda::GpuMat depth_gpu;
    depth_gpu.upload(depth);

    cv::cuda::GpuMat normals_fvigs(height, width, CV_32FC4);
    cv::cuda::GpuMat normals_original(height, width, CV_32FC4);

    launchBothKernels(depth_gpu, k, normals_fvigs, normals_original);

    cv::Mat h_fvigs;
    cv::Mat h_original;
    normals_fvigs.download(h_fvigs);
    normals_original.download(h_original);

    ASSERT_EQ(h_fvigs.rows, h_original.rows);
    ASSERT_EQ(h_fvigs.cols, h_original.cols);
    ASSERT_EQ(h_fvigs.type(), CV_32FC4);
    ASSERT_EQ(h_original.type(), CV_32FC4);

    const ErrorMetrics metrics = compareNormals(h_fvigs, h_original);

    ASSERT_GT(metrics.valid_count, 0);
    EXPECT_LE(metrics.max_abs, 5.0e-2f);
    EXPECT_LE(metrics.mean_abs, 5.0e-3);
}
