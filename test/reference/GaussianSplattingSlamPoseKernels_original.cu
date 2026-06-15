#include "GaussianSplattingSlamPoseKernels_original.hpp"

#include <Eigen/Dense>

#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>

namespace cg = cooperative_groups;

#define BLOCK_SIZE (GSS_BLOCK_X * GSS_BLOCK_Y)

namespace gaussian_splatting_slam_original
{
    using gaussian_splatting_slam::SplattedGaussian;

    inline __device__ void forwardPass(
        float3 &output_color,
        float &output_depth,
        float &final_T,
        uint32_t &n_contrib,
        int x,
        int y,
        uint32_t tile_id,
        SplattedGaussian *__restrict__ splattedGaussian_sh,
        uint32_t *__restrict__ gids_sh,
        const uint2 *__restrict__ ranges,
        const uint32_t *__restrict__ indices,
        const float3 *__restrict__ imgPositions,
        const float3 *__restrict__ imgInvSigmas,
        const float2 *__restrict__ pHats,
        const float3 *__restrict__ colors,
        const float *__restrict__ alphas,
        const float3 &bgColor,
        bool inside)
    {
        auto block = cg::this_thread_block();
        int tid = block.thread_rank();

        uint2 range = ranges[tile_id];
        int n = range.y - range.x;

        float3 color = make_float3(0.f, 0.f, 0.f);
        float depth = 0.f;
        uint32_t contributor = 0;
        uint32_t last_contributor = 0;
        float T = 1.f;
        bool done = !inside;

        for (int k = 0; k < (n + BLOCK_SIZE - 1) / BLOCK_SIZE; k++)
        {
            if (k * BLOCK_SIZE + tid < n)
            {
                uint32_t gid = indices[range.x + k * BLOCK_SIZE + tid];
                gids_sh[tid] = gid;
                splattedGaussian_sh[tid].position = imgPositions[gid];
                splattedGaussian_sh[tid].invSigma = imgInvSigmas[gid];
                splattedGaussian_sh[tid].color = colors[gid];
                splattedGaussian_sh[tid].alpha = alphas[gid];
                splattedGaussian_sh[tid].pHat = pHats[gid];
            }
            block.sync();

            for (int i = 0; !done && i + k * BLOCK_SIZE < n && i < BLOCK_SIZE; i++)
            {
                contributor++;
                const float dx = splattedGaussian_sh[i].position.x - x;
                const float dy = splattedGaussian_sh[i].position.y - y;
                const float v = splattedGaussian_sh[i].invSigma.x * dx * dx + 2.f * splattedGaussian_sh[i].invSigma.y * dx * dy + splattedGaussian_sh[i].invSigma.z * dy * dy;
                const float alpha_i = min(0.99f, splattedGaussian_sh[i].alpha * expf(-0.5f * v));
                if (alpha_i < 1.f / 255.f || v <= 0.f)
                    continue;
                float test_T = T * (1.f - alpha_i);
                if (test_T < 0.0001f)
                {
                    done = true;
                    continue;
                }
                color += splattedGaussian_sh[i].color * alpha_i * T;

                float d = splattedGaussian_sh[i].position.z + dx * splattedGaussian_sh[i].pHat.x + dy * splattedGaussian_sh[i].pHat.y;

                if (T > 0.5f && test_T < 0.5f)
                {
                    depth = d;
                }

                T = test_T;
                last_contributor = contributor;
            }
        }

        if (inside)
        {
            final_T = T;
            n_contrib = last_contributor;
            color += T * bgColor;
            output_color = color;
            output_depth = depth;
        }
    }

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
        uint32_t height)
    {
        int x = blockIdx.x * blockDim.x + threadIdx.x;
        int y = blockIdx.y * blockDim.y + threadIdx.y;

        auto block = cg::this_thread_block();
        int tid = block.thread_rank();

        __shared__ MotionTrackingData mtd_sh[BLOCK_SIZE];
        __shared__ SplattedGaussian splattedGaussian_sh[BLOCK_SIZE];
        __shared__ uint32_t gids_sh[BLOCK_SIZE];

        float JtJ_data[36];

        MotionTrackingData &mtd_i = mtd_sh[tid];

        Eigen::Map<Eigen::Matrix<float, 6, 6>> JtJ(JtJ_data);
        Eigen::Map<Eigen::Vector<float, 6>> Jtr(mtd_i.Jtr);

        JtJ.setZero();
        Jtr.setZero();

        const float imgDepth = tex2D<float>(texDepth, x, y);

        int tileId = blockIdx.y * numTiles.x + blockIdx.x;

        bool inside = x < width && y < height && imgDepth > 0.1f;

        float3 color = make_float3(0.f);
        float depth = 0.f;
        float final_T;
        uint32_t n_contrib;

        forwardPass(
            color,
            depth,
            final_T,
            n_contrib,
            x,
            y,
            tileId,
            splattedGaussian_sh,
            gids_sh,
            ranges,
            indices,
            imgPositions,
            imgInvSigmas,
            pHats,
            colors,
            alphas,
            bgColor,
            inside);

        uint2 range = ranges[tileId];
        int n = range.y - range.x;

        inside &= final_T < alphaThresh;

        uchar4 rgba = tex2D<uchar4>(texRGBA, x, y);

        Eigen::Vector3f color_eig(color.x, color.y, color.z);
        const Eigen::Vector3f color_error = color_eig - Eigen::Vector3f(rgba.x / 255.f, rgba.y / 255.f, rgba.z / 255.f);
        const float depth_error = imgDepth > 0.1f ? depth - imgDepth : 0.f;

        Eigen::Map<const Eigen::Vector3f> p_cam((float *)&cameraPose.position);
        Eigen::Map<const Eigen::Quaternionf> q_cam((float *)&cameraPose.orientation);
        Eigen::Vector3f ray((x - cameraParams.c.x) / cameraParams.f.x,
                            (y - cameraParams.c.y) / cameraParams.f.y,
                            1.f);

        Eigen::Matrix3f ray_cross{{0.f, -ray.z(), ray.y()},
                                  {ray.z(), 0.f, -ray.x()},
                                  {-ray.y(), ray.x(), 0.f}};

        Eigen::Matrix3f R = q_cam.toRotationMatrix();

        float T = 1.f;
        float prod_alpha = 1.f;

        if (inside)
        {
            float4 gradX = tex2D<float4>(texDx, x, y);
            float4 gradY = tex2D<float4>(texDy, x, y);

            Eigen::Matrix<float, 2, 3> dl_Img{{gradX.x, gradX.y, gradX.z},
                                              {gradY.x, gradY.y, gradY.z}};

            float lc = color_error.norm();
            float wc = lc < colorThresh ? 1.f : colorThresh / lc;

            Eigen::Matrix<float, 3, 2> Jt{{cameraParams.f.x / imgDepth, 0.f},
                                          {0.f, cameraParams.f.y / imgDepth},
                                          {-cameraParams.f.x * ray.x() / imgDepth, -cameraParams.f.y * ray.y() / imgDepth}};

            Eigen::Matrix3f Jt_cam = Jt * dl_Img;

            Eigen::Matrix3f JtJ_cam = wc * Jt_cam * Jt_cam.transpose();
            Eigen::Vector3f Jtr_cam = -wc * Jt_cam * color_error;

            float ld = fabsf(depth_error);
            float wd = ld < depthThresh ? 1.f : depthThresh / ld;
            wd /= imgDepth;
            JtJ_cam += wd * ray * ray.transpose();
            Jtr_cam += wd * depth_error * ray;

            Eigen::Matrix<float, 6, 3> Jpose;

            Jpose.block<3, 3>(0, 0) = R;
            Jpose.block<3, 3>(3, 0) = imgDepth * ray_cross;

            JtJ += Jpose * JtJ_cam * Jpose.transpose();
            Jtr += Jpose * Jtr_cam;
        }

        Eigen::Vector3f acc_c(0.f, 0.f, 0.f);
        float alpha_prev = 0.f;
        Eigen::Vector3f color_prev(0.f, 0.f, 0.f);

        for (int k = 0; k < (n + BLOCK_SIZE - 1) / BLOCK_SIZE; k++)
        {
            if (k * BLOCK_SIZE + tid < n)
            {
                uint32_t gid = indices[range.x + k * BLOCK_SIZE + tid];
                gids_sh[tid] = gid;
                splattedGaussian_sh[tid].position = imgPositions[gid];
                splattedGaussian_sh[tid].invSigma = imgInvSigmas[gid];
                splattedGaussian_sh[tid].color = colors[gid];
                splattedGaussian_sh[tid].alpha = alphas[gid];
                splattedGaussian_sh[tid].pHat = pHats[gid];
            }
            block.sync();

            for (int i = 0; inside && i + k * BLOCK_SIZE < n_contrib && i < BLOCK_SIZE; i++)
            {
                const float dx = splattedGaussian_sh[i].position.x - x;
                const float dy = splattedGaussian_sh[i].position.y - y;
                const float d = splattedGaussian_sh[i].position.z + dx * splattedGaussian_sh[i].pHat.x + dy * splattedGaussian_sh[i].pHat.y;
                const float v = splattedGaussian_sh[i].invSigma.x * dx * dx + 2.f * splattedGaussian_sh[i].invSigma.y * dx * dy + splattedGaussian_sh[i].invSigma.z * dy * dy;

                const float G = expf(-0.5f * v);
                const float alpha_i = min(0.99f, splattedGaussian_sh[i].alpha * G);
                if (alpha_i < 1.f / 255.f)
                    continue;

                Eigen::Vector3f d_alpha(splattedGaussian_sh[i].color.x * prod_alpha,
                                        splattedGaussian_sh[i].color.y * prod_alpha,
                                        splattedGaussian_sh[i].color.z * prod_alpha);

                acc_c += alpha_i * d_alpha;
                d_alpha -= (color_eig - acc_c) / (1.f - alpha_i);

                Eigen::Vector2f dl_mean2d(splattedGaussian_sh[i].invSigma.x * dx + splattedGaussian_sh[i].invSigma.y * dy,
                                          splattedGaussian_sh[i].invSigma.y * dx + splattedGaussian_sh[i].invSigma.z * dy);

                float lc = color_error.norm();
                float wc = lc < colorThresh ? 1.f : colorThresh / lc;

                Eigen::Vector3f ray((splattedGaussian_sh[i].position.x - cameraParams.c.x) / cameraParams.f.x,
                                    (splattedGaussian_sh[i].position.y - cameraParams.c.y) / cameraParams.f.y,
                                    1.f);

                Eigen::Matrix3f ray_cross{{0.f, -ray.z(), ray.y()},
                                          {ray.z(), 0.f, -ray.x()},
                                          {-ray.y(), ray.x(), 0.f}};

                Eigen::Matrix<float, 3, 2> Jt{{cameraParams.f.x / d, 0.f},
                                              {0.f, cameraParams.f.y / d},
                                              {-cameraParams.f.x * ray.x() / d, -cameraParams.f.y * ray.y() / d}};

                Eigen::Matrix3f Jt_cam = Jt * dl_mean2d * alpha_i * d_alpha.transpose();

                Eigen::Matrix3f JtJ_cam = wc * Jt_cam * Jt_cam.transpose();
                Eigen::Vector3f Jtr_cam = -wc * Jt_cam * color_error;

                float test_T = alpha_i;
                (void)test_T;

                Eigen::Matrix<float, 6, 3> Jpose;
                Jpose.block<3, 3>(0, 0) = R;
                Jpose.block<3, 3>(3, 0) = d * ray_cross;

                JtJ += Jpose * JtJ_cam * Jpose.transpose();
                Jtr += Jpose * Jtr_cam;

                prod_alpha *= (1.f - alpha_i);

                if (T > 0.5f && prod_alpha <= 0.5f && imgDepth > 0.5f)
                {
                    T = prod_alpha;
                }

                if (prod_alpha < 0.001f)
                {
                    break;
                }
            }
        }

        block.sync();

        int k = 0;
        for (int i = 0; i < 6; i++)
            for (int j = i; j < 6; j++, k++)
            {
                mtd_i.JtJ[k] = JtJ_data[6 * i + j];
            }

        block.sync();

        if (tid == 0)
        {
            MotionTrackingData &mtd0 = mtd_sh[0];
            for (int i = 0; i < 21; i++)
            {
                atomicAdd(&mtd->JtJ[i], mtd0.JtJ[i]);
            }
            for (int i = 0; i < 6; i++)
            {
                atomicAdd(&mtd->Jtr[i], mtd0.Jtr[i]);
            }
        }
    }
}