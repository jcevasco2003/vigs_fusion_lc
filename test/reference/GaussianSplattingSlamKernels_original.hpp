#include "GaussianSplattingSlamKernels.hpp"
#include <Eigen/Dense>
#include <Eigen/QR>

#include <cstdio>

#include <cub/cub.cuh>

#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>

namespace cg = cooperative_groups;

#define BLOCK_SIZE (GSS_BLOCK_X * GSS_BLOCK_Y)
#define NUM_WARPS (BLOCK_SIZE / 32)

// #define USE_MEAN_DEPTH
#define USE_MEDIAN_DEPTH

namespace gaussian_splatting_slam
{
    #define GSS_DEPTH_HS 4
    __global__ void computeNormalsFromDepth_kernel(
        float4 *normalsData,
        cudaTextureObject_t texDepth,
        const CameraParameters cameraParams,
        uint32_t width,
        uint32_t height,
        uint32_t normalsStep)
    {
        const int nbShPoints = (GSS_BLOCK_X + 2 * GSS_DEPTH_HS) * (GSS_BLOCK_Y + 2 * GSS_DEPTH_HS);
        __shared__ float disps[(GSS_BLOCK_X + 2 * GSS_DEPTH_HS) * (GSS_BLOCK_Y + 2 * GSS_DEPTH_HS)];

        auto block = cg::this_thread_block();
        int tid = block.thread_rank();

        for (int k = tid; k < nbShPoints; k += GSS_BLOCK_SIZE)
        {
            int x = blockIdx.x * blockDim.x - GSS_DEPTH_HS + k / (GSS_BLOCK_Y + 2 * GSS_DEPTH_HS);
            int y = blockIdx.y * blockDim.y - GSS_DEPTH_HS + k % (GSS_BLOCK_Y + 2 * GSS_DEPTH_HS);

            if (x < 0 || x >= width || y < 0 || y >= height)
            {
                disps[k] = 0.f;
            }
            else
            {
                float depth = tex2D<float>(texDepth, x, y);

                if (depth == 0.f)
                {
                    disps[k] = 0.f;
                }
                else
                {
                    disps[k] = 1. / depth;
                }
            }
        }

        block.sync();

        int x = blockIdx.x * blockDim.x + threadIdx.x;
        int y = blockIdx.y * blockDim.y + threadIdx.y;

        if (x >= width - 1 || y >= height - 1)
        {
            return;
        }

        double sx, sy, sz, sw;
        double sxx, sxy, sxz, syy, syz, szz;
        sx = sy = sz = sw = sxx = sxy = sxz = syy = syz = szz = 0.f;

        for (int j = -GSS_DEPTH_HS; j <= GSS_DEPTH_HS; j++)
        {
            for (int i = -GSS_DEPTH_HS; i <= GSS_DEPTH_HS; i++)
            {
                float disp = disps[(threadIdx.x + GSS_DEPTH_HS + j) * (blockDim.y + 2 * GSS_DEPTH_HS) + threadIdx.y + GSS_DEPTH_HS + i];
                if (disp > 0.f)
                {
                    const float w = 1.f; // expf(-0.5f*(i*i+j*j)*0.5f);
                    sx += w * j;
                    sy += w * i;
                    sz += w * disp;
                    sw += w;
                    sxx += w * j * j;
                    sxy += w * j * i;
                    sxz += w * j * disp;
                    syy += w * i * i;
                    syz += w * i * disp;
                    szz += w * disp * disp;
                }
            }
        }

        Eigen::Matrix3d A;
        A << sxx, sxy, sx,
            sxy, syy, sy,
            sx, sy, sw;

        Eigen::Vector3d b(sxz, syz, sz);

        float3 n = {0.f, 0.f, 0.f};

        float4 *normal_row = (float4 *)&((unsigned char *)normalsData)[y * normalsStep];

        if (A.determinant() > 1e-6)
        {
            Eigen::Vector3d plane = A.inverse() * b;

            n.x = plane.x() * cameraParams.f.x;
            n.y = plane.y() * cameraParams.f.y;
            n.z = plane.z() + plane.x() * (cameraParams.c.x - x) + plane.y() * (cameraParams.c.y - y);

            // n = {(float)plane.x(), (float)plane.y(), (float)plane.z()};
            n = normalize(n);
            if (n.z > 0.f)
            {
                n = -n;
            }
            normal_row[x] = make_float4(n.x, n.y, n.z, 1.f);
        }
        else
        {
            normal_row[x] = make_float4(0.f, 0.f, 0.f, 0.f);
        }
        // normal_row[x] = make_float4(p0.z, p0.z, p0.z, 0.f);
    }
}