#include "NormalsKernel_original.cuh"

#include <Eigen/Dense>
#include <cooperative_groups.h>

#include <cuda_utils/vector_math.cuh>

namespace cg = cooperative_groups;

#define GSS_BLOCK_X 16
#define GSS_BLOCK_Y 16
#define GSS_BLOCK_SIZE (GSS_BLOCK_X * GSS_BLOCK_Y)
#define GSS_DEPTH_HS 4

namespace gaussian_splatting_slam_original
{
__global__ void computeNormalsFromDepth_kernel(
    float4 *normalsData,
    cudaTextureObject_t texDepth,
    const gaussian_splatting_slam::CameraParameters cameraParams,
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

        if (x < 0 || x >= static_cast<int>(width) || y < 0 || y >= static_cast<int>(height))
        {
            disps[k] = 0.f;
        }
        else
        {
            float depth = tex2D<float>(texDepth, x, y);
            disps[k] = (depth == 0.f) ? 0.f : (1.f / depth);
        }
    }

    block.sync();

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= static_cast<int>(width) - 1 || y >= static_cast<int>(height) - 1)
    {
        return;
    }

    double sx = 0.0;
    double sy = 0.0;
    double sz = 0.0;
    double sw = 0.0;
    double sxx = 0.0;
    double sxy = 0.0;
    double sxz = 0.0;
    double syy = 0.0;
    double syz = 0.0;

    for (int j = -GSS_DEPTH_HS; j <= GSS_DEPTH_HS; j++)
    {
        for (int i = -GSS_DEPTH_HS; i <= GSS_DEPTH_HS; i++)
        {
            float disp = disps[(threadIdx.x + GSS_DEPTH_HS + j) * (blockDim.y + 2 * GSS_DEPTH_HS) + threadIdx.y + GSS_DEPTH_HS + i];
            if (disp > 0.f)
            {
                const float w = 1.f;
                sx += w * j;
                sy += w * i;
                sz += w * disp;
                sw += w;
                sxx += w * j * j;
                sxy += w * j * i;
                sxz += w * j * disp;
                syy += w * i * i;
                syz += w * i * disp;
            }
        }
    }

    Eigen::Matrix3d A;
    A << sxx, sxy, sx,
         sxy, syy, sy,
         sx,  sy,  sw;

    Eigen::Vector3d b(sxz, syz, sz);

    float3 n = {0.f, 0.f, 0.f};
    float4 *normal_row = reinterpret_cast<float4 *>(reinterpret_cast<unsigned char *>(normalsData) + y * normalsStep);

    if (A.determinant() > 1e-6)
    {
        Eigen::Vector3d plane = A.inverse() * b;

        n.x = static_cast<float>(plane.x() * cameraParams.f.x);
        n.y = static_cast<float>(plane.y() * cameraParams.f.y);
        n.z = static_cast<float>(plane.z() + plane.x() * (cameraParams.c.x - x) + plane.y() * (cameraParams.c.y - y));

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
}
}
