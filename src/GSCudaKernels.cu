#include <f_vigs_slam/GSCudaKernels.cuh>
#include <math.h>
#include <algorithm>
#include <math_functions.h>
#include <cub/cub.cuh>  // para reduce

#define TILE_SIZE 16
#define BLOCK_SIZE 256

namespace f_vigs_slam
{
    __device__ __constant__ int g_gpu_exp_eval_mode = static_cast<int>(GpuExpEvaluationMode::DEFAULT);

    void setGpuExpEvaluationMode(GpuExpEvaluationMode mode)
    {
        const int mode_value = static_cast<int>(mode);
        cudaMemcpyToSymbol(g_gpu_exp_eval_mode, &mode_value, sizeof(mode_value));
    }

#if 0
    enum PoseDebugStageCode
    {
        POSE_DBG_NONE = 0,
        POSE_DBG_IMG_DEPTH = 1,
        POSE_DBG_GAUSS_MAHALANOBIS = 2,
        POSE_DBG_GAUSS_ALPHA = 3,
        POSE_DBG_FORWARD_DEPTH = 4,
        POSE_DBG_COLOR_ERROR = 5,
        POSE_DBG_COLOR_WEIGHT = 6,
        POSE_DBG_GRADIENT = 7,
        POSE_DBG_JT = 8,
        POSE_DBG_DEPTH_WEIGHT = 9,
        POSE_DBG_JTJ_CAM = 10,
        POSE_DBG_JTR_CAM = 11,
        POSE_DBG_LOCAL_ACCUM = 12,
        POSE_DBG_BLOCK_REDUCE = 13
    };

    __device__ __forceinline__ void recordPoseDebug(
        PoseKernelDebugInfo *debug_info,
        int stage,
        int x,
        int y,
        float a,
        float b,
        float c,
        float d)
    {
        if (!debug_info)
        {
            return;
        }

        atomicAdd(&debug_info->total_nonfinite_events, 1u);
        if (stage >= 0 && stage < 16)
        {
            atomicAdd(&debug_info->stage_counts[stage], 1u);
        }

        if (atomicCAS(&debug_info->first_stage, POSE_DBG_NONE, stage) == POSE_DBG_NONE)
        {
            debug_info->first_x = x;
            debug_info->first_y = y;
            debug_info->first_a = a;
            debug_info->first_b = b;
            debug_info->first_c = c;
            debug_info->first_d = d;
        }
    }
    #endif

    // Implementamos los kernels CUDA para operaciones paralelizables.
    // El archivo queda organizado en:
    // - Funciones auxiliares
    // - Inicializacion de gaussianas desde RGB-D
    // - Forward pass (renderizacion)
    // - Backward pass (optimizacion)
    // - Eliminacion de outliers y densificacion
    // - Actualizacion de parametros
    // ============================================================================
    // 1. FUNCIONES AUXILIARES
    // ============================================================================

    __device__ inline float evalGaussian2DExponent(
        float dx, float dy,
        float inv_cov_xx, float inv_cov_yy, float inv_cov_xy)
    {
        return -0.5f * (inv_cov_xx * dx * dx + inv_cov_yy * dy * dy + 2.0f * inv_cov_xy * dx * dy);
    }

    __device__ inline float evaluate_exponential(float x)
    {
        if (g_gpu_exp_eval_mode == static_cast<int>(GpuExpEvaluationMode::TAYLOR))
        {
            // Polinomio de Taylor de orden 5 con clamp para evitar inestabilidades numéricas.
            const float xc = fmaxf(-6.0f, fminf(6.0f, x));
            const float x2 = xc * xc;
            const float x3 = x2 * xc;
            const float x4 = x2 * x2;
            const float x5 = x4 * xc;
            const float approx = 1.0f + xc + 0.5f * x2 + (1.0f / 6.0f) * x3 +
                                 (1.0f / 24.0f) * x4 + (1.0f / 120.0f) * x5;
            return fmaxf(0.0f, approx);
        }

        return __expf(x);
    }

    __device__ inline float evalGaussian2D(
        float dx, float dy,
        float inv_cov_xx, float inv_cov_yy, float inv_cov_xy)
    {
        return evaluate_exponential(evalGaussian2DExponent(dx, dy, inv_cov_xx, inv_cov_yy, inv_cov_xy));
    }

    __device__ inline float evalGaussian2DFromMahalanobis(float v)
    {
        return evaluate_exponential(-0.5f * v);
    }

    __device__ inline float mahalanobis2D(
        float dx, float dy,
        float inv_cov_xx, float inv_cov_yy, float inv_cov_xy)
    {
        return inv_cov_xx * dx * dx + inv_cov_yy * dy * dy + 2.0f * inv_cov_xy * dx * dy;
    }

    __device__ inline float evalGaussianWeight(
        float dx, float dy,
        const float3& inv_cov,
        float alpha)
    {
        float v = mahalanobis2D(dx, dy, inv_cov.x, inv_cov.z, inv_cov.y);

        if (v < 0.f) return 0.f;

        float w = alpha * evalGaussian2DFromMahalanobis(v);

        if (w < (1.f / 255.f)) return 0.f;

        return fminf(0.99f, w);
    }

    __device__ inline void invert2x2(
        float cov_xx, float cov_yy, float cov_xy,
        float &inv_cov_xx, float &inv_cov_yy, float &inv_cov_xy)
    {
        // Calculamos el determinante
        float det = cov_xx * cov_yy - cov_xy * cov_xy;
        if (det == 0.0f)
        {
            // Matriz singular, devolvemos identidad como fallback
            inv_cov_xx = 1.0f;
            inv_cov_yy = 1.0f;
            inv_cov_xy = 0.0f;
            return;
        }

        // Usamos la formula:
        // [ a  b ]^-1    =    1 / det * [ d  -b ]
        // [ c  d ]                      [ -c  a ]
        // nota: como cov es simetrica, b = c, por lo que la inversa es simetrica tambien

        float inv_det = 1.0f / det;

        // Invertimos la matriz 2x2
        inv_cov_xx =  cov_yy * inv_det;
        inv_cov_yy =  cov_xx * inv_det;
        inv_cov_xy = -cov_xy * inv_det;
    }

    __device__ inline float3 clampFloat3(const float3 &v, float lo, float hi)
    {
        return make_float3(
            fminf(hi, fmaxf(lo, v.x)),
            fminf(hi, fmaxf(lo, v.y)),
            fminf(hi, fmaxf(lo, v.z)));
    }

    // ============================================================================
    // 2. INICIALIZACION DE GAUSSIANAS DESDE RGB-D
    // ============================================================================
    __global__ void computeSobelRgb_kernel(
        const uchar4* input_rgb,
        size_t input_step,
        float4 *grad_x,
        size_t grad_x_step,
        float4 *grad_y,
        size_t grad_y_step,
        int width,
        int height)
    {
        int x = blockIdx.x * blockDim.x + threadIdx.x;
        int y = blockIdx.y * blockDim.y + threadIdx.y;

        int shared_width = blockDim.x + 2;
        int shared_height = blockDim.y + 2;
        // Para alinear con 16 bytes usamos una cuarta coordenada de relleno
        extern __shared__ float4 shared_tile[];

        int block_origin_x = blockIdx.x * blockDim.x;
        int block_origin_y = blockIdx.y * blockDim.y;

        for (int sy = threadIdx.y; sy < shared_height; sy += blockDim.y)
        {
            for (int sx = threadIdx.x; sx < shared_width; sx += blockDim.x)
            {
                int gx = clampIndex(block_origin_x + sx - 1, 0, width - 1);
                int gy = clampIndex(block_origin_y + sy - 1, 0, height - 1);
                int shared_idx = sy * shared_width + sx;

                const uchar4* row_ptr = reinterpret_cast<const uchar4*>(reinterpret_cast<const char*>(input_rgb) + gy * input_step);
                uchar4 pixel_u8 = row_ptr[gx];

                shared_tile[shared_idx] = make_float4(
                    pixel_u8.x * (1.0f / 255.0f), 
                    pixel_u8.y * (1.0f / 255.0f), 
                    pixel_u8.z * (1.0f / 255.0f), 
                    0.0f);
            }
        }
        __syncthreads();

        if (x >= width || y >= height)
        {
            return;
        }

        int local_x = threadIdx.x + 1;
        int local_y = threadIdx.y + 1;

        int idx00 = (local_y - 1) * shared_width + (local_x - 1);
        int idx01 = (local_y - 1) * shared_width + local_x;
        int idx02 = (local_y - 1) * shared_width + (local_x + 1);
        int idx10 = local_y * shared_width + (local_x - 1);
        int idx12 = local_y * shared_width + (local_x + 1);
        int idx20 = (local_y + 1) * shared_width + (local_x - 1);
        int idx21 = (local_y + 1) * shared_width + local_x;
        int idx22 = (local_y + 1) * shared_width + (local_x + 1);

        float4 p00 = shared_tile[idx00];
        float4 p01 = shared_tile[idx01];
        float4 p02 = shared_tile[idx02];
        float4 p10 = shared_tile[idx10];
        float4 p12 = shared_tile[idx12];
        float4 p20 = shared_tile[idx20];
        float4 p21 = shared_tile[idx21];
        float4 p22 = shared_tile[idx22];

        float4 gx;
        gx.x = -p00.x + p02.x - 2.0f * p10.x + 2.0f * p12.x - p20.x + p22.x;
        gx.y = -p00.y + p02.y - 2.0f * p10.y + 2.0f * p12.y - p20.y + p22.y;
        gx.z = -p00.z + p02.z - 2.0f * p10.z + 2.0f * p12.z - p20.z + p22.z;
        gx.w = 0.0f;


        float4 gy;
        gy.x = -p00.x - 2.0f * p01.x - p02.x + p20.x + 2.0f * p21.x + p22.x;
        gy.y = -p00.y - 2.0f * p01.y - p02.y + p20.y + 2.0f * p21.y + p22.y;
        gy.z = -p00.z - 2.0f * p01.z - p02.z + p20.z + 2.0f * p21.z + p22.z;
        gy.w = 0.0f;

        const float scale = 1.0f / 8.0f;

        gx.x *= scale;
        gx.y *= scale;
        gx.z *= scale;

        gy.x *= scale;
        gy.y *= scale;
        gy.z *= scale;

        char *gx_row_ptr = reinterpret_cast<char *>(grad_x) + y * grad_x_step;
        char *gy_row_ptr = reinterpret_cast<char *>(grad_y) + y * grad_y_step;
        float4 *gx_row = reinterpret_cast<float4 *>(gx_row_ptr);
        float4 *gy_row = reinterpret_cast<float4 *>(gy_row_ptr);

        gx_row[x] = gx;
        gy_row[x] = gy;
    }

    // ============================================================================
    // Cálculo de normales desde mapa de profundidad (GPU kernel)
    // ============================================================================
    /*
    __global__ void computeNormalsFromDepth_kernel(
        const float *depth,
        size_t depth_step,
        float3 *normals_out,
        size_t normals_step,
        int width,
        int height)
    {
        int x = blockIdx.x * blockDim.x + threadIdx.x;
        int y = blockIdx.y * blockDim.y + threadIdx.y;

        if (x >= width || y >= height) return;

        // Inicializar a [0, 0, 1] por defecto
        float3 normal = make_float3(0.0f, 0.0f, 1.0f);

        // Saltar píxeles de borde (necesitamos vecinos)
        if (x > 0 && x < width - 1 && y > 0 && y < height - 1)
        {
            // Acceso a los píxeles de profundidad
            const unsigned char *depth_row_center = 
                reinterpret_cast<const unsigned char *>(depth) + y * depth_step;
            const float *depth_row_center_f = reinterpret_cast<const float *>(depth_row_center);
            
            const unsigned char *depth_row_up = 
                reinterpret_cast<const unsigned char *>(depth) + (y - 1) * depth_step;
            const float *depth_row_up_f = reinterpret_cast<const float *>(depth_row_up);
            
            const unsigned char *depth_row_down = 
                reinterpret_cast<const unsigned char *>(depth) + (y + 1) * depth_step;
            const float *depth_row_down_f = reinterpret_cast<const float *>(depth_row_down);

            float z_c = depth_row_center_f[x];
            float z_l = depth_row_center_f[x - 1];
            float z_r = depth_row_center_f[x + 1];
            float z_u = depth_row_up_f[x];
            float z_d = depth_row_down_f[x];

            // Verificar que tenemos profundidad válida
            if (z_c > 0.01f && z_l > 0.01f && z_r > 0.01f && z_u > 0.01f && z_d > 0.01f)
            {
                // Calcular gradientes de profundidad (diferencias finitas)
                float dz_dx = (z_r - z_l) / 2.0f;
                float dz_dy = (z_d - z_u) / 2.0f;

                // Vectores tangentes en el espacio de imagen: (u, v, z)
                float3 tangent_x = make_float3(1.0f, 0.0f, dz_dx);
                float3 tangent_y = make_float3(0.0f, 1.0f, dz_dy);

                // Normal = tangent_x × tangent_y
                normal = cross(tangent_x, tangent_y);

                // Normalizar
                float norm = sqrtf(normal.x * normal.x + normal.y * normal.y + normal.z * normal.z);
                if (norm > 1e-6f)
                {
                    normal.x /= norm;
                    normal.y /= norm;
                    normal.z /= norm;
                }

                // Asegurar que normal apunta hacia la cámara (Z > 0, componente positiva)
                if (normal.z < 0.0f)
                {
                    normal.x = -normal.x;
                    normal.y = -normal.y;
                    normal.z = -normal.z;
                }
            }
        }

        // Escribir en el buffer de salida
        unsigned char *normals_row = reinterpret_cast<unsigned char *>(normals_out) + y * normals_step;
        float3 *normals_row_f = reinterpret_cast<float3 *>(normals_row);
        normals_row_f[x] = normal;
    }
    */

#define HALF_WINDOW 4  // tamaño del vecindario

__global__ void computeNormalsFromDepth_kernel(
    cudaTextureObject_t depth_tex,   // textura de depth 
    float4* normals_out,             // salida
    size_t normals_step,             // pitch en bytes
    int width,
    int height,
    IntrinsicParameters K)
{
    // ============================================================
    // 1. SHARED MEMORY TILE (con halo)
    // ============================================================
    constexpr int TILE_W = TILE_SIZE;
    constexpr int TILE_H = TILE_SIZE;

    constexpr int SH_W = TILE_W + 2 * HALF_WINDOW;
    constexpr int SH_H = TILE_H + 2 * HALF_WINDOW;
    constexpr int SH_SIZE = SH_W * SH_H;

    __shared__ float disp_tile[SH_SIZE]; // disparity = 1/depth

    // Thread id lineal dentro del bloque
    const int tid = threadIdx.y * blockDim.x + threadIdx.x;

    // ============================================================
    // 2. Carga del tile con halo en shared memory
    // ============================================================
    for (int idx = tid; idx < SH_SIZE; idx += blockDim.x * blockDim.y)
    {
        int local_y = idx / SH_W;
        int local_x = idx % SH_W;

        int global_x = blockIdx.x * blockDim.x - HALF_WINDOW + local_x;
        int global_y = blockIdx.y * blockDim.y - HALF_WINDOW + local_y;

        float disp = 0.0f;

        if (global_x >= 0 && global_x < width &&
            global_y >= 0 && global_y < height)
        {
            float d = getImageData<float>(depth_tex, global_x, global_y);

            if (d > 0.0f)
                disp = 1.0f / d;
        }

        disp_tile[local_y * SH_W + local_x] = disp;
    }

    __syncthreads();

    // ============================================================
    // 3. Pixel actual
    // ============================================================
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width - 1 || y >= height - 1)
        return;

    // ============================================================
    // 4. Acumuladores para ajuste de plano
    // ============================================================
    float sx = 0.f, sy = 0.f, sz = 0.f, sw = 0.f;
    float sxx = 0.f, sxy = 0.f, sxz = 0.f;
    float syy = 0.f, syz = 0.f;

    // ============================================================
    // 5. Recorrer ventana local
    // ============================================================
    for (int dy = -HALF_WINDOW; dy <= HALF_WINDOW; ++dy)
    {
        for (int dx = -HALF_WINDOW; dx <= HALF_WINDOW; ++dx)
        {
            int sh_x = threadIdx.x + HALF_WINDOW + dx;
            int sh_y = threadIdx.y + HALF_WINDOW + dy;

            float disp = disp_tile[sh_y * SH_W + sh_x];

            if (disp > 0.0f)
            {
                const float w = 1.0f;

                sx  += w * dx;
                sy  += w * dy;
                sz  += w * disp;
                sw  += w;

                sxx += w * dx * dx;
                sxy += w * dx * dy;
                sxz += w * dx * disp;

                syy += w * dy * dy;
                syz += w * dy * disp;
            }
        }
    }

    // ============================================================
    // 6. Sistema lineal A * plane = b
    // ============================================================
    float a00 = sxx, a01 = sxy, a02 = sx;
    float a10 = sxy, a11 = syy, a12 = sy;
    float a20 = sx,  a21 = sy,  a22 = sw;

    float b0 = sxz, b1 = syz, b2 = sz;

    float detA =
          a00 * (a11 * a22 - a12 * a21)
        - a01 * (a10 * a22 - a12 * a20)
        + a02 * (a10 * a21 - a11 * a20);

    float3 normal = make_float3(0.f, 0.f, 0.f);

    // ============================================================
    // 7. Resolver con Cramer
    // ============================================================
    if (fabsf(detA) > 1e-6f)
    {
        float inv_detA = 1.0f / detA;

        if (fabsf(inv_detA) > 1e8f) return;

        float3 plane;

        plane.x =
            (b0*(a11*a22 - a12*a21)
           - a01*(b1*a22 - a12*b2)
           + a02*(b1*a21 - a11*b2)) * inv_detA;

        plane.y =
            (a00*(b1*a22 - a12*b2)
           - b0*(a10*a22 - a12*a20)
           + a02*(a10*b2 - b1*a20)) * inv_detA;

        plane.z =
            (a00*(a11*b2 - b1*a21)
           - a01*(a10*b2 - b1*a20)
           + b0*(a10*a21 - a11*a20)) * inv_detA;
        
        // ========================================================
        // 8. Convertir plano -> normal en cámara
        // ========================================================
        float3 n;

        n.x = plane.x * K.f.x;
        n.y = plane.y * K.f.y;
        n.z = plane.z
            + plane.x * (K.c.x - x)
            + plane.y * (K.c.y - y);

        float norm = sqrtf(n.x*n.x + n.y*n.y + n.z*n.z);

        if (norm > 1e-8f)
        {
            n.x /= norm;
            n.y /= norm;
            n.z /= norm;

            // Convención: normal mirando hacia la cámara
            if (n.z > 0.f)
            {
                n.x = -n.x;
                n.y = -n.y;
                n.z = -n.z;
            }

            normal = n;
        }
    }

    // ============================================================
    // 9. Escritura (respetando pitch)
    // ============================================================
    float4 normal4 = make_float4(normal.x, normal.y, normal.z, 1.f);
    setImageData<float4>(normals_out, normals_step, x, y, normal4);
}

    __global__ void initGaussiansFromRgbd_kernel(
        float4 *positions,
        float4 *scales,
        float4 *orientations,
        float4 *colors,
        float *opacities,
        uint32_t *instanceCounter,
        uint32_t maxGaussians,

        cudaTextureObject_t rgb,
        cudaTextureObject_t depth,
        cudaTextureObject_t normals,
        cudaTextureObject_t initMask,


        int width,
        int height,

        IntrinsicParameters intrinsics,
        Pose cameraPose,

        uint32_t sample_dx,
        uint32_t sample_dy,

        float init_opacity,
        uint32_t use_mask,
        float scale_factor,
        Pose submap_pose_global)
    {
        // ============================================================
        // 1. Coordenadas de la grilla de muestreo (subsampleo)
        // ============================================================
        int sx = blockIdx.x * blockDim.x + threadIdx.x;
        int sy = blockIdx.y * blockDim.y + threadIdx.y;

        int sample_w = (width  + sample_dx - 1) / sample_dx;
        int sample_h = (height + sample_dy - 1) / sample_dy;

        if (sx >= sample_w || sy >= sample_h) return;

        // ============================================================
        // 2. Convertir celda -> pixel (centro de la celda)
        // ============================================================
        float u_f = (sx + 0.5f) * sample_dx;
        float v_f = (sy + 0.5f) * sample_dy;

        if (u_f >= width || v_f >= height) return;

        int u = static_cast<int>(u_f);
        int v = static_cast<int>(v_f);

        if (use_mask != 0u && getImageData<float>(initMask, u, v) <= 0.f) return;

        // ============================================================
        // 3. Lectura de datos (RGB-D + normal)
        // ============================================================
        uchar4 rgba = getImageData<uchar4>(rgb, u, v);   // BGRA
        float   z   = getImageData<float>(depth, u, v);

        // Filtrado de profundidad inválida
        if (!(z >= 0.5f)) return;

        float4 n4 = getImageData<float4>(normals, u, v);
        float3 n = make_float3(n4.x, n4.y, n4.z);

        // ============================================================
        // 4. Validación y normalización de la normal
        // ============================================================
        float norm_len = sqrtf(n.x * n.x + n.y * n.y + n.z * n.z);
        bool normal_valid = norm_len >= 1e-6f;

        if (normal_valid) {
            n.x /= norm_len;
            n.y /= norm_len;
            n.z /= norm_len;
        }

        // ============================================================
        // 5. Proyección pixel → coordenadas 3D (frame cámara)
        // ============================================================
        float x = (u - intrinsics.c.x) * z / intrinsics.f.x;
        float y = (v - intrinsics.c.y) * z / intrinsics.f.y;

        float3 pos_cam = make_float3(x, y, z);

        // Transformación a mundo
        float3 pos_world =
            cameraPose.position +
            rotateByQuaternion(cameraPose.orientation, pos_cam);

        // ============================================================
        // 6. Orientación (alinear normal con eje Z)
        // ============================================================
        if (normal_valid && n.z < 0.0f) {
            n.x = -n.x;
            n.y = -n.y;
            n.z = -n.z;
        }

        float4 q_from_normal = normal_valid
            ? quatFromTwoVectors(make_float3(0.f, 0.f, 1.f), n)
            : make_float4(0.f, 0.f, 0.f, 1.f);

        float4 q_world = quatMultiply(cameraPose.orientation, q_from_normal);

        float q_norm = sqrtf(q_world.x*q_world.x + q_world.y*q_world.y + 
                             q_world.z*q_world.z + q_world.w*q_world.w);
        if (q_norm < 1e-6f) return;
        q_world.x /= q_norm;
        q_world.y /= q_norm;
        q_world.z /= q_norm;
        q_world.w /= q_norm;

        // ============================================================
        // 7. Escala (depende de profundidad y sampling)
        // ============================================================
        float scale_x = scale_factor * sample_dx * z / intrinsics.f.x;
        float scale_y = scale_factor * sample_dy * z / intrinsics.f.y;

        if (scale_x <= 0.f || scale_y <= 0.f) return;

        float avg = 0.5f * (scale_x + scale_y);

        float3 scale = make_float3(
            avg,
            avg,
            0.1f * avg
        );
        
        if (scale.x <= 1e-6f || scale.y <= 1e-6f || scale.z <= 1e-6f) return;

        // ============================================================
        // 8. Reserva de índice
        // ============================================================
        uint32_t idx = atomicAggInc(instanceCounter);
        if (idx >= maxGaussians) return;

        // ============================================================
        // 8b. TRANSFORMACION A COORDENADAS LOCALES DEL SUBMAPA
        // ============================================================
        const float4 q_submap_inv = quaternionInverse(submap_pose_global.orientation);
        const float3 pos_local = rotateByQuaternion(
            q_submap_inv,
            pos_world - submap_pose_global.position);

        float4 q_local = quatMultiply(q_submap_inv, q_world);
        const float q_local_norm = sqrtf(q_local.x * q_local.x + q_local.y * q_local.y +
                                         q_local.z * q_local.z + q_local.w * q_local.w);
        if (q_local_norm < 1e-6f) return;
        q_local.x /= q_local_norm;
        q_local.y /= q_local_norm;
        q_local.z /= q_local_norm;
        q_local.w /= q_local_norm;

        // ============================================================
        // 9. Escritura de la gaussiana (en coordenadas LOCALES del submapa)
        // ============================================================
        positions[idx]    = make_float4(pos_local.x, pos_local.y, pos_local.z, 1.f);
        scales[idx]       = make_float4(scale.x, scale.y, scale.z, 0.f);
        orientations[idx] = q_local;

        // BGRA -> RGB normalizado
        colors[idx] = make_float4(
            rgba.x / 255.f,
            rgba.y / 255.f,
            rgba.z / 255.f,
            0.f
        );

        opacities[idx] = init_opacity;
    }

    __global__ void backprojectDepthImage_kernel(
        float3 *points_out,
        uint32_t *pointCounter,
        uint32_t maxPoints,
        const float *depth,
        size_t depth_step,
        int width,
        int height,
        IntrinsicParameters intrinsics,
        uint32_t sample_stride)
    {
        const int x = blockIdx.x * blockDim.x + threadIdx.x;
        const int y = blockIdx.y * blockDim.y + threadIdx.y;

        if (x >= width || y >= height)
        {
            return;
        }

        if (sample_stride > 1u &&
            ((x % static_cast<int>(sample_stride)) != 0 || (y % static_cast<int>(sample_stride)) != 0))
        {
            return;
        }

        const float z = getImageData<float>(depth, depth_step, x, y);
        if (!isfinite(z) || z <= 0.0f)
        {
            return;
        }

        const uint32_t idx = atomicAggInc(pointCounter);
        if (idx >= maxPoints)
        {
            return;
        }

        points_out[idx] = make_float3(
            (static_cast<float>(x) - intrinsics.c.x) * z / intrinsics.f.x,
            (static_cast<float>(y) - intrinsics.c.y) * z / intrinsics.f.y,
            z);
    }

    // ============================================================================
    // KERNEL PARA TRANSFORMAR GAUSSIANOS DE LOCALES A GLOBALES
    // ============================================================================
    
    __global__ void transformGaussians_localToGlobal_kernel(
        const float4 *positions_local,
        const float4 *orientations_local,
        float4 *positions_global,
        float4 *orientations_global,
        Pose submap_pose_global,
        uint32_t n_gaussians)
    {
        uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= n_gaussians) return;

        // ============================================================
        // 1. Cargar gaussiana local
        // ============================================================
        float4 pos_local_4 = positions_local[idx];
        float3 pos_local = make_float3(pos_local_4.x, pos_local_4.y, pos_local_4.z);
        
        float4 q_local = orientations_local[idx];

        // ============================================================
        // 2. Transformar posición a global: P_global = T + R * P_local
        // ============================================================
        const float3 pos_global = submap_pose_global.position +
                      rotateByQuaternion(submap_pose_global.orientation, pos_local);

        // ============================================================
        // 3. Transformar orientación: q_global = q_submap * q_local
        // ============================================================
        float4 q_global = quatMultiply(submap_pose_global.orientation, q_local);
        const float q_global_norm = sqrtf(q_global.x * q_global.x + q_global.y * q_global.y +
                          q_global.z * q_global.z + q_global.w * q_global.w);
        if (q_global_norm < 1e-6f) return;
        q_global.x /= q_global_norm;
        q_global.y /= q_global_norm;
        q_global.z /= q_global_norm;
        q_global.w /= q_global_norm;

        // ============================================================
        // 4. Escribir gaussiana transformada
        // ============================================================
        positions_global[idx] = make_float4(pos_global.x, pos_global.y, pos_global.z, 1.f);
        orientations_global[idx] = q_global;
    }

    __global__ void tintGaussianColors_kernel(
        float4 *colors,
        uint32_t n_gaussians,
        float accent_b,
        float accent_g,
        float accent_r,
        float blend,
        float gain)
    {
        const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= n_gaussians || !colors)
        {
            return;
        }

        float4 color = colors[idx];
        const float retain = 1.0f - blend;
        color.x = fminf(1.0f, fmaxf(0.0f, color.x * gain * retain + accent_b * blend));
        color.y = fminf(1.0f, fmaxf(0.0f, color.y * gain * retain + accent_g * blend));
        color.z = fminf(1.0f, fmaxf(0.0f, color.z * gain * retain + accent_r * blend));
        colors[idx] = color;
    }



    // ============================================================================
    // 3. FORWARD PASS KERNELS (RENDERIZACION)
    // ============================================================================
    
    __global__ void projectAndHashGaussians_kernel(
        float4 *positions_2d,
        float4 *covariances_2d,
        float4 *inv_covariances_2d,
        float2 *p_hats,
        float4 *normals,
        uint64_t *hashes,
        uint32_t *gaussian_indices,
        uint32_t *instance_counter,
        const float4 *positions_world,
        const float4 *scales,
        const float4 *orientations,
        Pose camera_pose,
        IntrinsicParameters intrinsics,
        float min_depth,
        uint2 tile_size,
        uint2 num_tiles,
        uint32_t n_gaussians,
        uint32_t width,
        uint32_t height)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= n_gaussians) return;

        // ============================================================
        // 1. Valores por DEFAULT
        // ============================================================
        positions_2d[idx]        = make_float4(-1.f, -1.f, -1.f, 0.f);
        covariances_2d[idx]      = make_float4(0.f, 0.f, 0.f, 0.f);
        inv_covariances_2d[idx]  = make_float4(0.f, 0.f, 0.f, 0.f);
        p_hats[idx]              = make_float2(0.f, 0.f);
        normals[idx]             = make_float4(0.f, 0.f, 1.f, 0.f);

        // ============================================================
        // 2. LOAD + VALIDATION (world space)
        // ============================================================
        float3 p_world = make_float3(
            positions_world[idx].x,
            positions_world[idx].y,
            positions_world[idx].z
        );
        float4 scale4 = scales[idx];
        float3 scale = make_float3(scale4.x, scale4.y, scale4.z);
        float4 q_gauss = orientations[idx];

        // ============================================================
        // 3. TRANSFORMACION A CAMARA
        // ============================================================
        float3 mu_cam = rotateByQuaternionInverse(
            camera_pose.orientation,
            p_world - camera_pose.position
        );

        if (mu_cam.z < min_depth) return;

        // ============================================================
        // 4. PROJECCION
        // ============================================================
        float invz = 1.f / mu_cam.z;
        float u = intrinsics.f.x * mu_cam.x * invz + intrinsics.c.x;
        float v = intrinsics.f.y * mu_cam.y * invz + intrinsics.c.y;

        if (u < 0.f || u > (float)(width - 1) ||
            v < 0.f || v > (float)(height - 1)) return;

        positions_2d[idx] = make_float4(u, v, mu_cam.z, 0.f);

        // ============================================================
        // 5. JACOBIANO
        // ============================================================
        float invz2 = invz * invz;

        float J[3][3] = {
            {intrinsics.f.x * invz, 0.f, -intrinsics.f.x * mu_cam.x * invz2},
            {0.f, intrinsics.f.y * invz, -intrinsics.f.y * mu_cam.y * invz2},
            {0.f, 0.f, 1.f}
        };

        float W[3][3];
        quaternionToMatrix(quaternionInverse(camera_pose.orientation), W);

        float R[3][3];
        quaternionToMatrix(q_gauss, R);

        float S[3][3] = {
            {scale.x, 0.f, 0.f},
            {0.f, scale.y, 0.f},
            {0.f, 0.f, scale.z}
        };

        // A = J * W * R * S
        float A[3][3];
        mult3x3(J, W, A);
        mult3x3(A, R, A);
        mult3x3(A, S, A);

        // ============================================================
        // 6. COVARIANZA 2D
        // ============================================================
        // Sigma_2d = [a b] con b = c por simetria
        //            [c d]
        float a = dot3(make_float3(A[0][0], A[0][1], A[0][2]),
                    make_float3(A[0][0], A[0][1], A[0][2]));
        float b = dot3(make_float3(A[0][0], A[0][1], A[0][2]),
                    make_float3(A[1][0], A[1][1], A[1][2]));
        float d = dot3(make_float3(A[1][0], A[1][1], A[1][2]),
                    make_float3(A[1][0], A[1][1], A[1][2]));

        covariances_2d[idx] = make_float4(a, b, d, 0.f);

        // ============================================================
        // 7. COVARIANZA INVERSA 2D
        // ============================================================
        float det = a * d - b * b;
        float inv_det = 1.f / det;

        float inv00 = d * inv_det;
        float inv01 = -b * inv_det;
        float inv11 = a * inv_det;

        inv_covariances_2d[idx] = make_float4(inv00, inv01, inv11, 0.f);

        // ============================================================
        // 8. TILE BOUNDING BOX
        // ============================================================
        float sigma_u = sqrtf(a);
        float sigma_v = sqrtf(d);

        float radius_x = 3.f * sigma_u;
        float radius_y = 3.f * sigma_v;

        int tile_min_x = (int)((u - radius_x) / tile_size.x);
        int tile_max_x = (int)((u + radius_x) / tile_size.x);
        int tile_min_y = (int)((v - radius_y) / tile_size.y);
        int tile_max_y = (int)((v + radius_y) / tile_size.y);

        if (tile_max_x < 0 || tile_max_y < 0 ||
            tile_min_x >= (int)num_tiles.x ||
            tile_min_y >= (int)num_tiles.y) return;

        tile_min_x = max(0, tile_min_x);
        tile_min_y = max(0, tile_min_y);
        tile_max_x = min((int)num_tiles.x - 1, tile_max_x);
        tile_max_y = min((int)num_tiles.y - 1, tile_max_y);

        // ============================================================
        // 9. P_HAT y NORMAL
        // ============================================================
        // Calculamos sigma = 
        float Sigma[3][3];
        for (int i = 0; i < 3; ++i)
            for (int j = i; j < 3; ++j)
            {
                Sigma[i][j] = A[i][0]*A[j][0] + A[i][1]*A[j][1] + A[i][2]*A[j][2];
                if (i != j) Sigma[j][i] = Sigma[i][j];
            }

        // invert3x3(Sigma, invSigma);
        Eigen::Matrix3f Sigma_eig;
        Sigma_eig << Sigma[0][0], Sigma[0][1], Sigma[0][2],
                     Sigma[1][0], Sigma[1][1], Sigma[1][2],
                     Sigma[2][0], Sigma[2][1], Sigma[2][2];
        Eigen::Matrix3f invSigma = Sigma_eig.inverse();

        // ---- p_hat ----
        float mu_norm = sqrtf(mu_cam.x*mu_cam.x + mu_cam.y*mu_cam.y + mu_cam.z*mu_cam.z);
        float p_hat_scale = mu_cam.z / (mu_norm * invSigma(2, 2));

        p_hats[idx] = make_float2(
            invSigma(0, 2) * p_hat_scale,
            invSigma(1, 2) * p_hat_scale
        );

        // ---- normal ----
        float nx = invSigma(2, 0) / invSigma(2, 2);
        float ny = invSigma(2, 1) / invSigma(2, 2);

        float3 normal = make_float3(
            -(J[0][0]*nx + J[1][0]*ny + J[2][0]),
            -(J[0][1]*nx + J[1][1]*ny + J[2][1]),
            -(J[0][2]*nx + J[1][2]*ny + J[2][2])
        );

        float norm = sqrtf(normal.x*normal.x + normal.y*normal.y + normal.z*normal.z);
        normals[idx] = make_float4(normal.x / norm, normal.y / norm, normal.z / norm, 0.f);

        // ============================================================
        // 10. GENERACION DE INSTANCIAS POR TILES
        // ============================================================
        for (int ty = tile_min_y; ty <= tile_max_y; ++ty)
        {
            for (int tx = tile_min_x; tx <= tile_max_x; ++tx)
            {
                uint32_t instance = atomicAggInc(instance_counter);

                gaussian_indices[instance] = idx;

                uint32_t tile_id = ty * num_tiles.x + tx;

                hashes[instance] =
                    ((uint64_t)tile_id << 32) |
                    (uint64_t)__float_as_uint(mu_cam.z);
            }
        }
    }

    __global__ void projectAndHashGaussians_kernel_vigs(
        float4 *positions_2d,
        float4 *covariances_2d,
        float4 *inv_covariances_2d,
        float2 *p_hats,
        float4 *normals,
        uint64_t *hashes,
        uint32_t *gaussian_indices,
        uint32_t *instance_counter,
        const float4 *positions_world,
        const float4 *scales,
        const float4 *orientations,
        Pose camera_pose,
        IntrinsicParameters intrinsics,
        float min_depth,
        uint2 tile_size,
        uint2 num_tiles,
        uint32_t n_gaussians,
        uint32_t width,
        uint32_t height)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= n_gaussians) return;

        positions_2d[idx]        = make_float4(-1.f, -1.f, -1.f, 0.f);
        covariances_2d[idx]      = make_float4(0.f, 0.f, 0.f, 0.f);
        inv_covariances_2d[idx]  = make_float4(0.f, 0.f, 0.f, 0.f);
        p_hats[idx]              = make_float2(0.f, 0.f);
        normals[idx]             = make_float4(0.f, 0.f, 1.f, 0.f);

        float3 p_world = make_float3(
            positions_world[idx].x,
            positions_world[idx].y,
            positions_world[idx].z
        );
        float4 scale4 = scales[idx];
        float3 scale = make_float3(scale4.x, scale4.y, scale4.z);
        float4 q_gauss = orientations[idx];

        float3 mu_cam = rotateByQuaternionInverse(
            camera_pose.orientation,
            p_world - camera_pose.position
        );

        if (mu_cam.z < min_depth) return;

        float invz = 1.f / mu_cam.z;
        float u = intrinsics.f.x * mu_cam.x * invz + intrinsics.c.x;
        float v = intrinsics.f.y * mu_cam.y * invz + intrinsics.c.y;

        if (u < 0.f || u > (float)(width - 1) ||
            v < 0.f || v > (float)(height - 1)) return;

        positions_2d[idx] = make_float4(u, v, mu_cam.z, 0.f);

        float invz2 = invz * invz;

        float J[3][3] = {
            {intrinsics.f.x * invz, 0.f, -intrinsics.f.x * mu_cam.x * invz2},
            {0.f, intrinsics.f.y * invz, -intrinsics.f.y * mu_cam.y * invz2},
            {0.f, 0.f, 1.f}
        };

        float W[3][3];
        quaternionToMatrix(quaternionInverse(camera_pose.orientation), W);

        float R[3][3];
        quaternionToMatrix(q_gauss, R);

        float S[3][3] = {
            {scale.x, 0.f, 0.f},
            {0.f, scale.y, 0.f},
            {0.f, 0.f, scale.z}
        };

        float A[3][3];
        mult3x3(J, W, A);
        mult3x3(A, R, A);
        mult3x3(A, S, A);

        float a = dot3(make_float3(A[0][0], A[0][1], A[0][2]),
                       make_float3(A[0][0], A[0][1], A[0][2]));
        float b = dot3(make_float3(A[0][0], A[0][1], A[0][2]),
                       make_float3(A[1][0], A[1][1], A[1][2]));
        float d = dot3(make_float3(A[1][0], A[1][1], A[1][2]),
                       make_float3(A[1][0], A[1][1], A[1][2]));

        covariances_2d[idx] = make_float4(a, b, d, 0.f);

        float det = a * d - b * b;
        float inv_det = 1.f / det;

        float inv00 = d * inv_det;
        float inv01 = -b * inv_det;
        float inv11 = a * inv_det;

        inv_covariances_2d[idx] = make_float4(inv00, inv01, inv11, 0.f);

        float sigma_u = sqrtf(a);
        float sigma_v = sqrtf(d);

        float radius_x = 3.f * sigma_u;
        float radius_y = 3.f * sigma_v;

        int tile_min_x = (int)((u - radius_x) / tile_size.x);
        int tile_max_x = (int)((u + radius_x) / tile_size.x);
        int tile_min_y = (int)((v - radius_y) / tile_size.y);
        int tile_max_y = (int)((v + radius_y) / tile_size.y);

        if (tile_max_x < 0 || tile_max_y < 0 ||
            tile_min_x >= (int)num_tiles.x ||
            tile_min_y >= (int)num_tiles.y) return;

        tile_min_x = max(0, tile_min_x);
        tile_min_y = max(0, tile_min_y);
        tile_max_x = min((int)num_tiles.x - 1, tile_max_x);
        tile_max_y = min((int)num_tiles.y - 1, tile_max_y);

        float Sigma[3][3];
        for (int i = 0; i < 3; ++i)
            for (int j = i; j < 3; ++j)
            {
                Sigma[i][j] = A[i][0] * A[j][0] + A[i][1] * A[j][1] + A[i][2] * A[j][2];
                if (i != j) Sigma[j][i] = Sigma[i][j];
            }

        // invert3x3(Sigma, invSigma);
        Eigen::Matrix3f Sigma_eig;
        Sigma_eig << Sigma[0][0], Sigma[0][1], Sigma[0][2],
                     Sigma[1][0], Sigma[1][1], Sigma[1][2],
                     Sigma[2][0], Sigma[2][1], Sigma[2][2];
        Eigen::Matrix3f invSigma = Sigma_eig.inverse();

        float mu_norm = sqrtf(mu_cam.x * mu_cam.x + mu_cam.y * mu_cam.y + mu_cam.z * mu_cam.z);
        float p_hat_scale = mu_cam.z / (mu_norm * invSigma(2, 2));

        p_hats[idx] = make_float2(
            invSigma(0, 2) * p_hat_scale,
            invSigma(1, 2) * p_hat_scale
        );

        float nx = invSigma(2, 0) / invSigma(2, 2);
        float ny = invSigma(2, 1) / invSigma(2, 2);

        float3 normal = make_float3(
            -(J[0][0] * nx + J[1][0] * ny + J[2][0]),
            -(J[0][1] * nx + J[1][1] * ny + J[2][1]),
            -(J[0][2] * nx + J[1][2] * ny + J[2][2])
        );

        float norm = sqrtf(normal.x * normal.x + normal.y * normal.y + normal.z * normal.z);
        normals[idx] = make_float4(normal.x / norm, normal.y / norm, normal.z / norm, 0.f);

        for (int ty = tile_min_y; ty <= tile_max_y; ++ty)
        {
            for (int tx = tile_min_x; tx <= tile_max_x; ++tx)
            {
                uint32_t instance = atomicAggInc(instance_counter);

                gaussian_indices[instance] = idx;

                uint32_t tile_id = ty * num_tiles.x + tx;

                hashes[instance] =
                    ((uint64_t)tile_id << 32) |
                    (uint64_t)__float_as_uint(mu_cam.z);
            }
        }
    }

    __global__ void computeIndicesRanges_kernel(
        uint2 *ranges,
        const uint64_t *hashes,
        uint32_t n_instances)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= n_instances) return;

        // Extraemos tile_id del hash (bits altos)
        // Se asume que los hashes están ORDENADOS por thrust::sort_by_key
        // n_instances = total_hashes (puede ser mayor a #gaussianas porque a priori cada una
        // puede cubrir varios tiles segun su radio)
        uint32_t tile_id = (uint32_t)(hashes[idx] >> 32);

        if (idx == 0)
        {
            ranges[tile_id].x = 0;
        }
        else
        {
            // Comprobar si el tile_id cambió respecto al anterior
            uint32_t prev_tile_id = (uint32_t)(hashes[idx - 1] >> 32);

            if (prev_tile_id != tile_id)
            {
                // Tile anterior termina aquí
                ranges[prev_tile_id].y = idx;
                // Nuevo tile empieza aquí
                ranges[tile_id].x = idx;
            }
        }

        // Último elemento: cierra el último tile
        if (idx == n_instances - 1)
        {
            ranges[tile_id].y = idx + 1;
        }
    }

// Auxiliar para forwardpass
struct __align__(16) RasterGauss2D
{
    float4 pos;        // (u, v, z, pad)
    float4 inv_cov;    // (xx, xy, yy, pad)
    float4 color;      // (r, g, b, pad)
    float2 pHat;
    float  alpha;
    float  pad;        // padding para alinear a 16 bytes
};

__global__ void forwardPassTileKernel(
    uchar3 *output_color,
    float *output_depth,
    size_t output_color_step,
    size_t output_depth_step,
    const uint32_t *tile_gaussian_indices,
    const uint2 *tile_ranges,
    const float4 *positions_2d,
    const float4 *inv_covariances_2d,
    const float4 *colors,
    const float *alphas,
    const float2 *pHats,
    float3 bg_color,
    int width,
    int height,
    int num_tiles_x,
    int num_tiles_y,
    uint32_t n_instances_max,
    uint32_t n_gaussians_max)
{
    // ============================================================
    // Shared memory
    // ============================================================
    __shared__ RasterGauss2D s_gauss[TILE_SIZE * TILE_SIZE];

    // ============================================================
    // Thread / pixel info
    // ============================================================
    const int px_i = blockIdx.x * TILE_SIZE + threadIdx.x;
    const int py_i = blockIdx.y * TILE_SIZE + threadIdx.y;
    const bool inside = (px_i < width && py_i < height);

    const float px = (float)px_i;
    const float py = (float)py_i;

    const int tid = threadIdx.y * TILE_SIZE + threadIdx.x;
    const int block_size = TILE_SIZE * TILE_SIZE;

    // ============================================================
    // Tile range
    // ============================================================
    const int tile_id = blockIdx.y * num_tiles_x + blockIdx.x;

    int range_start = min((int)tile_ranges[tile_id].x, (int)n_instances_max);
    int range_end   = min((int)tile_ranges[tile_id].y, (int)n_instances_max);
    if (range_end < range_start) range_end = range_start;

    const int n_tile = range_end - range_start;

    // ============================================================
    // Accumulators
    // ============================================================
    float3 color_acc = make_float3(0.f, 0.f, 0.f);
    float alpha_acc = 0.f;
    float depth = 1e10f;

    bool done = !inside;
    bool has_depth = false;

    // ============================================================
    // Main loop (batching)
    // ============================================================
    for (int base = 0; base < n_tile; base += block_size)
    {
        const int local_idx = base + tid;

        // --------------------------------------------------------
        // Load batch → shared memory
        // --------------------------------------------------------
        if (local_idx < n_tile)
        {
            const int g = tile_gaussian_indices[range_start + local_idx];

            RasterGauss2D g_sh;

            if (g < n_gaussians_max)
            {
                g_sh.pos     = positions_2d[g];
                g_sh.inv_cov = inv_covariances_2d[g];
                g_sh.color   = colors[g];
                g_sh.alpha   = alphas[g];
                g_sh.pHat    = pHats[g];
            }
            else
            {
                g_sh.pos     = make_float4(0,0,0,0);
                g_sh.inv_cov = make_float4(1,0,1,0);
                g_sh.color   = make_float4(0,0,0,0);
                g_sh.alpha   = 0.f;
                g_sh.pHat    = make_float2(0,0);
            }

            s_gauss[tid] = g_sh;
        }

        __syncthreads();

        // --------------------------------------------------------
        // Process batch
        // --------------------------------------------------------
        const int batch_count = min(block_size, n_tile - base);

        for (int i = 0; i < batch_count && !done; i++)
        {
            const RasterGauss2D& g = s_gauss[i];

            const float dx = px - g.pos.x;
            const float dy = py - g.pos.y;

            float3 inv_cov = make_float3(g.inv_cov.x, g.inv_cov.y, g.inv_cov.z);
            const float alpha_i = evalGaussianWeight(dx, dy, inv_cov, g.alpha);
            if (alpha_i == 0.f) continue;

            const float T_before = 1.f - alpha_acc;
            const float T_after  = T_before * (1.f - alpha_i);

            if (T_after < 1e-4f)
            {
                done = true;
                continue;
            }

            // Color
            color_acc.x += T_before * alpha_i * g.color.x;
            color_acc.y += T_before * alpha_i * g.color.y;
            color_acc.z += T_before * alpha_i * g.color.z;

            alpha_acc += T_before * alpha_i;

            // Depth (median)
            if (!has_depth && T_before > 0.5f && T_after <= 0.5f)
            {
                depth = g.pos.z + g.pHat.x * dx + g.pHat.y * dy;
                has_depth = true;
            }
        }

        __syncthreads();
    }

    // ============================================================
    // Output
    // ============================================================
    if (inside)
    {
        color_acc += (1.f - alpha_acc) * bg_color;

        uchar3 *row_color = (uchar3*)((char*)output_color + py_i * output_color_step);
        float  *row_depth = (float*)((char*)output_depth + py_i * output_depth_step);

        row_color[px_i] = make_uchar3(
            (unsigned char)(fminf(fmaxf(color_acc.x, 0.f), 1.f) * 255.f),
            (unsigned char)(fminf(fmaxf(color_acc.y, 0.f), 1.f) * 255.f),
            (unsigned char)(fminf(fmaxf(color_acc.z, 0.f), 1.f) * 255.f)
        );

        row_depth[px_i] = (alpha_acc < 0.9f) ? 0.f : depth;
    }
}

    /*  __global__ void forwardPassKernel(
        float3 *output_color,           // Salida: imagen RGB renderizada
        float *output_depth,            // Salida: mapa de profundidad
        const float2 *positions_2d,     // Entrada: posiciones proyectadas [n_gaussians]
        const float3 *covariances_2d,   // Entrada: covarianzas 2D [n_gaussians]
        const float3 *colors,           // Entrada: color RGB de cada Gaussian [n_gaussians]
        const float *alphas,            // Entrada: opacidad de cada Gaussian [n_gaussians]
        int width,
        int height,
        int n_gaussians)
    {
        int idx_x = blockIdx.x * blockDim.x + threadIdx.x;
        int idx_y = blockIdx.y * blockDim.y + threadIdx.y;

        if (idx_x >= width || idx_y >= height) return;

        int pixel_idx = idx_y * width + idx_x;
        float px = (float)idx_x;
        float py = (float)idx_y;

        float3 pixel_color = make_float3(0.0f, 0.0f, 0.0f);
        float pixel_alpha = 0.0f;
        float pixel_depth = 1e10f;



        for (int g = 0; g < n_gaussians; g++)
        {
            float2 pos_2d = positions_2d[g];
            float3 cov_2d = covariances_2d[g];
            float3 color = colors[g];
            float alpha = alphas[g];

            float dx = px - pos_2d.x;
            float dy = py - pos_2d.y;

            float inv_cov_xx, inv_cov_yy, inv_cov_xy;
            invert2x2(cov_2d.x, cov_2d.y, cov_2d.z, inv_cov_xx, inv_cov_yy, inv_cov_xy);

            float gauss_val = evalGaussian2D(dx, dy, inv_cov_xx, inv_cov_yy, inv_cov_xy);

            float weighted_alpha = alpha * gauss_val;

            pixel_color.x += (1.0f - pixel_alpha) * weighted_alpha * color.x;
            pixel_color.y += (1.0f - pixel_alpha) * weighted_alpha * color.y;
            pixel_color.z += (1.0f - pixel_alpha) * weighted_alpha * color.z;

            pixel_alpha += (1.0f - pixel_alpha) * weighted_alpha;

            if (pixel_alpha >= 0.99f) break; // En muchos casos la contribucion se vuelve casi nula en cierto punto
        }

        output_color[pixel_idx] = pixel_color;
        output_depth[pixel_idx] = pixel_depth;
    } */


    // ============================================================================
    // 4. BACKWARD PASS KERNELS (OPTIMIZACION)
    // ============================================================================

    // Equivalente a optimizePoseGN3_kernel
    __global__ void getRgbdPoseJacobians(
        PoseOptimizationRgbdData *output_posedata,
        const uint2 *ranges,
        const uint32_t *indices,
        const float4 *positions_2d,
        const float4 *inv_covariances_2d,
        const float2 *p_hats,
        const float4 *colors,
        const float *alphas,

        cudaTextureObject_t tex_rgb,
        cudaTextureObject_t tex_depth,
        cudaTextureObject_t tex_dx,
        cudaTextureObject_t tex_dy,

        Pose camera_pose,
        IntrinsicParameters intrinsics,
        float3 bg_color,

        float alpha_thresh,
        float color_thresh,
        float depth_thresh,

        int width, int height,
        int num_tiles_x, int num_tiles_y)
    {
        // ============================================================
        // 1. INDEXACIÓN (1 thread = 1 pixel)
        // ============================================================
        int x = blockIdx.x * TILE_SIZE + threadIdx.x;
        int y = blockIdx.y * TILE_SIZE + threadIdx.y;
        int tile_idx = blockIdx.y * num_tiles_x + blockIdx.x;
        int tid = threadIdx.y * TILE_SIZE + threadIdx.x;

        // ============================================================
        // 2. LECTURA DE DEPTH OBSERVADO
        // ============================================================
        float img_depth = 0.0f;
        bool in_bounds = (x < width && y < height);

        if (in_bounds) {
            img_depth = getImageData<float>(tex_depth, x, y);
        }

        bool inside = in_bounds && img_depth > 0.1f;

        // ============================================================
        // 3. ACUMULADORES LOCALES (por thread)
        // ============================================================
        float local_JtJ[21] = {0.0f};
        float local_Jtr[6]  = {0.0f};

        // ============================================================
        // 4. SHARED MEMORY: cache de gaussianas del tile
        // ============================================================
        __shared__ RasterGauss2D s_gauss[TILE_SIZE * TILE_SIZE];

        uint2 range = ranges[tile_idx];
        int n_total = range.y - range.x;
        int block_size = TILE_SIZE * TILE_SIZE;

        // ============================================================
        // 5. FORWARD PASS (RENDERING)
        // ============================================================
        float3 color = make_float3(0.0f, 0.0f, 0.0f);
        float depth  = 0.0f;
        float final_T = 1.0f;
        uint32_t n_contrib = 0;

        if (n_total > 0)
        {
            float T = 1.0f;
            uint32_t contributor = 0;
            uint32_t last_contributor = 0;
            bool done = !inside;

            // Barrido por batches de gaussianas
            for (int base = 0; base < n_total; base += block_size)
            {
                // --- cargar a shared ---
                int local_idx = base + tid;
                if (local_idx < n_total) {
                    uint32_t gid = indices[range.x + local_idx];
                    RasterGauss2D g_sh;
                    g_sh.pos     = positions_2d[gid];
                    g_sh.inv_cov = inv_covariances_2d[gid];
                    g_sh.color   = colors[gid];
                    g_sh.alpha   = alphas[gid];
                    g_sh.pHat    = p_hats[gid];
                    g_sh.pad     = 0.f;
                    s_gauss[tid] = g_sh;
                }
                __syncthreads();

                int batch_count = min(block_size, n_total - base);

                for (int i = 0; i < batch_count; ++i)
                {
                    if (done) continue;

                    contributor++;

                    const RasterGauss2D& g = s_gauss[i];
                    float dx = g.pos.x - x;
                    float dy = g.pos.y - y;

                    float4 invSigma = g.inv_cov;
                    float v = mahalanobis2D(dx, dy, invSigma.x, invSigma.y, invSigma.z);

                    float alpha_i = fminf(0.99f,
                        g.alpha * evalGaussian2DFromMahalanobis(v));

                    if (alpha_i < 1.f/255.f || v < 0.0f) continue;

                    float test_T = T * (1.0f - alpha_i);

                    if (test_T < 1e-4f) {
                        done = true;
                        continue;
                    }

                    // acumulación de color
                    float3 color_g = make_float3(g.color.x, g.color.y, g.color.z);
                    color += color_g * alpha_i * T;

                    // profundidad tipo "median"
                    float d = g.pos.z + dx * g.pHat.x + dy * g.pHat.y;

                    if (T > 0.5f && test_T < 0.5f)
                        depth = d;

                    T = test_T;
                    last_contributor = contributor;
                }
                __syncthreads();
            }

            if (inside) {
                final_T = T;
                n_contrib = last_contributor;
                color += T * bg_color;
            }
        }

        inside = inside && (final_T < alpha_thresh);

        // ============================================================
        // 6. ERROR + GRADIENTES DE IMAGEN
        // ============================================================
        float3 color_error = make_float3(0.0f, 0.0f, 0.0f);
        float depth_error = 0.0f;
        float wc = 0.0f;

        __shared__ float R_shared[3][3];

        if (threadIdx.x == 0 && threadIdx.y == 0) {
            quaternionToMatrix(camera_pose.orientation, R_shared);
        }
        __syncthreads();

        if (inside)
        {
            uchar4 rgba = getImageData<uchar4>(tex_rgb, x, y);
            const float inv_255 = 1.0f / 255.0f;

            float3 observed_c = make_float3(rgba.x * inv_255, rgba.y * inv_255, rgba.z * inv_255);

            color_error = color - observed_c;
            depth_error = (img_depth > 0.1f) ? (depth - img_depth) : 0.0f;

            float3 ray = make_float3(
                (x - intrinsics.c.x) / intrinsics.f.x,
                (y - intrinsics.c.y) / intrinsics.f.y,
                1.0f
            );

            // Huber color
            float color_loss = sqrtf(color_error.x * color_error.x +
                                color_error.y * color_error.y +
                                color_error.z * color_error.z);

            wc = (color_loss < color_thresh) ? 1.0f : color_thresh / color_loss;

            // Gradientes imagen (Sobel)
            float4 gradX4 = getImageData<float4>(tex_dx, x, y);
            float4 gradY4 = getImageData<float4>(tex_dy, x, y);

            float3 gradX = make_float3(gradX4.x, gradX4.y, gradX4.z);
            float3 gradY = make_float3(gradY4.x, gradY4.y, gradY4.z);

            float Jt_pix[6];
            float inv_img_depth = 1.0f / img_depth;
            Jt_pix[0] = intrinsics.f.x * inv_img_depth;
            Jt_pix[1] = 0.0f;
            Jt_pix[2] = 0.0f;
            Jt_pix[3] = intrinsics.f.y * inv_img_depth;
            Jt_pix[4] = -intrinsics.f.x * ray.x * inv_img_depth;
            Jt_pix[5] = -intrinsics.f.y * ray.y * inv_img_depth;

            float Jt_cam_pix[9];
            Jt_cam_pix[0] = Jt_pix[0] * gradX.x + Jt_pix[1] * gradY.x;
            Jt_cam_pix[1] = Jt_pix[0] * gradX.y + Jt_pix[1] * gradY.y;
            Jt_cam_pix[2] = Jt_pix[0] * gradX.z + Jt_pix[1] * gradY.z;
            Jt_cam_pix[3] = Jt_pix[2] * gradX.x + Jt_pix[3] * gradY.x;
            Jt_cam_pix[4] = Jt_pix[2] * gradX.y + Jt_pix[3] * gradY.y;
            Jt_cam_pix[5] = Jt_pix[2] * gradX.z + Jt_pix[3] * gradY.z;
            Jt_cam_pix[6] = Jt_pix[4] * gradX.x + Jt_pix[5] * gradY.x;
            Jt_cam_pix[7] = Jt_pix[4] * gradX.y + Jt_pix[5] * gradY.y;
            Jt_cam_pix[8] = Jt_pix[4] * gradX.z + Jt_pix[5] * gradY.z;

            float JtJ_cam_pix[9];
            for (int r = 0; r < 3; ++r) {
                for (int c = 0; c < 3; ++c) {
                    float sum = 0.0f;
                    for (int k = 0; k < 3; ++k) {
                        sum += Jt_cam_pix[r * 3 + k] * Jt_cam_pix[c * 3 + k];
                    }
                    JtJ_cam_pix[r * 3 + c] = wc * sum;
                }
            }

            float Jtr_cam_pix[3];
            Jtr_cam_pix[0] = -wc * (Jt_cam_pix[0] * color_error.x + Jt_cam_pix[1] * color_error.y + Jt_cam_pix[2] * color_error.z);
            Jtr_cam_pix[1] = -wc * (Jt_cam_pix[3] * color_error.x + Jt_cam_pix[4] * color_error.y + Jt_cam_pix[5] * color_error.z);
            Jtr_cam_pix[2] = -wc * (Jt_cam_pix[6] * color_error.x + Jt_cam_pix[7] * color_error.y + Jt_cam_pix[8] * color_error.z);

            float ld = fabsf(depth_error);
            float wd = (ld < depth_thresh) ? 1.0f : depth_thresh / ld;
            wd /= img_depth;
            JtJ_cam_pix[0] += wd * ray.x * ray.x;
            JtJ_cam_pix[1] += wd * ray.x * ray.y;
            JtJ_cam_pix[2] += wd * ray.x * ray.z;
            JtJ_cam_pix[3] += wd * ray.y * ray.x;
            JtJ_cam_pix[4] += wd * ray.y * ray.y;
            JtJ_cam_pix[5] += wd * ray.y * ray.z;
            JtJ_cam_pix[6] += wd * ray.z * ray.x;
            JtJ_cam_pix[7] += wd * ray.z * ray.y;
            JtJ_cam_pix[8] += wd * ray.z * ray.z;

            Jtr_cam_pix[0] += wd * depth_error * ray.x;
            Jtr_cam_pix[1] += wd * depth_error * ray.y;
            Jtr_cam_pix[2] += wd * depth_error * ray.z;

            float ray_cross[9];
            skewSymmetric(ray, ray_cross);

            float Jpose_pix[18];
            Jpose_pix[0] = R_shared[0][0];  Jpose_pix[1] = R_shared[0][1];  Jpose_pix[2] = R_shared[0][2];
            Jpose_pix[3] = R_shared[1][0];  Jpose_pix[4] = R_shared[1][1];  Jpose_pix[5] = R_shared[1][2];
            Jpose_pix[6] = R_shared[2][0];  Jpose_pix[7] = R_shared[2][1];  Jpose_pix[8] = R_shared[2][2];

            float z_ray_cross_pix[9];
            for (int r = 0; r < 3; ++r) {
                for (int c = 0; c < 3; ++c) {
                    z_ray_cross_pix[r * 3 + c] = img_depth * ray_cross[r * 3 + c];
                }
            }
            Jpose_pix[9]  = z_ray_cross_pix[0];  Jpose_pix[10] = z_ray_cross_pix[1];  Jpose_pix[11] = z_ray_cross_pix[2];
            Jpose_pix[12] = z_ray_cross_pix[3];  Jpose_pix[13] = z_ray_cross_pix[4];  Jpose_pix[14] = z_ray_cross_pix[5];
            Jpose_pix[15] = z_ray_cross_pix[6];  Jpose_pix[16] = z_ray_cross_pix[7];  Jpose_pix[17] = z_ray_cross_pix[8];

            float temp_pix[18];
            for (int r = 0; r < 6; ++r) {
                for (int c = 0; c < 3; ++c) {
                    float sum = 0.0f;
                    for (int k = 0; k < 3; ++k) {
                        sum += Jpose_pix[r * 3 + k] * JtJ_cam_pix[k * 3 + c];
                    }
                    temp_pix[r * 3 + c] = sum;
                }
            }

            int idx = 0;
            for (int r = 0; r < 6; ++r) {
                for (int c = r; c < 6; ++c) {
                    float sum = 0.0f;
                    for (int k = 0; k < 3; ++k) {
                        sum += temp_pix[r * 3 + k] * Jpose_pix[c * 3 + k];
                    }
                    local_JtJ[idx] += sum;
                    idx++;
                }
            }

            for (int r = 0; r < 6; ++r) {
                float sum = 0.0f;
                for (int k = 0; k < 3; ++k) {
                    sum += Jpose_pix[r * 3 + k] * Jtr_cam_pix[k];
                }
                local_Jtr[r] += sum;
            }

            float prod_alpha = 1.0f;
            float T = 1.0f;
            float3 acc_c = make_float3(0.0f, 0.0f, 0.0f);

        for (int base = 0; base < n_total; base += block_size)
        {
            int local_idx = base + tid;
            if (local_idx < n_total) {
                uint32_t gid = indices[range.x + local_idx];
                RasterGauss2D g_sh;
                g_sh.pos     = positions_2d[gid];
                g_sh.inv_cov = inv_covariances_2d[gid];
                g_sh.color   = colors[gid];
                g_sh.alpha   = alphas[gid];
                g_sh.pHat    = p_hats[gid];
                g_sh.pad     = 0.f;
                s_gauss[tid] = g_sh;
            }
            __syncthreads();

            int batch_count = min(block_size, n_total - base);

            for (int i = 0; i < batch_count && (i + base) < (int)n_contrib; ++i)
            {
                const RasterGauss2D& g = s_gauss[i];

                float dx = g.pos.x - (float)x;
                float dy = g.pos.y - (float)y;

                float4 invSigma = g.inv_cov;
                float v = mahalanobis2D(dx, dy, invSigma.x, invSigma.y, invSigma.z);

                float G = evalGaussian2DFromMahalanobis(v);
                float alpha_i = fminf(0.99f, g.alpha * G);

                if (alpha_i < (1.f / 255.f) || v < 0.0f)
                    continue;

                float d = g.pos.z + dx * g.pHat.x + dy * g.pHat.y;

                float3 d_alpha = make_float3(
                    g.color.x * prod_alpha,
                    g.color.y * prod_alpha,
                    g.color.z * prod_alpha
                );

                acc_c.x += alpha_i * d_alpha.x;
                acc_c.y += alpha_i * d_alpha.y;
                acc_c.z += alpha_i * d_alpha.z;

                float inv_one_minus = 1.0f / fmaxf(1e-6f, (1.0f - alpha_i));
                d_alpha.x -= (color.x - acc_c.x) * inv_one_minus;
                d_alpha.y -= (color.y - acc_c.y) * inv_one_minus;
                d_alpha.z -= (color.z - acc_c.z) * inv_one_minus;

                float2 dl_mean2d = make_float2(
                    invSigma.x * dx + invSigma.y * dy,
                    invSigma.y * dx + invSigma.z * dy
                );

                float3 ray_g = make_float3(
                    (g.pos.x - intrinsics.c.x) / intrinsics.f.x,
                    (g.pos.y - intrinsics.c.y) / intrinsics.f.y,
                    1.0f
                );

                float ray_cross_g[9];
                skewSymmetric(ray_g, ray_cross_g);

                float inv_d = 1.0f / fmaxf(1e-6f, d);

                float Jt[6];
                Jt[0] = intrinsics.f.x * inv_d;
                Jt[1] = 0.0f;
                Jt[2] = 0.0f;
                Jt[3] = intrinsics.f.y * inv_d;
                Jt[4] = -intrinsics.f.x * ray_g.x * inv_d;
                Jt[5] = -intrinsics.f.y * ray_g.y * inv_d;

                float jt_dl[3];
                jt_dl[0] = Jt[0] * dl_mean2d.x + Jt[1] * dl_mean2d.y;
                jt_dl[1] = Jt[2] * dl_mean2d.x + Jt[3] * dl_mean2d.y;
                jt_dl[2] = Jt[4] * dl_mean2d.x + Jt[5] * dl_mean2d.y;

                float scale = alpha_i;

                float Jt_cam[9];
                Jt_cam[0] = jt_dl[0] * scale * d_alpha.x;
                Jt_cam[1] = jt_dl[0] * scale * d_alpha.y;
                Jt_cam[2] = jt_dl[0] * scale * d_alpha.z;
                Jt_cam[3] = jt_dl[1] * scale * d_alpha.x;
                Jt_cam[4] = jt_dl[1] * scale * d_alpha.y;
                Jt_cam[5] = jt_dl[1] * scale * d_alpha.z;
                Jt_cam[6] = jt_dl[2] * scale * d_alpha.x;
                Jt_cam[7] = jt_dl[2] * scale * d_alpha.y;
                Jt_cam[8] = jt_dl[2] * scale * d_alpha.z;

                float JtJ_cam[9];
                for (int r = 0; r < 3; ++r) {
                    for (int c = 0; c < 3; ++c) {
                        float sum = 0.0f;
                        for (int k = 0; k < 3; ++k) {
                            sum += Jt_cam[r * 3 + k] * Jt_cam[c * 3 + k];
                        }
                        JtJ_cam[r * 3 + c] = wc * sum;
                    }
                }

                float Jtr_cam[3];
                Jtr_cam[0] = -wc * (Jt_cam[0] * color_error.x + Jt_cam[1] * color_error.y + Jt_cam[2] * color_error.z);
                Jtr_cam[1] = -wc * (Jt_cam[3] * color_error.x + Jt_cam[4] * color_error.y + Jt_cam[5] * color_error.z);
                Jtr_cam[2] = -wc * (Jt_cam[6] * color_error.x + Jt_cam[7] * color_error.y + Jt_cam[8] * color_error.z);

                prod_alpha *= (1.0f - alpha_i);

                if (T > 0.5f && prod_alpha <= 0.5f && img_depth > 0.5f)
                {
                    T = prod_alpha;

                    float ld = fabsf(depth_error);
                    float wd = (ld < depth_thresh) ? 1.0f : depth_thresh / ld;
                    wd /= img_depth;

                    JtJ_cam[0] += wd * ray_g.x * ray_g.x;
                    JtJ_cam[1] += wd * ray_g.x * ray_g.y;
                    JtJ_cam[2] += wd * ray_g.x * ray_g.z;
                    JtJ_cam[3] += wd * ray_g.y * ray_g.x;
                    JtJ_cam[4] += wd * ray_g.y * ray_g.y;
                    JtJ_cam[5] += wd * ray_g.y * ray_g.z;
                    JtJ_cam[6] += wd * ray_g.z * ray_g.x;
                    JtJ_cam[7] += wd * ray_g.z * ray_g.y;
                    JtJ_cam[8] += wd * ray_g.z * ray_g.z;

                    Jtr_cam[0] += wd * depth_error * ray_g.x;
                    Jtr_cam[1] += wd * depth_error * ray_g.y;
                    Jtr_cam[2] += wd * depth_error * ray_g.z;
                }

                float Jpose[18];
                Jpose[0] = R_shared[0][0];  Jpose[1] = R_shared[0][1];  Jpose[2] = R_shared[0][2];
                Jpose[3] = R_shared[1][0];  Jpose[4] = R_shared[1][1];  Jpose[5] = R_shared[1][2];
                Jpose[6] = R_shared[2][0];  Jpose[7] = R_shared[2][1];  Jpose[8] = R_shared[2][2];

                float d_ray_cross[9];
                for (int k = 0; k < 9; ++k)
                    d_ray_cross[k] = d * ray_cross_g[k];

                for (int k = 0; k < 9; ++k)
                    Jpose[9 + k] = d_ray_cross[k];

                float temp[18];
                for (int r = 0; r < 6; ++r) {
                    for (int c = 0; c < 3; ++c) {
                        float sum = 0.0f;
                        for (int k = 0; k < 3; ++k) {
                            sum += Jpose[r * 3 + k] * JtJ_cam[k * 3 + c];
                        }
                        temp[r * 3 + c] = sum;
                    }
                }

                int idx = 0;
                for (int r = 0; r < 6; ++r) {
                    for (int c = r; c < 6; ++c) {
                        float sum = 0.0f;
                        for (int k = 0; k < 3; ++k) {
                            sum += temp[r * 3 + k] * Jpose[c * 3 + k];
                        }
                        local_JtJ[idx++] += sum;
                    }
                }

                for (int r = 0; r < 6; ++r) {
                    float sum = 0.0f;
                    for (int k = 0; k < 3; ++k) {
                        sum += Jpose[r * 3 + k] * Jtr_cam[k];
                    }
                    local_Jtr[r] += sum;
                }

                if (prod_alpha < 0.001f)
                    break;
            }
            __syncthreads();
        }

            }
        // ========================================================================
        // 7: Block Reduction (tree reduction)
        // ========================================================================

        // Shared memory para reduction
        __shared__ float shared_JtJ[TILE_SIZE * TILE_SIZE][21];
        __shared__ float shared_Jtr[TILE_SIZE * TILE_SIZE][6];

        // Copiar locales a shared
        for (int i = 0; i < 21; i++) shared_JtJ[tid][i] = local_JtJ[i];
        for (int i = 0; i < 6; i++) shared_Jtr[tid][i] = local_Jtr[i];
        __syncthreads();

        // Tree reduction
        for (int s = (TILE_SIZE * TILE_SIZE) / 2; s > 0; s >>= 1) {
            if (tid < s) {
                for (int i = 0; i < 21; i++) {
                    shared_JtJ[tid][i] += shared_JtJ[tid + s][i];
                }
                for (int i = 0; i < 6; i++) {
                    shared_Jtr[tid][i] += shared_Jtr[tid + s][i];
                }
            }
            __syncthreads();
        }

        // ========================================================================
        // 8: Atomic Write a Memoria Global
        // ========================================================================

        if (tid == 0) {
            PoseOptimizationRgbdData &out = output_posedata[tile_idx];
            for (int i = 0; i < 21; i++) {
                atomicAdd(&out.JtJ[i], shared_JtJ[0][i]);
            }
            for (int i = 0; i < 6; i++) {
                atomicAdd(&out.Jtr[i], shared_Jtr[0][i]);
            }
        }
    }

    // Equivalente a optimizePoseGN3_fast_kernel (VIGS-Fusion) adaptado a f_vigs_slam.
    __global__ void getRgbdPoseJacobians_fast(
        PoseOptimizationRgbdData *output_posedata,
        const uint2 *ranges,
        const uint32_t *indices,
        const float4 *positions_2d,
        const float4 *inv_covariances_2d,
        const float2 *p_hats,
        const float4 *colors,
        const float *alphas,
        cudaTextureObject_t tex_rgb,
        cudaTextureObject_t tex_depth,
        cudaTextureObject_t tex_dx,
        cudaTextureObject_t tex_dy,
        Pose camera_pose,
        IntrinsicParameters intrinsics,
        float3 bg_color,
        float alpha_thresh,
        float color_thresh,
        float depth_thresh,
        int width,
        int height,
        int num_tiles_x,
        int num_tiles_y)
    {
        int x = blockIdx.x * blockDim.x + threadIdx.x;
        int y = blockIdx.y * blockDim.y + threadIdx.y;
        int tile_idx = blockIdx.y * num_tiles_x + blockIdx.x;
        int tid = threadIdx.y * blockDim.x + threadIdx.x;

        bool in_bounds = x < width && y < height;
        float img_depth = in_bounds ? getImageData<float>(tex_depth, x, y) : 0.0f;
        bool inside = in_bounds && img_depth > 0.1f;

        float local_JtJ[21] = {0.0f};
        float local_Jtr[6] = {0.0f};

        __shared__ RasterGauss2D s_gauss[BLOCK_SIZE];
        __shared__ uint32_t gids_sh[BLOCK_SIZE];

        uint2 range = ranges[tile_idx];
        int n = range.y - range.x;

        float3 color = make_float3(0.f, 0.f, 0.f);
        float depth = 0.f;
        float final_T = 1.f;
        uint32_t n_contrib = 0;
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
                s_gauss[tid].pos = positions_2d[gid];
                s_gauss[tid].inv_cov = inv_covariances_2d[gid];
                s_gauss[tid].color = colors[gid];
                s_gauss[tid].alpha = alphas[gid];
                s_gauss[tid].pHat = p_hats[gid];
            }
            __syncthreads();

            for (int i = 0; !done && i + k * BLOCK_SIZE < n && i < BLOCK_SIZE; i++)
            {
                contributor++;

                const float dx = s_gauss[i].pos.x - x;
                const float dy = s_gauss[i].pos.y - y;
                const float v = s_gauss[i].inv_cov.x * dx * dx +
                                2.f * s_gauss[i].inv_cov.y * dx * dy +
                                s_gauss[i].inv_cov.z * dy * dy;
                const float alpha_i = min(0.99f, s_gauss[i].alpha * expf(-0.5f * v));

                if (alpha_i < 1.f / 255.f || v <= 0.f)
                    continue;

                float test_T = T * (1.f - alpha_i);
                if (test_T < 0.0001f)
                {
                    done = true;
                    continue;
                }

                color.x += s_gauss[i].color.x * alpha_i * T;
                color.y += s_gauss[i].color.y * alpha_i * T;
                color.z += s_gauss[i].color.z * alpha_i * T;

                float d = s_gauss[i].pos.z + dx * s_gauss[i].pHat.x + dy * s_gauss[i].pHat.y;

                if (T > 0.5f && test_T < 0.5f)
                {
                    depth = d;
                }

                T = test_T;
                last_contributor = contributor;
            }
            __syncthreads();
        }

        if (inside)
        {
            final_T = T;
            n_contrib = last_contributor;
            color += T * bg_color;
        }

        inside &= final_T < alpha_thresh;

        (void)n_contrib;

        __shared__ float R_shared[3][3];
        if (threadIdx.x == 0 && threadIdx.y == 0)
        {
            quaternionToMatrix(camera_pose.orientation, R_shared);
        }
        __syncthreads();

        if (inside)
        {
            uchar4 rgba = getImageData<uchar4>(tex_rgb, x, y);
            float3 color_error = make_float3(
                color.x - rgba.x / 255.f,
                color.y - rgba.y / 255.f,
                color.z - rgba.z / 255.f);

            float depth_error = img_depth > 0.1f ? depth - img_depth : 0.f;

            float3 ray = make_float3(
                (x - intrinsics.c.x) / intrinsics.f.x,
                (y - intrinsics.c.y) / intrinsics.f.y,
                1.f);

            float ray_cross[9];
            skewSymmetric(ray, ray_cross);

            float lc = sqrtf(color_error.x * color_error.x +
                             color_error.y * color_error.y +
                             color_error.z * color_error.z);
            float wc = lc < color_thresh ? 1.f : color_thresh / lc;
            float4 gradX = getImageData<float4>(tex_dx, x, y);
            float4 gradY = getImageData<float4>(tex_dy, x, y);


            float Jt[6];
            Jt[0] = intrinsics.f.x / img_depth;
            Jt[1] = 0.f;
            Jt[2] = 0.f;
            Jt[3] = intrinsics.f.y / img_depth;
            Jt[4] = -intrinsics.f.x * ray.x / img_depth;
            Jt[5] = -intrinsics.f.y * ray.y / img_depth;


            float Jt_cam[9];
            Jt_cam[0] = Jt[0] * gradX.x + Jt[1] * gradY.x;
            Jt_cam[1] = Jt[0] * gradX.y + Jt[1] * gradY.y;
            Jt_cam[2] = Jt[0] * gradX.z + Jt[1] * gradY.z;
            Jt_cam[3] = Jt[2] * gradX.x + Jt[3] * gradY.x;
            Jt_cam[4] = Jt[2] * gradX.y + Jt[3] * gradY.y;
            Jt_cam[5] = Jt[2] * gradX.z + Jt[3] * gradY.z;
            Jt_cam[6] = Jt[4] * gradX.x + Jt[5] * gradY.x;
            Jt_cam[7] = Jt[4] * gradX.y + Jt[5] * gradY.y;
            Jt_cam[8] = Jt[4] * gradX.z + Jt[5] * gradY.z;

            float JtJ_cam[9];
            for (int r = 0; r < 3; ++r)
            {
                for (int c = 0; c < 3; ++c)
                {
                    float sum = 0.f;
                    for (int k = 0; k < 3; ++k)
                    {
                        sum += Jt_cam[r * 3 + k] * Jt_cam[c * 3 + k];
                    }
                    JtJ_cam[r * 3 + c] = wc * sum;
                }
            }

            float Jtr_cam[3];
            Jtr_cam[0] = -wc * (Jt_cam[0] * color_error.x + Jt_cam[1] * color_error.y + Jt_cam[2] * color_error.z);
            Jtr_cam[1] = -wc * (Jt_cam[3] * color_error.x + Jt_cam[4] * color_error.y + Jt_cam[5] * color_error.z);
            Jtr_cam[2] = -wc * (Jt_cam[6] * color_error.x + Jt_cam[7] * color_error.y + Jt_cam[8] * color_error.z);

            float ld = fabsf(depth_error);
            float wd = ld < depth_thresh ? 1.f : depth_thresh / ld;
            wd /= img_depth;


            JtJ_cam[0] += wd * ray.x * ray.x;
            JtJ_cam[1] += wd * ray.x * ray.y;
            JtJ_cam[2] += wd * ray.x * ray.z;
            JtJ_cam[3] += wd * ray.y * ray.x;
            JtJ_cam[4] += wd * ray.y * ray.y;
            JtJ_cam[5] += wd * ray.y * ray.z;
            JtJ_cam[6] += wd * ray.z * ray.x;
            JtJ_cam[7] += wd * ray.z * ray.y;
            JtJ_cam[8] += wd * ray.z * ray.z;

            Jtr_cam[0] += wd * depth_error * ray.x;
            Jtr_cam[1] += wd * depth_error * ray.y;
            Jtr_cam[2] += wd * depth_error * ray.z;



            float Jpose[18];
            Jpose[0] = R_shared[0][0];
            Jpose[1] = R_shared[0][1];
            Jpose[2] = R_shared[0][2];
            Jpose[3] = R_shared[1][0];
            Jpose[4] = R_shared[1][1];
            Jpose[5] = R_shared[1][2];
            Jpose[6] = R_shared[2][0];
            Jpose[7] = R_shared[2][1];
            Jpose[8] = R_shared[2][2];
            for (int i = 0; i < 9; ++i)
            {
                Jpose[9 + i] = img_depth * ray_cross[i];
            }

            float temp[18];
            for (int r = 0; r < 6; ++r)
            {
                for (int c = 0; c < 3; ++c)
                {
                    float sum = 0.f;
                    for (int k = 0; k < 3; ++k)
                    {
                        sum += Jpose[r * 3 + k] * JtJ_cam[k * 3 + c];
                    }
                    temp[r * 3 + c] = sum;
                }
            }

            int idx = 0;
            for (int r = 0; r < 6; ++r)
            {
                for (int c = r; c < 6; ++c)
                {
                    float sum = 0.f;
                    for (int k = 0; k < 3; ++k)
                    {
                        sum += temp[r * 3 + k] * Jpose[c * 3 + k];
                    }
                    local_JtJ[idx++] += sum;
                }
            }

            for (int r = 0; r < 6; ++r)
            {
                float sum = 0.f;
                for (int k = 0; k < 3; ++k)
                {
                    sum += Jpose[r * 3 + k] * Jtr_cam[k];
                }
                local_Jtr[r] += sum;
            }


        }

        __shared__ float shared_JtJ[BLOCK_SIZE][21];
        __shared__ float shared_Jtr[BLOCK_SIZE][6];

        for (int i = 0; i < 21; i++)
            shared_JtJ[tid][i] = local_JtJ[i];
        for (int i = 0; i < 6; i++)
            shared_Jtr[tid][i] = local_Jtr[i];
        __syncthreads();

        for (int s = BLOCK_SIZE / 2; s > 0; s >>= 1)
        {
            if (tid < s)
            {
                for (int i = 0; i < 21; i++)
                    shared_JtJ[tid][i] += shared_JtJ[tid + s][i];
                for (int i = 0; i < 6; i++)
                    shared_Jtr[tid][i] += shared_Jtr[tid + s][i];
            }
            __syncthreads();
        }

        if (tid == 0)
        {
            PoseOptimizationRgbdData &out = output_posedata[0];
            for (int i = 0; i < 21; i++)
                atomicAdd(&out.JtJ[i], shared_JtJ[0][i]);
            for (int i = 0; i < 6; i++)
                atomicAdd(&out.Jtr[i], shared_Jtr[0][i]);
        }
    }

    /*
    __global__ void getRgbdPoseJacobians_warping(
        PoseOptimizationRgbdData *output_posedata,
        const float3 *warped_rgb,
        const float *warped_depth,
        const float4 *grad_x,
        const float4 *grad_y,
        const float3 *observed_rgb,
        const float *observed_depth,
        size_t grad_x_step,
        size_t grad_y_step,
        size_t warped_rgb_step,
        size_t warped_depth_step,
        size_t observed_rgb_step,
        size_t observed_depth_step,
        CameraPose warp_pose,
        IntrinsicParameters intrinsics,
        float w_depth,
        float color_thresh,
        float depth_thresh,
        int width,
        int height,
        int num_tiles_x,
        int num_tiles_y)
    {
        int x = blockIdx.x * TILE_SIZE + threadIdx.x;
        int y = blockIdx.y * TILE_SIZE + threadIdx.y;
        int tile_idx = blockIdx.y * num_tiles_x + blockIdx.x;
        int tid = threadIdx.y * TILE_SIZE + threadIdx.x;

        float local_JtJ[21] = {0.0f};
        float local_Jtr[6] = {0.0f};

        bool inside = (x < width && y < height);

        float d_obs = 0.0f;
        float d_warp = 0.0f;
        float3 c_obs = make_float3(0.0f, 0.0f, 0.0f);
        float3 c_warp = make_float3(0.0f, 0.0f, 0.0f);
        float3 gx = make_float3(0.0f, 0.0f, 0.0f);
        float3 gy = make_float3(0.0f, 0.0f, 0.0f);

        if (inside)
        {
            d_obs = getImageData<float>(observed_depth, observed_depth_step, x, y);
            d_warp = getImageData<float>(warped_depth, warped_depth_step, x, y);
            c_obs = getImageData<float4>(observed_rgb, observed_rgb_step, x, y);
            c_warp = getImageData<float4>(warped_rgb, warped_rgb_step, x, y);
            float4 gx4 = getImageData<float4>(grad_x, grad_x_step, x, y);
            float4 gy4 = getImageData<float4>(grad_y, grad_y_step, x, y);
            gx = make_float3(gx4.x, gx4.y, gx4.z);
            gy = make_float3(gy4.x, gy4.y, gy4.z);

            inside = (d_obs > 0.1f && d_warp > 0.1f);
        }

        __shared__ float R_shared[3][3];
        if (threadIdx.x == 0 && threadIdx.y == 0) {
            quaternionToMatrix(warp_pose.orientation, R_shared);
        }
        __syncthreads();

        if (inside)
        {
            float3 color_error = make_float3(c_warp.x - c_obs.x,
                                             c_warp.y - c_obs.y,
                                             c_warp.z - c_obs.z);
            float depth_error = d_warp - d_obs;

            float color_loss = sqrtf(color_error.x * color_error.x +
                                     color_error.y * color_error.y +
                                     color_error.z * color_error.z);
            float wc = (color_loss < color_thresh) ? 1.0f : color_thresh / (color_loss + 1e-8f);

            float3 ray = make_float3((x - intrinsics.c.x) / intrinsics.f.x,
                                     (y - intrinsics.c.y) / intrinsics.f.y,
                                     1.0f);

            float ray_cross[9];
            ray_cross[0] = 0.0f;      ray_cross[1] = -ray.z;  ray_cross[2] = ray.y;
            ray_cross[3] = ray.z;     ray_cross[4] = 0.0f;    ray_cross[5] = -ray.x;
            ray_cross[6] = -ray.y;    ray_cross[7] = ray.x;   ray_cross[8] = 0.0f;

            float inv_d = 1.0f / d_obs;
            float Jt_pix[6];
            Jt_pix[0] = intrinsics.f.x * inv_d;
            Jt_pix[1] = 0.0f;
            Jt_pix[2] = 0.0f;
            Jt_pix[3] = intrinsics.f.y * inv_d;
            Jt_pix[4] = -intrinsics.f.x * ray.x * inv_d;
            Jt_pix[5] = -intrinsics.f.y * ray.y * inv_d;

            float Jt_cam_pix[9];
            Jt_cam_pix[0] = Jt_pix[0] * gx.x + Jt_pix[1] * gy.x;
            Jt_cam_pix[1] = Jt_pix[0] * gx.y + Jt_pix[1] * gy.y;
            Jt_cam_pix[2] = Jt_pix[0] * gx.z + Jt_pix[1] * gy.z;
            Jt_cam_pix[3] = Jt_pix[2] * gx.x + Jt_pix[3] * gy.x;
            Jt_cam_pix[4] = Jt_pix[2] * gx.y + Jt_pix[3] * gy.y;
            Jt_cam_pix[5] = Jt_pix[2] * gx.z + Jt_pix[3] * gy.z;
            Jt_cam_pix[6] = Jt_pix[4] * gx.x + Jt_pix[5] * gy.x;
            Jt_cam_pix[7] = Jt_pix[4] * gx.y + Jt_pix[5] * gy.y;
            Jt_cam_pix[8] = Jt_pix[4] * gx.z + Jt_pix[5] * gy.z;

            float JtJ_cam_pix[9];
            for (int r = 0; r < 3; ++r) {
                for (int c = 0; c < 3; ++c) {
                    float sum = 0.0f;
                    for (int k = 0; k < 3; ++k) {
                        sum += Jt_cam_pix[r * 3 + k] * Jt_cam_pix[c * 3 + k];
                    }
                    JtJ_cam_pix[r * 3 + c] = wc * sum;
                }
            }

            float Jtr_cam_pix[3];
            Jtr_cam_pix[0] = -wc * (Jt_cam_pix[0] * color_error.x + Jt_cam_pix[1] * color_error.y + Jt_cam_pix[2] * color_error.z);
            Jtr_cam_pix[1] = -wc * (Jt_cam_pix[3] * color_error.x + Jt_cam_pix[4] * color_error.y + Jt_cam_pix[5] * color_error.z);
            Jtr_cam_pix[2] = -wc * (Jt_cam_pix[6] * color_error.x + Jt_cam_pix[7] * color_error.y + Jt_cam_pix[8] * color_error.z);

            float ld = fabsf(depth_error);
            float wd = (ld < depth_thresh) ? 1.0f : depth_thresh / (ld + 1e-8f);
            wd = (w_depth > 0.0f ? w_depth : 1.0f) * wd / d_obs;

            JtJ_cam_pix[0] += wd * ray.x * ray.x;
            JtJ_cam_pix[1] += wd * ray.x * ray.y;
            JtJ_cam_pix[2] += wd * ray.x * ray.z;
            JtJ_cam_pix[3] += wd * ray.y * ray.x;
            JtJ_cam_pix[4] += wd * ray.y * ray.y;
            JtJ_cam_pix[5] += wd * ray.y * ray.z;
            JtJ_cam_pix[6] += wd * ray.z * ray.x;
            JtJ_cam_pix[7] += wd * ray.z * ray.y;
            JtJ_cam_pix[8] += wd * ray.z * ray.z;

            Jtr_cam_pix[0] += wd * depth_error * ray.x;
            Jtr_cam_pix[1] += wd * depth_error * ray.y;
            Jtr_cam_pix[2] += wd * depth_error * ray.z;

            float Jpose[18];
            Jpose[0] = R_shared[0][0];  Jpose[1] = R_shared[0][1];  Jpose[2] = R_shared[0][2];
            Jpose[3] = R_shared[1][0];  Jpose[4] = R_shared[1][1];  Jpose[5] = R_shared[1][2];
            Jpose[6] = R_shared[2][0];  Jpose[7] = R_shared[2][1];  Jpose[8] = R_shared[2][2];

            float z_ray_cross[9];
            for (int r = 0; r < 3; ++r) {
                for (int c = 0; c < 3; ++c) {
                    z_ray_cross[r * 3 + c] = d_obs * ray_cross[r * 3 + c];
                }
            }
            Jpose[9]  = z_ray_cross[0];  Jpose[10] = z_ray_cross[1];  Jpose[11] = z_ray_cross[2];
            Jpose[12] = z_ray_cross[3];  Jpose[13] = z_ray_cross[4];  Jpose[14] = z_ray_cross[5];
            Jpose[15] = z_ray_cross[6];  Jpose[16] = z_ray_cross[7];  Jpose[17] = z_ray_cross[8];

            float temp[18];
            for (int r = 0; r < 6; ++r) {
                for (int c = 0; c < 3; ++c) {
                    float sum = 0.0f;
                    for (int k = 0; k < 3; ++k) {
                        sum += Jpose[r * 3 + k] * JtJ_cam_pix[k * 3 + c];
                    }
                    temp[r * 3 + c] = sum;
                }
            }

            int idx = 0;
            for (int r = 0; r < 6; ++r) {
                for (int c = r; c < 6; ++c) {
                    float sum = 0.0f;
                    for (int k = 0; k < 3; ++k) {
                        sum += temp[r * 3 + k] * Jpose[c * 3 + k];
                    }
                    local_JtJ[idx++] += sum;
                }
            }

            for (int r = 0; r < 6; ++r) {
                float sum = 0.0f;
                for (int k = 0; k < 3; ++k) {
                    sum += Jpose[r * 3 + k] * Jtr_cam_pix[k];
                }
                local_Jtr[r] += sum;
            }
        }

        __shared__ float shared_JtJ[TILE_SIZE * TILE_SIZE][21];
        __shared__ float shared_Jtr[TILE_SIZE * TILE_SIZE][6];

        for (int i = 0; i < 21; ++i) shared_JtJ[tid][i] = local_JtJ[i];
        for (int i = 0; i < 6; ++i) shared_Jtr[tid][i] = local_Jtr[i];
        __syncthreads();

        for (int s = (TILE_SIZE * TILE_SIZE) / 2; s > 0; s >>= 1)
        {
            if (tid < s)
            {
                for (int i = 0; i < 21; ++i) {
                    shared_JtJ[tid][i] += shared_JtJ[tid + s][i];
                }
                for (int i = 0; i < 6; ++i) {
                    shared_Jtr[tid][i] += shared_Jtr[tid + s][i];
                }
            }
            __syncthreads();
        }

        if (tid == 0)
        {
            PoseOptimizationRgbdData &out = output_posedata[tile_idx];
            for (int i = 0; i < 21; ++i) {
                atomicAdd(&out.JtJ[i], shared_JtJ[0][i]);
            }
            for (int i = 0; i < 6; ++i) {
                atomicAdd(&out.Jtr[i], shared_Jtr[0][i]);
            }
        }
    }
    */




    // ============================================================================
    // 5. KERNELS DE COVISIBILIDAD
    // ============================================================================

    struct SplattedGaussian
    {
        float3 position;
        float3 invSigma;
        float alpha;
    };

    __global__ void computeGaussiansVisibility_kernel(
        unsigned char *__restrict__ visibilities,
        const uint2 *__restrict__ ranges,
        const uint32_t *__restrict__ indices,
        const float4 *__restrict__ imgPositions,
        const float4 *__restrict__ imgInvSigmas,
        const float *__restrict__ alphas,
        uint2 numTiles,
        uint32_t width,
        uint32_t height)
    {
        // ============================================================
        // 1. Pixel / tile actual
        // ============================================================
        int x = blockIdx.x * blockDim.x + threadIdx.x;
        int y = blockIdx.y * blockDim.y + threadIdx.y;

        int tid = threadIdx.y * blockDim.x + threadIdx.x;

        // ============================================================
        // 2. Cache local de gaussianas del tile
        // ============================================================
        __shared__ SplattedGaussian splattedGaussians[BLOCK_SIZE];
        __shared__ uint32_t gids[BLOCK_SIZE];

        int tileId = blockIdx.y * numTiles.x + blockIdx.x;
        uint2 range = ranges[tileId];
        int n = (int)(range.y - range.x);

        bool inside = x < (int)width && y < (int)height;

        // ============================================================
        // 3. Integracion de opacidad a lo largo del rayo
        // ============================================================
        float prod_alpha = 1.f;
        bool done = !inside;

        for (int k = 0; k < (n + BLOCK_SIZE - 1) / BLOCK_SIZE; k++)
        {
            // --------------------------------------------------------
            // 3.1 Carga cooperativa a shared memory
            // --------------------------------------------------------
            if (k * BLOCK_SIZE + tid < n)
            {
                uint32_t gid = indices[range.x + k * BLOCK_SIZE + tid];
                gids[tid] = gid;
                float4 img_pos4 = imgPositions[gid];
                splattedGaussians[tid].position = make_float3(img_pos4.x, img_pos4.y, img_pos4.z);
                float4 invSigma4 = imgInvSigmas[gid];
                splattedGaussians[tid].invSigma = make_float3(invSigma4.x, invSigma4.y, invSigma4.z);
                splattedGaussians[tid].alpha = alphas[gid];
            }
            __syncthreads();

            // --------------------------------------------------------
            // 3.2 Marcado de gaussianas visibles
            // --------------------------------------------------------
            for (int i = 0; !done && i + k * BLOCK_SIZE < n && i < BLOCK_SIZE; i++)
            {
                float dx = splattedGaussians[i].position.x - x;
                float dy = splattedGaussians[i].position.y - y;
                float alpha_i = min(0.99f, splattedGaussians[i].alpha * evalGaussian2D(
                    dx,
                    dy,
                    splattedGaussians[i].invSigma.x,
                    splattedGaussians[i].invSigma.z,
                    splattedGaussians[i].invSigma.y));

                if (alpha_i < 1.f / 255.f)
                    continue;

                prod_alpha *= (1.f - alpha_i);

                visibilities[gids[i]] = 1;

                if (prod_alpha < 0.5f)
                {
                    done = true;
                    continue;
                }
                prod_alpha = max(0.f, min(1.f, prod_alpha));
            }
        }
    }

    __global__ void computeGaussiansCovisibility_kernel(
        uint32_t *visibilityInter,
        uint32_t *visibilityUnion,
        unsigned char *visibilities1,
        unsigned char *visibilities2,
        uint32_t nbGaussians)
    {
        // ============================================================
        // 1. Un thread por gaussiana
        // ============================================================
        int idx = blockIdx.x * blockDim.x + threadIdx.x;

        if (idx >= nbGaussians) return;

        // ============================================================
        // 2. Interseccion / union de visibilidades
        // ============================================================
        unsigned char vis1 = visibilities1[idx];
        unsigned char vis2 = visibilities2[idx];

        if (vis1 | vis2) {
            atomicAggInc(visibilityUnion);
        }

        if (vis1 & vis2) {
            atomicAggInc(visibilityInter);
        }
    }

    // ============================================================================
    // 6. GESTION DE MAPA: PRUNE, OUTLIERS Y DENSIFICACION
    // ============================================================================

    __global__ void pruneGaussians_kernel(
        uint32_t *__restrict__ nbRemoved,
        unsigned char *__restrict__ states,
        const float4 *__restrict__ scales,   // <-- tu layout (float4)
        const float *__restrict__ alphas,
        float alphaThreshold,
        float scaleRatioThreshold,
        uint32_t nbGaussians)
    {
        // ============================================================
        // 1. Thread index
        // ============================================================
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= nbGaussians) return;

        unsigned char state = 0;

        // ============================================================
        // 2. Alpha pruning
        // ============================================================
        if (alphas[idx] < alphaThreshold)
        {
            state = 0xff;
        }
        else
        {
            // ========================================================
            // 3. Escalas
            // ========================================================
            float3 s = make_float3(
                scales[idx].x,
                scales[idx].y,
                scales[idx].z
            );

            // Ordenar s.x <= s.y <= s.z
            if (s.x > s.y) deviceSwap(s.x, s.y);
            if (s.y > s.z) deviceSwap(s.y, s.z);
            if (s.x > s.y) deviceSwap(s.x, s.y);

            // ========================================================
            // 4. Criterio de forma (ratio + tamaño mínimo)
            // ========================================================
            if (s.y / s.z < scaleRatioThreshold || s.z < 0.005f)
            {
                state = 0xff;
            }
        }

        // ============================================================
        // 5. Contador global
        // ============================================================
        if (state != 0)
        {
            atomicAggInc(nbRemoved);
        }

        // ============================================================
        // 6. Guardar estado
        // ============================================================
        states[idx] = state;
    }

    __global__ void computeOutliers_kernel(
        float *outlierProb,
        float *totalAlpha,
        const uint2 *ranges,
        const uint32_t *indices,
        const float4 *positions_2d,
        const float4 *inv_covariances_2d,
        const float2 *pHats,
        const float *alphas,
        const float *depth,
        size_t depth_step,
        uint2 numTiles,
        uint32_t width,
        uint32_t height,
        uint32_t n_gaussians_max,
        uint32_t n_instances_max)
    {
        // ============================================================
        // 1. Un thread por pixel
        // ============================================================
        int x = blockIdx.x * blockDim.x + threadIdx.x;
        int y = blockIdx.y * blockDim.y + threadIdx.y;
        if (x >= width || y >= height) return;

        // ============================================================
        // 2. Tile actual y rango de gaussianas
        // ============================================================
        int tileId = blockIdx.y * numTiles.x + blockIdx.x;
        uint2 range = ranges[tileId];
        uint32_t range_start_u = min(range.x, n_instances_max);
        uint32_t range_end_u = min(range.y, n_instances_max);
        if (range_end_u < range_start_u) return;

        // ============================================================
        // 3. Cache local por batches
        // ============================================================
        int n = static_cast<int>(range_end_u - range_start_u);
        __shared__ float4 s_positions[TILE_SIZE * TILE_SIZE];
        __shared__ float4 s_invSigmas[TILE_SIZE * TILE_SIZE];
        __shared__ float2 s_pHats[TILE_SIZE * TILE_SIZE];
        __shared__ uint32_t s_gids[TILE_SIZE * TILE_SIZE];
        __shared__ float s_alphas[TILE_SIZE * TILE_SIZE];

        int tid = threadIdx.y * TILE_SIZE + threadIdx.x;
        int block_size = TILE_SIZE * TILE_SIZE;
        float px = (float)x;
        float py = (float)y;

        // ============================================================
        // 4. Profundidad observada
        // ============================================================
        const char *depth_row = reinterpret_cast<const char *>(depth) + y * depth_step;
        float depth_obs = *(reinterpret_cast<const float *>(depth_row) + x);

        // ============================================================
        // 5. Procesamiento por batches
        // ============================================================
        for (int base = 0; base < n; base += block_size)
        {
            // --------------------------------------------------------
            // 5.1 Carga cooperativa a shared memory
            // --------------------------------------------------------
            int local_idx = base + tid;
            if (local_idx < n)
            {
                uint32_t global_idx = range_start_u + static_cast<uint32_t>(local_idx);
                uint32_t gid = (global_idx < n_instances_max) ? indices[global_idx] : n_gaussians_max;
                s_gids[tid] = gid;
                if (gid < n_gaussians_max)
                {
                    s_positions[tid] = positions_2d[gid];
                    s_invSigmas[tid] = inv_covariances_2d[gid];
                    s_pHats[tid] = pHats[gid];
                    s_alphas[tid] = alphas[gid];
                }
                else
                {
                    s_positions[tid] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
                    s_invSigmas[tid] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
                    s_pHats[tid] = make_float2(0.0f, 0.0f);
                    s_alphas[tid] = 0.0f;
                }
            }
            __syncthreads();

            // --------------------------------------------------------
            // 5.2 Acumulacion de alpha y deteccion de inconsistencias
            // --------------------------------------------------------
            int batch_count = min(block_size, n - base);
            for (int i = 0; i < batch_count; i++)
            {
                float4 pos = s_positions[i];
                float2 pHat = s_pHats[i];
                uint32_t gid = s_gids[i];
                if (gid >= n_gaussians_max) continue;

                // Calcular desplazamiento en imagen
                float dx = pos.x - px;
                float dy = pos.y - py;

                // Calcular profundidad renderizada: d = z + pHat·(x-u, y-v)
                float depth_rendered = pos.z + pHat.x * dx + pHat.y * dy;

                // Evaluar gaussiana 2D para obtener alpha_i
                float4 invSigma = s_invSigmas[i];
                
                float mahalanobis = mahalanobis2D(dx, dy, invSigma.x, invSigma.z, invSigma.y);
                float alpha_i = evalGaussian2DFromMahalanobis(mahalanobis);
                if (alpha_i < (1.0f / 255.0f) || mahalanobis <= 0.0f)
                {
                    continue;
                }

                // Acumular en totalAlpha
                atomicAdd(&totalAlpha[gid], alpha_i);

                // Detectar outliers: si hay mismatch significativo entre observado y renderizado
                if (depth_obs > 0.1f && depth_rendered < 0.8f * depth_obs)
                {
                    // Hay oclusión/mismatch: profundidad renderizada es mucho menor que observada
                    atomicAdd(&outlierProb[gid], alpha_i);
                }
            }

            __syncthreads();
        }
    }

    __global__ void removeOutliers_kernel(
        uint32_t *nbRemoved,
        unsigned char *states,
        const float *outlierProb,
        const float *totalAlpha,
        float threshold,
        uint32_t nbGaussians)
    {
        // ============================================================
        // 1. Un thread por gaussiana
        // ============================================================
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= nbGaussians) return;

        // ============================================================
        // 2. Regla de descarte por ratio outlier / alpha total
        // ============================================================
        unsigned char state = 0;

        if (totalAlpha[idx] > 1.0f && outlierProb[idx] / totalAlpha[idx] > threshold)
        {
            atomicAggInc(nbRemoved);
            state = 0xff;
        }

        states[idx] = state;

    }

    __global__ void computeDensityMask_kernel(
        float *maskData,
        const uint2 *__restrict__ ranges,
        const uint32_t *__restrict__ indices,
        const float4 *__restrict__ imgPositions,
        const float4 *__restrict__ imgInvSigmas,
        const float2 *__restrict__ pHats,
        const float *__restrict__ alphas,
        cudaTextureObject_t depthTex,
        uint2 numTiles,
        uint32_t width,
        uint32_t height,
        size_t mask_stride,
        uint32_t n_gaussians_max,
        uint32_t n_instances_max)
    {
        // ============================================================
        // 1. Thread → pixel
        // ============================================================
        int x = blockIdx.x * blockDim.x + threadIdx.x;
        int y = blockIdx.y * blockDim.y + threadIdx.y;

        if (x >= (int)width || y >= (int)height)
            return;

        // ============================================================
        // 2. Tile range
        // ============================================================
        int tileId = blockIdx.y * numTiles.x + blockIdx.x;
        uint2 range = ranges[tileId];

        uint32_t start = min(range.x, n_instances_max);
        uint32_t end   = min(range.y, n_instances_max);
        int n = (int)(end - start);

        // ============================================================
        // 3. Depth (texture)
        // ============================================================
        float depth_obs = getImageData<float>(depthTex, x, y);

        float prod_alpha = 1.0f;
        float depth_rendered = 0.0f;
        float T = 1.0f;

        bool done = (depth_obs < 0.5f);

        // ============================================================
        // 4. Shared memory
        // ============================================================
        __shared__ RasterGauss2D s_gauss[TILE_SIZE * TILE_SIZE];

        // ============================================================
        // 5. Loop por batches
        // ============================================================
        for (int base = 0; base < n; base += TILE_SIZE * TILE_SIZE)
        {
            int local_idx = base + threadIdx.y * TILE_SIZE + threadIdx.x;
            int s_idx     = threadIdx.y * TILE_SIZE + threadIdx.x;

            // --------------------------------------------------------
            // 5.1 Carga cooperativa a shared memory
            // --------------------------------------------------------
            if (local_idx < n)
            {
                uint32_t global_idx = start + (uint32_t)local_idx;
                uint32_t gid = (global_idx < n_instances_max) ? indices[global_idx] : n_gaussians_max;

                if (gid < n_gaussians_max)
                {
                    float4 pos  = imgPositions[gid];
                    float4 invS = imgInvSigmas[gid];

                    RasterGauss2D g;
                    g.pos  = pos;
                    g.inv_cov = invS;
                    // g.color = colors[gid]; no se usa
                    g.alpha = alphas[gid];
                    g.pHat  = pHats[gid];

                    s_gauss[s_idx] = g;
                }
                else
                {
                    // gauss inválida
                    s_gauss[s_idx].alpha = 0.0f;
                }
            }

            __syncthreads();

            int batch_count = min(TILE_SIZE * TILE_SIZE, n - base);

            // --------------------------------------------------------
            // 5.2 Integración por pixel
            // --------------------------------------------------------
            for (int i = 0; !done && i < batch_count; i++)
            {
                const RasterGauss2D &g = s_gauss[i];

                if (g.alpha <= 0.0f) continue;

                float dx = g.pos.x - (float)x;
                float dy = g.pos.y - (float)y;

                // Evaluación gaussiana encapsulada
                float G = evalGaussian2D(
                    dx, dy,
                    g.inv_cov.x,
                    g.inv_cov.z,
                    g.inv_cov.y);

                float alpha_i = fminf(0.99f, g.alpha * G);

                if (alpha_i < (1.0f / 255.0f))
                    continue;

                // ----------------------------------------------------
                // 5.3 Acumulación de transmitancia
                // ----------------------------------------------------
                prod_alpha *= (1.0f - alpha_i);

                // Cruce de mediana (T=0.5)
                if (T > 0.5f && prod_alpha < 0.5f)
                {
                    T = prod_alpha;

                    // depth reprojection encapsulada
                    depth_rendered =
                        g.pos.z +
                        g.pHat.x * dx +
                        g.pHat.y * dy;
                }

                if (prod_alpha < 1e-4f)
                {
                    done = true;
                    break;
                }
            }

            __syncthreads();
        }

        // ============================================================
        // 6. Decisión del mask
        // ============================================================
        // Convención de máscaras:
        // - M_h (alta frecuencia / zonas informativas) => valor alto en maskData.
        // - M_l (baja frecuencia / zonas planas) => valor cercano a 0.
        // Este kernel deja un score continuo y luego se umbraliza fuera.
        float val = 0.0f;

        if (depth_obs > 0.5f)
        {
            if (prod_alpha > 0.5f)
            {
                val = prod_alpha;
            }
            else
            {
                float depth_error = depth_rendered - depth_obs;

                if (depth_obs < depth_rendered &&
                    depth_error > 0.2f * depth_obs)
                {
                    val = 1.0f;
                }
            }
        }

        // ============================================================
        // 7. Escritura
        // ============================================================
        size_t idx = y * (mask_stride) + x;
        maskData[idx] = val;
    }

 __global__ void densifyGaussians_kernel(
    float4 *__restrict__ positions,
    float4 *__restrict__ scales,
    float4 *__restrict__ orientations,
    float4 *__restrict__ colors,
    float *__restrict__ alphas,
    uint32_t *__restrict__ instanceCounter,

    cudaTextureObject_t texColor,
    cudaTextureObject_t texDepth,
    cudaTextureObject_t texNormal,
    cudaTextureObject_t texMask,

    Pose cameraPose,
    Pose submapPoseGlobal,
    IntrinsicParameters intrinsics,

    uint32_t sample_dx,
    uint32_t sample_dy,
    uint32_t width,
    uint32_t height,
    float scale_factor,
    uint32_t maxGaussians)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    // ============================================================
    // 1. BLOQUE POR CELDA (subsampling)
    // ============================================================
    if (sample_dx > 1)
    {
        int u_min = x * sample_dx;
        int v_min = y * sample_dy;

        if (u_min >= (int)width || v_min >= (int)height)
            return;

        float3 img_pos = make_float3(0.f, 0.f, 0.f);
        float3 rgb_acc = make_float3(0.f, 0.f, 0.f);
        int n = 0;

        // --------------------------------------------------------
        // Acumulación dentro de la celda
        // --------------------------------------------------------
        for (int u = u_min; u < u_min + (int)sample_dx && u < (int)width; u++)
        {
            for (int v = v_min; v < v_min + (int)sample_dy && v < (int)height; v++)
            {
                // Solo aportan píxeles activos en la máscara de entrada.
                // Usualmente: texMask > 0 representa M_h y el resto actúa como M_l.
                if (getImageData<float>(texMask, u, v) > 0.f)
                {
                    float d = getImageData<float>(texDepth, u, v);
                    uchar4 color = getImageData<uchar4>(texColor, u, v);

                    img_pos.x += u;
                    img_pos.y += v;
                    img_pos.z += d;

                    rgb_acc.x += color.x / 255.f;
                    rgb_acc.y += color.y / 255.f;
                    rgb_acc.z += color.z / 255.f;

                    n++;
                }
            }
        }

        if (n < ((sample_dx * sample_dy) >> 1))
            return;

        uint32_t idx = atomicAggInc(instanceCounter);
        if (idx >= maxGaussians)
            return;

        float inv_n = 1.f / (float)n;

        img_pos.x *= inv_n;
        img_pos.y *= inv_n;
        img_pos.z *= inv_n;

        rgb_acc.x *= inv_n;
        rgb_acc.y *= inv_n;
        rgb_acc.z *= inv_n;

        // --------------------------------------------------------
        // Proyección cámara → mundo
        // --------------------------------------------------------
        float3 pos_cam = make_float3(
            img_pos.z * (img_pos.x - intrinsics.c.x) / intrinsics.f.x,
            img_pos.z * (img_pos.y - intrinsics.c.y) / intrinsics.f.y,
            img_pos.z);

        float3 pos_world =
            cameraPose.position +
            rotateByQuaternion(cameraPose.orientation, pos_cam);

        const float4 q_submap_inv = quaternionInverse(submapPoseGlobal.orientation);
        const float3 pos_local = rotateByQuaternion(
            q_submap_inv,
            pos_world - submapPoseGlobal.position);

        positions[idx] = make_float4(
            pos_local.x, pos_local.y, pos_local.z, 1.f);

        // --------------------------------------------------------
        // Escala
        // --------------------------------------------------------
        float scale_x = img_pos.z * sample_dx / intrinsics.f.x;
        float scale_y = img_pos.z * sample_dy / intrinsics.f.y;
        float s = 0.5f * (scale_x + scale_y) * scale_factor;

        scales[idx] = make_float4(s, s, 0.1f * s, 0.f);

        // --------------------------------------------------------
        // Normal → orientación
        // --------------------------------------------------------
        float4 nrm4 = tex2D<float4>(texNormal, img_pos.x, img_pos.y);
        float3 normal = make_float3(nrm4.x, nrm4.y, nrm4.z);

        if (normal.z < 0.f)
        {
            normal.x = -normal.x;
            normal.y = -normal.y;
            normal.z = -normal.z;
        }

        float4 q = quatFromTwoVectors(
            make_float3(0.f, 0.f, 1.f),
            normal);

        float4 q_world = quatMultiply(cameraPose.orientation, q);
        float4 q_local = quatMultiply(q_submap_inv, q_world);
        float q_local_norm = sqrtf(q_local.x * q_local.x + q_local.y * q_local.y +
                       q_local.z * q_local.z + q_local.w * q_local.w);
        if (q_local_norm < 1e-6f) return;
        q_local.x /= q_local_norm;
        q_local.y /= q_local_norm;
        q_local.z /= q_local_norm;
        q_local.w /= q_local_norm;
        orientations[idx] = q_local;

        // --------------------------------------------------------
        // Color + alpha
        // --------------------------------------------------------
        colors[idx] = make_float4(
            rgb_acc.x, rgb_acc.y, rgb_acc.z, 0.f);

        alphas[idx] = 1.f;
    }

    // ============================================================
    // 2. MODO PIXEL A PIXEL
    // ============================================================
    else
    {
        if (x >= (int)width || y >= (int)height)
            return;

        // En modo píxel a píxel, la máscara opera como compuerta binaria:
        // M_h pasa a densificación; M_l se descarta en este punto.
        if (tex2D<float>(texMask, x, y) <= 0.f)
            return;

        float d = tex2D<float>(texDepth, x, y);
        uchar4 color = tex2D<uchar4>(texColor, x, y);

        uint32_t idx = atomicAggInc(instanceCounter);
        if (idx >= maxGaussians)
            return;

        float3 pos_cam = make_float3(
            d * (x - intrinsics.c.x) / intrinsics.f.x,
            d * (y - intrinsics.c.y) / intrinsics.f.y,
            d);

        float3 pos_world =
            cameraPose.position +
            rotateByQuaternion(cameraPose.orientation, pos_cam);

        const float4 q_submap_inv = quaternionInverse(submapPoseGlobal.orientation);
        const float3 pos_local = rotateByQuaternion(
            q_submap_inv,
            pos_world - submapPoseGlobal.position);

        positions[idx] = make_float4(
            pos_local.x, pos_local.y, pos_local.z, 1.f);

        float scale_x = 0.8f * d * sample_dx / intrinsics.f.x;
        float scale_y = 0.8f * d * sample_dy / intrinsics.f.y;
        float s = 0.5f * (scale_x + scale_y) * scale_factor;

        scales[idx] = make_float4(s, s, 0.1f * s, 0.f);

        float4 nrm4 = tex2D<float4>(texNormal, x, y);
        float3 normal = make_float3(nrm4.x, nrm4.y, nrm4.z);

        if (normal.z < 0.f)
        {
            normal.x = -normal.x;
            normal.y = -normal.y;
            normal.z = -normal.z;
        }

        float4 q = quatFromTwoVectors(
            make_float3(0.f, 0.f, 1.f),
            normal);

        float4 q_world = quatMultiply(cameraPose.orientation, q);
        float4 q_local = quatMultiply(q_submap_inv, q_world);
        float q_local_norm = sqrtf(q_local.x * q_local.x + q_local.y * q_local.y +
                                   q_local.z * q_local.z + q_local.w * q_local.w);
        if (q_local_norm < 1e-6f)
            return;
        q_local.x /= q_local_norm;
        q_local.y /= q_local_norm;
        q_local.z /= q_local_norm;
        q_local.w /= q_local_norm;
        orientations[idx] = q_local;

        colors[idx] = make_float4(
            color.x / 255.f,
            color.y / 255.f,
            color.z / 255.f,
            0.f);

        alphas[idx] = 1.f;
    }
}

    // ============================================================================
    // 7. OPTIMIZACION DE KEYFRAMES
    // ============================================================================

    __global__ void perTileBucketCount(
        uint32_t* __restrict__ bucketCount,
        const uint2* __restrict__ tileRanges,
        int numTiles)
    {
        // ============================================================
        // 1. Un thread por tile
        // ============================================================
        int tileId = blockIdx.x * blockDim.x + threadIdx.x;
        if (tileId >= numTiles) return;

        // ============================================================
        // 2. Numero de gaussianas en el tile
        // ============================================================
        // Contamos las gaussianas en el tile
        uint2 range = tileRanges[tileId];
        int amount = range.y - range.x;

        // ============================================================
        // 3. Bucketizado fijo para el barrido por bloques
        // ============================================================
        // Agrupamos en buckets de tamaño 32
        constexpr uint32_t BUCKET_SIZE = 32;
        bucketCount[tileId] = (amount + BUCKET_SIZE - 1) / BUCKET_SIZE;
    } 

    __global__ void optimizeGaussiansForwardPass(
        const uint2 *ranges,
        const uint32_t *indices,
        const float4 *positions_2d,
        const float4 *inv_covariances_2d,
        const float2 *p_hats,
        const float4 *colors,
        const float *alphas,
        const uint32_t *bucketOffsets,
        uint32_t *bucket_to_tile,
        float *sampled_T,
        float3 *sampled_ar,
        float *final_T,
        uint32_t *n_contrib,
        uint32_t *max_contrib,
        float3 *output_color,
        float *output_depth,
        float3 *color_error,
        float *depth_error,
        cudaTextureObject_t observed_rgb,
        cudaTextureObject_t observed_depth,
        float3 bg_color,
        uint2 num_tiles,
        int width,
        int height)
    {
        // ============================================================
        // 1. Pixel
        // ============================================================
        int x = blockIdx.x * blockDim.x + threadIdx.x;
        int y = blockIdx.y * blockDim.y + threadIdx.y;
        int tId = threadIdx.x + threadIdx.y * blockDim.x;

        bool inside = (x < width && y < height);
        bool done = !inside;

        // ============================================================
        // 2. Tile + rango
        // ============================================================
        int tileId = blockIdx.y * num_tiles.x + blockIdx.x;
        uint2 range = ranges[tileId];
        int n = min((int)(range.y - range.x), BLOCK_SIZE);

        // ============================================================
        // 3. bucket_to_tile
        // ============================================================
        uint32_t bbm = (tileId == 0) ? 0 : bucketOffsets[tileId - 1];
        int bucketCount = (n + 31) / 32;

        for (int i = 0; i < (bucketCount + BLOCK_SIZE - 1) / BLOCK_SIZE; ++i)
        {
            int bIdx = i * BLOCK_SIZE + tId;
            if (bIdx < bucketCount)
            {
                bucket_to_tile[bbm + bIdx] = tileId;
            }
        }

        // ============================================================
        // 4. Inicialización
        // ============================================================
        float T = 1.0f;
        float3 color = make_float3(0.0f, 0.0f, 0.0f);
        float depth = 0.0f;

        uint32_t contributor = 0;
        uint32_t last_contributor = 0;

        // ============================================================
        // 5. Shared memory
        // ============================================================
        __shared__ RasterGauss2D s_gauss[BLOCK_SIZE];

        int n_batches = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;

        bbm = (tileId == 0) ? 0 : bucketOffsets[tileId - 1];

        // ============================================================
        // 6. Loop principal
        // ============================================================
        for (int batch = 0; batch < n_batches; ++batch)
        {
            int idx = batch * BLOCK_SIZE + tId;

            // ---- carga a shared ----
            if (idx < n)
            {
                uint32_t gid = indices[range.x + idx];

                RasterGauss2D g;
                g.pos = positions_2d[gid];
                g.inv_cov = inv_covariances_2d[gid];
                g.color = colors[gid];
                g.pHat = p_hats[gid];
                g.alpha = alphas[gid];

                s_gauss[tId] = g;
            }

            __syncthreads();

            int batch_count = min(BLOCK_SIZE, n - batch * BLOCK_SIZE);

            // ---- procesamiento ----
            for (int i = 0; i < batch_count && !done; ++i)
            {
                int global_idx = batch * BLOCK_SIZE + i;

                if (global_idx % 32 == 0)
                {
                    int sampleIdx = (bbm * BLOCK_SIZE) + tId;
                    sampled_T[sampleIdx] = T;
                    sampled_ar[sampleIdx] = color;
                    ++bbm;
                }

                contributor++;

                RasterGauss2D g = s_gauss[i];

                float dx = g.pos.x - (float)x;
                float dy = g.pos.y - (float)y;

                float v = mahalanobis2D(dx, dy, g.inv_cov.x, g.inv_cov.z, g.inv_cov.y);

                float alpha_i = fminf(0.99f, g.alpha * evalGaussian2DFromMahalanobis(v));

                if (alpha_i < (1.0f / 255.0f) || v < 0.0f)
                {
                    continue;
                }

                float test_T = T * (1.0f - alpha_i);

                if (test_T < 0.0001f)
                {
                    done = true;
                    continue;
                }

                float T_pre = T;

                float3 col = make_float3(g.color.x, g.color.y, g.color.z);
                color += T * alpha_i * col;

                if (T_pre > 0.5f && test_T <= 0.5f)
                {
                    depth = g.pos.z + g.pHat.x * dx + g.pHat.y * dy;
                }

                T = test_T;
                last_contributor = contributor;
            }

            __syncthreads();
        }

        // ============================================================
        // 7. Escritura final
        // ============================================================
        if (inside)
        {
            int pix_id = y * width + x;

            final_T[pix_id] = T;
            n_contrib[pix_id] = last_contributor;

            color += T * bg_color;

            output_color[pix_id] = color;
            output_depth[pix_id] = depth;

            uchar4 rgba = getImageData<uchar4>(observed_rgb, x, y);
            float3 obs = make_float3(rgba.x / 255.f,
                                    rgba.y / 255.f,
                                    rgba.z / 255.f);

            color_error[pix_id] = color - obs;

            float img_depth = getImageData<float>(observed_depth, x, y);
            depth_error[pix_id] = img_depth;
        }

        // ============================================================
        // 8. Reducción
        // ============================================================
        typedef cub::BlockReduce<uint32_t, BLOCK_SIZE,
                                cub::BLOCK_REDUCE_WARP_REDUCTIONS>
            BlockReduce;

        __shared__ typename BlockReduce::TempStorage temp_storage;

        last_contributor =
            BlockReduce(temp_storage).Reduce(last_contributor, cub::Max());

        if (tId == 0)
        {
            max_contrib[tileId] = last_contributor;
        }
    }

    __global__ void optimizeGaussiansPerGaussianPass(
        const uint2 *ranges,
        const uint32_t *indices,
        const float4 *positions_2d,
        const float4 *inv_covariances_2d,
        const float2 *p_hats,
        const float4 *colors,
        const float *alphas,
        const uint32_t *bucketOffsets,
        const uint32_t *bucket_to_tile,
        const float *sampled_T,
        const float3 *sampled_ar,
        const uint32_t *n_contrib,
        const uint32_t *max_contrib,
        const float3 *output_color,
        const float *output_depth,
        const float3 *color_error,
        const float *depth_error,
        DeltaGaussian2D *delta_gaussians,
        float w_depth,
        float w_dist,
        uint2 num_tiles,
        int width,
        int height,
        int num_buckets)
    {
        // ============================================================
        // 1. Warp mapping (igual que el original)
        // ============================================================
        uint32_t warp_id = threadIdx.x >> 5;
        uint32_t lane_id = threadIdx.x & 31;
        uint32_t warps_per_block = blockDim.x >> 5;

        uint32_t global_bucket_id =
            blockIdx.x * warps_per_block + warp_id;

        if (global_bucket_id >= (uint32_t)num_buckets)
            return;

        // ============================================================
        // 2. Resolver tile y bucket local
        // ============================================================
        uint32_t tileId = bucket_to_tile[global_bucket_id];
        uint2 range = ranges[tileId];

        uint32_t bucket_base =
            (tileId == 0) ? 0 : bucketOffsets[tileId - 1];

        uint32_t bucket_idx = global_bucket_id - bucket_base;

        // Early exit (igual al original)
        if (bucket_idx * 32 >= max_contrib[tileId])
            return;

        // ============================================================
        // 3. Índice de gaussiana dentro del tile
        // ============================================================
        uint32_t splat_idx_in_tile = bucket_idx * 32 + lane_id;
        uint32_t num_splats =
            min((uint32_t)(range.y - range.x), (uint32_t)BLOCK_SIZE);

        bool valid_splat = splat_idx_in_tile < num_splats;

        uint32_t gId = 0;

        // ============================================================
        // 4. Carga de parámetros de la gaussiana (registro)
        //    Uso RasterGauss2D para claridad (AoS)
        // ============================================================
        RasterGauss2D g;

        if (valid_splat)
        {
            gId = indices[range.x + splat_idx_in_tile];
            g.pos     = positions_2d[gId];
            g.inv_cov = inv_covariances_2d[gId];
            g.color   = colors[gId];
            g.pHat    = p_hats[gId];
            g.alpha   = alphas[gId];
        }

        // ============================================================
        // 5. Inicialización de acumulador de gradientes
        // ============================================================
        DeltaGaussian2D delta;
        delta.meanImg    = make_float2(0.0f, 0.0f);
        delta.invSigmaImg = make_float3(0.0f, 0.0f, 0.0f);
        delta.color      = make_float3(0.0f, 0.0f, 0.0f);
        delta.depth      = 0.0f;
        delta.alpha      = 0.0f;
        delta.pHat       = make_float2(0.0f, 0.0f);
        delta.n          = 0;

        // ============================================================
        // 6. Coordenadas del tile en imagen
        // ============================================================
        uint2 tile = make_uint2(tileId % num_tiles.x, tileId / num_tiles.x);
        uint2 pix_min = make_uint2(tile.x * TILE_SIZE, tile.y * TILE_SIZE);

        // ============================================================
        // 7. Estado warp (propagado con shuffle)
        // ============================================================
        float T = 0.0f;
        float last_contributor = 0.0f;
        float3 acc_c = make_float3(0.0f, 0.0f, 0.0f);
        float3 col_err = make_float3(0.0f, 0.0f, 0.0f);
        float3 color = make_float3(0.0f, 0.0f, 0.0f);
        float img_depth = 0.0f;
        float depth = 0.0f;

        const unsigned mask = 0xffffffffu;

        // ============================================================
        // 8. Loop principal (idéntico al original)
        // ============================================================
        for (int i = 0; i < BLOCK_SIZE + 31; ++i)
        {
            // Warp shift (pipeline)
            T = __shfl_up_sync(mask, T, 1);
            last_contributor = __shfl_up_sync(mask, last_contributor, 1);
            acc_c.x = __shfl_up_sync(mask, acc_c.x, 1);
            acc_c.y = __shfl_up_sync(mask, acc_c.y, 1);
            acc_c.z = __shfl_up_sync(mask, acc_c.z, 1);
            color.x = __shfl_up_sync(mask, color.x, 1);
            color.y = __shfl_up_sync(mask, color.y, 1);
            color.z = __shfl_up_sync(mask, color.z, 1);
            col_err.x = __shfl_up_sync(mask, col_err.x, 1);
            col_err.y = __shfl_up_sync(mask, col_err.y, 1);
            col_err.z = __shfl_up_sync(mask, col_err.z, 1);
            img_depth = __shfl_up_sync(mask, img_depth, 1);
            depth = __shfl_up_sync(mask, depth, 1);

            // ========================================================
            // 8.1 Pixel asociado a este lane
            // ========================================================
            int idx = i - (int)lane_id;

            int pix_x = (int)pix_min.x + idx % TILE_SIZE;
            int pix_y = (int)pix_min.y + idx / TILE_SIZE;

            uint32_t pix_id = width * pix_y + pix_x;

            // MISMA condición que el original
            bool valid_pixel = (pix_x < width && pix_y < height);

            // ========================================================
            // 8.2 Lane 0 carga estado guardado (forward pass)
            // ========================================================
            if (valid_splat && valid_pixel && lane_id == 0 &&
                idx >= 0 && idx < BLOCK_SIZE)
            {
                T = sampled_T[global_bucket_id * BLOCK_SIZE + idx];
                acc_c = sampled_ar[global_bucket_id * BLOCK_SIZE + idx];
                color = output_color[pix_id];
                depth = output_depth[pix_id];
                last_contributor = (float)n_contrib[pix_id];
                col_err = color_error[pix_id];
                img_depth = depth_error[pix_id];
            }

            // ========================================================
            // 8.3 Backprop por pixel
            // ========================================================
            if (valid_splat && valid_pixel &&
                idx >= 0 && idx < BLOCK_SIZE)
            {
                if (splat_idx_in_tile >= (uint32_t)last_contributor)
                    continue;

                float dx = g.pos.x - (float)pix_x;
                float dy = g.pos.y - (float)pix_y;

                float v =
                    g.inv_cov.x * dx * dx +
                    2.0f * g.inv_cov.y * dx * dy +
                    g.inv_cov.z * dy * dy;

                float G = evalGaussian2D(dx, dy,
                                        g.inv_cov.x,
                                        g.inv_cov.z,
                                        g.inv_cov.y);

                float alpha = fminf(0.99f, g.alpha * G);
                if (alpha < (1.0f / 255.0f))
                    continue;

                // dC/dalpha
                float3 d_alpha = make_float3(g.color.x * T,
                                            g.color.y * T,
                                            g.color.z * T);

                acc_c += d_alpha * alpha;

                float inv_one_minus = 1.0f / (1.0f - alpha);

                d_alpha.x -= (color.x - acc_c.x) * inv_one_minus;
                d_alpha.y -= (color.y - acc_c.y) * inv_one_minus;
                d_alpha.z -= (color.z - acc_c.z) * inv_one_minus;

                // aplicar gradiente de color
                d_alpha.x *= -col_err.x;
                d_alpha.y *= -col_err.y;
                d_alpha.z *= -col_err.z;

                float dl_alpha = d_alpha.x + d_alpha.y + d_alpha.z;
                float a_G_dl = alpha * dl_alpha;

                delta.n++;

                // gradientes principales
                delta.color.x -= alpha * T * col_err.x;
                delta.color.y -= alpha * T * col_err.y;
                delta.color.z -= alpha * T * col_err.z;
                delta.alpha -= a_G_dl;

                delta.meanImg.x -= a_G_dl * (g.inv_cov.x * dx + g.inv_cov.y * dy);
                delta.meanImg.y -= a_G_dl * (g.inv_cov.y * dx + g.inv_cov.z * dy);

                delta.invSigmaImg.x -= 0.5f * a_G_dl * dx * dx;
                delta.invSigmaImg.y -= 0.5f * a_G_dl * dx * dy;
                delta.invSigmaImg.z -= 0.5f * a_G_dl * dy * dy;

                float test_T = T * (1.0f - alpha);

                // depth loss
                if (T > 0.5f && test_T <= 0.5f)
                {
                    float depth_err =
                        (img_depth > 0.1f) ? (depth - img_depth) : 0.0f;

                    delta.depth -= w_depth * depth_err;

                    delta.meanImg.x -= w_depth * depth_err * g.pHat.x;
                    delta.meanImg.y -= w_depth * depth_err * g.pHat.y;

                    delta.pHat.x -= w_depth * depth_err * dx;
                    delta.pHat.y -= w_depth * depth_err * dy;
                }

                // dist regularization
                float di = g.pos.z + dx * g.pHat.x + dy * g.pHat.y;
                float dd = di - depth;

                float dist_coeff = w_dist * alpha * T * dd;

                delta.depth -= dist_coeff;
                delta.meanImg.x -= dist_coeff * g.pHat.x;
                delta.meanImg.y -= dist_coeff * g.pHat.y;

                delta.pHat.x -= dist_coeff * dx;
                delta.pHat.y -= dist_coeff * dy;

                T = test_T;
            }
        }

        // ============================================================
        // 9. Acumulación global (atomics)
        // ============================================================
        if (valid_splat && delta.n > 0)
        {
            // TEST: divido por n para normalizar gradientes acumulados
            float inv_n = 1.0f / float(delta.n);
            atomicAdd(&delta_gaussians[gId].depth, delta.depth * inv_n);
            atomicAdd(&delta_gaussians[gId].pHat.x, delta.pHat.x * inv_n);
            atomicAdd(&delta_gaussians[gId].pHat.y, delta.pHat.y * inv_n);
            atomicAdd(&delta_gaussians[gId].meanImg.x, delta.meanImg.x * inv_n);
            atomicAdd(&delta_gaussians[gId].meanImg.y, delta.meanImg.y * inv_n);
            atomicAdd(&delta_gaussians[gId].invSigmaImg.x, delta.invSigmaImg.x * inv_n);
            atomicAdd(&delta_gaussians[gId].invSigmaImg.y, delta.invSigmaImg.y * inv_n);
            atomicAdd(&delta_gaussians[gId].invSigmaImg.z, delta.invSigmaImg.z * inv_n);
            atomicAdd(&delta_gaussians[gId].color.x, delta.color.x * inv_n);
            atomicAdd(&delta_gaussians[gId].color.y, delta.color.y * inv_n);
            atomicAdd(&delta_gaussians[gId].color.z, delta.color.z * inv_n);
            atomicAdd(&delta_gaussians[gId].alpha, delta.alpha * inv_n);
            atomicAdd(&delta_gaussians[gId].n, 1);  // Contar contribuciones
        }
    }

    // ============================================================================
    // 8. DELTAS 3D Y BACKPROP GEOMETRICO
    // ============================================================================

    // Equivalente a computeDeltaGaussians3D_kernel en vigs-fusion
    __global__ void computeDeltaGaussians3D_kernel(
        DeltaGaussian3D *delta_gaussians_3d,
        const float4 *positions,
        const float4 *scales,
        const float4 *orientations,
        const float4 *colors,
        const float *alphas,
        const DeltaGaussian2D *delta_gaussians_2d, // Grad de color, depth, media2d, cov2d, pHat, depth
        Pose camera_pose,
        IntrinsicParameters intrinsics,
        float lambda_iso,
        int n_gaussians)
    {
        // ============================================================
        // 1. Thread index
        // ============================================================
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= n_gaussians) return;

        // ============================================================
        // 2. Fetch input gradients (2D)
        // ============================================================
        DeltaGaussian2D delta_2d = delta_gaussians_2d[idx];
        DeltaGaussian3D delta_3d;

        // Si no contribuye, no hay gradiente
        if (delta_2d.n == 0)
        {
            delta_gaussians_3d[idx].n = 0;
            return;
        }

        delta_3d.n = delta_2d.n;

        // Learning rate adaptativo (menos update si más contribuciones)
        const float eta_n = 1.0f / (5.0f + float(delta_2d.n));



        // Lista de gradientes a calcular:
        // Color: dL/dc_n = alpha_n * T_n
        // Opacidad: dL/dalpha_n = c_n * T_n - S_n
        // Media: dL/du_n
        // d

        // Color: dL/dc_n = eta_n * c_n
        // Opacidad: dL/dalpha_n = eta_n * alpha_n
        // Covarianza: dL/dSigma_n = - 0.5 * Sigma'^{-1} * delta_n * delta_n^T * Sigma'^{-1} (con delta_n = (x-u, y-v))
        // Media: dL/dmu = antiproyectar

        // Pose: dL/dt = dL/dmu * dmu/dt + dL/dSigma * dSigma/dt
        // Rotacion: dL/dR = dL/dM * S^T
        // escala: dL/dS = R^T dL/dM
        
        // Gradiente isotropico: dL/dSigma_iso = lambda_iso * (sigma_x - sigma_y)

        // ============================================================
        // 3. Gradientes triviales (color y alpha)
        // ============================================================
        delta_3d.color = eta_n * delta_2d.color;
        delta_3d.alpha = eta_n * delta_2d.alpha;

        // ============================================================
        // 4. Setup geométrico
        // ============================================================

        // Media en world
        float4 pos4 = positions[idx];
        Eigen::Vector3f mu_w(pos4.x, pos4.y, pos4.z);

        // Pose cámara
        Eigen::Map<const Eigen::Vector3f> c_w((float *)&camera_pose.position);
        Eigen::Map<const Eigen::Quaternionf> q_cw((float *)&camera_pose.orientation);

        // Media en cámara
        Eigen::Vector3f mu_c = q_cw.inverse() * (mu_w - c_w);

        // Rotación gaussiana
        Eigen::Map<const Eigen::Quaternionf> q_g((float *)&orientations[idx]);
        Eigen::Matrix3f R_g = q_g.toRotationMatrix();

        // Escala
        float4 scale4 = scales[idx];
        Eigen::Vector3f s(scale4.x, scale4.y, scale4.z);

        // ============================================================
        // 5. Jacobiano de proyección
        // ============================================================
        Eigen::Matrix<float, 3, 3> J{
            {intrinsics.f.x / mu_c.z(), 0.f, -intrinsics.f.x * mu_c.x() / (mu_c.z()*mu_c.z())},
            {0.f, intrinsics.f.y / mu_c.z(), -intrinsics.f.y * mu_c.y() / (mu_c.z()*mu_c.z())},
            {0.f, 0.f, 1.f}
        };

        // Rotación world → camera
        Eigen::Matrix3f R_cw = q_cw.inverse().toRotationMatrix();
        
        // ============================================================
        // 6. Forward intermedio (covarianza proyectada)
        // ============================================================
        // T = J * R_cw, Sigma' = T * (R_g S^2 R_g^T) * T^T
        const Eigen::Matrix<float, 3, 3> T = J * R_cw;
        
        const Eigen::Matrix3f RS = R_g * s.asDiagonal();
        const Eigen::Matrix<float, 3, 3> M = T * RS;

        Eigen::Matrix3f Sigma_prime = M * M.transpose();
        Eigen::Matrix3f Sigma_prime_inv = Sigma_prime.inverse();
        // Revisar si se puede mejorar para no invertir en cada iteracion.

        
        // ============================================================
        // 7. Gradiente de la media 3D (desde reproyección)
        // ============================================================
        // dL/dmu_3d = R_cw^T J^T dL/dmu_2d + dL/dz * [mu_c.x/z, mu_c.y/z, 1]^T

        Eigen::Vector3f dL_dmu_2d(delta_2d.meanImg.x,
                                delta_2d.meanImg.y,
                                0.f);

        Eigen::Vector3f dL_dmu_3d =
            q_cw * (
                J.transpose() * dL_dmu_2d +
                delta_2d.depth * Eigen::Vector3f(
                    mu_c.x()/mu_c.z(),
                    mu_c.y()/mu_c.z(),
                    1.f)
            );

        // ============================================================
        // 8. Término p_hat (distorsión angular)
        // ============================================================

        Eigen::Vector2f dL_dp_hat(delta_2d.pHat.x, delta_2d.pHat.y);

        float norm2_mucam = mu_c.dot(mu_c) + 1e-7f;
        float norm = sqrtf(norm2_mucam);
        float inv_norm3 = 1.f / (norm2_mucam * norm + 1e-12f);

        Eigen::Vector3f dp_hat_d_mu(
            -mu_c.x()*mu_c.z()*inv_norm3,
            -mu_c.y()*mu_c.z()*inv_norm3,
            (mu_c.x()*mu_c.x() + mu_c.y()*mu_c.y())*inv_norm3
        );

        dL_dmu_3d += q_cw * (
            dp_hat_d_mu *
            ((Sigma_prime_inv(2,0)*dL_dp_hat.x() +
            Sigma_prime_inv(2,1)*dL_dp_hat.y()) /
            Sigma_prime_inv(2,2))
        );

        // ============================================================
        // 9. Gradiente de covarianza 2D (desde inversa)
        // ============================================================        

        // dL/dSigma' a partir de dL/dSigma'^{-1} (conica 2D)
        float sigma_xx = Sigma_prime(0, 0) + 1e-3f;
        float sigma_xy = Sigma_prime(1, 0);
        float sigma_yy = Sigma_prime(1, 1) + 1e-3f;

        float denom = sigma_xx * sigma_yy - sigma_xy * sigma_xy + 1e-8f; 
        float denom2inv = 1.0f / ((denom * denom) + 1e-7f);

        float3 dL_dSigmaInv2d = delta_2d.invSigmaImg;

        float dL_dSigma_xx = denom2inv * (-sigma_yy * sigma_yy * dL_dSigmaInv2d.x 
                                          + 2 * sigma_xy * sigma_yy * dL_dSigmaInv2d.y 
                                          + (denom - sigma_xx * sigma_yy) * dL_dSigmaInv2d.z);

        float dL_dSigma_yy = denom2inv * (-sigma_xx * sigma_xx * dL_dSigmaInv2d.z 
                                          + 2 * sigma_xx * sigma_xy * dL_dSigmaInv2d.y  
                                          + (denom - sigma_xx * sigma_yy) * dL_dSigmaInv2d.x);

        float dL_dSigma_xy = denom2inv * 2 * (sigma_xy * sigma_yy * dL_dSigmaInv2d.x 
                                              - (denom + 2 * sigma_xy * sigma_xy) * dL_dSigmaInv2d.y 
                                              + sigma_xx * sigma_xy * dL_dSigmaInv2d.z);

        // ============================================================
        // 10. Backprop hacia T (covarianza proyectada)
        // ============================================================

        const Eigen::Matrix3f Sigma_3d = RS * (RS.transpose());

        // dL/dT desde dL/dSigma' (producto T * Sigma_3d * T^T)
        float dL_dT00 = 2 * (T(0, 0) * Sigma_3d(0, 0) + T(0, 1) * Sigma_3d(0, 1) + T(0, 2) * Sigma_3d(0, 2)) * dL_dSigma_xx +
                        (T(1, 0) * Sigma_3d(0, 0) + T(1, 1) * Sigma_3d(0, 1) + T(1, 2) * Sigma_3d(0, 2)) * dL_dSigma_xy;
        float dL_dT01 = 2 * (T(0, 0) * Sigma_3d(1, 0) + T(0, 1) * Sigma_3d(1, 1) + T(0, 2) * Sigma_3d(1, 2)) * dL_dSigma_xx +
                        (T(1, 0) * Sigma_3d(1, 0) + T(1, 1) * Sigma_3d(1, 1) + T(1, 2) * Sigma_3d(1, 2)) * dL_dSigma_xy;
        float dL_dT02 = 2 * (T(0, 0) * Sigma_3d(2, 0) + T(0, 1) * Sigma_3d(2, 1) + T(0, 2) * Sigma_3d(2, 2)) * dL_dSigma_xx +
                        (T(1, 0) * Sigma_3d(2, 0) + T(1, 1) * Sigma_3d(2, 1) + T(1, 2) * Sigma_3d(2, 2)) * dL_dSigma_xy;
        float dL_dT10 = 2 * (T(1, 0) * Sigma_3d(0, 0) + T(1, 1) * Sigma_3d(0, 1) + T(1, 2) * Sigma_3d(0, 2)) * dL_dSigma_yy +
                        (T(0, 0) * Sigma_3d(0, 0) + T(0, 1) * Sigma_3d(0, 1) + T(0, 2) * Sigma_3d(0, 2)) * dL_dSigma_xy;
        float dL_dT11 = 2 * (T(1, 0) * Sigma_3d(1, 0) + T(1, 1) * Sigma_3d(1, 1) + T(1, 2) * Sigma_3d(1, 2)) * dL_dSigma_yy +
                        (T(0, 0) * Sigma_3d(1, 0) + T(0, 1) * Sigma_3d(1, 1) + T(0, 2) * Sigma_3d(1, 2)) * dL_dSigma_xy;
        float dL_dT12 = 2 * (T(1, 0) * Sigma_3d(2, 0) + T(1, 1) * Sigma_3d(2, 1) + T(1, 2) * Sigma_3d(2, 2)) * dL_dSigma_yy +
                        (T(0, 0) * Sigma_3d(2, 0) + T(0, 1) * Sigma_3d(2, 1) + T(0, 2) * Sigma_3d(2, 2)) * dL_dSigma_xy;

        // ============================================================
        // 11. Backprop extra por p_hat (via Sigma^{-1})
        // ============================================================

        // Aporte de p_hat a dL/dSigma'^{-1} y dL/dSigma'
        float norm_mu_c = (mu_c.x() * mu_c.x() + mu_c.y() * mu_c.y() + mu_c.z() * mu_c.z()) + 1e-7f;
        float z_over_t = mu_c.z() / norm_mu_c;

        Eigen::Matrix3f dL_dSigmaInv;
        dL_dSigmaInv << 0.f, 0.f, 0.5f * z_over_t * dL_dp_hat.x() / Sigma_prime_inv(2, 2),
                        0.f, 0.f, 0.5f * z_over_t * dL_dp_hat.y() / Sigma_prime_inv(2, 2),
                        0.5f * z_over_t * dL_dp_hat.x() / Sigma_prime_inv(2, 2),
                        0.5f * z_over_t * dL_dp_hat.y() / Sigma_prime_inv(2, 2), 
                        -z_over_t * (dL_dp_hat.x() * Sigma_prime_inv(2, 0) 
                                     + dL_dp_hat.y() * Sigma_prime_inv(2, 1)) / 
                                     (Sigma_prime_inv(2, 2) * Sigma_prime_inv(2, 2));

        Eigen::Matrix3f dL_dSigma_prime = -Sigma_prime_inv * dL_dSigmaInv * Sigma_prime_inv;


        Eigen::Matrix3f TSigma3D = T * Sigma_3d;
        Eigen::Matrix3f Sigma3DT = Sigma_3d * T.transpose();
        dL_dT00 += TSigma3D.col(0).dot(dL_dSigma_prime.col(0)) + Sigma3DT.row(0).dot(dL_dSigma_prime.row(0));
        dL_dT01 += TSigma3D.col(1).dot(dL_dSigma_prime.col(0)) + Sigma3DT.row(1).dot(dL_dSigma_prime.row(0));
        dL_dT02 += TSigma3D.col(2).dot(dL_dSigma_prime.col(0)) + Sigma3DT.row(2).dot(dL_dSigma_prime.row(0));
        dL_dT10 += TSigma3D.col(0).dot(dL_dSigma_prime.col(1)) + Sigma3DT.row(0).dot(dL_dSigma_prime.row(1));
        dL_dT11 += TSigma3D.col(1).dot(dL_dSigma_prime.col(1)) + Sigma3DT.row(1).dot(dL_dSigma_prime.row(1));
        dL_dT12 += TSigma3D.col(2).dot(dL_dSigma_prime.col(1)) + Sigma3DT.row(2).dot(dL_dSigma_prime.row(1));

        // dL/dJ y dL/dmu_c (via dJ/dmu_c)
        float dL_dJ00 = R_cw(0, 0) * dL_dT00 + R_cw(0, 1) * dL_dT01 + R_cw(0, 2) * dL_dT02;
        float dL_dJ02 = R_cw(2, 0) * dL_dT00 + R_cw(2, 1) * dL_dT01 + R_cw(2, 2) * dL_dT02;
        float dL_dJ11 = R_cw(1, 0) * dL_dT10 + R_cw(1, 1) * dL_dT11 + R_cw(1, 2) * dL_dT12;
        float dL_dJ12 = R_cw(2, 0) * dL_dT10 + R_cw(2, 1) * dL_dT11 + R_cw(2, 2) * dL_dT12;

        // ============================================================
        // 12. Backprop hasta mu_c via J
        // ============================================================

        float inv_z = 1.f / mu_c.z();
        float inv_z2 = inv_z * inv_z;
        float inv_z3 = inv_z2 * inv_z;

        float dL_dmu_cx = -intrinsics.f.x * inv_z2 * dL_dJ02;
        float dL_dmu_cy = -intrinsics.f.y * inv_z2 * dL_dJ12;
        float dL_dmu_cz = -intrinsics.f.x * inv_z2 * dL_dJ00 - intrinsics.f.y * inv_z2 * dL_dJ11 + (2 * intrinsics.f.x * mu_c.x()) * inv_z3 * dL_dJ02 + (2 * intrinsics.f.y * mu_c.y()) * inv_z3 * dL_dJ12;

        dL_dmu_3d += q_cw * Eigen::Vector3f(dL_dmu_cx, dL_dmu_cy, dL_dmu_cz);

        // ============================================================
        // 13. Output: posición
        // ============================================================
        delta_3d.position.x = eta_n * dL_dmu_3d.x();
        delta_3d.position.y = eta_n * dL_dmu_3d.y();
        delta_3d.position.z = eta_n * dL_dmu_3d.z();

        // ============================================================
        // 14. Gradientes de escala y rotación
        // ============================================================

        // dL/dSigma_3d = T^T * dL/dSigma' * T
        Eigen::Matrix3f dL_dSigma_3d;
        dL_dSigma_3d(0, 0) = (T(0, 0) * T(0, 0) * dL_dSigma_xx 
                              + T(0, 0) * T(1, 0) * dL_dSigma_xy 
                              + T(1, 0) * T(1, 0) * dL_dSigma_yy);

        dL_dSigma_3d(1, 1) = (T(0, 1) * T(0, 1) * dL_dSigma_xx 
                              + T(0, 1) * T(1, 1) * dL_dSigma_xy 
                              + T(1, 1) * T(1, 1) * dL_dSigma_yy);

        dL_dSigma_3d(2, 2) = (T(0, 2) * T(0, 2) * dL_dSigma_xx 
                              + T(0, 2) * T(1, 2) * dL_dSigma_xy 
                              + T(1, 2) * T(1, 2) * dL_dSigma_yy);

        dL_dSigma_3d(1, 0) = dL_dSigma_3d(0, 1) = T(0, 0) * T(0, 1) * dL_dSigma_xx 
                                                  + 0.5f * (T(0, 0) * T(1, 1) 
                                                  + T(0, 1) * T(1, 0)) * dL_dSigma_xy 
                                                  + T(1, 0) * T(1, 1) * dL_dSigma_yy;

        dL_dSigma_3d(2, 0) = dL_dSigma_3d(0, 2) = T(0, 0) * T(0, 2) * dL_dSigma_xx 
                                                  + 0.5f * (T(0, 0) * T(1, 2) 
                                                  + T(0, 2) * T(1, 0)) * dL_dSigma_xy 
                                                  + T(1, 0) * T(1, 2) * dL_dSigma_yy;

        dL_dSigma_3d(2, 1) = dL_dSigma_3d(1, 2) = T(0, 2) * T(0, 1) * dL_dSigma_xx 
                                                  + 0.5f * (T(0, 1) * T(1, 2) 
                                                  + T(0, 2) * T(1, 1)) * dL_dSigma_xy 
                                                  + T(1, 1) * T(1, 2) * dL_dSigma_yy;

        dL_dSigma_3d += T.transpose() * dL_dSigma_prime * T;

        // dL/dS y dL/dR con Sigma_3d = R S^2 R^T
        Eigen::Matrix3f dL_dM = 2.0f * RS.transpose() * dL_dSigma_3d;
        Eigen::Vector3f dL_ds = Eigen::Vector3f(R_g(0, 0) * dL_dM(0, 0) + R_g(1, 0) * dL_dM(0, 1) + R_g(2, 0) * dL_dM(0, 2),
                                                R_g(0, 1) * dL_dM(1, 0) + R_g(1, 1) * dL_dM(1, 1) + R_g(2, 1) * dL_dM(1, 2),
                                                R_g(0, 2) * dL_dM(2, 0) + R_g(1, 2) * dL_dM(2, 1) + R_g(2, 2) * dL_dM(2, 2));

        const float4 s_val4 = scales[idx];
        const float3 s_val = make_float3(s_val4.x, s_val4.y, s_val4.z);

        // Regularizacion isotropica: dL_iso/ds
        float mean_s = (s_val.x + s_val.y + s_val.z) / 3.f;
        float3 dl_iso = make_float3(s_val.x - mean_s,
                        s_val.y - mean_s,
                        s_val.z - mean_s);
        dL_ds -= lambda_iso * (1.f / 3.f) * Eigen::Vector3f(2.f * dl_iso.x - dl_iso.y - dl_iso.z, -dl_iso.x + 2.f * dl_iso.y - dl_iso.z, -dl_iso.x - dl_iso.y + 2.f * dl_iso.z);

        // dL/dR en espacio tangente: -sum_i R_i x dL/dM_i
        dL_dM.row(0) *= s_val.x;
        dL_dM.row(1) *= s_val.y;
        dL_dM.row(2) *= s_val.z;

        Eigen::Vector3f dL_dtheta = -eta_n * (R_g.row(0).cross(dL_dM.col(0)) + R_g.row(1).cross(dL_dM.col(1)) + R_g.row(2).cross(dL_dM.col(2)));

        // ============================================================
        // 15. Output final
        // ============================================================
        delta_3d.orientation.x = dL_dtheta.x();
        delta_3d.orientation.y = dL_dtheta.y();
        delta_3d.orientation.z = dL_dtheta.z();

        delta_3d.scale.x = eta_n * dL_ds.x();
        delta_3d.scale.y = eta_n * dL_ds.y();
        delta_3d.scale.z = eta_n * dL_ds.z();

        // TEST: trabo los parametros
        //const float grad_clip = 10.0f;
        //delta_3d.position = clampFloat3(delta_3d.position, -grad_clip, grad_clip);
        //delta_3d.scale = clampFloat3(delta_3d.scale, -grad_clip, grad_clip);
        //delta_3d.orientation = clampFloat3(delta_3d.orientation, -grad_clip, grad_clip);
        //delta_3d.color = clampFloat3(delta_3d.color, -grad_clip, grad_clip);
        //delta_3d.alpha = fminf(grad_clip, fmaxf(-grad_clip, delta_3d.alpha));

        delta_gaussians_3d[idx] = delta_3d;
    }

    // ============================================================================
    // 9. ACTUALIZACION ADAM DE LOS PARAMETROS DE GAUSSIANAS
    // ============================================================================

    __inline__ __device__ float adamStep(float &m,
                                         float &v,
                                         float grad,
                                         const float eta,
                                         const float alpha1,
                                         const float beta1,
                                         const float beta1t,
                                         const float alpha2,
                                         const float beta2,
                                         const float beta2t,
                                         const float epsilon)
    {
        // Actualizamos momento y varianza
        m = alpha1 * grad + beta1 * m;
        v = alpha2 * grad * grad + beta2 * v;
        
        // Aplicamos la correccion
        float m_hat = beta1t * m;
        float v_hat = beta2t * v;
        
        // Devolvemos el siguiente paso
        return eta * m_hat * __frsqrt_rn(v_hat + epsilon);
    }

    __inline__ __device__ float3 adamStep(float3 &m,
                                          float3 &v,
                                          float3 grad,
                                          const float eta,
                                          const float alpha1,
                                          const float beta1,
                                          const float beta1t,
                                          const float alpha2,
                                          const float beta2,
                                          const float beta2t,
                                          const float epsilon)
    {
        float3 res;
        
        res.x = adamStep(m.x, v.x, grad.x,
                        eta,
                        alpha1, beta1, beta1t,
                        alpha2, beta2, beta2t,
                        epsilon);
        res.y = adamStep(m.y, v.y, grad.y,
                        eta,
                        alpha1, beta1, beta1t,
                        alpha2, beta2, beta2t,
                        epsilon);
        res.z = adamStep(m.z, v.z, grad.z,
                        eta,
                        alpha1, beta1, beta1t,
                        alpha2, beta2, beta2t,
                        epsilon);
        
        return res;
    }

    __global__ void updateGaussiansParametersAdam_kernel(
        float4 *positions,
        float4 *scales,
        float4 *orientations,
        float4 *colors,
        float *alphas,
        AdamStateGaussian3D *adam_states,
        const DeltaGaussian3D *deltas_3d,
        float adam_eta,
        float adam_beta1,
        float adam_beta2,
        float adam_eps,
        int n_gaussians)
    {
        // ============================================================
        // 1. Thread index
        // ============================================================
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= n_gaussians) {
            return;
        }

        // ============================================================
        // 2. Gradiente por gaussiana
        // ============================================================
        DeltaGaussian3D delta = deltas_3d[idx];
        if (delta.n == 0) {
            return; 
        }

        // Limita updates extremos para evitar saturaciones bruscas en color y opacidad.
        const float grad_clip = 10.0f;
        delta.position = clampFloat3(delta.position, -grad_clip, grad_clip);
        delta.scale = clampFloat3(delta.scale, -grad_clip, grad_clip);
        delta.orientation = clampFloat3(delta.orientation, -grad_clip, grad_clip);
        delta.color = clampFloat3(delta.color, -grad_clip, grad_clip);
        delta.alpha = fminf(grad_clip, fmaxf(-grad_clip, delta.alpha));

        AdamStateGaussian3D adam_state = adam_states[idx];

        // ============================================================
        // 3. Paso temporal del optimizador
        // ============================================================
        // Avanzamos un paso
        adam_state.k += 1.f;

        // ============================================================
        // 4. Hiperparametros del paso Adam
        // ============================================================
        // Hiperparametro alpha = 1 - beta
        float alpha1 = 1.f - adam_beta1;
        float alpha2 = 1.f - adam_beta2;

        // Correccion de bias: 1 / (1 - beta^t)
        float beta1t = __frcp_rn(1.f - __powf(adam_beta1, adam_state.k));
        float beta2t = __frcp_rn(1.f - __powf(adam_beta2, adam_state.k));

        // Cargamos parametros
        float4 position4 = positions[idx];
        float3 position = make_float3(position4.x, position4.y, position4.z);
        float4 scale4 = scales[idx];
        float3 scale = make_float3(scale4.x, scale4.y, scale4.z);
        float4 orientation = orientations[idx];
        float4 color4 = colors[idx];
        float3 color = make_float3(color4.x, color4.y, color4.z);
        float alpha = alphas[idx];

        // La posicion se actualiza directamente
        position += adamStep(adam_state.m_position,
                            adam_state.v_position,
                            delta.position,
                            adam_eta,
                            alpha1, adam_beta1, beta1t,
                            alpha2, adam_beta2, beta2t,
                            adam_eps);

        // La orientacion se actualiza en el espacio tangente de SO(3) para mantener la normalizacion del cuaternion
        float3 dq = adamStep(adam_state.m_orientation,
                            adam_state.v_orientation,
                            delta.orientation,
                            adam_eta,
                            alpha1, adam_beta1, beta1t,
                            alpha2, adam_beta2, beta2t,
                            adam_eps);
        Eigen::Map<Eigen::Quaternionf> q_gauss((float *)&orientation);
        q_gauss = q_gauss * Eigen::Quaternionf(1.f, 0.5f * dq.x, 0.5f * dq.y, 0.5f * dq.z);
        q_gauss.normalize();

        scale += adamStep(adam_state.m_scale,
                        adam_state.v_scale,
                        delta.scale,
                        adam_eta,
                        alpha1, adam_beta1, beta1t,
                        alpha2, adam_beta2, beta2t,
                        adam_eps);

        // El color se actualiza directamente
        color += adamStep(adam_state.m_color,
                         adam_state.v_color,
                         delta.color,
                         adam_eta,
                         alpha1, adam_beta1, beta1t,
                         alpha2, adam_beta2, beta2t,
                         adam_eps);

        // La opacidad se actualiza en el espacio logit para mantenerla en el rango (0, 1)
        float dalpha = adamStep(adam_state.m_alpha,
                               adam_state.v_alpha,
                               delta.alpha,
                               adam_eta,
                               alpha1, adam_beta1, beta1t,
                               alpha2, adam_beta2, beta2t,
                               adam_eps);

        float denom = max(1e-6f, alpha * (1.f - alpha));
        float alpha_s = __logf(alpha / (1.f - alpha)) + dalpha / denom;
        alpha = max(0.01f, min(0.99f, 1.f / (1.f + evaluate_exponential(-alpha_s))));

        //Actualizamos los parametros y clampeamos para mantener valores utiles e interpretables
        position4 = make_float4(position.x, position.y, position.z, 1.0f);
        float3 clamped_scale = vec3Max(make_float3(0.001f, 0.001f, 0.001f), scale);
        scale4 = make_float4(clamped_scale.x, clamped_scale.y, clamped_scale.z, 0.0f);
        float3 clamped_color = vec3Min(make_float3(1.f, 1.f, 1.f), vec3Max(make_float3(0.f, 0.f, 0.f), color));
        color4 = make_float4(clamped_color.x, clamped_color.y, clamped_color.z, 0.0f);
        positions[idx] = position4;
        scales[idx] = scale4;
        orientations[idx] = orientation;
        colors[idx] = color4;
        alphas[idx] = alpha;


        adam_states[idx] = adam_state;
    }

}