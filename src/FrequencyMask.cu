// ============================================================
// 1. PROPIOS
// ============================================================
#include <f_vigs_slam/FrequencyMask.cuh>
#include <f_vigs_slam/RepresentationClasses.hpp>

// ============================================================
// 2. CUDA / CUFFT
// ============================================================
#include <cuda_runtime.h>
#include <cufft.h>

// ============================================================
// 3. OPENCV
// ============================================================
#include <opencv2/imgproc.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudaimgproc.hpp>

// ============================================================
// 4. THRUST
// ============================================================
#include <thrust/device_vector.h>
#include <thrust/extrema.h>
#include <thrust/fill.h>
#include <thrust/reduce.h>
#include <thrust/copy.h>

// ============================================================
// 5. STL
// ============================================================
#include <algorithm>
#include <cmath>
#include <iostream>
#include <memory>

namespace f_vigs_slam
{
// ============================================================
// 1. VERIFICACIONES
// ============================================================
bool checkCuda(cudaError_t status, const char *where)
{
    if (status == cudaSuccess)
    {
        return true;
    }

    std::cerr << "[GSSlam][ERROR CUDA] " << where
              << " failed with " << cudaGetErrorString(status)
              << std::endl;
    return false;
}

bool checkCufft(cufftResult status, const char *where)
{
    if (status == CUFFT_SUCCESS)
    {
        return true;
    }

    std::cerr << "[GSSlam][ERROR cuFFT] " << where
              << " failed with code " << static_cast<int>(status)
              << std::endl;
    return false;
}

bool checkLaunch(const char *where)
{
    return checkCuda(cudaGetLastError(), where);
}

// ============================================================
// 2. CONTEXTO PERSISTENTE (PLANES + BUFFERS + TEXTURAS)
// ============================================================
struct FrequencyMaskContext
{
    int width = 0;
    int height = 0;

    int n = 0;
    int width_half = 0;
    int n_freq = 0;

    cufftHandle plan_r2c = 0;
    cufftHandle plan_c2r = 0;

    thrust::device_vector<float> d_real;
    thrust::device_vector<float> d_abs;
    thrust::device_vector<unsigned char> d_u8;
    thrust::device_vector<cufftComplex> d_freq;
    thrust::device_vector<unsigned int> d_hist_u32;
    thrust::device_vector<unsigned char> d_triangle_thr;

    cv::cuda::GpuMat high_mask_u8_gpu;
    cv::cuda::GpuMat low_mask_u8_gpu;

    // Textura persistente para Sobel/Laplacian
    cv::cuda::GpuMat gray_u8_gpu;
    std::unique_ptr<Texture<unsigned char>> gray_tex_wrapper = nullptr;

    ~FrequencyMaskContext()
    {
        // gray_tex_wrapper destruido automáticamente por unique_ptr
        if (plan_r2c != 0)
        {
            cufftDestroy(plan_r2c);
        }
        if (plan_c2r != 0)
        {
            cufftDestroy(plan_c2r);
        }
    }

    bool ensureGeometry(int w, int h)
    {
        if (w <= 0 || h <= 0)
        {
            return false;
        }

        if (w == width && h == height && plan_r2c != 0 && plan_c2r != 0)
        {
            return true;
        }

        width = w;
        height = h;

        n = width * height;
        width_half = width / 2 + 1;
        n_freq = width_half * height;

        d_real.resize(n);
        d_abs.resize(n);
        d_u8.resize(n);
        d_freq.resize(n_freq);
        d_hist_u32.resize(256);
        d_triangle_thr.resize(1);

        high_mask_u8_gpu.create(height, width, CV_8U);
        low_mask_u8_gpu.create(height, width, CV_8U);

        if (plan_r2c != 0)
        {
            cufftDestroy(plan_r2c);
            plan_r2c = 0;
        }
        if (plan_c2r != 0)
        {
            cufftDestroy(plan_c2r);
            plan_c2r = 0;
        }

        if (!checkCufft(cufftPlan2d(&plan_r2c, height, width, CUFFT_R2C),
                        "FrequencyMaskContext::cufftPlan2d R2C"))
        {
            return false;
        }

        if (!checkCufft(cufftPlan2d(&plan_c2r, height, width, CUFFT_C2R),
                        "FrequencyMaskContext::cufftPlan2d C2R"))
        {
            cufftDestroy(plan_r2c);
            plan_r2c = 0;
            return false;
        }

        // Crear buffer persistente en escala de grises (para Sobel/Laplacian con acceso por textura)
        gray_u8_gpu.create(height, width, CV_8U);

        // Crear envoltorio de textura (RAII gestiona su ciclo de vida)
        gray_tex_wrapper = std::make_unique<Texture<unsigned char>>(gray_u8_gpu, cudaFilterModePoint);
        if (!gray_tex_wrapper || gray_tex_wrapper->getTextureObject() == 0)
        {
            std::cerr << "[GSSlam][ERROR] FrequencyMaskContext::ensureGeometry - "
                      << "Failed to create gray_tex_wrapper" << std::endl;
            gray_tex_wrapper = nullptr;
            return false;
        }

        return true;
    }
};

FrequencyMaskContext &getFrequencyMaskContext()
{
    static FrequencyMaskContext context;
    return context;
}

// ============================================================
// 3. KERNELS
// ============================================================
__global__ void copyU8ToFloatLinearKernel(const unsigned char *src,
                                          size_t src_step,
                                          float *dst,
                                          int width,
                                          int height)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height)
    {
        return;
    }

    const unsigned char *row = src + static_cast<size_t>(y) * src_step;
    dst[y * width + x] = static_cast<float>(row[x]);
}

__global__ void applyHighpassR2CKernel(cufftComplex *spectrum,
                                       int width,
                                       int height,
                                       float inv_two_sigma2)
{
    // M_h: filtro de alta frecuencia en el dominio espectral.
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    const int width_half = width / 2 + 1;

    if (x >= width_half || y >= height)
    {
        return;
    }

    const float fx = static_cast<float>(x);
    const int y_folded = min(y, height - y);
    const float fy = static_cast<float>(y_folded);
    const float d2 = fx * fx + fy * fy;
    const float w = 1.0f - expf(-d2 * inv_two_sigma2);

    cufftComplex &c = spectrum[y * width_half + x];
    c.x *= w;
    c.y *= w;
}

__global__ void absScaleKernel(const float *src,
                               float *dst,
                               int n,
                               float scale)
{
    // Magnitud espacial normalizada antes del umbral triangular.
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
    {
        return;
    }

    dst[i] = fabsf(src[i] * scale);
}

__global__ void normalizeFloatToU8Kernel(const float *src,
                                         unsigned char *dst,
                                         int n,
                                         float min_val,
                                         float inv_range)
{
    // Prepara la señal para el histograma/threshold final en bytes.
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
    {
        return;
    }

    float v = (src[i] - min_val) * inv_range;
    v = fminf(fmaxf(v, 0.0f), 255.0f);
    dst[i] = static_cast<unsigned char>(v);
}

__global__ void thresholdLinearToGpuMatKernel(const unsigned char *src_linear,
                                              unsigned char *dst_u8,
                                              size_t dst_step,
                                              int width,
                                              int height,
                                              const unsigned char *thr_ptr)
{
    // M_h queda binarizada; M_l se obtiene como complemento fuera de este kernel.
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height)
    {
        return;
    }

    const unsigned char thr = thr_ptr[0];
    const unsigned char v = src_linear[y * width + x];
    unsigned char *row = dst_u8 + static_cast<size_t>(y) * dst_step;
    row[x] = (v > thr) ? 255u : 0u;
}

__global__ void computeHistogram256Kernel(const unsigned char *src_linear,
                                          int n,
                                          unsigned int *hist256)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
    {
        return;
    }

    atomicAdd(&hist256[src_linear[i]], 1u);
}

__global__ void sobelConvolutionKernel(cudaTextureObject_t gray_tex,
                                       float *dst_magnitude,
                                       int width,
                                       int height)
{
    // Kernels de Sobel:
    // Gx = [-1, 0, 1; -2, 0, 2; -1, 0, 1]
    // Gy = [-1, -2, -1; 0, 0, 0; 1, 2, 1]
    // Magnitud: S = sqrt(Sx^2 + Sy^2)

    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < 1 || x >= width - 1 || y < 1 || y >= height - 1)
    {
        if (x < width && y < height)
        {
            dst_magnitude[y * width + x] = 0.0f;
        }
        return;
    }

    // Leer vecindario 3x3 mediante textura (clamp automático en bordes)
    float p[9];
    for (int dy = -1; dy <= 1; ++dy)
    {
        for (int dx = -1; dx <= 1; ++dx)
        {
            p[(dy + 1) * 3 + (dx + 1)] = static_cast<float>(tex2D<unsigned char>(gray_tex, x + dx, y + dy));
        }
    }

    // Sobel X: [-1, 0, 1; -2, 0, 2; -1, 0, 1]
    const float sx = -p[0] + p[2] - 2.0f * p[3] + 2.0f * p[5] - p[6] + p[8];

    // Sobel Y: [-1, -2, -1; 0, 0, 0; 1, 2, 1]
    const float sy = -p[0] - 2.0f * p[1] - p[2] + p[6] + 2.0f * p[7] + p[8];

    // Magnitud
    const float magnitude = sqrtf(sx * sx + sy * sy);
    dst_magnitude[y * width + x] = magnitude;
}

__global__ void laplacianConvolutionKernel(cudaTextureObject_t gray_tex,
                                           float *dst_response,
                                           int width,
                                           int height)
{
    // Kernel Laplaciano (5 puntos, segunda derivada):
    // L = [0, 1, 0; 1, -4, 1; 0, 1, 0]
    // Respuesta: L_I = |I * L|

    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < 1 || x >= width - 1 || y < 1 || y >= height - 1)
    {
        if (x < width && y < height)
        {
            dst_response[y * width + x] = 0.0f;
        }
        return;
    }

    // Leer stencil de 5 puntos mediante textura (clamp automático en bordes)
    const float center = static_cast<float>(tex2D<unsigned char>(gray_tex, x, y));
    const float up = static_cast<float>(tex2D<unsigned char>(gray_tex, x, y - 1));
    const float down = static_cast<float>(tex2D<unsigned char>(gray_tex, x, y + 1));
    const float left = static_cast<float>(tex2D<unsigned char>(gray_tex, x - 1, y));
    const float right = static_cast<float>(tex2D<unsigned char>(gray_tex, x + 1, y));

    // Respuesta Laplaciana: center*(-4) + (up + down + left + right)*1
    const float response = -4.0f * center + up + down + left + right;

    dst_response[y * width + x] = fabsf(response);
}

__global__ void triangularThresholdFromHistogramKernel(const unsigned int *hist256,
                                                       unsigned char *thr_out)
{
    if (blockIdx.x != 0 || threadIdx.x != 0)
    {
        return;
    }

    int peak = 0;
    unsigned int peak_val = hist256[0];
    for (int i = 1; i < 256; ++i)
    {
        const unsigned int v = hist256[i];
        if (v > peak_val)
        {
            peak_val = v;
            peak = i;
        }
    }

    int x1 = 0;
    int x2 = 255;
    if (peak < 128)
    {
        x1 = peak;
        x2 = 255;
    }
    else
    {
        x1 = 0;
        x2 = peak;
    }

    const float y1 = static_cast<float>(hist256[x1]);
    const float y2 = static_cast<float>(hist256[x2]);
    const float a = y2 - y1;
    const float b = static_cast<float>(x1 - x2);
    const float c = static_cast<float>(x2) * y1 - static_cast<float>(x1) * y2;
    const float den = sqrtf(a * a + b * b);

    int best_idx = peak;
    float best_dist = -1.0f;

    if (den > 1e-12f)
    {
        if (peak < 128)
        {
            for (int i = peak; i < 255; ++i)
            {
                const float yi = static_cast<float>(hist256[i]);
                const float dist = fabsf(a * static_cast<float>(i) + b * yi + c) / den;
                if (dist > best_dist)
                {
                    best_dist = dist;
                    best_idx = i;
                }
            }
        }
        else
        {
            for (int i = 0; i < peak; ++i)
            {
                const float yi = static_cast<float>(hist256[i]);
                const float dist = fabsf(a * static_cast<float>(i) + b * yi + c) / den;
                if (dist > best_dist)
                {
                    best_dist = dist;
                    best_idx = i;
                }
            }
        }
    }

    best_idx = max(0, min(255, best_idx));
    thr_out[0] = static_cast<unsigned char>(best_idx);
}

FrequencyMaskGpuPair buildFrequencyMasksGpu(const cv::cuda::GpuMat &color_gpu, float sigma_ratio)
{
    FrequencyMaskGpuPair masks;

    // ============================================================
    // 1. SALIDA TEMPRANA
    // ============================================================
    if (color_gpu.empty())
    {
        return masks;
    }

    // ============================================================
    // 2. PREPARAR ESCALA DE GRISES EN GPU
    // ============================================================
    cv::cuda::GpuMat gray_u8;

    if (color_gpu.channels() == 4)
    {
        cv::cuda::cvtColor(color_gpu, gray_u8, cv::COLOR_BGRA2GRAY);
    }
    else if (color_gpu.channels() == 3)
    {
        cv::cuda::cvtColor(color_gpu, gray_u8, cv::COLOR_BGR2GRAY);
    }
    else if (color_gpu.channels() == 1)
    {
        if (color_gpu.type() == CV_8UC1)
        {
            gray_u8 = color_gpu;
        }
        else
        {
            color_gpu.convertTo(gray_u8, CV_8U);
        }
    }
    else
    {
        return masks;
    }

    if (gray_u8.empty())
    {
        return masks;
    }

    const int width = gray_u8.cols;
    const int height = gray_u8.rows;

    if (width < 2 || height < 2)
    {
        masks.high_mask_u8_gpu = cv::cuda::GpuMat(height, width, CV_8U);
        masks.high_mask_u8_gpu.setTo(cv::Scalar(255));
        masks.low_mask_u8_gpu = cv::cuda::GpuMat(height, width, CV_8U);
        masks.low_mask_u8_gpu.setTo(cv::Scalar(0));
        return masks;
    }

    // ============================================================
    // 3. CONTEXTO (CACHE DE PLANOS Y BUFFERS)
    // ============================================================
    FrequencyMaskContext &ctx = getFrequencyMaskContext();
    if (!ctx.ensureGeometry(width, height))
    {
        return masks;
    }

    dim3 block2d(16, 16);
    dim3 grid2d((width + block2d.x - 1) / block2d.x,
                (height + block2d.y - 1) / block2d.y);

    // ============================================================
    // 4. PREPROCESAMIENTO A FLOAT LINEAL
    // ============================================================
    copyU8ToFloatLinearKernel<<<grid2d, block2d>>>(
        gray_u8.ptr<unsigned char>(),
        gray_u8.step,
        thrust::raw_pointer_cast(ctx.d_real.data()),
        width,
        height);

    if (!checkLaunch("buildFrequencyHighMaskGpu::copyU8ToFloatLinearKernel"))
    {
        return masks;
    }

    // ============================================================
    // 5. FFT DIRECTA + PASA ALTOS + FFT INVERSA
    // ============================================================
    if (!checkCufft(cufftExecR2C(ctx.plan_r2c,
                                 thrust::raw_pointer_cast(ctx.d_real.data()),
                                 thrust::raw_pointer_cast(ctx.d_freq.data())),
                    "buildFrequencyHighMaskGpu::cufftExecR2C"))
    {
        return masks;
    }

    const float sigma = std::max(1.0f, sigma_ratio * static_cast<float>(std::min(width, height)));
    const float inv_two_sigma2 = 1.0f / (2.0f * sigma * sigma);

    dim3 grid_freq((ctx.width_half + block2d.x - 1) / block2d.x,
                   (height + block2d.y - 1) / block2d.y);
    applyHighpassR2CKernel<<<grid_freq, block2d>>>(
        thrust::raw_pointer_cast(ctx.d_freq.data()),
        width,
        height,
        inv_two_sigma2);

    if (!checkLaunch("buildFrequencyHighMaskGpu::applyHighpassR2CKernel"))
    {
        return masks;
    }

    if (!checkCufft(cufftExecC2R(ctx.plan_c2r,
                                 thrust::raw_pointer_cast(ctx.d_freq.data()),
                                 thrust::raw_pointer_cast(ctx.d_real.data())),
                    "buildFrequencyHighMaskGpu::cufftExecC2R"))
    {
        return masks;
    }

    // ============================================================
    // 6. POSTPROCESADO EN GPU
    // ============================================================
    const int threads = 256;
    const int blocks = (ctx.n + threads - 1) / threads;
    const float inv_n = 1.0f / static_cast<float>(ctx.n);

    absScaleKernel<<<blocks, threads>>>(
        thrust::raw_pointer_cast(ctx.d_real.data()),
        thrust::raw_pointer_cast(ctx.d_abs.data()),
        ctx.n,
        inv_n);

    if (!checkLaunch("buildFrequencyHighMaskGpu::absScaleKernel"))
    {
        return masks;
    }

    auto mm_it = thrust::minmax_element(ctx.d_abs.begin(), ctx.d_abs.end());
    const float min_val = *mm_it.first;
    const float max_val = *mm_it.second;

    if (!std::isfinite(min_val) || !std::isfinite(max_val))
    {
        return masks;
    }

    if (max_val - min_val > 1e-12f)
    {
        const float inv_range = 255.0f / (max_val - min_val);
        normalizeFloatToU8Kernel<<<blocks, threads>>>(
            thrust::raw_pointer_cast(ctx.d_abs.data()),
            thrust::raw_pointer_cast(ctx.d_u8.data()),
            ctx.n,
            min_val,
            inv_range);

        if (!checkLaunch("buildFrequencyHighMaskGpu::normalizeFloatToU8Kernel"))
        {
            return masks;
        }
    }
    else
    {
        thrust::fill(ctx.d_u8.begin(), ctx.d_u8.end(), static_cast<unsigned char>(0));
    }

    // ============================================================
    // 7. UMBRAL TRIANGULAR (GPU)
    // ============================================================
    cudaMemset(
        thrust::raw_pointer_cast(ctx.d_hist_u32.data()),
        0,
        256 * sizeof(unsigned int));

    computeHistogram256Kernel<<<blocks, threads>>>(
        thrust::raw_pointer_cast(ctx.d_u8.data()),
        ctx.n,
        thrust::raw_pointer_cast(ctx.d_hist_u32.data()));

    if (!checkLaunch("buildFrequencyHighMaskGpu::computeHistogram256Kernel"))
    {
        return masks;
    }

    triangularThresholdFromHistogramKernel<<<1, 1>>>(
        thrust::raw_pointer_cast(ctx.d_hist_u32.data()),
        thrust::raw_pointer_cast(ctx.d_triangle_thr.data()));

    if (!checkLaunch("buildFrequencyHighMaskGpu::triangularThresholdFromHistogramKernel"))
    {
        return masks;
    }

    // ============================================================
    // 7.X REFERENCIA CPU (comentada)
    // ============================================================
    // thrust::host_vector<unsigned char> h_u8 = ctx.d_u8;
    // if (h_u8.size() != static_cast<size_t>(ctx.n))
    // {
    //     return masks;
    // }
    // cv::Mat response_u8(height, width, CV_8U, h_u8.data());
    // cv::Mat triangle_bin_cpu;
    // const double triangle_thr_d = cv::threshold(
    //     response_u8,
    //     triangle_bin_cpu,
    //     0.0,
    //     255.0,
    //     cv::THRESH_BINARY | cv::THRESH_TRIANGLE);
    // const unsigned char triangle_thr = static_cast<unsigned char>(
    //     std::max(0.0, std::min(255.0, triangle_thr_d)));

    thresholdLinearToGpuMatKernel<<<grid2d, block2d>>>(
        thrust::raw_pointer_cast(ctx.d_u8.data()),
        ctx.high_mask_u8_gpu.ptr<unsigned char>(),
        ctx.high_mask_u8_gpu.step,
        width,
        height,
        thrust::raw_pointer_cast(ctx.d_triangle_thr.data()));

    if (!checkLaunch("buildFrequencyHighMaskGpu::thresholdLinearToGpuMatKernel"))
    {
        return masks;
    }

    const size_t non_zero = static_cast<size_t>(cv::cuda::countNonZero(ctx.high_mask_u8_gpu));
    if (non_zero == 0)
    {
        ctx.high_mask_u8_gpu.setTo(cv::Scalar(255));
    }

    // M_l = complemento de M_h.
    cv::cuda::bitwise_not(ctx.high_mask_u8_gpu, ctx.low_mask_u8_gpu);

    masks.high_mask_u8_gpu = ctx.high_mask_u8_gpu;
    masks.low_mask_u8_gpu = ctx.low_mask_u8_gpu;
    return masks;
}

cv::cuda::GpuMat buildFrequencyHighMaskGpu(const cv::cuda::GpuMat &color_gpu, float sigma_ratio)
{
    return buildFrequencyMasksGpu(color_gpu, sigma_ratio).high_mask_u8_gpu;
}

cv::Mat buildFrequencyHighMask(const cv::cuda::GpuMat &color_gpu, float sigma_ratio)
{
    cv::Mat high_mask_cpu;
    const cv::cuda::GpuMat high_mask_gpu = buildFrequencyHighMaskGpu(color_gpu, sigma_ratio);
    if (high_mask_gpu.empty())
    {
        return cv::Mat();
    }

    high_mask_gpu.download(high_mask_cpu);
    return high_mask_cpu;
}

FrequencyMaskGpuPair buildSobelMasksGpu(const cv::cuda::GpuMat &color_gpu)
{
    FrequencyMaskGpuPair masks;

    if (color_gpu.empty())
    {
        std::cerr << "[GSSlam][ERROR] buildSobelMasksGpu: color_gpu is empty" << std::endl;
        return masks;
    }

    // Preparar escala de grises en GPU
    cv::cuda::GpuMat gray_u8;

    if (color_gpu.channels() == 4)
    {
        cv::cuda::cvtColor(color_gpu, gray_u8, cv::COLOR_BGRA2GRAY);
    }
    else if (color_gpu.channels() == 3)
    {
        cv::cuda::cvtColor(color_gpu, gray_u8, cv::COLOR_BGR2GRAY);
    }
    else if (color_gpu.channels() == 1)
    {
        gray_u8 = color_gpu;
    }
    else
    {
        std::cerr << "[GSSlam][ERROR] buildSobelMasksGpu: Unsupported channel count: " << color_gpu.channels() << std::endl;
        return masks;
    }

    if (gray_u8.empty())
    {
        std::cerr << "[GSSlam][ERROR] buildSobelMasksGpu: Failed to convert to grayscale" << std::endl;
        return masks;
    }

    const int width = gray_u8.cols;
    const int height = gray_u8.rows;

    if (width < 3 || height < 3)
    {
        std::cerr << "[GSSlam][ERROR] buildSobelMasksGpu: Image too small for Sobel (need 3x3 kernels)" << std::endl;
        return masks;
    }

    // Obtener contexto persistente
    FrequencyMaskContext &ctx = getFrequencyMaskContext();
    if (!ctx.ensureGeometry(width, height))
    {
        std::cerr << "[GSSlam][ERROR] buildSobelMasksGpu: Failed to ensure geometry" << std::endl;
        return masks;
    }

    dim3 block2d(16, 16);
    dim3 grid2d((width + block2d.x - 1) / block2d.x,
                (height + block2d.y - 1) / block2d.y);

    // Copiar la imagen en gris al buffer persistente y recrear la textura (RAII)
    gray_u8.copyTo(ctx.gray_u8_gpu);
    ctx.gray_tex_wrapper = std::make_unique<Texture<unsigned char>>(ctx.gray_u8_gpu, cudaFilterModePoint);
    
    if (!ctx.gray_tex_wrapper || ctx.gray_tex_wrapper->getTextureObject() == 0)
    {
        std::cerr << "[GSSlam][ERROR] buildSobelMasksGpu - Failed to create texture" << std::endl;
        return masks;
    }

    // Aplicar convolución Sobel -> dst_magnitude (usando el objeto textura)
    sobelConvolutionKernel<<<grid2d, block2d>>>(
        ctx.gray_tex_wrapper->getTextureObject(),
        thrust::raw_pointer_cast(ctx.d_real.data()),
        width,
        height);

    if (!checkLaunch("buildSobelMasksGpu::sobelConvolutionKernel"))
    {
        return masks;
    }

    // Obtener mínimo y máximo para normalización
    auto minmaxptr = thrust::minmax_element(ctx.d_real.begin(), ctx.d_real.end());
    float min_val = *minmaxptr.first;
    float max_val = *minmaxptr.second;

    if (max_val <= min_val)
    {
        max_val = min_val + 1e-6f;
    }

    const float inv_range = 255.0f / (max_val - min_val);

    // Normalizar a [0, 255]
    dim3 block1d(256);
    dim3 grid1d((ctx.n + block1d.x - 1) / block1d.x);

    normalizeFloatToU8Kernel<<<grid1d, block1d>>>(
        thrust::raw_pointer_cast(ctx.d_real.data()),
        thrust::raw_pointer_cast(ctx.d_u8.data()),
        ctx.n,
        min_val,
        inv_range);

    if (!checkLaunch("buildSobelMasksGpu::normalizeFloatToU8Kernel"))
    {
        return masks;
    }

    // Construir histograma
    thrust::fill(ctx.d_hist_u32.begin(), ctx.d_hist_u32.end(), 0u);

    computeHistogram256Kernel<<<grid1d, block1d>>>(
        thrust::raw_pointer_cast(ctx.d_u8.data()),
        ctx.n,
        thrust::raw_pointer_cast(ctx.d_hist_u32.data()));

    if (!checkLaunch("buildSobelMasksGpu::computeHistogram256Kernel"))
    {
        return masks;
    }

    // Umbral triangular
    triangularThresholdFromHistogramKernel<<<1, 1>>>(
        thrust::raw_pointer_cast(ctx.d_hist_u32.data()),
        thrust::raw_pointer_cast(ctx.d_triangle_thr.data()));

    if (!checkLaunch("buildSobelMasksGpu::triangularThresholdFromHistogramKernel"))
    {
        return masks;
    }

    // Binarizar
    thresholdLinearToGpuMatKernel<<<grid2d, block2d>>>(
        thrust::raw_pointer_cast(ctx.d_u8.data()),
        reinterpret_cast<unsigned char *>(ctx.high_mask_u8_gpu.data),
        ctx.high_mask_u8_gpu.step,
        width,
        height,
        thrust::raw_pointer_cast(ctx.d_triangle_thr.data()));

    if (!checkLaunch("buildSobelMasksGpu::thresholdLinearToGpuMatKernel"))
    {
        return masks;
    }

    // M_l = complemento
    cv::cuda::bitwise_not(ctx.high_mask_u8_gpu, ctx.low_mask_u8_gpu);

    masks.high_mask_u8_gpu = ctx.high_mask_u8_gpu;
    masks.low_mask_u8_gpu = ctx.low_mask_u8_gpu;
    return masks;
}

FrequencyMaskGpuPair buildLaplacianMasksGpu(const cv::cuda::GpuMat &color_gpu)
{
    FrequencyMaskGpuPair masks;

    if (color_gpu.empty())
    {
        std::cerr << "[GSSlam][ERROR] buildLaplacianMasksGpu: color_gpu is empty" << std::endl;
        return masks;
    }

    // Preparar escala de grises en GPU
    cv::cuda::GpuMat gray_u8;

    if (color_gpu.channels() == 4)
    {
        cv::cuda::cvtColor(color_gpu, gray_u8, cv::COLOR_BGRA2GRAY);
    }
    else if (color_gpu.channels() == 3)
    {
        cv::cuda::cvtColor(color_gpu, gray_u8, cv::COLOR_BGR2GRAY);
    }
    else if (color_gpu.channels() == 1)
    {
        gray_u8 = color_gpu;
    }
    else
    {
        std::cerr << "[GSSlam][ERROR] buildLaplacianMasksGpu: Unsupported channel count: " << color_gpu.channels() << std::endl;
        return masks;
    }

    if (gray_u8.empty())
    {
        std::cerr << "[GSSlam][ERROR] buildLaplacianMasksGpu: Failed to convert to grayscale" << std::endl;
        return masks;
    }

    const int width = gray_u8.cols;
    const int height = gray_u8.rows;

    if (width < 3 || height < 3)
    {
        std::cerr << "[GSSlam][ERROR] buildLaplacianMasksGpu: Image too small for Laplacian (need 3x3 kernels)" << std::endl;
        return masks;
    }

    // Obtener contexto persistente
    FrequencyMaskContext &ctx = getFrequencyMaskContext();
    if (!ctx.ensureGeometry(width, height))
    {
        std::cerr << "[GSSlam][ERROR] buildLaplacianMasksGpu: Failed to ensure geometry" << std::endl;
        return masks;
    }

    dim3 block2d(16, 16);
    dim3 grid2d((width + block2d.x - 1) / block2d.x,
                (height + block2d.y - 1) / block2d.y);

    // Copiar la imagen en gris al buffer persistente y recrear la textura (RAII)
    gray_u8.copyTo(ctx.gray_u8_gpu);
    ctx.gray_tex_wrapper = std::make_unique<Texture<unsigned char>>(ctx.gray_u8_gpu, cudaFilterModePoint);
    
    if (!ctx.gray_tex_wrapper || ctx.gray_tex_wrapper->getTextureObject() == 0)
    {
        std::cerr << "[GSSlam][ERROR] buildLaplacianMasksGpu - Failed to create texture" << std::endl;
        return masks;
    }

    // Aplicar convolución Laplaciana -> dst_response (usando el objeto textura)
    laplacianConvolutionKernel<<<grid2d, block2d>>>(
        ctx.gray_tex_wrapper->getTextureObject(),
        thrust::raw_pointer_cast(ctx.d_real.data()),
        width,
        height);

    if (!checkLaunch("buildLaplacianMasksGpu::laplacianConvolutionKernel"))
    {
        return masks;
    }

    // Obtener mínimo y máximo para normalización
    auto minmaxptr = thrust::minmax_element(ctx.d_real.begin(), ctx.d_real.end());
    float min_val = *minmaxptr.first;
    float max_val = *minmaxptr.second;

    if (max_val <= min_val)
    {
        max_val = min_val + 1e-6f;
    }

    const float inv_range = 255.0f / (max_val - min_val);

    // Normalizar a [0, 255]
    dim3 block1d(256);
    dim3 grid1d((ctx.n + block1d.x - 1) / block1d.x);

    normalizeFloatToU8Kernel<<<grid1d, block1d>>>(
        thrust::raw_pointer_cast(ctx.d_real.data()),
        thrust::raw_pointer_cast(ctx.d_u8.data()),
        ctx.n,
        min_val,
        inv_range);

    if (!checkLaunch("buildLaplacianMasksGpu::normalizeFloatToU8Kernel"))
    {
        return masks;
    }

    // Construir histograma
    thrust::fill(ctx.d_hist_u32.begin(), ctx.d_hist_u32.end(), 0u);

    computeHistogram256Kernel<<<grid1d, block1d>>>(
        thrust::raw_pointer_cast(ctx.d_u8.data()),
        ctx.n,
        thrust::raw_pointer_cast(ctx.d_hist_u32.data()));

    if (!checkLaunch("buildLaplacianMasksGpu::computeHistogram256Kernel"))
    {
        return masks;
    }

    // Umbral triangular
    triangularThresholdFromHistogramKernel<<<1, 1>>>(
        thrust::raw_pointer_cast(ctx.d_hist_u32.data()),
        thrust::raw_pointer_cast(ctx.d_triangle_thr.data()));

    if (!checkLaunch("buildLaplacianMasksGpu::triangularThresholdFromHistogramKernel"))
    {
        return masks;
    }

    // Binarizar
    thresholdLinearToGpuMatKernel<<<grid2d, block2d>>>(
        thrust::raw_pointer_cast(ctx.d_u8.data()),
        reinterpret_cast<unsigned char *>(ctx.high_mask_u8_gpu.data),
        ctx.high_mask_u8_gpu.step,
        width,
        height,
        thrust::raw_pointer_cast(ctx.d_triangle_thr.data()));

    if (!checkLaunch("buildLaplacianMasksGpu::thresholdLinearToGpuMatKernel"))
    {
        return masks;
    }

    // M_l = complemento
    cv::cuda::bitwise_not(ctx.high_mask_u8_gpu, ctx.low_mask_u8_gpu);

    masks.high_mask_u8_gpu = ctx.high_mask_u8_gpu;
    masks.low_mask_u8_gpu = ctx.low_mask_u8_gpu;
    return masks;
}

FrequencyMaskGpuPair buildCannyMasksGpu(const cv::cuda::GpuMat &color_gpu,
                                         double threshold1,
                                         double threshold2)
{
    FrequencyMaskGpuPair masks;

    if (color_gpu.empty())
    {
        std::cerr << "[GSSlam][ERROR] buildCannyMasksGpu: color_gpu is empty" << std::endl;
        return masks;
    }

    // ============================================================
    // 1. CONVERTIR A ESCALA DE GRISES
    // ============================================================
    cv::cuda::GpuMat gray_u8;

    if (color_gpu.channels() == 4)
    {
        cv::cuda::cvtColor(color_gpu, gray_u8, cv::COLOR_BGRA2GRAY);
    }
    else if (color_gpu.channels() == 3)
    {
        cv::cuda::cvtColor(color_gpu, gray_u8, cv::COLOR_BGR2GRAY);
    }
    else if (color_gpu.channels() == 1)
    {
        gray_u8 = color_gpu;
    }
    else
    {
        std::cerr << "[GSSlam][ERROR] buildCannyMasksGpu: Unsupported number of channels: "
                  << color_gpu.channels() << std::endl;
        return masks;
    }

    if (gray_u8.empty())
    {
        std::cerr << "[GSSlam][ERROR] buildCannyMasksGpu: gray_u8 conversion failed" << std::endl;
        return masks;
    }

    // ============================================================
    // 2. APLICAR CANNY EDGE DETECTION (NVIDIA CUDA)
    // ============================================================
    auto canny_detector = cv::cuda::createCannyEdgeDetector(threshold1, threshold2);
    if (!canny_detector)
    {
        std::cerr << "[GSSlam][ERROR] buildCannyMasksGpu: Failed to create Canny detector" << std::endl;
        return masks;
    }

    cv::cuda::GpuMat edges_u8;
    canny_detector->detect(gray_u8, edges_u8);

    if (edges_u8.empty())
    {
        std::cerr << "[GSSlam][ERROR] buildCannyMasksGpu: Canny detection returned empty result" << std::endl;
        return masks;
    }

    // ============================================================
    // 3. ASIGNAR MÁSCARAS
    // ============================================================
    // M_h: bordes detectados (alta frecuencia)
    masks.high_mask_u8_gpu = edges_u8;

    // M_l: complemento de los bordes (baja frecuencia / regiones suaves)
    cv::cuda::bitwise_not(edges_u8, masks.low_mask_u8_gpu);

    return masks;
}

} // namespace f_vigs_slam
