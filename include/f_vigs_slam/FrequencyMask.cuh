#pragma once

#include <opencv2/core.hpp>
#include <opencv2/core/cuda.hpp>

namespace f_vigs_slam
{
// ============================================================
// MASCARA DE FRECUENCIA (API)
// ============================================================
// M_h: mascara de alta frecuencia.
// M_l: mascara de baja frecuencia.
// M_i: mascara de regiones faltantes en el mapa.
// M_m: mascara final de regiones faltantes.
struct FrequencyMaskGpuPair
{
	cv::cuda::GpuMat high_mask_u8_gpu;
	cv::cuda::GpuMat low_mask_u8_gpu;
};

// Construye M_h y M_l en GPU reutilizando buffers persistentes (FFT-based).
FrequencyMaskGpuPair buildFrequencyMasksGpu(const cv::cuda::GpuMat &color_gpu, float sigma_ratio);

// Devuelve solo M_h en GPU para compatibilidad con el flujo actual (FFT-based).
cv::cuda::GpuMat buildFrequencyHighMaskGpu(const cv::cuda::GpuMat &color_gpu, float sigma_ratio);

// Wrapper de compatibilidad: descarga M_h a CPU (FFT-based).
cv::Mat buildFrequencyHighMask(const cv::cuda::GpuMat &color_gpu, float sigma_ratio);

// Construye máscaras Sobel (primera derivada) en GPU.
FrequencyMaskGpuPair buildSobelMasksGpu(const cv::cuda::GpuMat &color_gpu);

// Construye máscaras Laplacian (segunda derivada) en GPU.
FrequencyMaskGpuPair buildLaplacianMasksGpu(const cv::cuda::GpuMat &color_gpu);

// Construye máscaras Canny (detección de bordes) en GPU.
FrequencyMaskGpuPair buildCannyMasksGpu(const cv::cuda::GpuMat &color_gpu,
                                         double threshold1 = 50.0,
                                         double threshold2 = 150.0);
}
