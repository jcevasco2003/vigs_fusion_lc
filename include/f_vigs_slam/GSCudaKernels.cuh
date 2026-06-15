#pragma once

#include <cuda_runtime.h>
#include <Eigen/Dense>
#include <opencv2/core/cuda.hpp>
#include "RepresentationClasses.hpp"
#include "CudaMathOperations.cuh"

namespace f_vigs_slam
{
    enum class GpuExpEvaluationMode : int
    {
        DEFAULT = 0,
        TAYLOR = 1
    };

    // Configura en runtime el modo de evaluación exponencial usado en kernels CUDA.
    void setGpuExpEvaluationMode(GpuExpEvaluationMode mode);

    struct DeltaGaussian2D
    {
        float2 meanImg;
        float3 invSigmaImg;
        float3 color;
        float depth;
        float alpha;
        float2 pHat;
        unsigned int n;
    };

    struct DeltaGaussian3D
    {
        float3 position;
        float3 scale;
        float3 orientation;
        float3 color;
        float alpha;
        int n;
    };

    // Declaramos kernels CUDA para operaciones paralelizables

    // =======================================================================
    // Inicializacion de Gaussianas desde RGB-D
    // =======================================================================

    // Equivalente a computeNormalsFromDepth_kernel en vigs-fusion
    /**
     * @brief Estima normales por píxel a partir de una imagen de profundidad.
     * @param depth [IN] Textura de profundidad en memoria lineal.
     * @param normals_out [OUT] Mapa de normales de salida (float3 por píxel).
     * @param normals_step [IN] Stride en bytes del mapa normals_out.
     * @param width [IN] Ancho de la imagen.
     * @param height [IN] Alto de la imagen.
     * @param intrinsics [IN] Parámetros intrínsecos de cámara.
     */
    __global__ void computeNormalsFromDepth_kernel(
        cudaTextureObject_t depth,
        float4 *normals_out,
        size_t normals_step,
        int width,
        int height,
        IntrinsicParameters intrinsics);


    // Equivalente a generateGaussians_kernel en vigs-fusion
    /**
     * @brief Inicializa gaussianas 3D desde imagen RGB-D y normales estimadas.
     * @param positions [OUT] Posiciones 3D inicializadas.
     * @param scales [OUT] Escalas iniciales por gaussiana.
     * @param orientations [OUT] Orientaciones iniciales (cuaternion).
     * @param colors [OUT] Colores iniciales normalizados.
     * @param opacities [OUT] Opacidades iniciales.
     * @param instanceCounter [INOUT] Contador global de gaussianas creadas.
     * @param maxGaussians [IN] Capacidad máxima de gaussianas.
     * @param rgb [IN] Imagen RGB de entrada.
     * @param depth [IN] Mapa de profundidad de entrada.
     * @param normals [IN] Mapa de normales por píxel.
     * @param width [IN] Ancho de imagen.
     * @param height [IN] Alto de imagen.
     * @param intrinsics [IN] Parámetros intrínsecos de cámara.
     * @param cameraPose [IN] Pose de cámara para transformar a mundo.
     * @param sample_dx [IN] Paso de muestreo horizontal.
     * @param sample_dy [IN] Paso de muestreo vertical.
     * @param init_opacity [IN] Opacidad inicial por gaussiana.
     */
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
        // Pose global del submapa para transformar P_global -> P_local
        Pose submap_pose_global);

    /**
     * @brief backprojectDepthImage_kernel
     * Convierte una imagen de profundidad en una nube de puntos 3D en el frame de cámara.
     * Cada thread procesa un píxel; los puntos válidos se compactan con un contador atómico.
     *
     * @param points_out [OUT] Puntos 3D compactados en memoria global.
     * @param pointCounter [INOUT] Contador de puntos válidos escritos.
     * @param maxPoints [IN] Capacidad máxima del buffer de salida.
     * @param depth [IN] Imagen de profundidad en memoria pitched.
     * @param depth_step [IN] Paso en bytes entre filas de depth.
     * @param width [IN] Ancho de la imagen.
     * @param height [IN] Alto de la imagen.
     * @param intrinsics [IN] Parámetros intrínsecos de la cámara.
     * @param sample_stride [IN] Muestreo espacial (1 = todos los píxeles).
     */
    __global__ void backprojectDepthImage_kernel(
        float3 *points_out,
        uint32_t *pointCounter,
        uint32_t maxPoints,
        const float *depth,
        size_t depth_step,
        int width,
        int height,
        IntrinsicParameters intrinsics,
        uint32_t sample_stride);

    /**
     * @brief Transforma gaussianos de coordenadas locales a globales.
     * Fase 5: Usado para renderizado global - convierte gaussianos en frame local
     * del submapa a frame global para agregación en global_view_submap.
     * 
     * @param positions_local [IN] Posiciones en coordenadas locales del submapa.
     * @param orientations_local [IN] Orientaciones en coordenadas locales.
     * @param positions_global [OUT] Posiciones transformadas a globales.
     * @param orientations_global [OUT] Orientaciones transformadas a globales.
     * @param submap_t_global Parte traslacional de T_global del submapa.
     * @param submap_R_col0/1/2 Columnas de matriz de rotación R de T_global.
     * @param n_gaussians Número de gaussianos a transformar.
     */
    __global__ void transformGaussians_localToGlobal_kernel(
        const float4 *positions_local,
        const float4 *orientations_local,
        float4 *positions_global,
        float4 *orientations_global,
        Pose submap_pose_global,
        uint32_t n_gaussians);

    /**
     * @brief Resalta un rango de gaussianas con un acento visual temporal.
     */
    __global__ void tintGaussianColors_kernel(
        float4 *colors,
        uint32_t n_gaussians,
        float accent_b,
        float accent_g,
        float accent_r,
        float blend,
        float gain);

    /**
     * @brief Calcula gradientes Sobel RGB por canal en X e Y.
     * @param input_rgb [IN] Imagen RGB de entrada en uchar4.
     * @param input_step [IN] Stride en bytes de input_rgb.
     * @param grad_x [OUT] Gradiente Sobel horizontal.
     * @param grad_x_step [IN] Stride en bytes de grad_x.
     * @param grad_y [OUT] Gradiente Sobel vertical.
     * @param grad_y_step [IN] Stride en bytes de grad_y.
     * @param width [IN] Ancho de la imagen.
     * @param height [IN] Alto de la imagen.
     */
    __global__ void computeSobelRgb_kernel(
        const uchar4* input_rgb,
        size_t input_step,
        float4 *grad_x,
        size_t grad_x_step,
        float4 *grad_y,
        size_t grad_y_step,
        int width,
        int height);


    // =======================================================================
    // Forward pass: renderiza las gaussianas 3D a imagen 2D con alpha-blending
    // =======================================================================

    /**
     * @brief countTilesPerGaussian_kernel
     * Cuenta cuántos tiles 2D cubre cada gaussiana considerando su radio
     * Usa la covarianza 2D para determinar el radio de cobertura (3-sigma)
     * 
     * Un thread por cada gaussiana
     * 
     * @param tile_counts Salida: número de tiles cubiertos [n_gaussians]
     * @param positions_2d Entrada: centros proyectados [n_gaussians]
     * @param covariances_2d Entrada: covarianzas 2D (xx, yy, xy) [n_gaussians]
     * @param radius_sigma Factor para bounding box (típicamente 3.0f para 3-sigma)
     */
    __global__ void countTilesPerGaussian_kernel(
        uint32_t *tile_counts,
        const float2 *positions_2d,
        const float3 *covariances_2d,
        int width,
        int height,
        int n_gaussians,
        int num_tiles_x,
        int num_tiles_y,
        float radius_sigma = 3.0f);

    /**
     * @brief generateTileHashes_kernel
     * Genera múltiples hashes para cada gaussiana (uno por tile que cubre)
     * Utiliza los offsets acumulativos para escribir en las posiciones correctas
     * 
     * Un thread por cada gaussiana
     * 
     * @param hashes Salida: array de hashes expandido [total_tiles_covered]
     * @param gaussian_indices Salida: índice de gaussiana para cada hash
     * @param tile_offsets Entrada: offsets acumulativos de tile_counts (resultado de exclusive scan)
     * @param positions_2d Entrada: centros proyectados [n_gaussians]
     * @param covariances_2d Entrada: covarianzas 2D [n_gaussians]
     * @param depths Entrada: profundidades Z [n_gaussians]
     * @param radius_sigma Factor para bounding box (típicamente 3.0f)
     */
    __global__ void generateTileHashes_kernel(
        uint64_t *hashes,
        uint32_t *gaussian_indices,
        const uint32_t *tile_offsets,
        const float2 *positions_2d,
        const float3 *covariances_2d,
        const float *depths,
        const float2 *pHats,
        int width,
        int height,
        int n_gaussians,
        int num_tiles_x,
        int num_tiles_y,
        float radius_sigma = 3.0f);

    // DEPRECATED: Ya no lo uso, en su lugar uso projectAndHashGaussians_kernel que hace ambas cosas en un solo paso
    /**
     * @brief projectGaussiansWorldToScreen_kernel
     * Proyecta gaussianas 3D (mundo) → 2D (pantalla)
     * Transforma de mundo a cámara, proyecta, y calcula covarianza 2D
     */
    __global__ void projectGaussiansWorldToScreen_kernel(
        float2 *positions_2d,
        float3 *covariances_2d,
        float3 *inv_covariances_2d,
        float *depths,
        float2 *p_hats,
        float3 *normals,
        const float4 *positions_world,
        const float4 *scales,
        const float4 *orientations,
        Pose camera_pose,
        IntrinsicParameters intrinsics,
        int width,
        int height,
        int n_gaussians);

    // Equivalente a computeScreenSpaceParamsAndHashes_kernel en vigs-fusion
    /**
     * @brief projectAndHashGaussians_kernel
     * Versión one-pass equivalente al kernel original de referencia:
     * proyecta gaussianas a pantalla, calcula covarianzas/pHat y genera hashes por tile.
     */
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
        uint32_t height);

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
        uint32_t height);

    /**
     * @brief projectGaussiansKernel - Proyectar gaussianas 3D a 2D con covarianzas
     * Proyecta los centros 3D de las gaussianas a coordenadas 2D de imagen
     * y transforma sus covarianzas de 3D a 2D usando el Jacobiano de proyeccion
     * 
     * Un thread por cada gaussiana
     */
    /*
    __global__ void projectGaussiansKernel(
        float2 *positions_2d,          // Salida: posiciones proyectadas [n_gaussians]
        float3 *covariances_2d,        // Salida: covarianzas 2D (xx, yy, xy) [n_gaussians]
        float *depths,                  // Salida: profundidad Z [n_gaussians]
        const float3 *positions_cam,    // Entrada: posiciones 3D en frame de camara
        const float3 *covariances_cam,  // Entrada: covarianzas 3D (xx, yy, zz)
        const float *intrinsics,        // Entrada: [fx, fy, cx, cy]
        int width,
        int height,
        int n_gaussians);
    */

    /**
     * @brief forwardPassKernel - Renderizar imagen final con alpha-blending
     * ---DESCARTADO por Tile Based Rendering---
     *
     *
     * Renderiza la imagen final componiendo todas las gaussianas con alpha-blending
     * Asume que las gaussianas estan ya ordenadas por profundidad (back-to-front)
     * 
     * Un thread por cada pixel de la imagen
     */
    //__global__ void forwardPassKernel(
    //    float3 *output_color,           // Salida: imagen RGB renderizada
    //    float *output_depth,            // Salida: mapa de profundidad
    //    const float2 *positions_2d,     // Entrada: posiciones proyectadas [n_gaussians]
    //    const float3 *covariances_2d,   // Entrada: covarianzas 2D [n_gaussians]
    //    const float3 *colors,           // Entrada: color RGB de cada Gaussian [n_gaussians]
    //    const float *alphas,            // Entrada: opacidad de cada Gaussian [n_gaussians]
    //    int width,
    //    int height,
    //    int n_gaussians);

    // Equivalente a computeIndicesRanges_kernel en vigs-fusion
    /**
     * @brief computeIndicesRanges_kernel
     * Extrae ranges [start, end) de un array de hashes ordenados
     * 
     * Despues de thrust::sort_by_key, este kernel genera un array de ranges
     * que especifica para cada tile dónde empiezan y terminan sus gaussianas
     * en el array ordenado. Esto permite acceso O(1) a las gaussianas de cada tile.
     * 
     * NOTA: n_instances = total_hashes (puede ser > n_gaussians por cobertura multi-tile)
     * 
     * @param ranges Salida: [start, end) en indices para cada tile [num_tiles]
     * @param hashes Entrada: hashes ya ordenados (después de sort_by_key)
     * @param n_instances Numero total de hashes (total_hashes)
     */
    __global__ void computeIndicesRanges_kernel(
        uint2 *ranges,
        const uint64_t *hashes,
        uint32_t n_instances);

    
    // Equivalente a rasterizeGaussians_kernel en vigs-fusion
    /**
     * @brief forwardPassTileKernel
     * Renderiza la imagen usando tile-based parallelism
     * 
     * La imagen se divide en tiles de TILE_SIZE x TILE_SIZE
     * Cada bloque CUDA procesa un tile
     * Los threads dentro del bloque cargan gaussianas relevantes en shared memory
     * y realizan alpha-blending para cada pixel del tile
     * 
     * Grid: (num_tiles_x, num_tiles_y)
     * Block: (TILE_SIZE, TILE_SIZE) - cada thread procesa un pixel
     * 
     * @param output_color Salida: imagen RGB renderizada [width*height]
     * @param output_depth Salida: mapa de profundidad [width*height]
     * @param tile_gaussian_indices Entrada: lista de indices de gaussianas por tile
     * @param tile_ranges Entrada: rangos [start, end) en tile_gaussian_indices
     * @param positions_2d Entrada: centros proyectados [n_gaussians]
    * @param inv_covariances_2d Entrada: inversas de covarianza 2D [n_gaussians]
     * @param colors Entrada: colores RGB [n_gaussians]
     * @param alphas Entrada: opacidades [n_gaussians]
     * @param depths Entrada: profundidades Z [n_gaussians]
     */
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
        uint32_t n_gaussians_max); 


    /**
     *
     *
     *
     *
     *
     *
     */
    
    // ========================================================================
    // Backward pass: computa gradientes y residuos
    // ========================================================================

    // Equivalente a optimizePoseGN3_kernel en vigs-fusion
    /**
     * @brief Calcula J^T*J (Hessiana) y J^T*r (gradiente) para optimización de pose RGB-D.
     * @param output_posedata [OUT] Acumuladores JtJ[21] y Jtr[6] por tile
     * @param ranges [IN] Rangos [start, end) por tile
     * @param indices [IN] Indices ordenados de gaussianas
     * @param positions_2d [IN] Posiciones proyectadas (u, v)
     * @param inv_covariances_2d [IN] Covarianzas 2D invertidas
     * @param p_hats [IN] Coeficientes para profundidad por pixel
     * @param colors [IN] Colores RGB
     * @param alphas [IN] Opacidades
     * @param tex_rgb [IN] Imagen RGB observada (cudaTextureObject_t)
     * @param tex_depth [IN] Mapa de profundidad observado (cudaTextureObject_t)
     * @param tex_grad_x [IN] Gradiente en X (cudaTextureObject_t)
     * @param tex_grad_y [IN] Gradiente en Y (cudaTextureObject_t)
     * @param camera_pose [IN] Pose actual de la camara
     * @param intrinsics [IN] Parametros intrinsecos de camara
     * @param bg_color [IN] Color de fondo
     * @param alpha_thresh [IN] Umbral de alpha para evaluar residuales
     * @param color_thresh [IN] Umbral de Huber para color
     * @param depth_thresh [IN] Umbral de Huber para profundidad
     * @param width [IN] Ancho de imagen
     * @param height [IN] Alto de imagen
     * @param num_tiles_x [IN] Numero de tiles en X
     * @param num_tiles_y [IN] Numero de tiles en Y
     */
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
        cudaTextureObject_t tex_grad_x,
        cudaTextureObject_t tex_grad_y,
        Pose camera_pose,
        IntrinsicParameters intrinsics,
        float3 bg_color,
        float alpha_thresh,
        float color_thresh,
        float depth_thresh,
        int width, int height,
        int num_tiles_x, int num_tiles_y
    );

    /**
     * @brief Variante fast equivalente a optimizePoseGN3_fast_kernel de VIGS-Fusion.
     */
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
        cudaTextureObject_t tex_grad_x,
        cudaTextureObject_t tex_grad_y,
        Pose camera_pose,
        IntrinsicParameters intrinsics,
        float3 bg_color,
        float alpha_thresh,
        float color_thresh,
        float depth_thresh,
        int width, int height,
        int num_tiles_x, int num_tiles_y
    );

    /**
     * @brief Variante para warping single-rendering.
     *
     * Usa una referencia renderizada fija (warped_rgb/warped_depth) y la compara
     * contra la observación actual (observed_rgb/observed_depth) para acumular
     * JtJ y Jtr por tile en espacio de pose cámara local.
     */
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
        Pose warp_pose,
        IntrinsicParameters intrinsics,
        float w_depth,
        float color_thresh,
        float depth_thresh,
        int width,
        int height,
        int num_tiles_x,
        int num_tiles_y
    );

    // =======================================================================
    // Funciones auxiliares
    // =======================================================================

    __device__ inline float evalGaussian2DExponent(
        float dx, float dy,
        float inv_cov_xx, float inv_cov_yy, float inv_cov_xy);

    /**
     * @brief evalGaussian2D - Evaluar gaussiana 2D en un punto
     * Funcion auxiliar para evaluar una Gaussiana 2D en un punto dado
     * Calcula exp(-0.5 * d^T * Σ^-1 * d) de forma inline
    *
     * @param dx, dy desplazamiento relativo desde el centro del Gaussian
     * @param inv_cov_xx, inv_cov_yy, inv_cov_xy elementos de Σ^-1 preinvertida
     * @return valor de la Gaussiana 2D
     */
    __device__ inline float evalGaussian2D(
        float dx, float dy,
        float inv_cov_xx, float inv_cov_yy, float inv_cov_xy);

    /**
     * @brief invert2x2 - Invertir matriz 2x2 de forma rápida
     * Invierte una matriz 2x2 de covarianza de forma rapida
     * Usamos formula explicita porque es mas eficiente que metodos generales
     */
    __device__ inline void invert2x2(
        float cov_xx, float cov_yy, float cov_xy,
        float &inv_xx, float &inv_yy, float &inv_xy);

    // =======================================================================
    // Covisibilidad: Deteccion de gaussianas visibles en dos frames
    // =======================================================================

    // Equivalente a computeGaussiansVisibility_kernel en vigs-fusion
    /**
     * @brief computeGaussiansVisibility_kernel - Detectar gaussianas visibles en frame
     * Marca cuáles gaussianas son visibles (contribuyen al alpha-blending) en un frame dado.
     * 
     * Para cada pixel (x,y), itera sobre gaussianas y marca como visible si:
     * - La gaussiana cubre el pixel con alpha significativo (>1/255)
     * - La gaussiana contribuye antes de que se alcance opacity = 0.5 (saturación)
     * 
     * El algoritmo simula el proceso de rasterización:
     * - Iteración back-to-front (por profundidad)
     * - Acumulación de opacidad: T_i = T_{i-1} * (1 - alfa_i)
     * - Se detiene cuando T < 0.5 (imagen suficientemente opaca)
     * 
     * SALIDA: visibilities[gid] = 1 si gaussiana 'gid' es visible, 0 si no
     * 
     * @param visibilities      [OUT] Array de marks: 1=visible, 0=no visible [nbGaussians]
     * @param ranges            [IN]  Rango de índices por tile [numTiles_x * numTiles_y]
     * @param indices           [IN]  Índices ordenados de gaussianas [totalGaussians]
     * @param imgPositions      [IN]  Posiciones proyectadas (u, v, depth) [nbGaussians]
     * @param imgInvSigmas      [IN]  Covarianzas 2D invertidas (u², uv, v²) [nbGaussians]
     * @param alphas            [IN]  Opacidad de cada gaussiana [nbGaussians]
     * @param numTiles          [IN]  Dimensiones de grilla de tiles (x, y)
     * @param width             [IN]  Ancho de imagen (píxeles)
     * @param height            [IN]  Alto de imagen (píxeles)
     */
    __global__ void computeGaussiansVisibility_kernel(
        unsigned char *visibilities,
        const uint2 *ranges,
        const uint32_t *indices,
        const float4 *imgPositions,
        const float4 *imgInvSigmas,
        const float *alphas,
        uint2 numTiles,
        uint32_t width,
        uint32_t height);

    // Equivalente a computeGaussiansCovisibility_kernel en vigs-fusion
    /**
     * @brief computeGaussiansCovisibility_kernel - Calcular covisibilidad entre frames
     * Computa intersección y unión de dos conjuntos de visibilidad.
     * 
     * MATEMÁTICA:
     * Para cada gaussiana i:
     * - Si vis1[i] || vis2[i]: incrementa counter UNIÓN
     * - Si vis1[i] && vis2[i]: incrementa counter INTERSECCIÓN
     * 
     *  Ratio = |intersección| / |unión|
     * - Ratio cercano a 1: dos frames ven aprox. las mismas gaussianas (buena covisibilidad)
     * - Ratio cercano a 0: frames ven conjuntos muy diferentes (poca covisibilidad)
     * 
     * SALIDA: usa atomicAggInc() para actualizar contadores globales de forma thread-safe
     * 
     * @param visibilityInter   [OUT] Contador global de gaussianas en INTERSECCIÓN
     * @param visibilityUnion   [OUT] Contador global de gaussianas en UNIÓN
     * @param visibilities1     [IN]  Array de visibilidad del frame 1 [nbGaussians]
     * @param visibilities2     [IN]  Array de visibilidad del frame 2 [nbGaussians]
     * @param nbGaussians       [IN]  Número total de gaussianas
     */
    __global__ void computeGaussiansCovisibility_kernel(
        uint32_t *visibilityInter,
        uint32_t *visibilityUnion,
        unsigned char *visibilities1,
        unsigned char *visibilities2,
        uint32_t nbGaussians);

    // =======================================================================
    // Gestión de Mapa: Prune, Outliers y Densificación
    // =======================================================================

    // Equivalente a pruneGaussians_kernel en vigs-fusiob
    /**
     * @brief pruneGaussians_kernel - Eliminar gaussianas de baja calidad
     * Elimina gaussianas que ya no son útiles según criterios de calidad:
     * 1. Opacidad muy baja (alpha < threshold): no contribuyen visualmente
     * 2. Covarianza degenerada: ratio entre escalas muy bajo o escala máxima muy pequeña
     * 
     * CRITERIOS (basados en VIGS-Fusion):
     * - alpha < 0.05: eliminar (contribuye <5% opacidad)
     * - s_medio / s_max < 0.05: gaussiana muy plana (degenerada)
     * - s_max < 0.005: gaussiana muy pequeña (ruido)
     * 
     * SALIDA: states[i] = 0xff si debe eliminarse, 0 si debe mantenerse
     * 
     * @param nbRemoved         [OUT] Contador atómico de gaussianas eliminadas
     * @param states            [OUT] Array de estados: 0=keep, 0xff=remove [nbGaussians]
     * @param scales            [IN]  Escalas 3D [nbGaussians]
     * @param alphas            [IN]  Opacidades [nbGaussians]
     * @param alphaThreshold    [IN]  Umbral de opacidad mínima (por defecto 0.05)
     * @param scaleRatioThreshold [IN] Umbral de ratio escala media/max (por defecto 0.05)
     * @param nbGaussians       [IN]  Número total de gaussianas
     */
    __global__ void pruneGaussians_kernel(
        uint32_t *nbRemoved,
        unsigned char *states,
        const float4 *scales,
        const float *alphas,
        float alphaThreshold,
        float scaleRatioThreshold,
        uint32_t nbGaussians);

    // Equivalente a computeOutliers_kernel en vigs-fusion
    /**
     * @brief computeOutliers_kernel - Detectar gaussianas outlier por profundidad
     * Detecta gaussianas outliers comparando profundidad renderizada vs observada.
     * 
     * ALGORITMO:
     * Para cada pixel (x,y) en el tile:
     * 1. Renderizar profundidad de cada gaussiana: d = z_center + pHat·(x-u, y-v)
     * 2. Si depth_observed > 0.1 && depth_rendered < 0.8 * depth_observed:
     *    → La gaussiana está "flotando" delante de la superficie real
     *    → Acumular outlierProb[gid] += alpha_i
     * 3. Acumular totalAlpha[gid] += alpha_i (normalización)
     * 
     * SALIDA: Arrays outlierProb y totalAlpha para posterior decisión de eliminación
     * 
     * @param outlierProb       [OUT] Probabilidad acumulada de outlier [nbGaussians]
     * @param totalAlpha        [OUT] Alpha total acumulado [nbGaussians]
     * @param ranges            [IN]  Rangos de índices por tile
     * @param indices           [IN]  Índices ordenados de gaussianas
    * @param positions_2d      [IN]  Posiciones proyectadas (u, v)
    * @param inv_covariances_2d [IN] Covarianzas 2D invertidas
    * @param depths            [IN]  Profundidades por gaussiana
    * @param alphas            [IN]  Opacidades por gaussiana
     * @param pHats             [IN]  Coeficientes de variación de profundidad
    * @param depth             [IN]  Profundidad observada (puntero GPU)
    * @param depth_step        [IN]  Paso en bytes entre filas de depth
     * @param numTiles          [IN]  Dimensiones de grilla de tiles
     * @param width             [IN]  Ancho de imagen
     * @param height            [IN]  Alto de imagen
     */
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
        uint32_t n_instances_max);

    // Equivalente a removeOutliers_kernel en vigs-fusion
    /**
     * @brief removeOutliers_kernel - Eliminar gaussianas outlier
     * Decide qué gaussianas eliminar según el ratio outlierProb/totalAlpha.
     * 
     * CRITERIO:
     * Si (totalAlpha[i] > 1.0) && (outlierProb[i] / totalAlpha[i] > threshold):
     *   → La gaussiana es visible Y está mayormente "delante" de la superficie
     *   → Marcar para eliminar: states[i] = 0xff
     * 
     * threshold típico: 0.6 (60% de sus contribuciones son outliers)
     * 
     * @param nbRemoved         [OUT] Contador atómico de gaussianas eliminadas
     * @param states            [OUT] Array de estados: 0=keep, 0xff=remove
     * @param outlierProb       [IN]  Probabilidad acumulada de outlier
     * @param totalAlpha        [IN]  Alpha total acumulado
     * @param threshold         [IN]  Umbral de ratio (típ. 0.6)
     * @param nbGaussians       [IN]  Número total de gaussianas
     */
    __global__ void removeOutliers_kernel(
        uint32_t *nbRemoved,
        unsigned char *states,
        const float *outlierProb,
        const float *totalAlpha,
        float threshold,
        uint32_t nbGaussians);


    // Equivalente a computeDensityMask_kernel en vigs-fusion
    /**
     * @brief computeDensityMask_kernel - Generar mask de densidad por pixel
     * Genera un mask float por pixel para guiar densificacion.
     * maskData: [height x width] en float (step en bytes).
     *
     * @param maskData   [OUT] Mask por pixel (float) en GPU
     * @param ranges     [IN]  Rangos de indices por tile
     * @param indices    [IN]  Indices ordenados de gaussianas
     * @param imgPositions [IN] Posiciones proyectadas (u, v)
     * @param imgInvSigmas [IN] Covarianzas 2D invertidas
     * @param pHats      [IN]  Coeficientes de variacion de profundidad
     * @param alphas     [IN]  Opacidades por gaussiana
     * @param depthTex   [IN]  Profundidad observada (textura CUDA)
     * @param numTiles   [IN]  Dimensiones de la grilla de tiles
     * @param width      [IN]  Ancho de imagen
     * @param height     [IN]  Alto de imagen
     * @param mask_stride [IN] Paso en elementos (float) entre filas
    */
    __global__ void computeDensityMask_kernel(
        float *__restrict__ maskData,
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
        uint32_t n_instances_max);


    // Equivalente a densifyGaussians_kernel en vigs-fusion
    /**
     * @brief densifyGaussians_kernel - Generar nuevas gaussianas
     * Genera nuevas gaussianas a partir de un mask de densidad y RGB-D.
     * 
     * Para cada celda de muestreo (sample_dx x sample_dy):
     * - Si el mask marca suficiente area, promedia color/profundidad
     * - Reproyecta a 3D y crea una nueva gaussiana
     * 
     * @param positions       [INOUT] Posiciones 3D (se agregan al final)
     * @param scales          [INOUT] Escalas 3D
     * @param orientations    [INOUT] Orientaciones (quaternions)
     * @param colors          [INOUT] Colores RGB
     * @param alphas          [INOUT] Opacidades
     * @param instanceCounter [INOUT] Contador global (iniciar con nbGaussians)
     * @param rgb             [IN]  Imagen color (uchar3, BGR)
     * @param rgb_step        [IN]  Paso en bytes entre filas de rgb
     * @param depth           [IN]  Imagen de profundidad (float)
     * @param depth_step      [IN]  Paso en bytes entre filas de depth
    * @param normals         [IN]  Imagen de normales (float3) o nullptr
     * @param normals_step    [IN]  Paso en bytes entre filas de normals
     * @param mask            [IN]  Mask de densidad (float)
     * @param mask_step       [IN]  Paso en bytes entre filas de mask
     * @param cameraPose      [IN]  Pose de camara
     * @param intrinsics      [IN]  Intrinsecos de camara
     * @param sample_dx       [IN]  Tamaño de celda en x
     * @param sample_dy       [IN]  Tamaño de celda en y
     * @param width           [IN]  Ancho de imagen
     * @param height          [IN]  Alto de imagen
     * @param maxGaussians    [IN]  Capacidad maxima del array
     */
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
        uint32_t maxGaussians);

    // Equivalente a perTileBucketCount en vigs-fusion
    /**
     * @brief perTileBucketCount - Contar buckets por tile
     * Cuenta la cantidad de buckets por tile para el pase de optimizacion.
     * Cada bucket agrupa hasta 32 gaussianas del tile.
     *
     * @param bucketCount [OUT] Buckets por tile [numTiles]
     * @param tileRanges  [IN]  Rangos [start, end) por tile
     * @param numTiles    [IN]  Numero total de tiles
     */
    __global__ void perTileBucketCount(
        uint32_t *bucketCount,
        const uint2 *tileRanges,
        int numTiles);

    // Equivalente a optimizeGaussiansForwardPass en vigs-fusion
    /**
     * @brief optimizeGaussiansForwardPass - Pase hacia adelante de optimización
     * Forward pass por tile: compone color/profundidad, calcula errores y buffers auxiliares.
     *
     * @param ranges           [IN]  Rangos [start, end) por tile
     * @param indices          [IN]  Indices ordenados de gaussianas
     * @param positions_2d     [IN]  Centros proyectados (u, v)
     * @param covariances_2d   [IN]  Covarianzas 2D (xx, yy, xy)
     * @param inv_covariances_2d [IN] Covarianzas 2D invertidas
     * @param p_hats           [IN]  Coeficientes de profundidad por pixel
     * @param colors           [IN]  Colores RGB
     * @param alphas           [IN]  Opacidades
     * @param per_tile_buckets [IN]  Buckets por tile
     * @param bucket_to_tile   [OUT] Mapa bucket -> tile
     * @param sampled_T        [OUT] Transmitancias muestreadas por bucket
     * @param sampled_ar       [OUT] Acumuladores de color por bucket
     * @param final_T          [OUT] Transmitancia final por pixel
     * @param n_contrib        [OUT] Numero de contribuciones por pixel
     * @param max_contrib      [OUT] Max contribuciones por tile
     * @param output_color     [OUT] Color renderizado por pixel
     * @param output_depth     [OUT] Profundidad renderizada por pixel
     * @param color_error      [OUT] Error de color por pixel
     * @param depth_error      [OUT] Error de profundidad por pixel
     * @param observed_rgb     [IN]  RGB observado (float3)
     * @param observed_depth   [IN]  Depth observado (float)
     * @param bg_color         [IN]  Color de fondo
     * @param num_tiles        [IN]  Dimensiones de tiles (x, y)
     * @param width            [IN]  Ancho de imagen
     * @param height           [IN]  Alto de imagen
     */
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
        int height);

    // Equivalente a optimizeGaussiansPerGaussianPass en vigs-fusion
    /**
     * @brief optimizeGaussiansPerGaussianPass - Pase hacia atrás por gaussiana
     * Backward pass por bucket: acumula gradientes por gaussiana (DeltaGaussian2D).
     *
     * @param ranges           [IN]  Rangos [start, end) por tile
     * @param indices          [IN]  Indices ordenados de gaussianas
     * @param positions_2d     [IN]  Centros proyectados (u, v)
     * @param inv_covariances_2d [IN] Covarianzas 2D invertidas
     * @param p_hats           [IN]  Coeficientes de profundidad por pixel
    * @param depths           [IN]  Profundidades Z
     * @param colors           [IN]  Colores RGB
     * @param alphas           [IN]  Opacidades
    * @param bucketOffsets    [IN]  Prefix sum de buckets por tile
     * @param bucket_to_tile   [IN]  Mapa bucket -> tile
     * @param sampled_T        [IN]  Transmitancias muestreadas
     * @param sampled_ar       [IN]  Acumuladores de color muestreados
     * @param n_contrib        [IN]  Numero de contribuciones por pixel
     * @param max_contrib      [IN]  Max contribuciones por tile
     * @param output_color     [IN]  Color renderizado por pixel
     * @param output_depth     [IN]  Profundidad renderizada por pixel
     * @param color_error      [IN]  Error de color por pixel
     * @param depth_error      [IN]  Error de profundidad por pixel
     * @param delta_gaussians  [OUT] Acumulador de gradientes 2D por gaussiana
     * @param w_depth          [IN]  Peso de profundidad
     * @param w_dist           [IN]  Peso de regularizacion/distancia
     * @param num_tiles        [IN]  Dimensiones de tiles (x, y)
     * @param width            [IN]  Ancho de imagen
     * @param height           [IN]  Alto de imagen
     * @param num_buckets      [IN]  Numero total de buckets
     */
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
        int num_buckets);

    // Equivalente a computeDeltaGaussians3D_kernel en vigs-fusion
    /**
     * @brief computeDeltaGaussians3D_kernel - Convertir gradientes 2D a 3D
     * Convierte gradientes 2D (DeltaGaussian2D) a gradientes 3D por gaussiana.
     *
     * @param delta_gaussians_3d [OUT] Gradientes 3D acumulados
     * @param positions          [IN]  Medias 3D
     * @param scales             [IN]  Escalas 3D
     * @param orientations       [IN]  Orientaciones (quat)
     * @param colors             [IN]  Colores
     * @param alphas             [IN]  Opacidades
     * @param delta_gaussians_2d [IN]  Gradientes 2D
     * @param camera_pose        [IN]  Pose de camara
     * @param intrinsics         [IN]  Intrinsecos
     * @param lambda_iso         [IN]  Peso de regularizacion isotropica
     * @param n_gaussians        [IN]  Numero de gaussianas
     */
    __global__ void computeDeltaGaussians3D_kernel(
        DeltaGaussian3D *delta_gaussians_3d,
        const float4 *positions,
        const float4 *scales,
        const float4 *orientations,
        const float4 *colors,
        const float *alphas,
        const DeltaGaussian2D *delta_gaussians_2d,
        Pose camera_pose,
        IntrinsicParameters intrinsics,
        float lambda_iso,
        int n_gaussians);

    // =======================================================================
    // Algoritmo ADAM de optimizacion
    // =======================================================================
    /**
     * @brief adamStep - Actualizar estimaciones de momentum y RMSprop, aplicar corrección de sesgo
     * 
     * Computa un paso de actualización de Adam para un solo parámetro escalar.
     * Actualiza m = (1-beta1)*grad + beta1*m y v = (1-beta2)*grad^2 + beta2*v
     * Y luego aplica corrección de sesgo: m_hat = m / (1 - beta1^t), v_hat = v / (1 - beta2^t)
     * 
     * @param m Momentum estimate (first moment), updated in-place
     * @param v RMSprop estimate (second moment), updated in-place
     * @param grad Gradient value
     * @param eta Learning rate
     * @param alpha1 (1 - beta1) coefficient for momentum
     * @param beta1 Momentum decay rate
     * @param beta1t Bias correction term 1/(1 - beta1^t)
     * @param alpha2 (1 - beta2) coefficient for RMSprop
     * @param beta2 RMSprop decay rate
     * @param beta2t Bias correction term 1/(1 - beta2^t)
     * @param epsilon Numerical stability constant
     * @return Parameter update: eta * m_hat / sqrt(v_hat + eps)
     */
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
                                          const float epsilon);

    /**
     * @brief adamStep (para float3) - Actualizar momentum y RMSprop para vector 3D
     * 
     * Aplica adamStep elemento a elemento en cada componente de un vector 3D.
     * 
     * @param m 3D momentum estimate, updated in-place
     * @param v 3D RMSprop estimate, updated in-place
     * @param grad 3D gradient vector
     * @param eta Learning rate
     * @param alpha1 (1 - beta1) coefficient for momentum
     * @param beta1 Momentum decay rate
     * @param beta1t Bias correction term 1/(1 - beta1^t)
     * @param alpha2 (1 - beta2) coefficient for RMSprop
     * @param beta2 RMSprop decay rate
     * @param beta2t Bias correction term 1/(1 - beta2^t)
     * @param epsilon Numerical stability constant
     * @return Parameter update: eta * m_hat / sqrt(v_hat + eps) per component
     */
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
                                           const float epsilon);

    // Equivalente a updateGaussiansParametersAdam_kernel en vigs-fusion
    /**
     * @brief updateGaussiansParametersAdam_kernel - Actualizar parámetros con Adam
     * Aplica una actualizacion Adam a los parametros 3D de gaussianas.
     *
     * @param positions    [IN/OUT] Medias 3D
     * @param scales       [IN/OUT] Escalas 3D
     * @param orientations [IN/OUT] Orientaciones (quat)
     * @param colors       [IN/OUT] Colores
     * @param alphas       [IN/OUT] Opacidades
     * @param adam_states  [IN/OUT] Estado Adam por gaussiana
     * @param deltas_3d    [IN]     Gradientes 3D
     * @param adam_eta     [IN]     Tasa de aprendizaje
     * @param adam_beta1   [IN]     Beta1
     * @param adam_beta2   [IN]     Beta2
     * @param adam_eps     [IN]     Epsilon
     * @param n_gaussians  [IN]     Numero de gaussianas
     */
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
        int n_gaussians);
} // namespace f_vigs_slam