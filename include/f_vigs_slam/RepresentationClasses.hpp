#pragma once

#include <thrust/device_vector.h>
#include <thrust/copy.h>
#include <thrust/tuple.h>
#include <Eigen/Geometry>
#include <opencv2/core/cuda.hpp>
#include <cuda_runtime.h>
#include <memory>
#include <utility>
#include <vector>
#include <unordered_map>
#include <iostream>
#include <cmath>

// Definimos estructuras de datos auxiliares.

namespace f_vigs_slam
{
    /**
     * @brief Gaussian
     * Representa una gaussiana 3D con sus parámetros principales.
     * 
     * NOTA ARQUITECTURAL: Todas las coordenadas (position, scale, orientation)
     * se almacenan en coordenadas LOCALES del submapa. Para obtener coordenadas
     * globales, se debe aplicar la transformación T_submap_global del submapa.
     */
    struct Gaussian
    {
        float3 position;    /// Media (en coordenadas LOCALES del submapa)
        float3 scale;       /// Varianza (asumimos que no hay correlacion)
        float4 orientation; /// Quaternion (x, y, z, w) (en coordenadas LOCALES)
        float3 color;       /// Color RGB [0,1]
        float opacity;      /// Opacidad [0,1]

        // Constructores
        __host__ __device__ Gaussian()
            : position(make_float3(0.f, 0.f, 0.f)),
              scale(make_float3(1.f, 1.f, 1.f)),
              orientation(make_float4(0.f, 0.f, 0.f, 1.f)),
              color(make_float3(1.f, 1.f, 1.f)),
              opacity(1.f)
        {
        }

        __host__ __device__ Gaussian(float3 pos,
                                     float3 scl,
                                     float4 ori,
                                     float3 col,
                                     float op)
            : position(pos),
              scale(scl),
              orientation(ori),
              color(col),
              opacity(op)
        {
        }
    };

    struct AdamStateGaussian3D
    {
        float3 m_position;
        float3 v_position;
        float3 m_scale;
        float3 v_scale;
        float3 m_orientation;
        float3 v_orientation;
        float3 m_color;
        float3 v_color;
        float m_alpha;
        float v_alpha;
        float k;
    };

    /**
     * @brief Gaussians
     * Contenedor SoA para parámetros de gaussianas con iteradores zip.
     */
    struct Gaussians
    {
        thrust::device_vector<float4> positions;
        thrust::device_vector<float4> scales;
        thrust::device_vector<float4> orientations;
        thrust::device_vector<float4> colors;
        thrust::device_vector<float> opacities;

        void resize(size_t len)
        {
            positions.resize(len);
            scales.resize(len);
            orientations.resize(len);
            colors.resize(len);
            opacities.resize(len);
        }

        typedef thrust::tuple<
            thrust::device_vector<float4>::iterator,
            thrust::device_vector<float4>::iterator,
            thrust::device_vector<float4>::iterator,
            thrust::device_vector<float4>::iterator,
            thrust::device_vector<float>::iterator>
            Tuple;

        typedef thrust::zip_iterator<Tuple> iterator;

        iterator begin()
        {
            return thrust::make_zip_iterator(thrust::make_tuple(
                positions.begin(),
                scales.begin(),
                orientations.begin(),
                colors.begin(),
                opacities.begin()));
        }

        iterator end()
        {
            return thrust::make_zip_iterator(thrust::make_tuple(
                positions.end(),
                scales.end(),
                orientations.end(),
                colors.end(),
                opacities.end()));
        }
    };


    template<typename T>
    cudaTextureObject_t createTextureObject(const cv::cuda::GpuMat& img)
    {
        // ============================================================
        // 1. RESOURCE DESCRIPTION
        // ============================================================
        cudaResourceDesc resDesc{};
        resDesc.resType = cudaResourceTypePitch2D;

        resDesc.res.pitch2D.devPtr = const_cast<void*>(
            static_cast<const void*>(img.ptr())
        );

        resDesc.res.pitch2D.desc = cudaCreateChannelDesc<T>();
        resDesc.res.pitch2D.width = img.cols;
        resDesc.res.pitch2D.height = img.rows;
        resDesc.res.pitch2D.pitchInBytes = img.step;

        // ============================================================
        // 2. TEXTURE DESCRIPTION
        // ============================================================
        cudaTextureDesc texDesc{};
        texDesc.addressMode[0] = cudaAddressModeClamp;
        texDesc.addressMode[1] = cudaAddressModeClamp;

        texDesc.filterMode = cudaFilterModePoint;   // no interpolación
        texDesc.readMode = cudaReadModeElementType;
        texDesc.normalizedCoords = 0;               // coords en píxeles

        // ============================================================
        // 3. CREATE
        // ============================================================
        cudaTextureObject_t tex = 0;

        cudaError_t err = cudaCreateTextureObject(&tex, &resDesc, &texDesc, nullptr);
        if (err != cudaSuccess)
        {
            std::cerr << "[CUDA ERROR] createTextureObject: "
                    << cudaGetErrorString(err) << std::endl;
            return 0;
        }

        return tex;
    }

    inline void destroyTextureVector(std::vector<cudaTextureObject_t>& tex_vec)
    {
        for (auto& tex : tex_vec)
        {
            if (tex != 0)  // 0 = textura inválida
            {
                cudaDestroyTextureObject(tex);
                tex = 0;   // evitar dangling handle
            }
        }

        tex_vec.clear(); // liberar el vector
    }

    template<typename T>
    void updateTexture(cudaTextureObject_t& tex, const cv::cuda::GpuMat& img)
    {
        if (tex) {
            cudaDestroyTextureObject(tex);
            tex = 0;
        }
        tex = createTextureObject<T>(img);
    }

    /**
     * @brief Texture
     * Template class that encapsulates CUDA texture object management with RAII.
     * Automatically handles creation and destruction of texture objects.
     */
    template <typename T>
    class Texture
    {
    private:
        cudaTextureObject_t textureObject = 0;
        cudaTextureFilterMode filterMode = cudaFilterModePoint;

    public:
        /**
         * @brief Constructor that creates a CUDA texture object from a GpuMat.
         * @param img The GPU matrix to create the texture from
         * @param filterMode_ The texture filter mode (default: cudaFilterModePoint)
         */
        Texture(const cv::cuda::GpuMat &img, cudaTextureFilterMode filterMode_ = cudaFilterModePoint)
        {
            filterMode = filterMode_;
            if (img.empty())
            {
                std::cerr << "[CUDA ERROR] Texture constructor: empty GpuMat" << std::endl;
                textureObject = 0;
                return;
            }

            struct cudaResourceDesc resDesc;
            memset(&resDesc, 0, sizeof(resDesc));
            resDesc.resType = cudaResourceTypePitch2D;
            resDesc.res.pitch2D.devPtr = const_cast<void*>(
                static_cast<const void*>(img.ptr())
            );
            resDesc.res.pitch2D.width = img.cols;
            resDesc.res.pitch2D.height = img.rows;
            resDesc.res.pitch2D.pitchInBytes = img.step;
            resDesc.res.pitch2D.desc = cudaCreateChannelDesc<T>();
            
            struct cudaTextureDesc texDesc;
            memset(&texDesc, 0, sizeof(texDesc));
            texDesc.addressMode[0] = cudaAddressModeClamp;
            texDesc.addressMode[1] = cudaAddressModeClamp;
            texDesc.filterMode = filterMode;
            texDesc.readMode = cudaReadModeElementType;
            texDesc.normalizedCoords = 0;

            cudaError_t err = cudaCreateTextureObject(&textureObject, &resDesc, &texDesc, NULL);
            if (err != cudaSuccess)
            {
                std::cerr << "[CUDA ERROR] Texture constructor: "
                        << cudaGetErrorString(err) << std::endl;
                textureObject = 0;
            }
        }

        Texture(const Texture&) = delete;
        Texture& operator=(const Texture&) = delete;

        Texture(Texture&& other) noexcept
            : textureObject(other.textureObject), filterMode(other.filterMode)
        {
            other.textureObject = 0;
        }

        Texture& operator=(Texture&& other) noexcept
        {
            if (this != &other)
            {
                if (textureObject != 0)
                {
                    cudaDestroyTextureObject(textureObject);
                }
                textureObject = other.textureObject;
                filterMode = other.filterMode;
                other.textureObject = 0;
            }
            return *this;
        }

        /**
         * @brief Destructor that destroys the CUDA texture object.
         */
        ~Texture()
        {
            if (textureObject != 0)
            {
                cudaDestroyTextureObject(textureObject);
                textureObject = 0;
            }
        }

        /**
         * @brief Get reference to the underlying CUDA texture object.
         * @return Reference to the cudaTextureObject_t
         */
        inline cudaTextureObject_t &getTextureObject() { return textureObject; }

        /**
         * @brief Get const reference to the underlying CUDA texture object.
         * @return Const reference to the cudaTextureObject_t
         */
        inline const cudaTextureObject_t &getTextureObject() const { return textureObject; }

    }; // class Texture

    struct IntrinsicParameters
    {
        // f longitud focal, c centro optico
        float2 f, c;
    };

    struct Pose {
        // Pose de la camara representada por posicion y orientacion
        // Quaternion formato: (x, y, z, w)
        
        float3 position;     // Posicion (x, y, z)
        float4 orientation;  // Quaternion (x, y, z, w)

        // Constructor por defecto: identidad
        __device__ __host__ Pose() 
            : position(make_float3(0.0f, 0.0f, 0.0f)),
              orientation(make_float4(0.0f, 0.0f, 0.0f, 1.0f)) {}

        // Constructor con valores
        __device__ __host__ Pose(float3 pos, float4 quat)
            : position(pos), orientation(quat) {}

        // Pose::Identity devuelve la pose base con pos en origen y sin rotacion
        __device__ __host__ static Pose Identity() {
            float3 pos = make_float3(0.0f, 0.0f, 0.0f);
            float4 quat = make_float4(0.0f, 0.0f, 0.0f, 1.0f);
            return Pose(pos, quat);
        }
        
    };

    // ============================================================
    // FASE 7: FUNCIONES HELPER PARA TRANSFORMACIONES DE POSES
    // ============================================================
    
    /**
     * @brief Multiplica dos poses en CPU: result = T1 * T2
     * Fórmula SE(3): (R1, t1) * (R2, t2) = (R1*R2, R1*t2 + t1)
     * Para quaternios: q_result = q1 * q2
     */
    inline Pose composePoses(const Pose& T1, const Pose& T2)
    {
        // Multiplicación de quaternios: q1 * q2
        float4 q1 = T1.orientation;
        float4 q2 = T2.orientation;
        
        float4 q_result;
        q_result.x = q1.w * q2.x + q1.x * q2.w + q1.y * q2.z - q1.z * q2.y;
        q_result.y = q1.w * q2.y - q1.x * q2.z + q1.y * q2.w + q1.z * q2.x;
        q_result.z = q1.w * q2.z + q1.x * q2.y - q1.y * q2.x + q1.z * q2.w;
        q_result.w = q1.w * q2.w - q1.x * q2.x - q1.y * q2.y - q1.z * q2.z;
        
        // Rotación de t2 por q1 + t1
        float3 t2 = T2.position;
        float qx = q1.x, qy = q1.y, qz = q1.z, qw = q1.w;
        float xx = qx*qx, yy = qy*qy, zz = qz*qz;
        float xy = qx*qy, xz = qx*qz, xw = qx*qw;
        float yz = qy*qz, yw = qy*qw, zw = qz*qw;
        
        float3 R1_t2;
        R1_t2.x = (1 - 2*(yy + zz)) * t2.x + 2*(xy - zw) * t2.y + 2*(xz + yw) * t2.z;
        R1_t2.y = 2*(xy + zw) * t2.x + (1 - 2*(xx + zz)) * t2.y + 2*(yz - xw) * t2.z;
        R1_t2.z = 2*(xz - yw) * t2.x + 2*(yz + xw) * t2.y + (1 - 2*(xx + yy)) * t2.z;
        
        float3 t_result;
        t_result.x = T1.position.x + R1_t2.x;
        t_result.y = T1.position.y + R1_t2.y;
        t_result.z = T1.position.z + R1_t2.z;
        
        // Normalize quaternion to avoid numerical drift
        float qnorm = std::sqrt(q_result.x*q_result.x + q_result.y*q_result.y + q_result.z*q_result.z + q_result.w*q_result.w);
        if (qnorm > 1e-12f) {
            q_result.x /= qnorm;
            q_result.y /= qnorm;
            q_result.z /= qnorm;
            q_result.w /= qnorm;
        }

        return Pose(t_result, q_result);
    }
    
    /**
     * @brief Invierte una pose en CPU: result = T^{-1}
     * Fórmula SE(3): (R, t)^{-1} = (R^T, -R^T * t)
     * Para quaternios: q^{-1} = conj(q) / |q|^2 = conj(q) si normalizado
     */
    inline Pose invertPose(const Pose& T)
    {
        // Invertir quaternion: conjugado
        float4 q = T.orientation;
        float4 q_inv;
        q_inv.x = -q.x;
        q_inv.y = -q.y;
        q_inv.z = -q.z;
        q_inv.w = q.w;
        
        // Invertir traslación: -R^T * t
        float3 t = T.position;
        float qx = q.x, qy = q.y, qz = q.z, qw = q.w;
        float xx = qx*qx, yy = qy*qy, zz = qz*qz;
        float xy = qx*qy, xz = qx*qz, xw = qx*qw;
        float yz = qy*qz, yw = qy*qw, zw = qz*qw;
        
        float3 t_inv;
        t_inv.x = -((1 - 2*(yy + zz)) * t.x + 2*(xy + zw) * t.y + 2*(xz - yw) * t.z);
        t_inv.y = -(2*(xy - zw) * t.x + (1 - 2*(xx + zz)) * t.y + 2*(yz + xw) * t.z);
        t_inv.z = -(2*(xz + yw) * t.x + 2*(yz - xw) * t.y + (1 - 2*(xx + yy)) * t.z);
        
        return Pose(t_inv, q_inv);
    }

    struct KeyframeData
    {
        // ===== GPU DATA =====
        cv::cuda::GpuMat color_img;
        cv::cuda::GpuMat depth_img;
        cv::cuda::GpuMat normal_img;

        // TEXTURAS
        std::shared_ptr<Texture<uchar4>> color_tex;
        std::shared_ptr<Texture<float>> depth_tex;
        std::shared_ptr<Texture<float4>> normal_tex;

        // ===== POSE (Marco Local del Submapa) =====
        Pose T_relative;                    // Pose relativa al submapa padre
        Pose T_global_cached;               // Pose global (pre-computada, relativa a origin global)
        bool pose_cache_valid = false;      // Flag de validez del caché
        IntrinsicParameters intrinsics;

        // ===== META =====
        uint32_t keyframe_id = 0;
        double timestamp = 0.0;

        // ===== Descriptor NetVLAD =====
        std::vector<float> netvlad_descriptor;  // CPU descriptor
        thrust::device_vector<float> netvlad_descriptor_gpu;  // GPU descriptor (kept in VRAM during detection)

        // ===== Descriptor GPU Methods =====
        inline void copyDescriptorToGpu()
        {
            if (!netvlad_descriptor.empty()) {
                netvlad_descriptor_gpu = thrust::device_vector<float>(netvlad_descriptor.begin(), netvlad_descriptor.end());
            }
        }
        
        inline void copyDescriptorFromGpu()
        {
            if (!netvlad_descriptor_gpu.empty()) {
                netvlad_descriptor.resize(netvlad_descriptor_gpu.size());
                thrust::copy(netvlad_descriptor_gpu.begin(), netvlad_descriptor_gpu.end(), netvlad_descriptor.begin());
            }
        }
        
        inline bool hasGpuDescriptor() const { return !netvlad_descriptor_gpu.empty(); }
        inline bool hasCpuDescriptor() const { return !netvlad_descriptor.empty(); }

        // ===== HELPERS =====
        inline Texture<uchar4>& getColorTex() { return *color_tex; }
        inline Texture<float>& getDepthTex() { return *depth_tex; }
        inline Texture<float4>& getNormalTex() { return *normal_tex; }
        inline const Texture<uchar4>& getColorTex() const { return *color_tex; }
        inline const Texture<float>& getDepthTex() const { return *depth_tex; }
        inline const Texture<float4>& getNormalTex() const { return *normal_tex; }
        inline int getWidth() const { return color_img.cols; }
        inline int getHeight() const { return color_img.rows; }
        inline IntrinsicParameters getIntrinsics() const { return intrinsics; }
        
        // Accesores de pose (retrocompatibilidad)
        inline Pose getPose() const { return pose_cache_valid ? T_global_cached : Pose::Identity(); }
        inline Pose getRelativePose() const { return T_relative; }
        inline Pose getGlobalPose() const { return pose_cache_valid ? T_global_cached : Pose::Identity(); }
        
        // Actualiza la pose global basada en la pose global del submapa padre
        // Fórmula: T_global_keyframe = T_submap_global * T_keyframe_relative
        inline void updateGlobalPose(const Pose& submap_global_pose)
        {
            if (pose_cache_valid) {
                // Si ya está válida, verificar si el submapa cambió
                // Por ahora, siempre recomputar para ser conservador
            }
            
            // Componer poses: T_global = T_submap * T_relative
            T_global_cached = composePoses(submap_global_pose, T_relative);
            pose_cache_valid = true;
        }
        
        // Invalida el caché global (debe llamarse si se modifica T_relative)
        inline void invalidatePoseCache()
        {
            pose_cache_valid = false;
        }
        
        // Establece la pose relativa e invalida caché
        inline void setRelativePose(const Pose& rel_pose)
        {
            T_relative = rel_pose;
            invalidatePoseCache();
        }
        // Guarda un descriptor copiado en el keyframe.
        inline void setDescriptor(const std::vector<float>& desc) { netvlad_descriptor = desc; }
        // Mueve un descriptor al keyframe sin copia extra.
        inline void setDescriptor(std::vector<float>&& desc) { netvlad_descriptor = std::move(desc); }
        // Devuelve el descriptor asociado al keyframe.
        inline const std::vector<float>& getDescriptor() const { return netvlad_descriptor; }
        // Indica si el keyframe ya tiene descriptor asignado.
        inline bool hasDescriptor() const { return !netvlad_descriptor.empty(); }

        KeyframeData(
            const cv::cuda::GpuMat& rgb,
            const cv::cuda::GpuMat& depth,
            const cv::cuda::GpuMat& normal,

            const Pose& p,
            const IntrinsicParameters& K,
            uint32_t id,
            double ts)
        {
            // Guardar imágenes
            color_img = rgb.clone();
            depth_img = depth.clone();
            normal_img = normal.clone();

            T_relative = p;  // Inicialmente se asume que p es la pose relativa
            pose_cache_valid = false;  // Caché no valido hasta que se actualice
            intrinsics = K;
            keyframe_id = id;
            timestamp = ts;

            // color_img CV_8UC4 (BGRA)
            color_tex = std::make_shared<Texture<uchar4>>(color_img);
            depth_tex = std::make_shared<Texture<float>>(depth_img);
            normal_tex = std::make_shared<Texture<float4>>(normal_img);
        }
        ~KeyframeData() = default;
    };

    struct ImuData
    {
        // En este struct guardamos la informacion medida y los parametros
        // del modelo dinamico del IMU

        // Mediciones
        Eigen::Vector3d Acc;
        Eigen::Vector3d Gyro;

        // Ruido (noise) y modelo de camino (walk) de acelerometro y giroscopio
        double acc_n;
        double gyr_n;
        double acc_w;
        double gyr_w;
        double g_norm;
    };

    struct PoseOptimizationRgbdData {
        
        // En este struct guardamos la jacobiana*residuo Jtr y la aproximacion
        // del hessiano JtJ para el factor RGB-D

        float Jtr[6];
        float JtJ[21]; // matriz 6x6 simetrica, guardamos solo triangulo sup

        // Definimos el operador suma para GPU y CPU como la suma matricial comun
        // para ambos terminos en la misma operacion
        __device__ __host__ inline PoseOptimizationRgbdData &operator+=(const PoseOptimizationRgbdData &d)
        {
            #pragma unroll
            for(int i=0; i<6; i++)
            {
                Jtr[i]+=d.Jtr[i];
            }
            #pragma unroll
            for(int i=0; i<21; i++)
            {
                JtJ[i]+=d.JtJ[i];
            }

            return *this;
        }
    };

    #if 0
    struct PoseKernelDebugInfo {
        // 0 means no error recorded yet. Stage codes are defined in GSCudaKernels.cu.
        int first_stage = 0;
        int first_x = -1;
        int first_y = -1;
        float first_a = 0.f;
        float first_b = 0.f;
        float first_c = 0.f;
        float first_d = 0.f;

        unsigned int total_nonfinite_events = 0u;
        unsigned int stage_counts[16] = {0u};
    };
    #endif
    
    struct PoseOptimizationMetrics {
        // En este struct guardamos metricas de optimizacion de pose

        // NO IMPLEMENTADA AUN
    };


    inline void add_unique(std::vector<int> &vec, int value)
    {
        if (std::find(vec.begin(), vec.end(), value) == vec.end())
        {
            vec.push_back(value);
        }
    }


    /**
     * @brief Submap
     * Estructura que agrupa un conjunto de Gaussianas 3D y keyframes locales.
     * Cada submapa forma una unidad independiente de mapeo y tracking.
     * 
     * ARQUITECTURA DE POSES:
     * - T_relative: Transformacion de este submapa respecto al submapa anterior (cadena)
     * - T_global_cached: Pose acumulada en el sistema global (T_global = T_prev * T_relative)
     * - Keyframes y gaussianas: Almacenados en coordenadas LOCALES del submapa
     * - Para obtener poses globales: aplicar T_global_cached a coordenadas locales
     */
    struct Submap
    {
        // ===== GPU GAUSSIANS (SoA) =====
        // NOTA: Todas las posiciones, escalas y orientaciones estan en coordenadas LOCALES
        Gaussians gaussians;          // Estructura SoA de GPU con positions, scales, etc. (locales)
        uint32_t gaussians_count = 0; // Numero actual de gaussianas activas
        uint32_t max_gaussians = 100'000; // Maximo reservado
        thrust::device_vector<AdamStateGaussian3D> adam_states;

        // ===== KEYFRAMES =====
        std::vector<KeyframeData> keyframes;  // Keyframes asociados a este submapa (con poses relativas)
        std::vector<size_t> keyframe_gaussian_counts; // Count de gaussianas al agregar cada keyframe

        // ===== POSE (Marco Local relativo a cadena de submapas) =====
        Pose T_relative;                    // Pose relativa a Submap[i-1] (cadena: T_0to1 * T_1to2 * ...)
        Pose T_global_cached;               // Pose acumulada en el sistema global
        bool pose_cache_valid = false;      // Flag de validez del cache

        // ===== SUBMAPA METADATA =====
        uint32_t submap_id = 0;           // ID unico global
        Pose first_frame_pose_local;      // Pose del primer frame EN COORDENADAS LOCALES (para comparaciones de threshold)
        double timestamp_creation = 0.0;  // Timestamp de creacion
        float accumulated_translation_uncertainty_m = 0.0f;   // Incertidumbre acumulada en traslacion
        float accumulated_rotation_uncertainty_deg = 0.0f;    // Incertidumbre acumulada en rotacion
        float min_descriptor_similarity = 1.0f;               // Minima similaridad coseno intra-submapa
        float self_similarity_percentile_score = 0.0f;        // s_self del submapa (percentil)
        bool has_descriptor_similarity_stats = false;

        // ===== CONSTRUCTORES =====
        Submap(uint32_t id = 0, uint32_t max_gauss = 1000000);

        // ===== HELPERS =====
        // Indica si el submapa no tiene gaussianas activas.
        inline bool isEmpty() const { return gaussians_count == 0; }
        
        // Devuelve cuantas gaussianas activas tiene el submapa.
        inline uint32_t getGaussiansCount() const { return gaussians_count; }
        
        inline void clear()
        {
            gaussians_count = 0;
            keyframes.clear();
            keyframe_gaussian_counts.clear();
            accumulated_translation_uncertainty_m = 0.0f;
            accumulated_rotation_uncertainty_deg = 0.0f;
            min_descriptor_similarity = 1.0f;
            self_similarity_percentile_score = 0.0f;
            has_descriptor_similarity_stats = false;
            pose_cache_valid = false;
        }
        
        // Retorna la pose global cached (relativa al origin del sistema)
        inline Pose getGlobalPose() const { return pose_cache_valid ? T_global_cached : Pose::Identity(); }
        
        // Retorna la pose relativa (respecto al submapa anterior)
        inline Pose getRelativePose() const { return T_relative; }
        
        // Establece la pose relativa del submapa (respecto a submapa anterior en cadena)
        inline void setRelativePose(const Pose& rel_pose)
        {
            T_relative = rel_pose;
            invalidatePoseCache();
        }
        
        // Actualiza la pose global cached basada en la pose del submapa anterior
        // Formula: T_global[i] = T_global[i-1] * T_relative[i]
        // Actualiza la pose global cached basada en la pose global del submapa anterior
        // Fórmula: T_global[i] = T_global[i-1] * T_relative[i]
        inline void updateGlobalPose(const Pose& prev_global_pose)
        {
            // Composición: T_global = T_prev * T_relative
            T_global_cached = composePoses(prev_global_pose, T_relative);
            pose_cache_valid = true;
        }
        
        // Invalida el cache global (debe llamarse cuando cambia T_relative o cadena anterior)
        inline void invalidatePoseCache()
        {
            pose_cache_valid = false;
            // Propagar invalidacion a keyframes
            for (auto& kf : keyframes) {
                kf.invalidatePoseCache();
            }
        }

        // Peso inversamente proporcional a la incertidumbre para PGO.
        inline float uncertaintyWeight() const
        {
            const float score = accumulated_translation_uncertainty_m +
                                0.02f * accumulated_rotation_uncertainty_deg;
            return 1.0f / (1.0f + std::max(0.0f, score));
        }

        // Calcula la distancia Euclidea del frame actual respecto al primer frame del submapa
        // (usando pose en COORDENADAS LOCALES del submapa)
        inline float getDistanceToFirstFrameLocal(const Pose& current_pose_local) const
        {
            float3 diff = make_float3(
                current_pose_local.position.x - first_frame_pose_local.position.x,
                current_pose_local.position.y - first_frame_pose_local.position.y,
                current_pose_local.position.z - first_frame_pose_local.position.z
            );
            return std::sqrt(diff.x * diff.x + diff.y * diff.y + diff.z * diff.z);
        }

        // Calcula la diferencia angular, en grados, respecto al primer frame del submapa
        // (usando pose en COORDENADAS LOCALES del submapa)
        inline float getRotationToFirstFrameLocal(const Pose& current_pose_local) const
        {
            // dot product de quaterniones: (q1 . q2) = x1*x2 + y1*y2 + z1*z2 + w1*w2
            float dot = current_pose_local.orientation.x * first_frame_pose_local.orientation.x +
                       current_pose_local.orientation.y * first_frame_pose_local.orientation.y +
                       current_pose_local.orientation.z * first_frame_pose_local.orientation.z +
                       current_pose_local.orientation.w * first_frame_pose_local.orientation.w;
            
            // Clampar para evitar artefactos numericos
            dot = std::max(-1.0f, std::min(1.0f, dot));
            
            // Angulo relativo estandar entre quaterniones: 2*acos(|dot|)
            float angle_rad = 2.0f * std::acos(std::abs(dot));
            
            // Convertir a grados
            return std::abs(angle_rad) * 180.0f / 3.14159265359f;
        }
        
        // Versiones legacy que usan poses globales (para compatibilidad durante transicion)
        // NOTA: Estas se removeran despues de refactorizar GSSlam
        static inline float getDistanceToFirstFrame(const Submap* submap, const Pose& current_pose)
        {
            if (!submap) return 0.0f;
            // Version legacy: asumir que current_pose es global y first_frame_pose_local es referencia
            float3 diff = make_float3(
                current_pose.position.x - submap->first_frame_pose_local.position.x,
                current_pose.position.y - submap->first_frame_pose_local.position.y,
                current_pose.position.z - submap->first_frame_pose_local.position.z
            );
            return std::sqrt(diff.x * diff.x + diff.y * diff.y + diff.z * diff.z);
        }

        static inline float getRotationToFirstFrame(const Submap* submap, const Pose& current_pose)
        {
            if (!submap) return 0.0f;
            float dot = current_pose.orientation.x * submap->first_frame_pose_local.orientation.x +
                       current_pose.orientation.y * submap->first_frame_pose_local.orientation.y +
                       current_pose.orientation.z * submap->first_frame_pose_local.orientation.z +
                       current_pose.orientation.w * submap->first_frame_pose_local.orientation.w;
            dot = std::max(-1.0f, std::min(1.0f, dot));
            float angle_rad = 2.0f * std::acos(std::abs(dot));
            return std::abs(angle_rad) * 180.0f / 3.14159265359f;
        }
    };

    /**
     * @brief PoseTransformCache
     * Estructura que mantiene las transformaciones precalculadas de poses globales
     * para todos los submapas. Esto evita recalcular la cadena de transformaciones
     * en cada operacion.
     * 
     * NOTA: Se actualiza cuando:
     * - Se crea un nuevo submapa
     * - Se modifica la pose relativa de un submapa
     * - Se detecta loop closure y se aplica PGO
     */
    struct PoseTransformCache
    {
        // ===== DATA =====
        // Mapeo de submap_id -> pose global precalculada
        std::unordered_map<uint32_t, Pose> submap_global_poses;
        bool cache_valid = false;

        // ===== METHODS =====
        
        // Actualiza el cache a partir de la cadena de submapas
        inline void updateFromSubmapChain(const std::vector<std::shared_ptr<Submap>>& submaps)
        {
            submap_global_poses.clear();
            
            Pose accumulated_pose = Pose::Identity();
            
            for (const auto& submap : submaps) {
                if (!submap) continue;
                
                // T_global[i] = T_global[i-1] * T_relative[i]
                // Nota: La transformacion completa se hara en GSSlam (por ahora placeholder)
                submap_global_poses[submap->submap_id] = accumulated_pose;
                accumulated_pose = submap->T_relative;  // Placeholder
            }
            
            cache_valid = true;
        }
        
        // Obtiene la transformacion global para un submapa dado
        inline Pose getSubmapGlobalPose(uint32_t submap_id) const
        {
            auto it = submap_global_poses.find(submap_id);
            if (it != submap_global_poses.end()) {
                return it->second;
            }
            return Pose::Identity();
        }
        
        // Invalida el cache (debe llamarse cuando cambia algun submapa)
        inline void invalidate()
        {
            cache_valid = false;
            submap_global_poses.clear();
        }
        
        // Verifica si el cache es valido
        inline bool isValid() const { return cache_valid; }
    };

}