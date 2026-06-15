#pragma once

// ============================================================
// STL
// ============================================================
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <fstream>
#include <memory>
#include <mutex>
#include <queue>
#include <set>
#include <string>
#include <thread>

// ============================================================
// Dependencias extra
// ============================================================
#include <opencv2/core.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudafilters.hpp>
#include <thrust/device_vector.h>
#include <ceres/ceres.h>

// ============================================================
// Propias
// ============================================================
#include "f_vigs_slam/GSCudaKernels.cuh"
#include "f_vigs_slam/KeyframeSelector.hpp"
#include "f_vigs_slam/RepresentationClasses.hpp"
#include "f_vigs_slam/Preintegration.hpp"
#include "f_vigs_slam/MarginalizationFactor.hpp"
//#include "f_vigs_slam/NetVLADWrapper.hpp"
#include "f_vigs_slam/LoopClosureModule.hpp"
#include "third_party/concurrentqueue/concurrentqueue.h"

// ============================================================

namespace f_vigs_slam
{
    // Forward declarations
    class RgbdPoseCostFunction;
    class ImuCostFunction;
    class NetVLADWrapper;
    struct FrequencyMaskGpuPair;

    // ============================================================
    // Descriptor Extraction Task (for async NetVLAD processing)
    // ============================================================
    struct DescriptorExtractionTask
    {
        int submap_idx;          // Índice del submapa
        int keyframe_id;         // ID local del keyframe dentro del submapa
        cv::Mat color_image;     // Copia de la imagen RGB (CPU)
        
        DescriptorExtractionTask() : submap_idx(-1), keyframe_id(-1) {}
        DescriptorExtractionTask(int s_idx, int kf_id, const cv::Mat& img)
            : submap_idx(s_idx), keyframe_id(kf_id), color_image(img.clone()) {}
    };

    // Event sent when a submap is closed and ready for detection
    struct SubmapClosedEvent
    {
        int submap_id = -1;
        int64_t timestamp = 0; // optional frame or time index
    };

    struct SubmapDebugInfo
    {
        int submap_id = -1;
        size_t gaussian_count = 0;
        size_t keyframe_count = 0;
        Pose first_frame_global_pose = Pose::Identity();
    };

    // ============================================================
    // Objetos auxiliares
    // ============================================================

    enum class PoseEstimationMethod
    {
        DirectRendering = 0,
        WarpingSingleRendering = 1
    };

    enum class RegionSamplingStrategy
    {
        VigsFusion = 0,
        FgsFourier = 1,
        SobelEdge = 2,
        LaplacianEdge = 3,
        FastPlaceholder = 4,
        CannyEdge = 5
    };

    // ============================================================
    // Estructura para métricas temporales
    // ============================================================
    struct GSSlamGPUTimings
    {
        float init_gaussians_ms = 0.0f;          // Inicialización de gaussianas
        float prepare_rasterization_ms = 0.0f;   // Proyección + hashing
        float rasterize_ms = 0.0f;                // Rasterización tile-based
        float pyramid_build_ms = 0.0f;            // Construcción de pirámide
        float image_copy_ms = 0.0f;               // Copia de imágenes GPU
        float keyword_optimization_ms = 0.0f;     // Optimización de gaussianas
        float densification_ms = 0.0f;            // Densificación
        float pruning_ms = 0.0f;                  // Poda de gaussianas
        float total_frame_ms = 0.0f;              // Tiempo total del frame
    };

    struct GSSlamCPUTimings
    {
        float lock_wait_ms = 0.0f;
        float imu_predict_ms = 0.0f;
        float pose_track_ms = 0.0f;
        float pose_refine_ms = 0.0f;
        float outlier_ms = 0.0f;
        float map_ops_ms = 0.0f;
        float loop_extract_ms = 0.0f;
        float loop_detect_ms = 0.0f;
        float loop_verify_ms = 0.0f;
        float loop_pgo_ms = 0.0f;
        float loop_total_ms = 0.0f;
        float loop_lock_hold_ms = 0.0f;
        int loop_edges = 0;
        float marginalization_ms = 0.0f;
        float total_frame_ms = 0.0f;
        int gaussian_count = 0;
        int densify_added = 0;
        int prune_removed = 0;
        int outliers_removed = 0;
    };

    // struct GSSlamTimingSummary
    // {
    //     int frame_index = 0;
    //     double lock_wait_ms = 0.0;
    //     double total_cpu_ms = 0.0;
    //     double total_gpu_ms = -1.0;
    //     double init_copy_ms = 0.0;
    //     double predict_ms = 0.0;
    //     double optimize_track_ms = 0.0;
    //     double remove_outliers_ms = 0.0;
    //     double keyframe_ms = 0.0;
    //     double keyframe_refine_ms = 0.0;
    //     double keyframe_prune_ms = 0.0;
    //     double keyframe_add_ms = 0.0;
    //     double keyframe_densify_ms = 0.0;
    //     double marginalization_ms = 0.0;
    //     bool keyframe_added = false;
    //     std::string jump_source_stage = "none";
    //     double jump_source_dpos_m = 0.0;
    //     double jump_source_drot_deg = 0.0;
    //     double bias_ba_norm = 0.0;
    //     double bias_bg_norm = 0.0;
    //     double bias_ba_delta_norm = 0.0;
    //     double bias_bg_delta_norm = 0.0;
    //     uint32_t invalid_gaussian_pos_count = 0;
    //     uint32_t invalid_gaussian_scale_count = 0;
    //     uint32_t invalid_gaussian_opacity_count = 0;
    //     double predict_dpos_m = 0.0;
    //     double predict_drot_deg = 0.0;
    //     double preint_sum_dt_s = 0.0;
    //     int preint_samples = 0;
    //     uint64_t imu_rejected_dt_total = 0;
    //     uint64_t imu_invalid_dt_total = 0;
    //     uint64_t imu_accepted_dt_total = 0;
    // };

    // =================================================================================================================
    // Clase principal, constituye el core del sistema. 
    // Maneja todo lo que tiene que ver con gaussianas y optimizacion
    // =================================================================================================================

    class GSSlam
    {
    public:
        GSSlam();
        ~GSSlam();

        void rasterize(Submap* submap, const Pose &camera_pose, const IntrinsicParameters &intrinsics, int width, int height);
        void densify(Submap* submap, const KeyframeData &keyframe);
        void addKeyframe(Submap* submap);
        inline bool getIsInitialized() const { return isInitialized; }

        cv::cuda::GpuMat rendered_rgb_gpu_;
        cv::cuda::GpuMat rendered_depth_gpu_;

        // ===== CONFIG =====
        void setIntrinsics(const IntrinsicParameters &params);
        void setGaussInitSizePx(int size_px);
        void setGaussInitScale(float scale);
        inline void setDepthScale(float scale) { depth_scale_ = std::max(1e-6f, scale); }
        inline void setFrequencyGuidedDensification(bool enable)
        {
            densification_strategy_ = enable
                ? RegionSamplingStrategy::FgsFourier
                : RegionSamplingStrategy::VigsFusion;
        }

        inline void setFrequencyGuidedInitialization(bool enable)
        {
            initialization_strategy_ = enable
                ? RegionSamplingStrategy::FgsFourier
                : RegionSamplingStrategy::VigsFusion;
        }

        void setInitializationStrategy(const std::string &strategy_name);
        void setDensificationStrategy(const std::string &strategy_name);
        std::string getInitializationStrategyName() const;
        std::string getDensificationStrategyName() const;

        inline void setFrequencyGuidedSampling(int high_px, int low_px)
        {
            fgs_sample_high_px_ = std::max(1, high_px);
            fgs_sample_low_px_ = std::max(fgs_sample_high_px_, low_px);
        }

        inline void setFrequencyGuidedScaleFactors(float high, float low)
        {
            fgs_scale_high_ = std::max(1e-5f, high);
            fgs_scale_low_ = std::max(1e-5f, low);
        }

        inline void setFrequencyGuidedHighpassSigmaRatio(float ratio)
        {
            fgs_highpass_sigma_ratio_ = std::clamp(ratio, 1e-4f, 1.0f);
        }

        inline void setCannyThresholds(double threshold1, double threshold2)
        {
            canny_threshold1_ = std::clamp(threshold1, 0.0, 255.0);
            canny_threshold2_ = std::clamp(threshold2, 0.0, 255.0);
            // Ensure threshold1 < threshold2
            if (canny_threshold1_ > canny_threshold2_)
            {
                std::swap(canny_threshold1_, canny_threshold2_);
            }
        }

        void setCamToImuExtrinsics(const Eigen::Vector3d &t_imu_cam,
                                   const Eigen::Quaterniond &q_imu_cam);
        void initialize(const Pose &pose_imu, const Pose &pose_cam);

        // ===== OPT PARAMS =====
        inline void setPoseIterations(int it) { pose_iterations_ = std::max(1, it); }
        inline void setGaussianIterations(int it) { gaussian_iterations_ = std::max(1, it); }
        inline void setEtaPose(float eta) { eta_pose_ = std::max(1e-5f, eta); }
        inline void setEtaGaussian(float eta) { eta_gaussian_ = std::max(1e-5f, eta); }
        inline void setAdamParameters(float eta, float beta1, float beta2, float epsilon)
        {
            adam_eta_ = std::max(1e-8f, eta);
            adam_beta1_ = std::clamp(beta1, 0.0f, 0.999999f);
            adam_beta2_ = std::clamp(beta2, 0.0f, 0.999999f);
            adam_eps_ = std::max(1e-12f, epsilon);
        }

        inline void setKeyframeSelectionConfig(const KeyframeSelectionConfig &config)
        { keyframe_selection_config_ = config; }

        inline void setPoseEstimationMethod(PoseEstimationMethod method)
        { pose_estimation_method_ = method; }

        inline void setPoseResidualThresholds(float a, float c, float d)
        {
            pose_alpha_thresh_ = std::max(1e-5f, a);
            pose_color_thresh_ = std::max(1e-5f, c);
            pose_depth_thresh_ = std::max(1e-5f, d);
        }

        inline void setUseDerivFilters(bool use_deriv_filters)
        {
            use_deriv_filters_ = use_deriv_filters;
        }

        inline void setGaussianErrorWeights(float wd, float wt)
        {
            gaussian_w_depth_ = std::max(0.0f, wd);
            gaussian_w_dist_ = std::max(0.0f, wt);
        }

        inline void setCovisibilityThreshold(float t)
        {
            covisibility_threshold_ = std::clamp(t, 0.0f, 1.0f);
        }

        inline void setSubmapTransitionThresholds(float distance_m, float rotation_deg)
        {
            submap_dist_threshold_m_ = std::max(0.0f, distance_m);
            submap_rot_threshold_deg_ = std::max(0.0f, rotation_deg);
        }

        inline float getSubmapDistanceThresholdM() const
        {
            return submap_dist_threshold_m_;
        }

        inline float getSubmapRotationThresholdDeg() const
        {
            return submap_rot_threshold_deg_;
        }

        inline void setImuRepropagationThresholds(double ba, double bg)
        {
            imu_reprop_ba_thresh_ = std::max(1e-9, ba);
            imu_reprop_bg_thresh_ = std::max(1e-9, bg);
        }

        inline void setLoopClosureParameters(float self_similarity_percentile,
                                             int min_votes_for_loop_closure,
                                             int loop_min_submap_difference,
                                             int max_descriptor_batch_size)
        {
            loop_self_similarity_percentile_ = std::clamp(self_similarity_percentile, 0.0f, 100.0f);
            min_votes_for_loop_closure_ = std::max(1, min_votes_for_loop_closure);
            loop_min_submap_difference_ = std::max(0, loop_min_submap_difference);
            max_descriptor_batch_size_ = std::max(1, max_descriptor_batch_size);
        }

        inline void setLoopVerifyThresholds(float max_distance_m, float max_rotation_deg)
        {
            loop_verify_max_distance_m_ = std::max(0.0f, max_distance_m);
            loop_verify_max_rotation_deg_ = std::max(0.0f, max_rotation_deg);
        }

        inline void setLoopClosureModuleConfig(const LoopClosureConfig &config)
        {
            if (loop_closure_)
            {
                loop_closure_->setConfiguration(config);
            }
        }

        inline void setLoopDiagnosticsMode(bool enabled)
        {
            loop_diagnostics_mode_ = enabled;
        }

        inline void setMetricsPrintIntervalFrames(int frames)
        {
            metrics_print_interval_frames_ = std::max(1, frames);
        }

        // ===== CORE =====
        void compute(const cv::Mat &bgr,
                     const cv::Mat &depth);
        void testGaussians(const cv::Mat &bgr,
                   const cv::Mat &depth);

        // Returns the camera pose in GLOBAL coordinates (composes submap global and local pose)
        Pose getCameraPose() const;
        Pose getAccumulatedOdomPose() const;
        // Returns the current camera pose in the LOCAL frame of the active submap.
        inline Pose getCurrentLocalPose() const { return current_pose_; }
        // Debug info for each submap: id, counts and first-frame global pose.
        std::vector<SubmapDebugInfo> getSubmapDebugInfo() const;
        // Devuelve las poses globales de todos los keyframes en orden cronológico
        std::vector<Pose> getAllKeyframeGlobalPoses() const;
        // Devuelve la pose global del primer frame de cada submapa (en orden de submaps_)
        std::vector<Pose> getSubmapFirstFrameGlobalPoses() const;
        // Debug: eleva el segundo submapa en z (metros) y recomputa la cadena desde el índice 1
        void debugLiftSecondSubmapAndUpdate(float dz);
        void setGpuExpEvaluation(const std::string &mode);
        inline const IntrinsicParameters& getIntrinsics() const { return intrinsics_; }

        // ===== TIMING METRICS =====
        const GSSlamGPUTimings& getLastGPUTimings() const { return last_gpu_timings_; }
        void resetGPUTimings() { last_gpu_timings_ = GSSlamGPUTimings(); }
        const GSSlamCPUTimings& getLastCPUTimings() const { return last_cpu_timings_; }
        void resetCPUTimings() { last_cpu_timings_ = GSSlamCPUTimings(); }

        // ===== GAUSSIANS =====
        // Equivalente a generateGaussians() de vigs-fusion
        void initializeGaussiansFromRgbd(Submap* submap, const Pose& pose);

        // ===== GAUSSIAN ACCESSORS (por submapa) =====
        bool hasGaussians(Submap* submap) const;
        uint32_t getGaussiansCount(Submap* submap) const;
        uint32_t getGaussianData(Submap* submap, float4* positions, float4* colors, uint32_t max_n);

        // ===== GAUSSIAN ACCESSORS (globales - todos los submapas) =====
        bool hasGaussiansGlobal() const;
        uint32_t getGaussiansCountGlobal() const;
        uint32_t getGaussianDataGlobal(float4* positions, float4* colors, uint32_t max_n);

        // ===== SUBMAP ACCESSORS =====
        // Devuelve el submapa activo para escritura.
        Submap* getCurrentSubmap();
        // Devuelve el submapa activo en contexto const.
        const Submap* getCurrentSubmap() const;

        // ===== RENDER =====
        void prepareRasterization(Submap* submap, const Pose &camera_pose,
                          const IntrinsicParameters &intrinsics,
                          int width,
                          int height);
        bool renderView(Submap* submap, const Pose&, const IntrinsicParameters&,
                int, int,
                cv::cuda::GpuMat&, cv::cuda::GpuMat&,
                bool visualize_current_submap = false);

        // ===== OPTIMIZATION =====
        void optimizeGaussians(int, float);

        // Equivalente a optimizeGaussiansKeyframe2() en vigs-fusion
        void optimizeGaussiansKeyframe(Submap* submap, const KeyframeData&);
        
        void optimizeWithCeres(int = 0, int = -1);
        void computeRgbdPoseJacobians(Eigen::Matrix<double, 6, 6> &JtJ,
                                     Eigen::Matrix<double, 6, 1> &Jtr,
                                     int level,
                                     const Eigen::Vector3d &P_imu,
                                     const Eigen::Quaterniond &Q_imu);

        // ===== IMU =====
        void addImuMeasurement(double, const Eigen::Vector3d&, const Eigen::Vector3d&);
        void processImu(double, const ImuData&);
        void initializeImu(const ImuData&);
        void setImuBias(const Eigen::Vector3d&, const Eigen::Vector3d&);

        inline const double* getImuPose() const { return P_cur_; }
        inline const double* getImuVelocity() const { return VB_cur_; }
        inline void getBiases(Eigen::Vector3d& ba, Eigen::Vector3d& bg) const {
            ba = Eigen::Vector3d(VB_cur_[3], VB_cur_[4], VB_cur_[5]);
            bg = Eigen::Vector3d(VB_cur_[6], VB_cur_[7], VB_cur_[8]);
        }

        bool getDebugUpdateGlobalPoses() const
        {
            return debug_update_global_poses_;
        }

        int getDebugUpdatePeriodFrames() const
        {
            return debug_update_period_frames_;
        }

        float getDebugUpdateDzM() const
        {
            return debug_update_dz_m_;
        }

    protected:

        // ===== INTERNAL PIPELINE =====
        void updateIntrinsicsPyramid();
        void initializeFirstFrame(const cv::Mat&, const cv::Mat&, const Pose&);
        void initWarping(const Pose&);
        void updateCameraPoseFromImu();
        Pose computeCameraPoseFromImuState() const;
        // For outlier removal, pruning, covisibility, etc.
        void removeOutliers();
        float computeCovisibilityRatio();
        void prune(Submap* submap);

        // For image copying
        void initAndCopyImgs(const cv::Mat &rgb, const cv::Mat &depth);

        // For optimization thread
        void optimizationLoop();
        // ===== Loop closure pipeline =====
        bool extractDescriptor(const DescriptorExtractionTask &task);
        bool processLoopDetectionEvent(int &loop_cycle);
        bool processSubmapSimilarities(int &loop_cycle);
        bool verifyLoop(std::vector<LoopEdge> &batch_edges);
        bool registerSubmaps(const LoopCandidate &candidate, LoopEdge &edge_out);
        bool poseGraphOptimization(std::vector<LoopEdge> &batch_edges);
        // Worker principal: extracción + similitud + verificación + registro + PGO
        void loopDetectionAndClosureThread();

        // ============================================================
        // ===== SUBMAP HELPERS =====
        // ============================================================
        // Crea un submapa nuevo y lo marca como activo.
        void createNewSubmap();
        // Marca el submapa activo como cerrado sin abrir uno nuevo.
        void finalizeCurrentSubmap();
        // Comprueba si el tracking actual ya justifica abrir un submapa nuevo.
        bool checkSubmapTransition(const Pose& current_pose);
        // Convierte una pose global al marco local del submapa.
        Pose toSubmapLocalPose(const Submap* submap, const Pose& global_pose) const;
        // Distancia del pose actual al primer frame del submapa indicado.
        float getDistanceToFirstFrame(const Submap* submap, const Pose& current_pose) const;
        // Giro del pose actual respecto al primer frame del submapa indicado.
        float getRotationToFirstFrame(const Submap* submap, const Pose& current_pose) const;
        // Recalcula el total global de gaussianas sumando todos los submapas.
        void refreshGlobalGaussiansCount();

        // Se llama cuando: submapa creado, PGO aplicado, o pose relativa modificada
        void updateSubmapChainGlobalPoses();
        void updateSubmapChainGlobalPosesFromIndex(size_t start_idx);
        // Invalida el cache de poses globales (marca para recomputo)
        void invalidateSubmapPoseCache();
        // Invalida el caché de poses globales desde start_idx en adelante (cascada de cambios)
        void invalidateSubmapPoseCacheFromIndex(size_t start_idx);
        // Recomputa T_relative de todos los submapas dado sus T_global correctos (usado post-PGO)
        void updateSubmapChainRelativePoses(const std::vector<Pose>& corrected_global_poses);

        // ============================================================
        // ===== CORE STATE (SUBMAPAS) =====
        // ============================================================
        std::vector<std::shared_ptr<Submap>> submaps_;     // Vector de todos los submapas
        int current_submap_idx_ = -1;                      // Índice del submapa actual en edición
        uint32_t next_submap_id_ = 0;                      // ID para el próximo submapa
        int global_gaussians_count_ = 0;                   // Total global de gaussianas en todos los submapas
        bool current_submap_finalized_ = false;            // Evita notificar dos veces el mismo cierre final
        bool replace_initial_submap_once_ = true;          // Reemplaza submapa 0 por 1 una sola vez
        
        // Thresholds para transición de submapas (valores de LoopSplat)
        float submap_dist_threshold_m_ = 0.5f;             // default: 0.5m
        float submap_rot_threshold_deg_ = 50.0f;           // default: 50°

        // Referencia global (inicialmente identidad)
        Pose base_pose_global_ = Pose::Identity();
        
        // Cache de transformaciones globales precalculadas para submapas
        // Evita recalcular la cadena de transformaciones en cada operacion
        PoseTransformCache submap_pose_cache_;

        bool isInitialized;
        bool first_image_;
        bool test_gaussians_initialized_ = false;
        int nb_images_processed_;
        bool intrinsics_set_ = false;
        bool runtime_params_logged_ = false;
        bool visual_system_valid_frame_ = true;

        // Helper para debug.
        bool gaussian_debug_enabled_ = true;
        uint32_t gaussian_debug_index_ = 0;
        int gaussian_debug_print_period_ = 50;
        int opt_iteration_ = 0;

        Eigen::Vector3d t_imu_cam_;
        Eigen::Quaterniond q_imu_cam_;

        int current_keyframe_idx_ = -1;
        int iterationsSinceDensification_ = 0;

        Pose initial_pose_imu_ = Pose::Identity();
        Pose initial_pose_cam_ = Pose::Identity();
        bool has_initial_pose_ = false;

        // ============================================================
        // ===== AUXILIARES PARA SWAPS =====
        // ============================================================
        thrust::device_vector<float4> new_positions;
        thrust::device_vector<float4> new_scales;
        thrust::device_vector<float4> new_orientations;
        thrust::device_vector<float4> new_colors;
        thrust::device_vector<float> new_opacities;
        thrust::device_vector<AdamStateGaussian3D> new_adam_states;

        // ============================================================
        // ===== CONFIG =====
        // ============================================================

        IntrinsicParameters intrinsics_;

        float depth_scale_ = 1.0f;
        int gauss_init_size_px_ = 7;
        float gauss_init_opacity_ = 0.5f;
        float gauss_init_scale_ = 0.5f;
        int pose_iterations_ = 5;
        int gaussian_iterations_ = 10;
        float eta_pose_ = 0.01f;
        float eta_gaussian_ = 0.002f;
        float adam_eta_ = 1e-3f;
        float adam_beta1_ = 0.9f;
        float adam_beta2_ = 0.999f;
        float adam_eps_ = 1e-8f;

        RegionSamplingStrategy initialization_strategy_ = RegionSamplingStrategy::VigsFusion;
        RegionSamplingStrategy densification_strategy_ = RegionSamplingStrategy::VigsFusion;
        int fgs_sample_high_px_ = 2;
        int fgs_sample_low_px_ = 6;
        float fgs_scale_high_ = 0.8f;
        float fgs_scale_low_ = 1.0f;
        float fgs_highpass_sigma_ratio_ = 0.06f;
        double canny_threshold1_ = 50.0;
        double canny_threshold2_ = 150.0;

        KeyframeSelectionConfig keyframe_selection_config_;
        PoseEstimationMethod pose_estimation_method_ = PoseEstimationMethod::DirectRendering;

        float pose_alpha_thresh_ = 0.1f;
        float pose_color_thresh_ = 0.2f;
        float pose_depth_thresh_ = 0.4f;
        float gaussian_w_depth_ = 1.0f;
        float gaussian_w_dist_ = 0.1f;
        float covisibility_threshold_ = 0.95f;
        double imu_reprop_ba_thresh_ = 0.10;
        double imu_reprop_bg_thresh_ = 0.01;

        int last_nb_instances_;

        std::deque<double> track_time_buffer_;
        std::deque<double> map_time_buffer_;
        std::deque<double> psnr_buffer_;
        size_t frame_count_ = 0;
        size_t metrics_print_interval_frames_ = 10;
        double last_psnr_db_ = std::numeric_limits<double>::quiet_NaN();

        Pose current_pose_ = Pose::Identity();
        Pose global_pose_correction_ = Pose::Identity();

        uint2 tile_size_;
        float3 bg_color_;

        uint2 num_tiles_;


        // ============================================================
        // ===== GPU BUFFERS =====
        // ============================================================
        thrust::device_vector<uint32_t> instance_counter_;
        thrust::device_vector<uint32_t> instance_counter_screen_;

        cv::cuda::GpuMat rgb_gpu_;
        cv::cuda::GpuMat depth_gpu_;
        
        cv::cuda::GpuMat density_mask_;
        std::shared_ptr<Texture<float>> density_mask_tex_ = nullptr;

        thrust::device_vector<float4> positions_2d_;
        thrust::device_vector<float4> covariances_2d_;
        thrust::device_vector<float4> inv_covariances_2d_;
        thrust::device_vector<float2> p_hats_;
        thrust::device_vector<float4> normals_2d_;

        thrust::device_vector<uint32_t> tile_counts_;
        thrust::device_vector<uint32_t> tile_offsets_;
        thrust::device_vector<uint64_t> hashes_;
        thrust::device_vector<uint2> tile_ranges_;

        thrust::device_vector<uint32_t> gaussian_indices_;

        thrust::device_vector<float4> gaussian_gradients_;
        thrust::device_vector<float> opacity_gradients_;

        thrust::device_vector<unsigned char> d_keyframeVis_;
        thrust::device_vector<unsigned char> d_frameVis_;

        thrust::device_vector<uint32_t> d_visUnion_;
        thrust::device_vector<uint32_t> d_visInter_;

        thrust::device_vector<float4> d_imgPositions_;

        // Optimizacion
        thrust::device_vector<uint32_t> opt_bucket_offsets_;
        thrust::device_vector<uint32_t> opt_bucket_to_tile_;

        thrust::device_vector<float> opt_sampled_T_;
        thrust::device_vector<float3> opt_sampled_ar_;

        thrust::device_vector<float> opt_final_T_;
        thrust::device_vector<uint32_t>   opt_n_contrib_;
        thrust::device_vector<uint32_t>   opt_max_contrib_;

        thrust::device_vector<float3> opt_output_color_;
        thrust::device_vector<float>  opt_output_depth_;

        thrust::device_vector<float3> opt_color_error_;
        thrust::device_vector<float> opt_depth_error_;

        thrust::device_vector<DeltaGaussian2D> opt_delta_gaussians_2d_;
        thrust::device_vector<DeltaGaussian3D> opt_delta_gaussians_3d_;

        // ============================================================
        // Buffers de compaction
        // ============================================================
        thrust::device_vector<float4> prune_positions_;
        thrust::device_vector<float4> prune_scales_;
        thrust::device_vector<float4> prune_orientations_;
        thrust::device_vector<float4> prune_colors_;
        thrust::device_vector<float>  prune_opacities_;

        thrust::device_vector<unsigned char> prune_states_;

        // ============================================================
        // ===== IMAGE PYRAMIDS =====
        // ============================================================

        int nb_pyr_levels_ = 3;

        std::vector<cv::cuda::GpuMat> pyr_color_;   // CV_8UC4
        std::vector<cv::cuda::GpuMat> pyr_depth_;   // CV_32FC1
        std::vector<cv::cuda::GpuMat> pyr_normals_; // CV_32FC4

        std::vector<cv::cuda::GpuMat> pyr_dx_;      // CV_32FC4
        std::vector<cv::cuda::GpuMat> pyr_dy_;      // CV_32FC4

        bool use_deriv_filters_ = true;

        cv::Ptr<cv::cuda::Filter> deriv_dx_filter_;
        cv::Ptr<cv::cuda::Filter> deriv_dy_filter_;

        // Warping
        std::vector<cv::cuda::GpuMat> pyr_color_warping_;
        std::vector<cv::cuda::GpuMat> pyr_depth_warping_;

        std::vector<IntrinsicParameters> pyr_intrinsics_;

        // ============================================================
        // ===== CUDA TEXTURES (pyramid) =====
        // ============================================================

        std::vector<std::shared_ptr<Texture<uchar4>>> pyr_color_tex_;
        std::vector<std::shared_ptr<Texture<float>>> pyr_depth_tex_;
        std::vector<std::shared_ptr<Texture<float4>>> pyr_normals_tex_;

        std::vector<std::shared_ptr<Texture<float4>>> pyr_dx_tex_;
        std::vector<std::shared_ptr<Texture<float4>>> pyr_dy_tex_;

        // Para warping
        //std::vector<std::shared_ptr<Texture<uchar4>>> pyr_color_warping_tex_;
        // std::vector<std::shared_ptr<Texture<float>>> pyr_depth_warping_tex_;

        // ============================================================
        // ===== IMU STATE =====
        // ============================================================
        bool imu_initialized_ = false;
        Preintegration* preint_ = nullptr;
        std::shared_ptr<Preintegration> preint_shared_;

        double last_imu_time_;
        ImuData last_imu_;


        double P_cur_[7];
        double P_prev_[7];
        double VB_cur_[9];
        double VB_prev_[9];

        // ============================================================
        // ===== CERES =====
        // ============================================================
        ceres::Problem problem_;
        ceres::Solver::Options options_;
        ceres::Solver::Summary summary_;

        RgbdPoseCostFunction* visual_cost_ = nullptr;
        ImuCostFunction* imu_cost_ = nullptr;
        MarginalizationFactor* marginalization_cost_ = nullptr;

        ceres::ResidualBlockId visual_residual_block_id_;
        ceres::ResidualBlockId imu_residual_block_id_;
        ceres::ResidualBlockId marginalization_residual_block_id_;

        bool imu_residual_added_ = false;

        MarginalizationInfo marginalization_info_;

        // ============================================================
        // ===== KEYFRAMES =====
        // ============================================================
        std::vector<KeyframeData> keyframes_;
        KeyframeSelector keyframe_selector_;

        // ============================================================
        // ===== THREADING =====
        // ============================================================
        std::thread optimize_thread_;
        std::atomic<bool> stop_optimization_{false};
        mutable std::mutex optimization_mutex_;
        
        // ===== NEW: Dedicated threads for GPU-accelerated loop closure =====
        std::thread loop_detection_thread_;      // Thread 3: GPU descriptor matching
        std::thread loop_verification_thread_;   // Thread 4: Verification/Registration
        std::thread pgo_thread_;                 // Thread 5: PGO optimization
        // Thread combinado: detection + verification + PGO
        std::thread loop_detection_and_closure_thread_;
        std::atomic<bool> stop_loop_detection_{false};
        std::atomic<bool> stop_loop_verification_{false};
        std::atomic<bool> stop_pgo_{false};
        std::condition_variable loop_work_cv_;
        std::mutex loop_work_mutex_;

        mutable std::mutex submap_pose_mutex_;
        
        // ===== Inter-thread communication queues (lock-free) =====
        moodycamel::ConcurrentQueue<LoopCandidate> loop_candidates_queue_;
        moodycamel::ConcurrentQueue<LoopEdge> verified_edges_queue_;
        moodycamel::ConcurrentQueue<SubmapClosedEvent> submap_closed_queue_;
        std::set<uint64_t> pending_loop_candidate_pairs_;
        
        // ===== Keyframe notification flag =====
        std::atomic<bool> new_keyframe_added_to_submap_{false};
        
        std::atomic<uint64_t> loop_kf_counter_{0};
        std::atomic<float> loop_last_extract_ms_{0.0f};
        std::atomic<float> loop_last_detect_ms_{0.0f};
        std::atomic<float> loop_last_verify_ms_{0.0f};
        std::atomic<float> loop_last_pgo_ms_{0.0f};
        std::atomic<float> loop_last_total_ms_{0.0f};
        std::atomic<float> loop_last_lock_hold_ms_{0.0f};
        std::atomic<int> loop_last_edges_{0};
        
        #moodycamel::ConcurrentQueue<DescriptorExtractionTask> descriptor_extraction_queue_;
        
        // Descriptores NetVLAD almacenados de forma persistente en GPU (append-only)
        thrust::device_vector<float> descriptor_database_gpu_;      // Pool persistente de descriptores
        std::vector<std::pair<int,int>> descriptor_to_keyframe_map; // Mapeo: DB index → (submap_id, keyframe_id)
        size_t descriptor_database_offset_ = 0;                     // Offset del siguiente descriptor a escribir (in floats)
        static constexpr size_t MAX_DESCRIPTORS_COUNT = 4096;
        size_t descriptor_database_count_ = 0;
        int descriptor_dim_ = 4096;
        
        // Derived: max GPU memory = MAX_DESCRIPTORS_COUNT * descriptor_dim_ floats
        // = 4096 * 4096 = 16,777,216 floats = ~64 MB

        // global keyframe index -> submap index local keyframe index
        size_t global_keyframe_counter_ = 0;
        std::vector<int> global_keyframe_to_submap_idx_;
        std::vector<int> global_keyframe_to_local_kf_idx_;

        // Config de loop closure
        float loop_self_similarity_percentile_ = 50.0f;
        int min_votes_for_loop_closure_ = 2;
        int loop_min_submap_difference_ = 1;
        int max_descriptor_batch_size_ = 1024;
        float loop_verify_max_distance_m_ = 1.0f;
        float loop_verify_max_rotation_deg_ = 15.0f;
        bool loop_diagnostics_mode_ = false;
        bool loop_diagnostics_executed_ = false;
        
          // Streams separados para overlapping de operaciones GPU
        cudaStream_t retrieval_stream_ = nullptr;     // Stream para descriptor matching (Thread 3)
        cudaStream_t render_stream_ = nullptr;        // Stream para rendering (Main thread)
        cudaStream_t pgo_stream_ = nullptr;           // Stream para operaciones PGO (Thread 5)
        
        static constexpr int PGO_BATCH_SIZE_MIN = 4;        // Ejecutar PGO si se acumulan 4+ edges
        static constexpr float PGO_TIMEOUT_MS = 500.0f;      // O si pasan 500ms desde el último PGO
        std::chrono::steady_clock::time_point last_pgo_time_; // Timestamp del último PGO ejecutado
        
        // ============================================================
        // ===== CUDA EVENTS PARA TIMING =====
        // ============================================================
        cudaEvent_t cuda_evt_init_start_, cuda_evt_init_end_;
        cudaEvent_t cuda_evt_pyramid_start_, cuda_evt_pyramid_end_;
        cudaEvent_t cuda_evt_prepare_rast_start_, cuda_evt_prepare_rast_end_;
        cudaEvent_t cuda_evt_rasterize_start_, cuda_evt_rasterize_end_;
        cudaEvent_t cuda_evt_keyframe_opt_start_, cuda_evt_keyframe_opt_end_;
        cudaEvent_t cuda_evt_densify_start_, cuda_evt_densify_end_;
        cudaEvent_t cuda_evt_prune_start_, cuda_evt_prune_end_;
        cudaEvent_t cuda_evt_imgcopy_start_, cuda_evt_imgcopy_end_;
        cudaEvent_t cuda_evt_frame_start_, cuda_evt_frame_end_;

        void initCudaEvents();
        void destroyCudaEvents();
        float getCudaEventElapsedMs(cudaEvent_t start, cudaEvent_t end) const;
        
        void startLoopThreadsIfNeeded();

        // Debug
        void runLoopDiagnosticsSmokeTest(const cv::Mat &rgb, const cv::Mat &depth);
        Pose makeLoopDiagnosticsPose(size_t index) const;
        void buildLoopDiagnosticsRgbdFrame(size_t index,
                           const cv::Mat &rgb,
                           const cv::Mat &depth,
                           cv::Mat &out_rgb,
                           cv::Mat &out_depth) const;

        GSSlamGPUTimings last_gpu_timings_;
        GSSlamCPUTimings last_cpu_timings_;

        // ============================================================
        // ===== LOOP CLOSURE (NetVLAD externo por ROS2) =====
        // ============================================================
        std::unique_ptr<LoopClosureModule> loop_closure_;
        std::unique_ptr<NetVLADWrapper> netvlad_;

        // ============================================================
        // ===== DEBUG / TIMING =====
        // ============================================================
        // GSSlamTimingSummary last_timing_summary_;
        // GSSlamTimingSummary summary_;

        // Debug
        bool debug_update_global_poses_ = true;
        int debug_update_period_frames_ = 20;
        float debug_update_dz_m_ = 0.5f; // meters per update

        // std::ofstream imu_diag_csv_;
        std::atomic<uint64_t> imu_rejected_dt_total_{0};
        std::vector<size_t> keyframe_gaussian_counts_;

        RegionSamplingStrategy parseRegionSamplingStrategy(const std::string &strategy_name) const;
        static const char* regionSamplingStrategyName(RegionSamplingStrategy strategy);
        bool buildStrategyFrequencyMasks(const cv::cuda::GpuMat &color_gpu,
                                         RegionSamplingStrategy strategy,
                                         FrequencyMaskGpuPair &frequency_masks_gpu,
                                         const char *stage_tag) const;
    };
}
