#pragma once

#include <algorithm>
#include <cmath>
#include <vector>
#include <memory>
#include <Eigen/Dense>
#include <thrust/device_vector.h>
#include "f_vigs_slam/RepresentationClasses.hpp"
//#include "f_vigs_slam/NetVLADWrapper.hpp"

namespace f_vigs_slam
{
    // Forward declarations
    //class NetVLADWrapper;
    struct Submap;

    /**
     * @brief LoopCandidate
     * Candidato de loop detectado en GPU (encolado para verificación).
     * Contiene solo los índices de submapa y la similitud.
     */
    struct LoopCandidate
    {
        int source_submap_idx;  // Índice en vector submaps
        int target_submap_idx;  // Índice en vector submaps
        float similarity;       // Similitud coseno GPU
        int votes = 0;          // Votos acumulados para el submapa histórico
        
        LoopCandidate() = default;
        LoopCandidate(int src, int tgt, float sim) 
            : source_submap_idx(src), target_submap_idx(tgt), similarity(sim) {}
    };

    // --- Small pose utilities exposed for multiple compilation units ---
    inline float poseTranslationError(const Pose &a, const Pose &b)
    {
        const float dx = a.position.x - b.position.x;
        const float dy = a.position.y - b.position.y;
        const float dz = a.position.z - b.position.z;
        return std::sqrt(dx * dx + dy * dy + dz * dz);
    }

    inline float normalizeAngleDeg(float angle_deg)
    {
        float wrapped = std::remainder(angle_deg, 360.0f);
        if (wrapped <= -180.0f)
        {
            wrapped += 360.0f;
        }
        else if (wrapped > 180.0f)
        {
            wrapped -= 360.0f;
        }
        return wrapped;
    }

    inline float poseRotationDeg(const Pose &a, const Pose &b)
    {
        const float dot = std::clamp(
            a.orientation.x * b.orientation.x +
            a.orientation.y * b.orientation.y +
            a.orientation.z * b.orientation.z +
            a.orientation.w * b.orientation.w,
            -1.0f,
            1.0f);
        constexpr float kRadToDeg = 57.2957795131f;
        return std::abs(normalizeAngleDeg(2.0f * std::acos(std::abs(dot)) * kRadToDeg));
    }

    inline float translationError(const Pose &a, const Pose &b)
    {
        return poseTranslationError(a, b);
    }

    inline float poseDistance(const Pose &a, const Pose &b)
    {
        return translationError(a, b);
    }

    inline float poseRotationErrorDeg(const Pose &a, const Pose &b)
    {
        return poseRotationDeg(a, b);
    }

    /**
     * @brief LoopEdge
     * Representa una restricción de cierre de bucle entre dos submapas.
     */
    struct LoopEdge
    {
        int source_submap_id;
        int target_submap_id;
        Pose relative_pose;  // T_source_from_target (matches inv(T_source_global) * T_target_global)
        float confidence;    // [0, 1], confianza de la arista
        float visual_similarity = 0.0f;
        float source_uncertainty = 0.0f;
        float target_uncertainty = 0.0f;
        int num_inliers = 0; // Número de correspondencias válidas
        // 6x6 information matrix (inverse covariance) for the edge in the
        // order [tx,ty,tz, rx,ry,rz] (rotation as minimal 3-vector).
        Eigen::Matrix<float, 6, 6> information_matrix = Eigen::Matrix<float,6,6>::Identity();
    };

    struct LoopClosureConfig
    {
        float loop_confidence_threshold = 0.5f;
        float min_similarity_floor = 0.55f;
        int max_submaps_to_compare = 10;
        int max_keyframes_per_submap = 12;
        int frustum_overlap_min_visible_keyframes = 1;
        int min_submap_gap = 2;
        float imu_max_anchor_distance_m = 8.0f;
        float imu_max_anchor_rotation_deg = 135.0f;
        float imu_max_uncertainty_score = 30.0f;
        float geometric_overlap_threshold = 0.2f;

        size_t open3d_max_points = 20000;
        size_t open3d_min_points = 30;
        double open3d_voxel_size_m = 0.03;
        size_t open3d_min_downsampled_points = 20;
        double open3d_normal_radius_scale = 2.0;
        int open3d_normal_max_nn = 30;
        double open3d_fpfh_radius_scale = 5.0;
        int open3d_fpfh_max_nn = 100;
        double open3d_ransac_distance_scale = 1.5;
        int open3d_ransac_n = 4;
        int open3d_ransac_max_iteration = 4000000;
        double open3d_ransac_confidence = 0.999;
        double open3d_icp_threshold_scale = 0.8;
        double open3d_icp_min_distance_m = 1.5;
        int open3d_icp_max_iteration = 30;
        bool apply_pgo_updates = true;
        // Safety thresholds: maximum allowed per-submap translation/rotation when applying PGO
        // If any corrected pose moves more than these thresholds, the PGO update will be rejected.
        float pgo_max_translation_apply_m = 1.0f; // meters
        float pgo_max_rotation_apply_deg = 30.0f; // degrees
        // Which backend to use for Pose Graph Optimization: "ceres" or "open3d"
        // Default: "ceres"
        std::string pgo_backend = "ceres";
    };

    /**
     * @brief LoopClosureModule
     * Módulo para detección y procesamiento de cierre de bucles.
     * Integra NetVLAD para detección y Pose Graph Optimization para corrección global.
     */
    class LoopClosureModule
    {
    public:
        LoopClosureModule();
        ~LoopClosureModule();

        void setConfiguration(const LoopClosureConfig &config);
        const LoopClosureConfig &getConfiguration() const;

        // ===== LOOP DETECTION =====
        /**
         * Detecta loops entre submapas usando NetVLAD + filtros geométricos.
         * Retorna lista de aristas de loop válidas.
         */
        std::vector<LoopEdge> detectLoops(
            const std::vector<std::shared_ptr<Submap>>& submaps,
            float geometric_overlap_threshold = 0.2f,
            float self_similarity_percentile = 50.0f,
            int max_candidates_per_query = 20);

        // ===== POSE GRAPH OPTIMIZATION =====
        struct PGOResult
        {
            std::vector<Pose> corrected_submap_poses;  // Poses optimizadas [submap_id] -> Pose
            float residual_error = 0.0f;               // Error residual final
            bool converged = false;                    // ¿Convergió la optimización?
        };

        /**
         * Optimiza el grafo de poses globales usando aristas odométricas + loops.
         * Retorna poses corregidas para cada submapa.
         */
        PGOResult optimizePoseGraph(
            const std::vector<std::shared_ptr<Submap>>& submaps,
            const std::vector<LoopEdge>& loop_edges,
            int max_iterations = 20);

        // ===== HELPER FUNCTIONS (Public for thread verification) =====
        /**
         * Registra dos submapas usando sus keyframes.
         * Retorna T_target_from_source estimado.
         */
        bool registerSubmaps(
            const Submap* source,
            const Submap* target,
            float visual_similarity,
            Pose& estimated_T,
            float& confidence,
            int& num_inliers,
            float& mean_residual);

        /**
         * Registra dos submapas usando un criterio guiado por overlap.
         * Retorna T_target_from_source estimado.
         */
        bool registerSubmapsByOverlap(
            const Submap* source,
            const Submap* target,
            float visual_similarity,
            Pose& estimated_T,
            float& confidence,
            int& num_inliers,
            float& mean_residual);

        // ===== Open3D-based registration =====
        // Frustum-selected global Gaussian clouds + ICP refinement (PointToPlane)
        bool registerSubmapsOpen3D(
            const Submap* source,
            const Submap* target,
            float visual_similarity,
            const Pose& init_guess,
            Pose& estimated_T,
            float& confidence,
            int& num_inliers,
            float& mean_residual);

        /**
         * Registra dos submapas usando Open3D y construye directamente una LoopEdge
         * lista para alimentar la PGO.
         */
        bool registerSubmapsOpen3D(
            const Submap* source,
            const Submap* target,
            float visual_similarity,
            const Pose& init_guess,
            LoopEdge& edge_out,
            float& mean_residual);

    private:
        LoopClosureConfig config_{};

        // ===== CONFIGURATION =====
        /**
         * Calcula overlap ratio entre dos submapas.
         * Usado para filtrado adicional de loops.
         */
        float computeOverlapRatio(const Submap* submap1, const Submap* submap2);
    };

    // ============================================================
    // GPU-accelerated loop detection functions
    // ============================================================

    /**
     * @brief Compute cosine similarities between query and stored descriptors on GPU.
     * 
     * @param query_desc_gpu Query descriptor on GPU [desc_dim]
     * @param all_descs_gpu Stored descriptors on GPU [num_stored * desc_dim]
     * @param num_stored Number of stored descriptors
    * @param stream CUDA stream for async execution (default: NULL = synchronous)
     * @return Similarities [num_stored]
     */
    thrust::device_vector<float> computeCosineSimilarities_GPU(
        const thrust::device_vector<float>& query_desc_gpu,
        const thrust::device_vector<float>& all_descs_gpu,
        int num_stored,
        cudaStream_t stream = nullptr);

    /**
     * @brief Extract top-k indices and similarities using GPU sort.
     * 
     * @param similarities Device vector of similarities [num_stored]
     * @param k Number of top results
     * @param topk_indices Output indices of top-k
     * @param topk_sims Output top-k similarities
     * @param stream CUDA stream for async execution (default: NULL = synchronous)
     */
    void selectTopk_GPU(
        const thrust::device_vector<float>& similarities,
        int k,
        thrust::device_vector<int>& topk_indices,
         thrust::device_vector<float>& topk_sims,
         cudaStream_t stream = nullptr);

} // namespace f_vigs_slam
