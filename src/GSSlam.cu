// ============================================================
// 1. PROPIOS
// ============================================================
#include <f_vigs_slam/GSSlam.cuh>
#include <f_vigs_slam/GSCudaKernels.cuh>
#include <f_vigs_slam/FrequencyMask.cuh>
#include <f_vigs_slam/RgbdPoseCost.hpp>
#include <f_vigs_slam/ImuCostFunction.hpp>
#include <f_vigs_slam/PoseLocalParameterization.hpp>
#include <f_vigs_slam/NetVLADWrapper.hpp>
#include <f_vigs_slam/LoopClosureModule.hpp>
#include <f_vigs_slam/MetricsHelper.hpp>

// ============================================================
// 2. CUDA
// ============================================================
#include <cuda_runtime.h>
// ============================================================
// 3. OPENCV
// ============================================================
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudawarping.hpp>

// ============================================================
// 4. THRUST
// ============================================================
#include <thrust/device_ptr.h>
#include <thrust/fill.h>
#include <thrust/host_vector.h>
#include <thrust/sort.h>
#include <thrust/transform.h>
#include <thrust/count.h>

// ============================================================
// 5. CERES
// ============================================================
#include <ceres/ceres.h>
#include <Eigen/Geometry>

// ============================================================
// 6. STL
// ============================================================
#include <algorithm>
#include <array>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <limits>
#include <random>
#include <sstream>
#include <stdexcept>
#include <thread>
#include <unordered_map>
#include <vector>

// ============================================================
// DEBUG LOGGING MACROS
// ============================================================
#ifndef F_VIGS_SLAM_VERBOSE_DEBUG
#define F_VIGS_SLAM_VERBOSE_DEBUG 0
#endif

#if F_VIGS_SLAM_VERBOSE_DEBUG
#define DEBUG_LOG(func_name, msg) \
    std::cerr << "[DEBUG " << func_name << "] " << msg << std::endl;

#define DEBUG_LOG_BUFFER(func_name, buf_name, size) \
    std::cerr << "[DEBUG " << func_name << "] " << #buf_name << ".size() = " << size << std::endl;

#define DEBUG_LOG_VALUE(func_name, var_name, value) \
    std::cerr << "[DEBUG " << func_name << "] " << #var_name << " = " << value << std::endl;

#define VALIDATE_BUFFER_SIZE(func_name, buf, expected_size) \
    if (buf.size() < static_cast<size_t>(expected_size)) { \
        std::cerr << "[ERROR " << func_name << "] " << #buf << " size mismatch!" << std::endl; \
        std::cerr << "  Expected: " << expected_size << ", Got: " << buf.size() << std::endl; \
        return; \
    }

#define VALIDATE_COUNTER_BOUNDS(func_name, counter, max_val) \
    if (counter > max_val) { \
        std::cerr << "[ERROR " << func_name << "] Counter overflow detected!" << std::endl; \
        std::cerr << "  Counter: " << counter << ", Max: " << max_val << std::endl; \
        return; \
    }

#define CUDA_CHECK_KERNEL(func_name) \
    { \
        cudaError_t err = cudaPeekAtLastError(); \
        if (err != cudaSuccess) { \
            std::cerr << "[CUDA ERROR " << func_name << "] Launch error: " \
                      << cudaGetErrorString(err) << std::endl; \
            std::cerr << "  This error occurred before kernel execution" << std::endl; \
        } \
        if (false) { \
            std::cerr << "[CUDA CHECK " << func_name << "] Kernel executed successfully." << std::endl; \
        } \
    }
#else
#define DEBUG_LOG(func_name, msg) do {} while (0)
#define DEBUG_LOG_BUFFER(func_name, buf_name, size) do {} while (0)
#define DEBUG_LOG_VALUE(func_name, var_name, value) do {} while (0)

#define VALIDATE_BUFFER_SIZE(func_name, buf, expected_size) \
    if (buf.size() < static_cast<size_t>(expected_size)) { \
        std::cerr << "[ERROR " << func_name << "] " << #buf << " size mismatch!" << std::endl; \
        std::cerr << "  Expected: " << expected_size << ", Got: " << buf.size() << std::endl; \
        return; \
    }

#define VALIDATE_COUNTER_BOUNDS(func_name, counter, max_val) \
    if (counter > max_val) { \
        std::cerr << "[ERROR " << func_name << "] Counter overflow detected!" << std::endl; \
        std::cerr << "  Counter: " << counter << ", Max: " << max_val << std::endl; \
        return; \
    }

#define CUDA_CHECK_KERNEL(func_name) \
    do { \
        cudaError_t _cuda_err = cudaPeekAtLastError(); \
        if (_cuda_err != cudaSuccess) { \
            std::cerr << "[CUDA ERROR " << func_name << "] Launch error: " \
                      << cudaGetErrorString(_cuda_err) << std::endl; \
        } if (false) { \
            std::cerr << "[CUDA CHECK " << func_name << "] Kernel launched successfully." << std::endl; \
        } \
    } while (0)
#endif

namespace f_vigs_slam
{
    Submap::Submap(uint32_t id, uint32_t max_gauss)
        : submap_id(id),
          max_gaussians(max_gauss),
          T_relative(Pose::Identity()),
          T_global_cached(Pose::Identity()),
          pose_cache_valid(true),
          first_frame_pose_local(Pose::Identity())
    {
        gaussians.resize(max_gaussians);
        adam_states.resize(max_gaussians);
        cudaMemset(thrust::raw_pointer_cast(adam_states.data()), 0, max_gaussians * sizeof(AdamStateGaussian3D));
    }

    namespace
    {
        thread_local bool g_debug_pose_update_trace = false;

        std::string poseToString(const Pose &pose)
        {
            std::ostringstream oss;
            oss << std::fixed << std::setprecision(4)
                << "pos=(" << pose.position.x << ", " << pose.position.y << ", " << pose.position.z << ") "
                << "quat=(" << pose.orientation.x << ", " << pose.orientation.y << ", "
                << pose.orientation.z << ", " << pose.orientation.w << ")";
            return oss.str();
        }

        void printPoseTrace(const char *tag, const Pose &pose)
        {
            std::cout << "[GSSlam][POSE] " << tag << ' ' << poseToString(pose) << std::endl;
        }

        Pose packedPoseToPose(const double *packed_pose)
        {
            Pose pose;
            pose.position = make_float3(
                static_cast<float>(packed_pose[0]),
                static_cast<float>(packed_pose[1]),
                static_cast<float>(packed_pose[2]));
            pose.orientation = make_float4(
                static_cast<float>(packed_pose[3]),
                static_cast<float>(packed_pose[4]),
                static_cast<float>(packed_pose[5]),
                static_cast<float>(packed_pose[6]));
            return pose;
        }

        void poseToPackedPose(const Pose &pose, double *packed_pose)
        {
            packed_pose[0] = pose.position.x;
            packed_pose[1] = pose.position.y;
            packed_pose[2] = pose.position.z;
            packed_pose[3] = pose.orientation.x;
            packed_pose[4] = pose.orientation.y;
            packed_pose[5] = pose.orientation.z;
            packed_pose[6] = pose.orientation.w;
        }

        void applyWorldDeltaToPackedPose(const Pose &world_delta, double *packed_pose)
        {
            const Pose current_pose = packedPoseToPose(packed_pose);
            const Pose updated_pose = composePoses(world_delta, current_pose);
            poseToPackedPose(updated_pose, packed_pose);
        }

        struct F0DepthStats
        {
            int64_t total = 0;
            int64_t finite = 0;
            int64_t positive = 0;
            double min_v = std::numeric_limits<double>::infinity();
            double max_v = -std::numeric_limits<double>::infinity();
            double sum = 0.0;
        };

        struct F0GradStats
        {
            int64_t total = 0;
            int64_t finite_dx = 0;
            int64_t finite_dy = 0;
            double norm_sum = 0.0;
            double norm_max = 0.0;
        };

        F0DepthStats computeDepthStatsCpu(const cv::Mat &depth)
        {
            F0DepthStats s;
            s.total = static_cast<int64_t>(depth.rows) * static_cast<int64_t>(depth.cols);

            for (int y = 0; y < depth.rows; ++y)
            {
                const float *row = depth.ptr<float>(y);
                for (int x = 0; x < depth.cols; ++x)
                {
                    const float d = row[x];
                    ++s.finite;
                    if (d > 0.0f)
                    {
                        ++s.positive;
                        s.min_v = std::min(s.min_v, static_cast<double>(d));
                        s.max_v = std::max(s.max_v, static_cast<double>(d));
                        s.sum += static_cast<double>(d);
                    }
                }
            }

            if (s.positive == 0)
            {
                s.min_v = 0.0;
                s.max_v = 0.0;
            }
            return s;
        }

        F0GradStats computeGradStatsCpu(const cv::Mat &dx, const cv::Mat &dy)
        {
            F0GradStats s;
            s.total = static_cast<int64_t>(dx.rows) * static_cast<int64_t>(dx.cols);

            for (int y = 0; y < dx.rows; ++y)
            {
                const cv::Vec4f *row_dx = dx.ptr<cv::Vec4f>(y);
                const cv::Vec4f *row_dy = dy.ptr<cv::Vec4f>(y);
                for (int x = 0; x < dx.cols; ++x)
                {
                    const cv::Vec4f gx = row_dx[x];
                    const cv::Vec4f gy = row_dy[x];

                    ++s.finite_dx;
                    ++s.finite_dy;
                    const double n = std::sqrt(
                        static_cast<double>(gx[0]) * gx[0] + static_cast<double>(gx[1]) * gx[1] +
                        static_cast<double>(gx[2]) * gx[2] + static_cast<double>(gy[0]) * gy[0] +
                        static_cast<double>(gy[1]) * gy[1] + static_cast<double>(gy[2]) * gy[2]);
                    s.norm_sum += n;
                    s.norm_max = std::max(s.norm_max, n);
                }
            }
            return s;
        }

        float cosineSimilarityHost(const std::vector<float> &a, const std::vector<float> &b)
        {
            if (a.size() != b.size() || a.empty())
            {
                return 0.0f;
            }

            float dot = 0.0f;
            float na = 0.0f;
            float nb = 0.0f;
            for (size_t i = 0; i < a.size(); ++i)
            {
                dot += a[i] * b[i];
                na += a[i] * a[i];
                nb += b[i] * b[i];
            }

            const float denom = std::sqrt(na) * std::sqrt(nb);
            if (denom < 1e-9f)
            {
                return 0.0f;
            }
            return dot / denom;
        }

        #if 0
        void printF0Diag(const std::string &node,
                         const std::string &stage,
                         const std::string &payload)
        {
            std::cerr << "F0DIAG {\"node\":\"" << node
                      << "\",\"stage\":\"" << stage
                      << "\"," << payload << "}" << std::endl;
        }

        const char *poseDebugStageName(int stage)
        {
            switch (stage)
            {
            case 1: return "img_depth";
            case 2: return "gauss_mahalanobis";
            case 3: return "gauss_alpha";
            case 4: return "forward_depth";
            case 5: return "color_error";
            case 6: return "color_weight";
            case 7: return "gradient";
            case 8: return "jt";
            case 9: return "depth_weight";
            case 10: return "jtj_cam";
            case 11: return "jtr_cam";
            case 12: return "local_accum";
            case 13: return "block_reduce";
            default: return "none";
            }
        }
        #endif

        struct HasDeltaGaussian3D
        {
            __host__ __device__ bool operator()(const DeltaGaussian3D &delta) const
            {
                return delta.n > 0;
            }
        };

        struct IsInvalidFloat3
        {
            __host__ __device__ bool operator()(const float3 &v) const
            {
                return !isfinite(v.x) || !isfinite(v.y) || !isfinite(v.z);
            }
        };

        struct IsInvalidFloat4
        {
            __host__ __device__ bool operator()(const float4 &v) const
            {
                return !isfinite(v.x) || !isfinite(v.y) || !isfinite(v.z);
            }
        };

        struct IsInvalidFloat
        {
            __host__ __device__ bool operator()(const float &v) const
            {
                return !isfinite(v);
            }
        };

        struct IsLowOpacity
        {
            float threshold;

            __host__ __device__ explicit IsLowOpacity(float t)
                : threshold(t)
            {
            }

            __host__ __device__ bool operator()(const float &v) const
            {
                return v <= threshold;
            }
        };

        void countInvalidGaussians(const thrust::device_vector<float4> &positions,
                       const thrust::device_vector<float4> &scales,
                       const thrust::device_vector<float4> &orientations,
                                   const thrust::device_vector<float> &opacities,
                                   uint32_t n_gaussians,
                                   uint32_t &invalid_pos,
                                   uint32_t &invalid_scale,
                                   uint32_t &invalid_orientation,
                                   uint32_t &invalid_opacity)
        {
            const uint32_t n_checked = std::min<uint32_t>({
                n_gaussians,
                static_cast<uint32_t>(positions.size()),
                static_cast<uint32_t>(scales.size()),
                static_cast<uint32_t>(orientations.size()),
                static_cast<uint32_t>(opacities.size())});

            if (n_checked == 0)
            {
                invalid_pos = 0;
                invalid_scale = 0;
                invalid_orientation = 0;
                invalid_opacity = 0;
                return;
            }

            invalid_pos = static_cast<uint32_t>(
                thrust::count_if(positions.begin(), positions.begin() + n_checked, IsInvalidFloat4()));
            invalid_scale = static_cast<uint32_t>(
                thrust::count_if(scales.begin(), scales.begin() + n_checked, IsInvalidFloat4()));
            invalid_orientation = static_cast<uint32_t>(
                thrust::count_if(orientations.begin(), orientations.begin() + n_checked, IsInvalidFloat4()));
            invalid_opacity = static_cast<uint32_t>(
                thrust::count_if(opacities.begin(), opacities.begin() + n_checked, IsInvalidFloat()));
        }

    } // namespace

    GSSlam::GSSlam()
        : isInitialized(false),
        first_image_(true),
        nb_images_processed_(0),
        next_submap_id_(0),
        current_submap_idx_(-1)
    {
        // ============================================================
        // 1. SUBMAPAS
        // ============================================================
        // Nota: la creación del primer submapa se difiere hasta la inicialización
        // para evitar que el submapa inicial se cree en un frame distinto (p.ej. IMU)
        // createNewSubmap();

        // Asegura modo por defecto para evaluación exponencial en kernels CUDA.
        setGpuExpEvaluation("DEFAULT");

        // Inicializar NetVLAD y LoopClosure
        netvlad_ = std::make_unique<NetVLADWrapper>();
        loop_closure_ = std::make_unique<LoopClosureModule>();

        // No external async manager: use internal threads and lock-free queues
        // CUDA streams for GPU parallelism
        cudaStreamCreate(&retrieval_stream_);
        cudaStreamCreate(&render_stream_);
        cudaStreamCreate(&pgo_stream_);
        
        // PGO batching: initialize last_pgo_time to now
        last_pgo_time_ = std::chrono::steady_clock::now();

        // Start the combined loop worker early so it is always live, even before the first keyframe.
        stop_loop_detection_.store(false, std::memory_order_release);
        if (!loop_detection_and_closure_thread_.joinable()) {
            loop_detection_and_closure_thread_ = std::thread(&GSSlam::loopDetectionAndClosureThread, this);
        }
        
        // Tamaño máximo por submapa
        constexpr uint32_t MAX_GAUSSIANS_PER_SUBMAP = 100000;

        // ============================================================
        // 2. BUFFERS AUXILIARES
        // ============================================================
        positions_2d_.resize(MAX_GAUSSIANS_PER_SUBMAP);
        covariances_2d_.resize(MAX_GAUSSIANS_PER_SUBMAP);
        inv_covariances_2d_.resize(MAX_GAUSSIANS_PER_SUBMAP);
        p_hats_.resize(MAX_GAUSSIANS_PER_SUBMAP);
        normals_2d_.resize(MAX_GAUSSIANS_PER_SUBMAP);

        tile_counts_.resize(MAX_GAUSSIANS_PER_SUBMAP);
        tile_offsets_.resize(MAX_GAUSSIANS_PER_SUBMAP);

        hashes_.resize(MAX_GAUSSIANS_PER_SUBMAP * 80);
        gaussian_indices_.resize(MAX_GAUSSIANS_PER_SUBMAP * 80);

        // ============================================================
        // 3. CONTADORES GPU
        // ============================================================
        instance_counter_.resize(1);
        cudaMemset(thrust::raw_pointer_cast(instance_counter_.data()), 0, sizeof(uint32_t));

        instance_counter_screen_.resize(1);

        // ============================================================
        // 3b. GPU DESCRIPTOR DATABASE (for loop closure)
        // ============================================================
        size_t max_gpu_floats = MAX_DESCRIPTORS_COUNT * descriptor_dim_;
        descriptor_database_gpu_.resize(max_gpu_floats);
        std::cout << "[GSSlam::constructor] Initialized GPU descriptor database: "
                  << MAX_DESCRIPTORS_COUNT << " descriptors × " << descriptor_dim_ << " dims = "
                  << max_gpu_floats << " floats (~" 
                  << (max_gpu_floats * sizeof(float) / (1024*1024)) << " MB)" << std::endl;

        // ============================================================
        // 4. GRADIENTES + OPTIMIZACION
        // ============================================================
        gaussian_gradients_.resize(MAX_GAUSSIANS_PER_SUBMAP);
        opacity_gradients_.resize(MAX_GAUSSIANS_PER_SUBMAP);

        new_positions.resize(MAX_GAUSSIANS_PER_SUBMAP);
        new_scales.resize(MAX_GAUSSIANS_PER_SUBMAP);
        new_orientations.resize(MAX_GAUSSIANS_PER_SUBMAP);
        new_colors.resize(MAX_GAUSSIANS_PER_SUBMAP);
        new_opacities.resize(MAX_GAUSSIANS_PER_SUBMAP);
        new_adam_states.resize(MAX_GAUSSIANS_PER_SUBMAP);

        // ============================================================
        // 5. PARAMS RASTERIZACION
        // ============================================================
        tile_size_ = make_uint2(16, 16);
        bg_color_ = make_float3(0.5f, 0.5f, 0.5f);
        srand48(static_cast<long>(std::chrono::high_resolution_clock::now().time_since_epoch().count()));

        // ============================================================
        // 6. PIRAMIDE DE IMAGENES
        // ============================================================
        pyr_color_.resize(nb_pyr_levels_);
        pyr_depth_.resize(nb_pyr_levels_);
        pyr_dx_.resize(nb_pyr_levels_);
        pyr_dy_.resize(nb_pyr_levels_);
        deriv_dx_filter_ = cv::cuda::createDerivFilter(CV_8UC4, CV_32FC4, 1, 0, 3, true, 1.0 / 255.0);
        deriv_dy_filter_ = cv::cuda::createDerivFilter(CV_8UC4, CV_32FC4, 0, 1, 3, true, 1.0 / 255.0);
        // pyr_color_warping_.resize(nb_pyr_levels_);
        // pyr_depth_warping_.resize(nb_pyr_levels_);

        // ============================================================
        // 7. IMU Y PREINTEGRACION
        // ============================================================

        // Pose actual
        P_cur_[0] = P_cur_[1] = P_cur_[2] = 0.0;
        P_cur_[3] = P_cur_[4] = P_cur_[5] = 0.0;
        P_cur_[6] = 1.0;

        // Pose previa
        P_prev_[0] = P_prev_[1] = P_prev_[2] = 0.0;
        P_prev_[3] = P_prev_[4] = P_prev_[5] = 0.0;
        P_prev_[6] = 1.0;

        // Velocidad + biases
        for (int i = 0; i < 9; ++i) {
            VB_cur_[i] = 0.0;
            VB_prev_[i] = 0.0;
        }

        // Preintegración
        preint_ = new Preintegration();
        preint_shared_ = std::shared_ptr<Preintegration>(preint_, [](Preintegration*) {});

        // ============================================================
        // 8. CERES
        // ============================================================
        visual_cost_ = new RgbdPoseCostFunction(this);
        imu_cost_ = new ImuCostFunction(preint_shared_, imu_reprop_ba_thresh_, imu_reprop_bg_thresh_);
        marginalization_cost_ = new MarginalizationFactor();

        problem_.AddParameterBlock(P_prev_, 7, new PoseLocalParameterization());

        problem_.AddParameterBlock(P_cur_, 7);
        problem_.SetParameterization(P_cur_, new PoseLocalParameterization());

        problem_.AddParameterBlock(VB_prev_, 9);
        problem_.AddParameterBlock(VB_cur_, 9);

        visual_residual_block_id_ =
            problem_.AddResidualBlock(visual_cost_, nullptr, P_cur_);

        imu_residual_block_id_ =
            problem_.AddResidualBlock(imu_cost_, nullptr, P_prev_, VB_prev_, P_cur_, VB_cur_);

        imu_residual_added_ = true;

        // Marginalización
        marginalization_info_.addResidualBlockInfo(
            new ResidualBlockInfo(visual_cost_, nullptr, {P_cur_}, {}));

        marginalization_info_.addResidualBlockInfo(
            new ResidualBlockInfo(imu_cost_, nullptr,
                                {P_prev_, VB_prev_, P_cur_, VB_cur_},
                                {0, 1}));

        marginalization_info_.init();

        std::unordered_map<long, double*> addr_shift;
        addr_shift[reinterpret_cast<long>(P_cur_)] = P_prev_;
        addr_shift[reinterpret_cast<long>(VB_cur_)] = VB_prev_;

        std::vector<double*> params =
            marginalization_info_.getParameterBlocks(addr_shift);

        marginalization_cost_->init(&marginalization_info_);

        marginalization_info_.addResidualBlockInfo(
            new ResidualBlockInfo(marginalization_cost_, nullptr, params, {}));

        marginalization_residual_block_id_ =
            problem_.AddResidualBlock(marginalization_cost_, nullptr, params);

        options_.linear_solver_type = ceres::DENSE_QR;
        options_.max_num_iterations = pose_iterations_;
        options_.minimizer_progress_to_stdout = false;

        // ============================================================
        // 9. VISIBILIDAD / COVISIBILIDAD
        // ============================================================
        d_keyframeVis_.resize(MAX_GAUSSIANS_PER_SUBMAP);
        d_frameVis_.resize(MAX_GAUSSIANS_PER_SUBMAP);
        d_visUnion_.resize(1);
        d_visInter_.resize(1);
        d_imgPositions_.resize(MAX_GAUSSIANS_PER_SUBMAP);

        // ============================================================
        // 10. FINAL
        // ============================================================
        isInitialized = true;

        // Inicializar CUDA events para timing
        initCudaEvents();

        //std::cout << "[GSSlam] GSSlam initialized successfully." << std::endl;
    }
        

    GSSlam::~GSSlam()
    {
        // ============================================================
        // Destruir CUDA events
        // ============================================================
        destroyCudaEvents();

        // ============================================================
        // PHASE 1: Destroy CUDA streams
        // ============================================================
        if (retrieval_stream_ != nullptr) {
            cudaStreamDestroy(retrieval_stream_);
            retrieval_stream_ = nullptr;
        }
        if (render_stream_ != nullptr) {
            cudaStreamDestroy(render_stream_);
            render_stream_ = nullptr;
        }
        if (pgo_stream_ != nullptr) {
            cudaStreamDestroy(pgo_stream_);
            pgo_stream_ = nullptr;
        }

        // ============================================================
        // Thread
        // ============================================================
        stop_optimization_.store(true);
            stop_loop_detection_.store(true);
            stop_loop_verification_.store(true);
            stop_pgo_.store(true);
        loop_work_cv_.notify_all();
        // No condition variables used; the loop closure worker polls the concurrent queue

        if (optimize_thread_.joinable()) {
            optimize_thread_.join();
        }
        // Original individual loop threads are not used in combined mode; skip joining them
        // if (loop_detection_thread_.joinable()) {
        //     loop_detection_thread_.join();
        // }
        // if (loop_verification_thread_.joinable()) {
        //     loop_verification_thread_.join();
        // }
        // if (pgo_thread_.joinable()) {
        //     pgo_thread_.join();
        // }
        if (loop_detection_and_closure_thread_.joinable()) {
            loop_detection_and_closure_thread_.join();
        }

        // ============================================================
        // Preintegracion
        // ============================================================
        if (preint_) {
            delete preint_;
            preint_ = nullptr;
        }

        // ============================================================
        // Costos
        // ============================================================
        // Si estás seguro de ownership:
        delete visual_cost_;
        delete imu_cost_;
        delete marginalization_cost_;

        
    }

    void GSSlam::createNewSubmap()
    {
        const Pose current_local_pose = current_pose_;
        uint32_t submap_id = next_submap_id_++;
        submaps_.push_back(std::make_shared<Submap>(submap_id));
        current_submap_idx_ = static_cast<int>(submaps_.size()) - 1;
        current_submap_finalized_ = false;

        if (current_submap_idx_ > 0) {
            const Pose prev_global = submaps_[static_cast<size_t>(current_submap_idx_ - 1)]->getGlobalPose();
            submaps_[static_cast<size_t>(current_submap_idx_)]->T_relative = current_local_pose;
            submaps_[static_cast<size_t>(current_submap_idx_)]->pose_cache_valid = false;
        }

        // Start the new submap at the identity of its own local frame.
        current_pose_ = Pose::Identity();

        current_keyframe_idx_ = -1;
        refreshGlobalGaussiansCount();
        
        // Actualizar cadena de poses globales
        updateSubmapChainGlobalPoses();

        // Notify loop detection that the previous submap was closed
        if (current_submap_idx_ > 0) {
            auto prev = submaps_[static_cast<size_t>(current_submap_idx_ - 1)];
            if (prev) {
                SubmapClosedEvent ev{prev->submap_id, static_cast<int64_t>(nb_images_processed_)};
                submap_closed_queue_.enqueue(ev);
                loop_work_cv_.notify_one();
            }
        }

        // One-shot behavior requested for debugging: when index 1 is created,
        // remove index 0 and keep the new submap in its place. Disable flag so
        // it does not repeat in a loop.
        if (replace_initial_submap_once_ && current_submap_idx_ == 1 && submaps_.size() >= 2)
        {
            submaps_.erase(submaps_.begin());
            current_submap_idx_ = 0;
            // Fijamos el indice del submapa 0 en 0
            submaps_[0]->submap_id = 0;
            next_submap_id_ = 1;

            //auto &first = submaps_.front();
            //if (first)
            //{
            //    first->T_relative = Pose::Identity();
            //    first->pose_cache_valid = false;
            //}

            //updateSubmapChainGlobalPoses();
            refreshGlobalGaussiansCount();
            replace_initial_submap_once_ = false;

            std::cout << "[GSSlam] [DEBUG] Replaced initial submap: dropped previous index 0,"
                      << " new current index=" << current_submap_idx_
                      << " total_submaps=" << submaps_.size() << std::endl;
        }

         // Extra diagnostic context to trace unexpected submap creations at startup
         printf("[GSSlam] Created submap #%u (index %d, total_submaps=%zu) nb_images=%lld first_image=%d has_initial_pose=%d\n"
             "           current_local_pose.pos=(%f,%f,%f) quat=(%f,%f,%f,%f)\n",
             submap_id,
             current_submap_idx_,
             submaps_.size(),
             static_cast<long long>(nb_images_processed_),
             first_image_ ? 1 : 0,
             has_initial_pose_ ? 1 : 0,
             current_local_pose.position.x,
             current_local_pose.position.y,
             current_local_pose.position.z,
             current_local_pose.orientation.x,
             current_local_pose.orientation.y,
             current_local_pose.orientation.z,
             current_local_pose.orientation.w);
    }

    void GSSlam::finalizeCurrentSubmap()
    {
        if (current_submap_finalized_ || current_submap_idx_ < 0 || current_submap_idx_ >= static_cast<int>(submaps_.size()))
        {
            return;
        }

        auto current = submaps_[static_cast<size_t>(current_submap_idx_)];
        if (!current)
        {
            return;
        }

        SubmapClosedEvent ev{current->submap_id, static_cast<int64_t>(nb_images_processed_)};
        submap_closed_queue_.enqueue(ev);
        loop_work_cv_.notify_one();
        current_submap_finalized_ = true;

        printf("[GSSlam] Finalized submap #%u (index %d, total_submaps=%zu)\n",
               current->submap_id,
               current_submap_idx_,
               submaps_.size());
    }

    void GSSlam::refreshGlobalGaussiansCount()
    {
        uint64_t total = 0;
        // Capture size at iteration time to avoid vector reallocation issues
        const size_t num_submaps = submaps_.size();
        for (size_t i = 0; i < num_submaps; ++i)
        {
            const auto &submap = submaps_[i];
            if (submap)
            {
                total += submap->getGaussiansCount();
            }
        }
        global_gaussians_count_ = static_cast<int>(total);
    }

    void GSSlam::setGpuExpEvaluation(const std::string &mode)
    {
        std::string normalized = mode;
        std::transform(normalized.begin(), normalized.end(), normalized.begin(),
                       [](unsigned char c) { return static_cast<char>(std::toupper(c)); });

        GpuExpEvaluationMode exp_mode = GpuExpEvaluationMode::DEFAULT;
        if (normalized == "TAYLOR")
        {
            exp_mode = GpuExpEvaluationMode::TAYLOR;
        }

        setGpuExpEvaluationMode(exp_mode);
    }

    void GSSlam::updateSubmapChainGlobalPoses()
    {
        updateSubmapChainGlobalPosesFromIndex(0);
    }

    void GSSlam::updateSubmapChainGlobalPosesFromIndex(size_t start_idx)
    {
        // ============================================================
        // Actualiza la cadena de transformaciones globales
        // Formula: T_global[i] = T_global[i-1] * T_relative[i]
        // IMPORTANTE: Recalcula TODOS los submapas desde start_idx hasta el final
        // para que los cambios cascaden correctamente. NO hay submapas que saltarse.
        // ============================================================
        
        // Mutex para poses de submapas
        //std::lock_guard<std::mutex> lock(submap_pose_mutex_);
        
        if (submaps_.empty()) {
            return;
        }

        if (start_idx >= submaps_.size()) {
            return;
        }

        if (g_debug_pose_update_trace)
        {
            std::cout << "[GSSlam][DEBUG_LIFT] updateSubmapChainGlobalPosesFromIndex(start_idx="
                      << start_idx << ") current_submap_idx=" << current_submap_idx_ << std::endl;
            printPoseTrace("current_pose_(local before update)", current_pose_);
            if (current_submap_idx_ >= 0 && static_cast<size_t>(current_submap_idx_) < submaps_.size())
            {
                const auto &active = submaps_[static_cast<size_t>(current_submap_idx_)];
                if (active)
                {
                    printPoseTrace("active_submap.T_global_cached(before)", active->getGlobalPose());
                    printPoseTrace("active_submap.T_relative(before)", active->T_relative);
                }
            }
        }

        // Preserve current local pose for the active submap if the update range affects it
        bool preserve_current_local_pose = false;
        Pose current_pose_local = Pose::Identity();

        if (current_submap_idx_ >= 0 && static_cast<size_t>(current_submap_idx_) >= start_idx &&
            static_cast<size_t>(current_submap_idx_) < submaps_.size())
        {
            current_pose_local = current_pose_;
            preserve_current_local_pose = true;
        }

        // Actualizar submapa en start_idx (y el primero si start_idx == 0)
        if (start_idx == 0)
        {
            // El primer submapa tiene pose global = identidad
            //submaps_[0]->T_global_cached = Pose::Identity();
            submaps_[0]->pose_cache_valid = true;

            for (auto& kf : submaps_[0]->keyframes) {
                kf.updateGlobalPose(submaps_[0]->T_global_cached);
            }
        }
        else
        {
            // Si start_idx > 0, recalcular el submapa en start_idx basado en el anterior
            if (submaps_[start_idx] && submaps_[start_idx - 1])
            {
                Pose T_prev_global = submaps_[start_idx - 1]->T_global_cached;
                Pose T_relative = submaps_[start_idx]->T_relative;
                
                submaps_[start_idx]->T_global_cached = composePoses(T_prev_global, T_relative);
                submaps_[start_idx]->pose_cache_valid = true;
                
                for (auto& kf : submaps_[start_idx]->keyframes) {
                    kf.updateGlobalPose(submaps_[start_idx]->T_global_cached);
                }
            }
        }

        // Recalcular TODOS los submapas posteriores a start_idx
        size_t begin_update_idx = (start_idx == 0) ? 1 : (start_idx + 1);
        for (size_t i = begin_update_idx; i < submaps_.size(); ++i) {
            if (!submaps_[i] || !submaps_[i-1]) continue;
            
            // Composición correcta: T_global[i] = T_global[i-1] * T_relative[i]
            Pose T_prev_global = submaps_[i-1]->T_global_cached;
            Pose T_relative = submaps_[i]->T_relative;
            
            submaps_[i]->T_global_cached = composePoses(T_prev_global, T_relative);
            submaps_[i]->pose_cache_valid = true;
            
            // Actualizar keyframes con nueva pose global del submapa
            for (auto& kf : submaps_[i]->keyframes) {
                kf.updateGlobalPose(submaps_[i]->T_global_cached);
            }
        }

        // Actualizar caché de transformaciones precalculadas
        submap_pose_cache_.updateFromSubmapChain(submaps_);

        if (g_debug_pose_update_trace)
        {
            for (size_t i = 0; i < submaps_.size(); ++i)
            {
                if (!submaps_[i])
                {
                    continue;
                }

                std::cout << "[GSSlam][DEBUG_LIFT] submap[" << i << "] "
                          << poseToString(submaps_[i]->getGlobalPose()) << " rel="
                          << poseToString(submaps_[i]->T_relative) << std::endl;
            }

            if (current_submap_idx_ >= 0 && static_cast<size_t>(current_submap_idx_) < submaps_.size())
            {
                const auto &active = submaps_[static_cast<size_t>(current_submap_idx_)];
                if (active)
                {
                    const Pose camera_global = getCameraPose();
                    printPoseTrace("active_submap.T_global_cached(after)", active->getGlobalPose());
                    printPoseTrace("current_pose_(local after update)", current_pose_);
                    printPoseTrace("getCameraPose()(global after update)", camera_global);
                }
            }
        }

        (void)preserve_current_local_pose;
        (void)current_pose_local;
    }

    // Debug helper: lift the second submap (index 1) by dz meters in Z
    // and recompute the global pose chain from index 1. Prints global poses.
    void GSSlam::debugLiftSecondSubmapAndUpdate(float dz)
    {
        const bool prev_debug_trace = g_debug_pose_update_trace;
        g_debug_pose_update_trace = true;

        if (submaps_.size() < 2) {
            printf("[GSSlam][DEBUG] Not enough submaps to lift (need >=2)\n");
            g_debug_pose_update_trace = prev_debug_trace;
            return;
        }

        auto &second = submaps_[1];
        if (!second) {
            std::cout << "[GSSlam][DEBUG] Second submap is null" << std::endl;
            g_debug_pose_update_trace = prev_debug_trace;
            return;
        }

        const Pose second_global_before = second->getGlobalPose();

        std::cout << "[GSSlam][DEBUG_LIFT] BEFORE lift dz=" << dz << std::endl;
        printPoseTrace("submap[1].T_relative(before)", second->T_relative);
        printPoseTrace("current_pose_(before lift)", current_pose_);
        printPoseTrace("getCameraPose()(before lift)", getCameraPose());

        // Modify the relative pose of the second submap:
        // lift it and rotate it 40 degrees to the right around Z.
        second->T_relative.position.y -= dz;

        const float yaw_angle_deg = 40.0f;
        const float yaw_angle_rad = yaw_angle_deg * 3.14159265358979323846f / 180.0f;
        const Eigen::AngleAxisf yaw_rotation(yaw_angle_rad, Eigen::Vector3f::UnitZ());
        const Eigen::Quaternionf yaw_quaternion(yaw_rotation);

        Pose yaw_pose = Pose::Identity();
        yaw_pose.orientation = make_float4(yaw_quaternion.x(),
                           yaw_quaternion.y(),
                           yaw_quaternion.z(),
                           yaw_quaternion.w());
        second->T_relative = composePoses(second->T_relative, yaw_pose);

        printPoseTrace("submap[1].T_relative(after lift+rotate)", second->T_relative);
        std::cout << "[GSSlam][DEBUG] Rotated submap 1 by " << yaw_angle_deg
              << " degrees around Z-axis" << std::endl;
        
        // Invalidar en cascada: el submapa 1 y todos los posteriores
        invalidateSubmapPoseCacheFromIndex(1);

        std::cout << "[GSSlam][DEBUG] Lifted submap 1 by " << dz 
                  << " meters. Recomputing chain from index 1" << std::endl;
        updateSubmapChainGlobalPosesFromIndex(1);

        // Print resulting global positions for inspection in logs/viewer
        for (size_t i = 0; i < submaps_.size(); ++i) {
            if (!submaps_[i]) continue;
            const Pose g = submaps_[i]->getGlobalPose();
            std::cout << "[GSSlam][DEBUG] submap " << i 
                      << " global position=(" << g.position.x 
                      << ", " << g.position.y 
                      << ", " << g.position.z << ")" << std::endl;
        }

        // Keep current_pose_ as the local pose of the active submap.
        // The global pose is obtained through getCameraPose().

        printPoseTrace("current_pose_(after lift)", current_pose_);
        printPoseTrace("getCameraPose()(after lift)", getCameraPose());

        if (current_submap_idx_ >= 1 && static_cast<size_t>(current_submap_idx_) < submaps_.size())
        {
            const Pose second_global_after = second->getGlobalPose();
            const Pose world_delta = composePoses(second_global_after, invertPose(second_global_before));

            std::cout << "[DEBUG_LIFT] P_cur BEFORE="
                    << " pos=("
                    << P_cur_[0] << ", "
                    << P_cur_[1] << ", "
                    << P_cur_[2] << ")"
                    << " quat=("
                    << P_cur_[3] << ", "
                    << P_cur_[4] << ", "
                    << P_cur_[5] << ", "
                    << P_cur_[6] << ")"
                    << std::endl;

            std::cout << "[DEBUG_LIFT] P_prev BEFORE="
                    << " pos=("
                    << P_prev_[0] << ", "
                    << P_prev_[1] << ", "
                    << P_prev_[2] << ")"
                    << " quat=("
                    << P_prev_[3] << ", "
                    << P_prev_[4] << ", "
                    << P_prev_[5] << ", "
                    << P_prev_[6] << ")"
                    << std::endl;

            applyWorldDeltaToPackedPose(world_delta, P_cur_);
            applyWorldDeltaToPackedPose(world_delta, P_prev_);

            std::cout << "[DEBUG_LIFT] P_cur AFTER="
                    << " pos=("
                    << P_cur_[0] << ", "
                    << P_cur_[1] << ", "
                    << P_cur_[2] << ")"
                    << " quat=("
                    << P_cur_[3] << ", "
                    << P_cur_[4] << ", "
                    << P_cur_[5] << ", "
                    << P_cur_[6] << ")"
                    << std::endl;

            std::cout << "[DEBUG_LIFT] P_prev AFTER="
                    << " pos=("
                    << P_prev_[0] << ", "
                    << P_prev_[1] << ", "
                    << P_prev_[2] << ")"
                    << " quat=("
                    << P_prev_[3] << ", "
                    << P_prev_[4] << ", "
                    << P_prev_[5] << ", "
                    << P_prev_[6] << ")"
                    << std::endl;

            std::cout << "[GSSlam][DEBUG_LIFT] Applied world delta to IMU state: "
                    << poseToString(world_delta) << std::endl;
        }

        g_debug_pose_update_trace = prev_debug_trace;
    }

    void GSSlam::invalidateSubmapPoseCache()
    {
        // Invalida el caché de poses globales de todos los submapas
        // Se llama cuando: PGO actualiza poses, se crea submapa, etc.
        
        for (auto& submap : submaps_) {
            if (submap) {
                submap->invalidatePoseCache();
            }
        }
        submap_pose_cache_.invalidate();
    }
    
    void GSSlam::invalidateSubmapPoseCacheFromIndex(size_t start_idx)
    {
        // Invalida el caché de poses globales de todos los submapas desde start_idx
        // en adelante. Esto asegura que los cambios cascaden correctamente cuando
        // se modifica la pose relativa de un submapa (e.g., por PGO).
        
        if (start_idx >= submaps_.size()) {
            return;
        }
        
        for (size_t i = start_idx; i < submaps_.size(); ++i) {
            if (submaps_[i]) {
                submaps_[i]->invalidatePoseCache();
            }
        }
        submap_pose_cache_.invalidate();
    }

    void GSSlam::updateSubmapChainRelativePoses(const std::vector<Pose>& corrected_global_poses)
    {
        // ============================================================
        // FASE 6: Recomputar T_relative a partir de T_global (post-PGO)
        // ============================================================
        // Dado: T_global[0], T_global[1], ..., T_global[n]
        // Calcular: T_relative[i] tal que T_global[i] = T_global[i-1] * T_relative[i]
        // Formula: T_relative[i] = T_global[i-1]^{-1} * T_global[i]
        
        if (corrected_global_poses.size() != submaps_.size()) {
            std::cerr << "[ERROR] updateSubmapChainRelativePoses: size mismatch. "
                      << "poses=" << corrected_global_poses.size()
                      << ", submapas=" << submaps_.size() << std::endl;
            return;
        }

        std::cout << "[GSSlam][PGO][chain_start]"
                  << " submaps=" << submaps_.size()
                  << " corrected_poses=" << corrected_global_poses.size()
                  << " current_submap_idx=" << current_submap_idx_
                  << std::endl;

        std::vector<Pose> previous_global_poses(submaps_.size(), Pose::Identity());
        for (size_t i = 0; i < submaps_.size(); ++i)
        {
            if (submaps_[i])
            {
                previous_global_poses[i] = submaps_[i]->getGlobalPose();
            }
        }

        auto poseChanged = [](const Pose &a, const Pose &b) -> bool
        {
            const float dx = a.position.x - b.position.x;
            const float dy = a.position.y - b.position.y;
            const float dz = a.position.z - b.position.z;
            const float dpos = std::sqrt(dx * dx + dy * dy + dz * dz);

            const float dot = std::abs(a.orientation.x * b.orientation.x +
                                       a.orientation.y * b.orientation.y +
                                       a.orientation.z * b.orientation.z +
                                       a.orientation.w * b.orientation.w);
            const float clamped_dot = std::min(1.0f, std::max(0.0f, dot));
            constexpr float kRadToDeg = 57.29577951308232f;
            const float drot_deg = 2.0f * std::acos(clamped_dot) * kRadToDeg;
            return dpos > 1e-6f || drot_deg > 1e-4f;
        };

        size_t first_changed = submaps_.size();
        size_t last_changed = 0;
        for (size_t i = 0; i < submaps_.size(); ++i)
        {
            if (!submaps_[i]) {
                continue;
            }

            if (poseChanged(submaps_[i]->getGlobalPose(), corrected_global_poses[i]))
            {
                first_changed = std::min(first_changed, i);
                last_changed = std::max(last_changed, i);
            }
        }

        if (first_changed == submaps_.size())
        {
            std::cout << "[GSSlam][PGO][chain_recompose] no_pose_changes_detected" << std::endl;
            return;
        }

        std::cout << "[GSSlam][PGO][chain_span]"
                  << " first_changed=" << first_changed
                  << " last_changed=" << last_changed
                  << " current_submap_idx=" << current_submap_idx_
                  << std::endl;

        Pose current_submap_global_before = Pose::Identity();
        bool shift_imu_state = false;
        if (current_submap_idx_ >= 0 &&
            static_cast<size_t>(current_submap_idx_) < submaps_.size() &&
            static_cast<size_t>(current_submap_idx_) >= first_changed) {
            const auto &cur = submaps_[static_cast<size_t>(current_submap_idx_)];
            if (cur && cur->pose_cache_valid) {
                current_submap_global_before = cur->getGlobalPose();
                shift_imu_state = true;
            }
        }

        // Nota: no se preserva current_pose_ antes del cambio porque el submapa actual
        // también puede tener su T_global recalculado. Se recalculará después.
        for (size_t i = first_changed; i <= last_changed; ++i) {
            if (!submaps_[i]) {
                continue;
            }
            
            const Pose& T_global_i = corrected_global_poses[i];
            submaps_[i]->T_global_cached = T_global_i;
            submaps_[i]->pose_cache_valid = true;
            
            if (i == 0) {
                // Primer submapa: T_relative[0] = Identity
                submaps_[i]->T_relative = Pose::Identity();
            } else {
                // i > 0: T_relative[i] = T_global[i-1]^{-1} * T_global[i]
                const Pose T_global_prev = (i > first_changed)
                    ? corrected_global_poses[i - 1]
                    : submaps_[i - 1]->getGlobalPose();
                submaps_[i]->T_relative = composePoses(invertPose(T_global_prev), T_global_i);
            }
        }

        const bool corrected_chain_finite = std::all_of(
            corrected_global_poses.begin(),
            corrected_global_poses.end(),
            [](const Pose &pose)
            {
                return std::isfinite(pose.position.x) && std::isfinite(pose.position.y) && std::isfinite(pose.position.z) &&
                       std::isfinite(pose.orientation.x) && std::isfinite(pose.orientation.y) && std::isfinite(pose.orientation.z) &&
                       std::isfinite(pose.orientation.w);
            });

        std::cout << "[GSSlam][PGO][chain_validate]"
                  << " corrected_chain_finite=" << (corrected_chain_finite ? 1 : 0)
                  << std::endl;
        
        // Repropaga poses globales desde el primer submapa afectado hasta el submapa actual
        // (incluyendo el submapa actual si exista).
        updateSubmapChainGlobalPosesFromIndex(first_changed);

        size_t changed_count = 0;
        for (size_t i = first_changed; i <= last_changed; ++i)
        {
            if (!submaps_[i])
            {
                continue;
            }

            const Pose before_pose = previous_global_poses[i];
            const Pose after_pose = submaps_[i]->getGlobalPose();
            const float dpos = std::sqrt(
                std::pow(before_pose.position.x - after_pose.position.x, 2.0f) +
                std::pow(before_pose.position.y - after_pose.position.y, 2.0f) +
                std::pow(before_pose.position.z - after_pose.position.z, 2.0f));
            const float dot = std::abs(before_pose.orientation.x * after_pose.orientation.x +
                                       before_pose.orientation.y * after_pose.orientation.y +
                                       before_pose.orientation.z * after_pose.orientation.z +
                                       before_pose.orientation.w * after_pose.orientation.w);
            const float drot_deg = 2.0f * std::acos(std::min(1.0f, std::max(0.0f, dot))) * 57.2957795131f;

            if (dpos > 1e-6f || drot_deg > 1e-4f)
            {
                ++changed_count;
            }
            /*
            std::cout << "[GSSlam][PGO][submap_update] submap_id=" << submaps_[i]->submap_id
                      << " before=" << poseToString(before_pose)
                      << " after=" << poseToString(after_pose)
                      << " dpos_m=" << dpos
                      << " drot_deg=" << drot_deg
                      << std::endl;
            */
        }

        std::cout << "[GSSlam][PGO][chain_recompose] first_changed=" << first_changed
                  << " last_changed=" << last_changed
                  << " changed_count=" << changed_count
                  << std::endl;

        if (shift_imu_state)
        {
            const auto &cur_after = submaps_[static_cast<size_t>(current_submap_idx_)];
            if (cur_after && cur_after->pose_cache_valid)
            {
                const Pose current_submap_global_after = cur_after->getGlobalPose();
                const Pose world_delta = composePoses(current_submap_global_after,
                                                      invertPose(current_submap_global_before));

                {
                    //std::lock_guard<std::mutex> lock(optimization_mutex_);
                    applyWorldDeltaToPackedPose(world_delta, P_cur_);
                    applyWorldDeltaToPackedPose(world_delta, P_prev_);
                }

                std::cout << "[GSSlam][PGO] Applied world delta to IMU state after submap update: "
                          << poseToString(world_delta) << std::endl;
            }
        }

    }

    Submap* GSSlam::getCurrentSubmap()
    {
        if (current_submap_idx_ < 0 || current_submap_idx_ >= static_cast<int>(submaps_.size())) {
            return nullptr;
        }
        return submaps_[current_submap_idx_].get();
    }

    Pose GSSlam::getAccumulatedOdomPose() const
    {
        return getCameraPose();
    }

    Pose GSSlam::getCameraPose() const
    {
        if (current_submap_idx_ >= 0 && static_cast<size_t>(current_submap_idx_) < submaps_.size()) {
            const auto &sub = submaps_[static_cast<size_t>(current_submap_idx_)];
            if (sub) {
                return composePoses(sub->getGlobalPose(), current_pose_);
            }
        }
        return current_pose_;
    }

    std::vector<Pose> GSSlam::getAllKeyframeGlobalPoses() const
    {
        std::vector<Pose> out;
        out.reserve(256);
        for (const auto &submap_ptr : submaps_) {
            if (!submap_ptr) continue;
            const Pose submap_global = submap_ptr->getGlobalPose();
            for (const auto &kf : submap_ptr->keyframes) {
                // Prefer global cached keyframe pose when available.
                Pose kf_global = kf.pose_cache_valid
                    ? kf.getGlobalPose()
                    : composePoses(submap_global, kf.getRelativePose());
                out.push_back(kf_global);
            }
        }
        return out;
    }

    std::vector<Pose> GSSlam::getSubmapFirstFrameGlobalPoses() const
    {
        std::vector<Pose> out;
        out.reserve(submaps_.size());
        for (const auto &submap_ptr : submaps_) {
            if (!submap_ptr) continue;
            Pose first_local = submap_ptr->first_frame_pose_local;
            Pose first_global = composePoses(submap_ptr->getGlobalPose(), first_local);
            out.push_back(first_global);
        }
        return out;
    }

    std::vector<SubmapDebugInfo> GSSlam::getSubmapDebugInfo() const
    {
        std::vector<SubmapDebugInfo> out;
        out.reserve(submaps_.size());

        for (const auto &submap_ptr : submaps_) {
            if (!submap_ptr) {
                continue;
            }
            SubmapDebugInfo info;
            info.submap_id = static_cast<int>(submap_ptr->submap_id);
            info.gaussian_count = static_cast<size_t>(submap_ptr->getGaussiansCount());
            info.keyframe_count = submap_ptr->keyframes.size();
            info.first_frame_global_pose = composePoses(submap_ptr->getGlobalPose(), submap_ptr->first_frame_pose_local);
            out.push_back(info);
        }

        return out;
    }

    float GSSlam::getDistanceToFirstFrame(const Submap* submap, const Pose& current_pose) const
    {
        if (!submap) {
            return 0.0f;
        }
        return submap->getDistanceToFirstFrameLocal(toSubmapLocalPose(submap, current_pose));
    }

    float GSSlam::getRotationToFirstFrame(const Submap* submap, const Pose& current_pose) const
    {
        if (!submap) {
            return 0.0f;
        }
        return submap->getRotationToFirstFrameLocal(toSubmapLocalPose(submap, current_pose));
    }

    Pose GSSlam::toSubmapLocalPose(const Submap* submap, const Pose& global_pose) const
    {
        if (!submap) {
            return global_pose;
        }
        return composePoses(invertPose(submap->getGlobalPose()), global_pose);
    }

    bool GSSlam::checkSubmapTransition(const Pose& current_pose)
    {
        Submap* submap = getCurrentSubmap();
        if (!submap) {
            return false;
        }

        const float distance_m = getDistanceToFirstFrame(submap, current_pose);
        const float rotation_deg = getRotationToFirstFrame(submap, current_pose);

        return distance_m > submap_dist_threshold_m_ || rotation_deg > submap_rot_threshold_deg_;
    }

    void GSSlam::buildLoopDiagnosticsRgbdFrame(size_t index,
                                               const cv::Mat &rgb,
                                               const cv::Mat &depth,
                                               cv::Mat &out_rgb,
                                               cv::Mat &out_depth) const
    {
        if (index == 0 || index == 7)
        {
            out_rgb = rgb.clone();
            out_depth = depth.clone();
            return;
        }

        out_rgb = rgb.clone();
        if (!out_rgb.empty())
        {
            const int channels = out_rgb.channels();
            const cv::Scalar tint(
                40.0 + 24.0 * static_cast<double>(index),
                180.0 - 12.0 * static_cast<double>(index),
                60.0 + 28.0 * static_cast<double>(index),
                channels == 4 ? 255.0 : 0.0);

            cv::Mat overlay(out_rgb.size(), out_rgb.type(), tint);
            cv::addWeighted(out_rgb, 0.35, overlay, 0.65, 0.0, out_rgb);

            const cv::Point center(out_rgb.cols / 2, out_rgb.rows / 2);
            const int radius = std::max(10, std::min(out_rgb.cols, out_rgb.rows) / 5);
            const cv::Scalar white(channels == 4 ? 255.0 : 255.0,
                                   channels == 4 ? 255.0 : 255.0,
                                   channels == 4 ? 255.0 : 255.0,
                                   channels == 4 ? 255.0 : 0.0);
            cv::circle(out_rgb, center, radius, white, 2, cv::LINE_AA);

            const std::string label = "DIAG " + std::to_string(index + 1);
            cv::putText(out_rgb,
                        label,
                        cv::Point(20, std::max(30, out_rgb.rows / 4)),
                        cv::FONT_HERSHEY_SIMPLEX,
                        1.0,
                        white,
                        2,
                        cv::LINE_AA);
        }

        out_depth = depth.clone();
        if (!out_depth.empty() && out_depth.channels() == 1)
        {
            if (out_depth.depth() == CV_32F)
            {
                const float bias = 0.01f * static_cast<float>(index);
                for (int y = 0; y < out_depth.rows; ++y)
                {
                    float *row = out_depth.ptr<float>(y);
                    const float y_term = 0.00025f * static_cast<float>(y);
                    for (int x = 0; x < out_depth.cols; ++x)
                    {
                        row[x] += bias + 0.00015f * static_cast<float>(x) + y_term;
                    }
                }
            }
            else
            {
                cv::Mat shifted_depth;
                out_depth.convertTo(shifted_depth, out_depth.type(), 1.0, static_cast<double>(index));
                out_depth = shifted_depth;
            }
        }
    }

    Pose GSSlam::makeLoopDiagnosticsPose(size_t index) const
    {
        constexpr float kPi = 3.14159265358979323846f;
        constexpr float kTwoPi = 6.28318530717958647692f;
        constexpr float kRadius = 2.0f;
        constexpr float kDriftStep = 0.1f;

        const float angle = (index < 7)
            ? (kTwoPi * static_cast<float>(index) / 7.0f)
            : 0.0f;
        const float radius = kRadius + kDriftStep * static_cast<float>(index);
        const Eigen::Vector3f forward_w(-std::sin(angle), std::cos(angle), 0.0f);
        const Eigen::Vector3f up_w(1.0f, 0.0f, 0.0f);
        Eigen::Vector3f right_w = up_w.cross(forward_w);
        if (right_w.squaredNorm() < 1e-8f)
        {
            right_w = Eigen::Vector3f::UnitX();
        }
        right_w.normalize();

        Eigen::Vector3f true_up_w = forward_w.cross(right_w);
        if (true_up_w.squaredNorm() < 1e-8f)
        {
            true_up_w = up_w;
        }
        true_up_w.normalize();

        Eigen::Matrix3f R_cw;
        R_cw.col(0) = right_w;
        R_cw.col(1) = true_up_w;
        R_cw.col(2) = forward_w.normalized();

        Eigen::Quaternionf q_cw(R_cw);
        q_cw.normalize();

        Pose pose = Pose::Identity();
        pose.position.x = radius * std::cos(angle);
        pose.position.y = radius * std::sin(angle);
        pose.position.z = index * 0.1f;
        pose.orientation.x = q_cw.x();
        pose.orientation.y = q_cw.y();
        pose.orientation.z = q_cw.z();
        pose.orientation.w = q_cw.w();
        return pose;
    }

    void GSSlam::runLoopDiagnosticsSmokeTest(const cv::Mat &rgb, const cv::Mat &depth)
    {

        std::cout << "[GSSlam][DIAG] Running legacy loop diagnostics smoke test" << std::endl;
        // Creamos un submapa inicial con la pose inicial de la cámara (basada en IMU) y luego vamos creando submapas nuevos con pequeñas transformaciones
        // para simular un escenario de loop closure. No se aplican actualizaciones de PGO en este modo, solo se verifica que el sistema maneja correctamente la creación de submapas
        createNewSubmap();
        createNewSubmap();
        Submap* submap = getCurrentSubmap();
        if (submap)
        {
            // Inicializar pose IMU
            P_cur_[0] = initial_pose_imu_.position.x;
            P_cur_[1] = initial_pose_imu_.position.y;
            P_cur_[2] = initial_pose_imu_.position.z;
            P_cur_[3] = initial_pose_imu_.orientation.x;  // qx
            P_cur_[4] = initial_pose_imu_.orientation.y;  // qy
            P_cur_[5] = initial_pose_imu_.orientation.z;  // qz
            P_cur_[6] = initial_pose_imu_.orientation.w;  // qw

            for (int i = 0; i < 7; ++i) {
                P_prev_[i] = P_cur_[i];
            }

            if (submap) {
                current_pose_ = composePoses(invertPose(submap->getGlobalPose()), initial_pose_cam_);
            } else {
                current_pose_ = initial_pose_cam_;
            }

            if (submap) {
                submap->first_frame_pose_local = current_pose_;
                {
                    Pose cam_global = composePoses(submap->getGlobalPose(), current_pose_);
                    initializeGaussiansFromRgbd(submap, cam_global);
                }
                addKeyframe(submap);
            }

            if (!optimize_thread_.joinable()) {
                stop_optimization_.store(false);
                optimize_thread_ = std::thread(&GSSlam::optimizationLoop, this);
            }

            // ===== Loop diagnostics mode: synthetic duplicated submaps =====
            if (loop_diagnostics_mode_ && !loop_diagnostics_executed_) {
                std::cout << "[GSSlam][DIAG] Running loop diagnostics synthetic scenario" << std::endl;

                auto printPoseBlock = [](const char *label, const Pose &relative_pose, const Pose &global_pose) {
                    std::cout << "[GSSlam][DIAG][POSE] " << label
                              << " rel=" << poseToString(relative_pose)
                              << " abs=" << poseToString(global_pose)
                              << std::endl;
                };

                auto printSubmapBlock = [&](size_t index, const std::shared_ptr<Submap> &submap_ptr) {
                    if (!submap_ptr) {
                        std::cout << "[GSSlam][DIAG][SUBMAP] index=" << index << " null" << std::endl;
                        return;
                    }

                    std::cout << "[GSSlam][DIAG][SUBMAP] index=" << index
                              << " submap_id=" << submap_ptr->submap_id
                              << " gaussians=" << submap_ptr->getGaussiansCount()
                              << " keyframes=" << submap_ptr->keyframes.size()
                              << std::endl;
                    printPoseBlock((std::string("submap[") + std::to_string(index) + "]").c_str(),
                                   submap_ptr->getRelativePose(),
                                   submap_ptr->getGlobalPose());

                    for (size_t kf_idx = 0; kf_idx < submap_ptr->keyframes.size(); ++kf_idx) {
                        const auto &kf = submap_ptr->keyframes[kf_idx];
                        printPoseBlock((std::string("submap[") + std::to_string(index) + "].keyframe[" + std::to_string(kf_idx) + "]").c_str(),
                                       kf.getRelativePose(),
                                       kf.getGlobalPose());
                    }
                };

            const bool apply_pgo_updates = loop_closure_ && loop_closure_->getConfiguration().apply_pgo_updates;

            if (apply_pgo_updates)
            {
                std::cout << "[GSSlam][DIAG] Running loop diagnostics synthetic 8-submap loop scenario" << std::endl;

                Submap* first_submap = getCurrentSubmap();
                if (!first_submap)
                {
                    optimization_mutex_.unlock();
                    return;
                }

                if (!optimize_thread_.joinable()) {
                    stop_optimization_.store(false);
                    optimize_thread_ = std::thread(&GSSlam::optimizationLoop, this);
                }

                constexpr size_t kSubmapCount = 8;
                std::array<Pose, kSubmapCount> absolute_poses{};
                for (size_t i = 0; i < kSubmapCount; ++i)
                {
                    absolute_poses[i] = makeLoopDiagnosticsPose(i);
                }

                first_submap->setRelativePose(absolute_poses[0]);
                updateSubmapChainGlobalPoses();

                cv::Mat diag_rgb;
                cv::Mat diag_depth;

                current_pose_ = initial_pose_cam_;
                //first_submap->first_frame_pose_local = Pose::Identity();
                buildLoopDiagnosticsRgbdFrame(0, rgb, depth, diag_rgb, diag_depth);
                initAndCopyImgs(diag_rgb, diag_depth);
                //initializeGaussiansFromRgbd(first_submap, absolute_poses[0]);
                addKeyframe(first_submap);

                Pose previous_absolute_pose = absolute_poses[0];
                for (size_t i = 1; i < kSubmapCount; ++i)
                {
                    current_pose_ = composePoses(invertPose(previous_absolute_pose), absolute_poses[i]);
                    createNewSubmap();

                    Submap* submap = getCurrentSubmap();
                    if (!submap)
                    {
                        optimization_mutex_.unlock();
                        return;
                    }

                    current_pose_ = Pose::Identity();
                    //submap->first_frame_pose_local = current_pose_;
                    buildLoopDiagnosticsRgbdFrame(i, rgb, depth, diag_rgb, diag_depth);
                    initAndCopyImgs(diag_rgb, diag_depth);
                    //initializeGaussiansFromRgbd(submap, absolute_poses[i]);
                    // Solo inicio gaussianas en el ultimo
                    if (i == kSubmapCount - 1) {
                        //Obtengo el submapa 0 desde el array
                        initializeGaussiansFromRgbd(submaps_[0].get(), absolute_poses[0]);

                        submap->gaussians = submaps_[0]->gaussians;
                        submap->gaussians_count = submaps_[0]->gaussians_count;
                        submap->adam_states = submaps_[0]->adam_states;
                    }
                    addKeyframe(submap);

                    previous_absolute_pose = absolute_poses[i];
                }

                refreshGlobalGaussiansCount();
                finalizeCurrentSubmap();

                updateSubmapChainGlobalPoses();

                loop_diagnostics_executed_ = true;
                first_image_ = false;
                optimization_mutex_.unlock();
                return;
            }

                Submap* first_submap = getCurrentSubmap();
                if (first_submap) {
                    auto first_submap_ptr = submaps_.empty() ? std::shared_ptr<Submap>{} : submaps_.front();
                    printSubmapBlock(0, first_submap_ptr);

                    // Print de su id
                    std::cout << "[GSSlam][DIAG] First submap ID: " << first_submap->submap_id << std::endl;

                    finalizeCurrentSubmap();

                    Pose saved_pose = current_pose_;
                    current_pose_.position.x = current_pose_.position.x - 0.80f;
                    current_pose_.position.y = current_pose_.position.y + 0.3f;
                    current_pose_.position.z = current_pose_.position.z + 0.30f;
                    current_pose_.orientation.x = 0.0f;
                    current_pose_.orientation.y = 0.0f;
                    current_pose_.orientation.z = 0.0f;
                    current_pose_.orientation.w = 1.0f;

                    //Eigen::AngleAxisf rot_yaw(5.0f * static_cast<float>(M_PI) / 180.0f, Eigen::Vector3f::UnitY());
                    //Eigen::AngleAxisf rot_pitch(-3.0f * static_cast<float>(M_PI) / 180.0f, Eigen::Vector3f::UnitX());
                    //Eigen::Quaternionf rot = rot_yaw * rot_pitch;
                    //Eigen::Quaternionf cam_rot(current_pose_.orientation.w,
                    //                          current_pose_.orientation.x,
                    //                          current_pose_.orientation.y,
                    //                          current_pose_.orientation.z);
                    //cam_rot = cam_rot * rot;
                    //current_pose_.orientation.x = cam_rot.x();
                    //current_pose_.orientation.y = cam_rot.y();
                    //current_pose_.orientation.z = cam_rot.z();
                    //current_pose_.orientation.w = cam_rot.w();

                    createNewSubmap();
                    Submap* second_submap = getCurrentSubmap();
                    if (second_submap) {
                        // Print de su id
                        std::cout << "[GSSlam][DIAG] Second submap ID: " << second_submap->submap_id << std::endl;

                        // Para el diagnostico, el segundo submapa copia exactamente
                        // las gaussianas del primero en vez de reconstruirlas desde RGB-D.
                        second_submap->gaussians = first_submap->gaussians;
                        second_submap->gaussians_count = first_submap->gaussians_count;
                        second_submap->adam_states = first_submap->adam_states;
                        refreshGlobalGaussiansCount();

                        addKeyframe(second_submap);

                        finalizeCurrentSubmap();

                        printSubmapBlock(1, submaps_.size() > 1 ? submaps_[1] : std::shared_ptr<Submap>{});
                    }
                    createNewSubmap();
                    Submap* third_submap = getCurrentSubmap();
                    if (third_submap) {
                        // El tercer submapa queda deliberadamente vacio: sin gaussianas ni keyframes.
                    }

                    current_pose_ = saved_pose;
                    loop_diagnostics_executed_ = true;
                    first_image_ = false;
                    optimization_mutex_.unlock();
                    return;
                }
            }
        }

        optimization_mutex_.unlock();
    }

    void GSSlam::initCudaEvents()
    {
        cudaEventCreate(&cuda_evt_init_start_);
        cudaEventCreate(&cuda_evt_init_end_);
        cudaEventCreate(&cuda_evt_pyramid_start_);
        cudaEventCreate(&cuda_evt_pyramid_end_);
        cudaEventCreate(&cuda_evt_prepare_rast_start_);
        cudaEventCreate(&cuda_evt_prepare_rast_end_);
        cudaEventCreate(&cuda_evt_rasterize_start_);
        cudaEventCreate(&cuda_evt_rasterize_end_);
        cudaEventCreate(&cuda_evt_keyframe_opt_start_);
        cudaEventCreate(&cuda_evt_keyframe_opt_end_);
        cudaEventCreate(&cuda_evt_densify_start_);
        cudaEventCreate(&cuda_evt_densify_end_);
        cudaEventCreate(&cuda_evt_prune_start_);
        cudaEventCreate(&cuda_evt_prune_end_);
        cudaEventCreate(&cuda_evt_imgcopy_start_);
        cudaEventCreate(&cuda_evt_imgcopy_end_);
        cudaEventCreate(&cuda_evt_frame_start_);
        cudaEventCreate(&cuda_evt_frame_end_);
    }

    void GSSlam::destroyCudaEvents()
    {
        cudaEventDestroy(cuda_evt_init_start_);
        cudaEventDestroy(cuda_evt_init_end_);
        cudaEventDestroy(cuda_evt_pyramid_start_);
        cudaEventDestroy(cuda_evt_pyramid_end_);
        cudaEventDestroy(cuda_evt_prepare_rast_start_);
        cudaEventDestroy(cuda_evt_prepare_rast_end_);
        cudaEventDestroy(cuda_evt_rasterize_start_);
        cudaEventDestroy(cuda_evt_rasterize_end_);
        cudaEventDestroy(cuda_evt_keyframe_opt_start_);
        cudaEventDestroy(cuda_evt_keyframe_opt_end_);
        cudaEventDestroy(cuda_evt_densify_start_);
        cudaEventDestroy(cuda_evt_densify_end_);
        cudaEventDestroy(cuda_evt_prune_start_);
        cudaEventDestroy(cuda_evt_prune_end_);
        cudaEventDestroy(cuda_evt_imgcopy_start_);
        cudaEventDestroy(cuda_evt_imgcopy_end_);
        cudaEventDestroy(cuda_evt_frame_start_);
        cudaEventDestroy(cuda_evt_frame_end_);
    }

    float GSSlam::getCudaEventElapsedMs(cudaEvent_t start, cudaEvent_t end) const
    {
        float ms = 0.0f;
        cudaEventSynchronize(end);
        cudaEventElapsedTime(&ms, start, end);
        return ms;
    }

    void GSSlam::setIntrinsics(const IntrinsicParameters &params)
    {
        intrinsics_ = params;
        updateIntrinsicsPyramid();
        intrinsics_set_ = true;
    }

    void GSSlam::updateIntrinsicsPyramid()
    {
        if (nb_pyr_levels_ <= 0) return;

        pyr_intrinsics_.resize(nb_pyr_levels_);

        for (int i = 0; i < nb_pyr_levels_; ++i)
        {
            float scale = 1.0f / (1 << i);

            pyr_intrinsics_[i] = intrinsics_;
            pyr_intrinsics_[i].f.x *= scale;
            pyr_intrinsics_[i].f.y *= scale;
            pyr_intrinsics_[i].c.x *= scale;
            pyr_intrinsics_[i].c.y *= scale;
        }
    }

    void GSSlam::setGaussInitSizePx(int size_px)
    {
        gauss_init_size_px_ = std::max(1, size_px);
    }

    void GSSlam::setGaussInitScale(float scale)
    {
        gauss_init_scale_ = std::max(1e-5f, scale);
    }

    RegionSamplingStrategy GSSlam::parseRegionSamplingStrategy(const std::string &strategy_name) const
    {
        std::string normalized = strategy_name;
        std::transform(normalized.begin(), normalized.end(), normalized.begin(),
                       [](unsigned char c) { return static_cast<char>(std::tolower(c)); });

        if (normalized == "vigs_fusion" || normalized == "original") {
            return RegionSamplingStrategy::VigsFusion;
        }
        if (normalized == "fgs" || normalized == "fourier") {
            return RegionSamplingStrategy::FgsFourier;
        }
        if (normalized == "fast_placeholder" || normalized == "placeholder") {
            return RegionSamplingStrategy::FastPlaceholder;
        }
        if (normalized == "sobel" || normalized == "sobel_edge") {
            return RegionSamplingStrategy::SobelEdge;
        }
        if (normalized == "laplacian" || normalized == "laplacian_edge") {
            return RegionSamplingStrategy::LaplacianEdge;
        }
        if (normalized == "canny" || normalized == "canny_edge") {
            return RegionSamplingStrategy::CannyEdge;
        }

        std::cerr << "[GSSlam][WARN] Unknown strategy '" << strategy_name
                  << "'. Falling back to vigs_fusion." << std::endl;
        return RegionSamplingStrategy::VigsFusion;
    }

    const char* GSSlam::regionSamplingStrategyName(RegionSamplingStrategy strategy)
    {
        switch (strategy)
        {
        case RegionSamplingStrategy::VigsFusion:
            return "vigs_fusion";
        case RegionSamplingStrategy::FgsFourier:
            return "fgs";
        case RegionSamplingStrategy::FastPlaceholder:
            return "fast_placeholder";
        case RegionSamplingStrategy::SobelEdge:
            return "sobel";
        case RegionSamplingStrategy::LaplacianEdge:
            return "laplacian";
        case RegionSamplingStrategy::CannyEdge:
            return "canny";
        default:
            return "vigs_fusion";
        }
    }

    void GSSlam::setInitializationStrategy(const std::string &strategy_name)
    {
        initialization_strategy_ = parseRegionSamplingStrategy(strategy_name);
    }

    void GSSlam::setDensificationStrategy(const std::string &strategy_name)
    {
        densification_strategy_ = parseRegionSamplingStrategy(strategy_name);
    }

    std::string GSSlam::getInitializationStrategyName() const
    {
        return regionSamplingStrategyName(initialization_strategy_);
    }

    std::string GSSlam::getDensificationStrategyName() const
    {
        return regionSamplingStrategyName(densification_strategy_);
    }

    bool GSSlam::buildStrategyFrequencyMasks(const cv::cuda::GpuMat &color_gpu,
                                             RegionSamplingStrategy strategy,
                                             FrequencyMaskGpuPair &frequency_masks_gpu,
                                             const char *stage_tag) const
    {
        if (strategy == RegionSamplingStrategy::FgsFourier)
        {
            frequency_masks_gpu = buildFrequencyMasksGpu(color_gpu, fgs_highpass_sigma_ratio_);
            return !frequency_masks_gpu.high_mask_u8_gpu.empty();
        }

        if (strategy == RegionSamplingStrategy::SobelEdge)
        {
            frequency_masks_gpu = buildSobelMasksGpu(color_gpu);
            return !frequency_masks_gpu.high_mask_u8_gpu.empty();
        }

        if (strategy == RegionSamplingStrategy::LaplacianEdge)
        {
            frequency_masks_gpu = buildLaplacianMasksGpu(color_gpu);
            return !frequency_masks_gpu.high_mask_u8_gpu.empty();
        }

        if (strategy == RegionSamplingStrategy::CannyEdge)
        {
            frequency_masks_gpu = buildCannyMasksGpu(color_gpu, canny_threshold1_, canny_threshold2_);
            return !frequency_masks_gpu.high_mask_u8_gpu.empty();
        }

        if (strategy == RegionSamplingStrategy::FastPlaceholder)
        {
            static std::atomic<bool> warned_init_once{false};
            static std::atomic<bool> warned_dens_once{false};
            std::atomic<bool> *warn_flag =
                (std::string(stage_tag) == "initialization") ? &warned_init_once : &warned_dens_once;

            bool expected = false;
            if (warn_flag->compare_exchange_strong(expected, true))
            {
                std::cerr << "[GSSlam][WARN] Strategy 'fast_placeholder' is not implemented for "
                          << stage_tag
                          << ". Falling back to vigs_fusion behavior." << std::endl;
            }
            return false;
        }

        return false;
    }

    void GSSlam::setCamToImuExtrinsics(const Eigen::Vector3d &t_imu_cam,
                                       const Eigen::Quaterniond &q_imu_cam)
    {
        // Notación A_b_c a la transformación A del sistema c al b
        t_imu_cam_ = t_imu_cam;
        q_imu_cam_ = q_imu_cam.normalized();
    }

    void GSSlam::initialize(const Pose &pose_imu, const Pose &pose_cam)
    {
        initial_pose_imu_ = pose_imu;
        initial_pose_cam_ = pose_cam;
        has_initial_pose_ = true;
    }

    // ========================================================================
    // Acceso seguro a gaussianas de un submapa (copia a buffers CPU)
    // ========================================================================
    uint32_t GSSlam::getGaussianData(Submap* submap, float4* positions, float4* colors, uint32_t max_n)
    {
        if (!submap || max_n == 0) {
            return 0;
        }

        std::lock_guard<std::mutex> lock(optimization_mutex_);

        // Ensure all pending GPU operations complete before copying data
        cudaDeviceSynchronize();

        const uint32_t n_gauss = submap->getGaussiansCount();
        const uint32_t n = std::min(n_gauss, max_n);

        // ============================================================
        // Validaciones
        // ============================================================
        if (submap->gaussians.positions.size() < n || submap->gaussians.colors.size() < n) {
            std::cerr << "[ERROR] getGaussianData size mismatch: "
                      << "positions.size()=" << submap->gaussians.positions.size()
                      << ", colors.size()=" << submap->gaussians.colors.size()
                      << ", requested=" << n << std::endl;
            return 0;
        }

        // ============================================================
        // Copia GPU → CPU
        // ============================================================
        try {
            thrust::copy(submap->gaussians.positions.begin(),
                         submap->gaussians.positions.begin() + n,
                         positions);

            thrust::copy(submap->gaussians.colors.begin(),
                         submap->gaussians.colors.begin() + n,
                         colors);
        } catch (const std::exception &e) {
            std::cerr << "[ERROR] getGaussianData thrust::copy failed: " << e.what() << std::endl;
            return 0;
        }

        return n;
    }

    // ========================================================================
    // VERSIONES GLOBALES (todos los submapas)
    // ========================================================================

    bool GSSlam::hasGaussiansGlobal() const
    {
        return global_gaussians_count_ > 0;
    }

    uint32_t GSSlam::getGaussiansCountGlobal() const
    {
        return static_cast<uint32_t>(std::max(0, global_gaussians_count_));
    }

    uint32_t GSSlam::getGaussianDataGlobal(float4* positions,
                                          float4* colors,
                                          uint32_t max_n)
    {
        // Llena un vector buffer con todas las gaussianas de todos los submapas
        optimization_mutex_.lock();

        // Ensure all pending GPU operations complete before copying data
        cudaDeviceSynchronize();
        
        uint32_t total_copied = 0;

        for (const auto& submap : submaps_) {
            if (!submap || total_copied >= max_n) {
                break;
            }

            const uint32_t n_gauss = submap->getGaussiansCount();
            if (n_gauss == 0) {
                continue;
            }

            const uint32_t n_to_copy = std::min(n_gauss, max_n - total_copied);

            // ============================================================
            // Validaciones
            // ============================================================
            if (submap->gaussians.positions.size() < n_to_copy ||
                submap->gaussians.colors.size() < n_to_copy) {
                std::cerr << "[ERROR] Submap gaussian size mismatch: "
                          << "positions.size()=" << submap->gaussians.positions.size()
                          << ", colors.size()=" << submap->gaussians.colors.size()
                          << ", requested=" << n_to_copy << std::endl;
                optimization_mutex_.unlock();
                return total_copied;
            }

            // ============================================================
            // Copia GPU → CPU (offset en el buffer)
            // ============================================================
            try {
                thrust::copy(submap->gaussians.positions.begin(),
                            submap->gaussians.positions.begin() + n_to_copy,
                            positions + total_copied);

                thrust::copy(submap->gaussians.colors.begin(),
                            submap->gaussians.colors.begin() + n_to_copy,
                            colors + total_copied);
            } catch (const std::exception &e) {
                std::cerr << "[ERROR] thrust::copy failed: " << e.what() << std::endl;
                optimization_mutex_.unlock();
                return total_copied;
            }

            total_copied += n_to_copy;
        }

        optimization_mutex_.unlock();
        return total_copied;
    }

    void GSSlam::initializeGaussiansFromRgbd(Submap* submap, const Pose &cameraPose)
    {
        cudaEventRecord(cuda_evt_init_start_);
        //std::cout << "[GSSlam] Initializing gaussians from RGB-D (GPU pipeline)..." << std::endl;
        // ============================================================
        // 1. Early exit
        // ============================================================
        if (rgb_gpu_.empty() || depth_gpu_.empty()) {
            std::cerr << "[GSSlam][ERROR] RGB or depth GPU data is empty." << std::endl;
            return;
        }

        const int width  = rgb_gpu_.cols;
        const int height = rgb_gpu_.rows;

        if (width <= 0 || height <= 0) {
            std::cerr << "[GSSlam][ERROR] Invalid image dimensions: " << width << "x" << height << std::endl;
            return;
        }

        // ============================================================
        // 2. Reset contador de gaussianas (GPU)
        // ============================================================
        cudaMemset(thrust::raw_pointer_cast(instance_counter_.data()),
                0,
                sizeof(uint32_t));

        const auto launch_init_pass = [&](uint32_t sample_px,
                                          cudaTextureObject_t mask_tex,
                                          uint32_t use_mask,
                                          float scale_factor)
        {
            const int sample_w = (width + static_cast<int>(sample_px) - 1) / static_cast<int>(sample_px);
            const int sample_h = (height + static_cast<int>(sample_px) - 1) / static_cast<int>(sample_px);

            dim3 block(16, 16);
            dim3 grid((sample_w + block.x - 1) / block.x,
                      (sample_h + block.y - 1) / block.y);

            const Pose submap_global = submap->getGlobalPose();

            initGaussiansFromRgbd_kernel<<<grid, block>>>(
                thrust::raw_pointer_cast(submap->gaussians.positions.data()),
                thrust::raw_pointer_cast(submap->gaussians.scales.data()),
                thrust::raw_pointer_cast(submap->gaussians.orientations.data()),
                thrust::raw_pointer_cast(submap->gaussians.colors.data()),
                thrust::raw_pointer_cast(submap->gaussians.opacities.data()),
                thrust::raw_pointer_cast(instance_counter_.data()),
                submap->max_gaussians,

                pyr_color_tex_[0]->getTextureObject(),
                pyr_depth_tex_[0]->getTextureObject(),
                pyr_normals_tex_[0]->getTextureObject(),
                mask_tex,

                width,
                height,
                intrinsics_,
                cameraPose,

                sample_px,
                sample_px,
                gauss_init_opacity_,
                use_mask,
                scale_factor,
                submap_global);

            CUDA_CHECK_KERNEL("initGaussiansFromRgbd_kernel");
        };

        if (initialization_strategy_ == RegionSamplingStrategy::VigsFusion)
        {
            launch_init_pass(static_cast<uint32_t>(gauss_init_size_px_),
                             0,
                             0u,
                             0.8f);
        }
        else
        {
            // ============================================================
            // 3. FRECUENCIA (M_h / M_l)
            // ============================================================
            // M_h: regiones de alta frecuencia.
            // M_l: regiones de baja frecuencia.
            FrequencyMaskGpuPair frequency_masks_gpu;
            const bool masks_ready = buildStrategyFrequencyMasks(
                pyr_color_[0], initialization_strategy_, frequency_masks_gpu, "initialization");

            if (!masks_ready)
            {
                launch_init_pass(static_cast<uint32_t>(gauss_init_size_px_),
                                 0,
                                 0u,
                                 0.8f);
            }
            else
            {
                cv::cuda::GpuMat high_gpu;
                cv::cuda::GpuMat low_gpu;
                frequency_masks_gpu.high_mask_u8_gpu.convertTo(high_gpu, CV_32F, 1.0 / 255.0);
                frequency_masks_gpu.low_mask_u8_gpu.convertTo(low_gpu, CV_32F, 1.0 / 255.0);

                auto high_tex = std::make_shared<Texture<float>>(high_gpu);
                auto low_tex = std::make_shared<Texture<float>>(low_gpu);

                launch_init_pass(static_cast<uint32_t>(fgs_sample_high_px_),
                                 high_tex->getTextureObject(),
                                 1u,
                                 fgs_scale_high_);

                launch_init_pass(static_cast<uint32_t>(fgs_sample_low_px_),
                                 low_tex->getTextureObject(),
                                 1u,
                                 fgs_scale_low_);
            }
        }

        // ============================================================
        // 5. Descargar número de gaussianas generadas
        // ============================================================
        uint32_t host_count = 0;

        cudaMemcpy(&host_count,
            thrust::raw_pointer_cast(instance_counter_.data()),
            sizeof(uint32_t),
            cudaMemcpyDeviceToHost);

        submap->gaussians_count = std::min(host_count, submap->max_gaussians);
        refreshGlobalGaussiansCount();

        uint32_t invalid_pos = 0;
        uint32_t invalid_scale = 0;
        uint32_t invalid_orientation = 0;
        uint32_t invalid_opacity = 0;
        countInvalidGaussians(
            submap->gaussians.positions,
            submap->gaussians.scales,
            submap->gaussians.orientations,
            submap->gaussians.opacities,
            submap->gaussians_count,
            invalid_pos,
            invalid_scale,
            invalid_orientation,
            invalid_opacity);

        if (nb_images_processed_ == 0)
        {
            static std::atomic<bool> f0_gauss_logged{false};
            bool expected = false;
            if (f0_gauss_logged.compare_exchange_strong(expected, true))
            {
                std::ostringstream oss;
                oss << "\"frame\":" << nb_images_processed_
                    << ",\"n_gaussians\":" << submap->gaussians_count
                    << ",\"invalid_pos\":" << invalid_pos
                    << ",\"invalid_scale\":" << invalid_scale
                    << ",\"invalid_orientation\":" << invalid_orientation
                    << ",\"invalid_opacity\":" << invalid_opacity;
                // printF0Diag("f_vigs_slam", "gaussians", oss.str());
            }
        }

        (void)invalid_pos;
        (void)invalid_scale;
        (void)invalid_orientation;
        (void)invalid_opacity;

        cudaEventRecord(cuda_evt_init_end_);
        last_gpu_timings_.init_gaussians_ms = getCudaEventElapsedMs(cuda_evt_init_start_, cuda_evt_init_end_);
    }

    void GSSlam::prepareRasterization(Submap* submap, const Pose &camera_pose,
                                    const IntrinsicParameters &intrinsics,
                                    int width,
                                    int height)
    {
        cudaEventRecord(cuda_evt_prepare_rast_start_);
        //std::cout << "[GSSlam] Preparing rasterization (projection + hashing)..." << std::endl;
        // ============================================================
        // 0. EARLY EXIT
        // ============================================================
        last_nb_instances_ = 0;

        if (!submap || submap->getGaussiansCount() == 0 || width <= 0 || height <= 0) {
            std::cerr << "[GSSlam][ERROR] Invalid parameters for rasterization." << std::endl;
            return;
        }

        // ============================================================
        // 1. CONFIGURAR GRID DE TILES
        // ============================================================
        num_tiles_ = make_uint2(
            (width  + tile_size_.x - 1) / tile_size_.x,
            (height + tile_size_.y - 1) / tile_size_.y
        );

        const uint32_t num_tiles_total = num_tiles_.x * num_tiles_.y;

        tile_ranges_.resize(num_tiles_total);
        thrust::fill(tile_ranges_.begin(), tile_ranges_.end(), make_uint2(0u, 0u));

        // ============================================================
        // 2. RESET CONTADORES GPU
        // ============================================================
        cudaMemset(thrust::raw_pointer_cast(instance_counter_.data()),
                0,
                sizeof(uint32_t));

        // ============================================================
        // 3. PREPARAR KERNEL LAUNCH
        // ============================================================
        const uint32_t n_gauss = submap->getGaussiansCount();
        constexpr uint32_t kMaxTilesPerGaussian = 80u;

        // Ensure dynamic buffers can hold the current gaussian count.
        if (positions_2d_.size() < n_gauss)
        {
            positions_2d_.resize(n_gauss);
            covariances_2d_.resize(n_gauss);
            inv_covariances_2d_.resize(n_gauss);
            p_hats_.resize(n_gauss);
            normals_2d_.resize(n_gauss);

            // Visibility buffers are also indexed by gaussian id.
            d_keyframeVis_.resize(n_gauss);
            d_frameVis_.resize(n_gauss);
            d_imgPositions_.resize(n_gauss);

            // Gradients/aux buffers used by optimization and pruning paths.
            gaussian_gradients_.resize(n_gauss);
            opacity_gradients_.resize(n_gauss);
            new_positions.resize(n_gauss);
            new_scales.resize(n_gauss);
            new_orientations.resize(n_gauss);
            new_colors.resize(n_gauss);
            new_opacities.resize(n_gauss);
            new_adam_states.resize(n_gauss);
        }

        const size_t required_instances = static_cast<size_t>(n_gauss) * static_cast<size_t>(kMaxTilesPerGaussian);
        if (hashes_.size() < required_instances)
        {
            hashes_.resize(required_instances);
            gaussian_indices_.resize(required_instances);
        }

        dim3 block(128);
        dim3 grid((n_gauss + block.x - 1) / block.x);

        // ============================================================
        // 4. RAW POINTERS
        // ============================================================
        auto *positions_2d_ptr        = thrust::raw_pointer_cast(positions_2d_.data());
        auto *covariances_2d_ptr      = thrust::raw_pointer_cast(covariances_2d_.data());
        auto *inv_covariances_2d_ptr  = thrust::raw_pointer_cast(inv_covariances_2d_.data());
        auto *p_hats_ptr              = thrust::raw_pointer_cast(p_hats_.data());
        auto *normals_2d_ptr          = thrust::raw_pointer_cast(normals_2d_.data());

        auto *hashes_ptr              = thrust::raw_pointer_cast(hashes_.data());
        auto *gaussian_indices_ptr    = thrust::raw_pointer_cast(gaussian_indices_.data());
        auto *instance_counter_ptr    = thrust::raw_pointer_cast(instance_counter_.data());

        auto *positions_world_ptr     = thrust::raw_pointer_cast(submap->gaussians.positions.data());
        auto *scales_ptr              = thrust::raw_pointer_cast(submap->gaussians.scales.data());
        auto *orientations_ptr        = thrust::raw_pointer_cast(submap->gaussians.orientations.data());

        // ============================================================
        // 5. PROJECTION + HASHING
        // ============================================================

        projectAndHashGaussians_kernel<<<grid, block>>>(
            positions_2d_ptr,
            covariances_2d_ptr,
            inv_covariances_2d_ptr,
            p_hats_ptr,
            normals_2d_ptr,
            hashes_ptr,
            gaussian_indices_ptr,
            instance_counter_ptr,
            positions_world_ptr,
            scales_ptr,
            orientations_ptr,
            camera_pose,
            intrinsics,
            0.4f,
            tile_size_,
            num_tiles_,
            n_gauss,
            static_cast<uint32_t>(width),
            static_cast<uint32_t>(height)
        );

        CUDA_CHECK_KERNEL("projectAndHashGaussians");

        // ============================================================
        // 6. OBTENER NÚMERO DE INSTANCIAS
        // ============================================================
        uint32_t nb_instances = 0;
        cudaMemcpy(&nb_instances,
                instance_counter_ptr,
                sizeof(uint32_t),
                cudaMemcpyDeviceToHost);

        if (nb_instances == 0) {
            return;
        }

        // ============================================================
        // 7. SORT POR TILE + DEPTH
        // ============================================================
        thrust::sort_by_key(
            hashes_.begin(),
            hashes_.begin() + nb_instances,
            gaussian_indices_.begin()
        );

        // ============================================================
        // 8. COMPUTAR RANGOS POR TILE
        // ============================================================
        computeIndicesRanges_kernel<<<(nb_instances + 255) / 256, 256>>>(
            thrust::raw_pointer_cast(tile_ranges_.data()),
            thrust::raw_pointer_cast(hashes_.data()),
            nb_instances
        );
        CUDA_CHECK_KERNEL("computeIndicesRanges");

        if (nb_images_processed_ == 0)
        {
            static std::atomic<bool> f0_raster_logged{false};
            bool expected = false;
            if (f0_raster_logged.compare_exchange_strong(expected, true))
            {
                thrust::host_vector<uint2> host_ranges = tile_ranges_;
                uint32_t non_empty_tiles = 0;
                for (size_t i = 0; i < host_ranges.size(); ++i)
                {
                    if (host_ranges[i].y > host_ranges[i].x)
                    {
                        ++non_empty_tiles;
                    }
                }
                const uint2 tile0 = host_ranges.empty() ? make_uint2(0u, 0u) : host_ranges[0];
                const uint32_t tile0_instances = (tile0.y >= tile0.x) ? (tile0.y - tile0.x) : 0u;

                std::ostringstream oss;
                oss << "\"frame\":" << nb_images_processed_
                    << ",\"width\":" << width
                    << ",\"height\":" << height
                    << ",\"tiles_total\":" << num_tiles_total
                    << ",\"tiles_non_empty\":" << non_empty_tiles
                    << ",\"nb_instances\":" << nb_instances
                    << ",\"tile0_instances\":" << tile0_instances;
                // printF0Diag("f_vigs_slam", "raster", oss.str());
            }
        }

        // ============================================================
        // 9. FINAL
        // ============================================================
        last_nb_instances_ = nb_instances;

        cudaEventRecord(cuda_evt_prepare_rast_end_);
        last_gpu_timings_.prepare_rasterization_ms = getCudaEventElapsedMs(cuda_evt_prepare_rast_start_, cuda_evt_prepare_rast_end_);
        //std::cout << "[GSSlam] [INFO prepareRasterization] Prepared " << last_nb_instances_ << " instances for rasterization." << std::endl;
    }
    
    void GSSlam::rasterize(Submap* submap, const Pose &camera_pose,
                        const IntrinsicParameters &intrinsics,
                        int width, int height)
    {
        cudaEventRecord(cuda_evt_rasterize_start_);
        //std::cout << "[GSSlam] Rasterizing view..." << std::endl;
        // ============================================================
        // 1. Early exit
        // ============================================================
        if (!submap || submap->getGaussiansCount() == 0 || width <= 0 || height <= 0) {
            std::cerr << "[GSSlam][ERROR] Invalid parameters for rasterization." << std::endl;
            return;
        }

        // ============================================================
        // 2. Preparación (proyección + binning + sorting)
        // ============================================================
        prepareRasterization(submap, camera_pose, intrinsics, width, height);

        const bool has_tile0 = !tile_ranges_.empty();
        const uint2 tile0_range = has_tile0 ? tile_ranges_[0] : make_uint2(0u, 0u);
        const uint32_t tile0_instances =
            (tile0_range.y >= tile0_range.x) ? (tile0_range.y - tile0_range.x) : 0u;

        if (last_nb_instances_ == 0) {
            static uint32_t no_visible_warn_count = 0;
            ++no_visible_warn_count;

            if (rendered_rgb_gpu_.empty() ||
                rendered_rgb_gpu_.cols != width ||
                rendered_rgb_gpu_.rows != height)
            {
                rendered_rgb_gpu_.create(height, width, CV_8UC3);
                rendered_depth_gpu_.create(height, width, CV_32FC1);
            }

            rendered_rgb_gpu_.setTo(cv::Scalar(
                static_cast<double>(bg_color_.x * 255.0f),
                static_cast<double>(bg_color_.y * 255.0f),
                static_cast<double>(bg_color_.z * 255.0f)));
            rendered_depth_gpu_.setTo(std::numeric_limits<float>::infinity());

            if (no_visible_warn_count <= 5 || (no_visible_warn_count % 60) == 0) {
                std::cerr << "[GSSlam][WARN rasterize-empty] no visible gaussians"
                          << " n_gaussians=" << submap->getGaussiansCount()
                          << " width=" << width
                          << " height=" << height
                          << " tile0_range=[" << tile0_range.x << "," << tile0_range.y << "]"
                          << " tile0_instances=" << tile0_instances
                          << " pose_t=[" << camera_pose.position.x << ", "
                          << camera_pose.position.y << ", "
                          << camera_pose.position.z << "]"
                          << " note=render_buffers_not_overwritten_on_empty"
                          << " warn_count=" << no_visible_warn_count
                          << std::endl;
            }
            return;
        }

        // ============================================================
        // 3. Buffers de salida
        // ============================================================
        if (rendered_rgb_gpu_.empty() ||
            rendered_rgb_gpu_.cols != width ||
            rendered_rgb_gpu_.rows != height)
        {
            rendered_rgb_gpu_.create(height, width, CV_8UC3);
            rendered_depth_gpu_.create(height, width, CV_32FC1);
        }

        // ============================================================
        // 4. Configuración del kernel
        // ============================================================
        const dim3 block(tile_size_.x, tile_size_.y);
        const dim3 grid(num_tiles_.x, num_tiles_.y);

        // ============================================================
        // 5. Punteros GPU
        // ============================================================
        auto* rgb_ptr        = rendered_rgb_gpu_.ptr<uchar3>();
        auto* depth_ptr      = rendered_depth_gpu_.ptr<float>();

        auto* indices_ptr    = thrust::raw_pointer_cast(gaussian_indices_.data());
        auto* ranges_ptr     = thrust::raw_pointer_cast(tile_ranges_.data());
        auto* pos2d_ptr      = thrust::raw_pointer_cast(positions_2d_.data());
        auto* inv_cov_ptr    = thrust::raw_pointer_cast(inv_covariances_2d_.data());
        auto* colors_ptr     = thrust::raw_pointer_cast(submap->gaussians.colors.data());
        auto* opacity_ptr    = thrust::raw_pointer_cast(submap->gaussians.opacities.data());
        auto* p_hats_ptr     = thrust::raw_pointer_cast(p_hats_.data());

        // ============================================================
        // 6. Rasterización (tile-based forward pass)
        // ============================================================
        forwardPassTileKernel<<<grid, block>>>(
            rgb_ptr,
            depth_ptr,
            rendered_rgb_gpu_.step,
            rendered_depth_gpu_.step,
            indices_ptr,
            ranges_ptr,
            pos2d_ptr,
            inv_cov_ptr,
            colors_ptr,
            opacity_ptr,
            p_hats_ptr,
            bg_color_,
            width,
            height,
            num_tiles_.x,
            num_tiles_.y,
            last_nb_instances_,
            submap->getGaussiansCount()
        );

        CUDA_CHECK_KERNEL("forwardPassTileKernel");

        cudaEventRecord(cuda_evt_rasterize_end_);
        last_gpu_timings_.rasterize_ms = getCudaEventElapsedMs(cuda_evt_rasterize_start_, cuda_evt_rasterize_end_);
    }

    bool GSSlam::renderView(Submap* submap, const Pose &camera_pose,
                            const IntrinsicParameters &intrinsics,
                            int width, int height,
                            cv::cuda::GpuMat &rendered_rgb,
                            cv::cuda::GpuMat &rendered_depth,
                            bool visualize_current_submap)
    {
        optimization_mutex_.lock();

        Submap* submap_to_render = submap;

        // ============================================================
        // 1. Validaciones
        // ============================================================
        if (!intrinsics_set_) {
            optimization_mutex_.unlock();
            return false;
        }

        // Si no se pasa submapa, renderiza el mapa global (todos los submapas).
        if (!submap_to_render)
        {
            // Ensure all pending GPU operations complete before aggregating data
            cudaDeviceSynchronize();

            Submap* current_submap = visualize_current_submap ? getCurrentSubmap() : nullptr;
            
            uint32_t total_count = 0;
            for (const auto &sm : submaps_)
            {
                if (sm)
                {
                    total_count += sm->getGaussiansCount();
                }
            }

            if (total_count == 0)
            {
                optimization_mutex_.unlock();
                return false;
            }

            static Submap global_view_submap(0, 1);
            if (global_view_submap.max_gaussians < total_count)
            {
                global_view_submap.max_gaussians = total_count;
                global_view_submap.gaussians.resize(total_count);
            }

            uint32_t offset = 0;
            for (const auto &sm : submaps_)
            {
                if (!sm || sm->getGaussiansCount() == 0)
                {
                    continue;
                }

                const uint32_t n = sm->getGaussiansCount();
                
                // Validate size before copying
                if (sm->gaussians.positions.size() < n ||
                    sm->gaussians.scales.size() < n ||
                    sm->gaussians.orientations.size() < n ||
                    sm->gaussians.colors.size() < n ||
                    sm->gaussians.opacities.size() < n)
                {
                    std::cerr << "[ERROR] renderView size mismatch in submap aggregation. "
                              << "positions=" << sm->gaussians.positions.size()
                              << ", scales=" << sm->gaussians.scales.size()
                              << ", orientations=" << sm->gaussians.orientations.size()
                              << ", colors=" << sm->gaussians.colors.size()
                              << ", opacities=" << sm->gaussians.opacities.size()
                              << ", requested=" << n << std::endl;
                    continue;
                }
                
                try {
                    // ============================================================
                    // Transformar gaussianas de locales a globales
                    // ============================================================
                    Pose submap_global = sm->getGlobalPose();
                    
                    // Llamar kernel CUDA para transformar
                    const int block_size = 256;
                    const int num_blocks = (n + block_size - 1) / block_size;
                    
                    transformGaussians_localToGlobal_kernel<<<num_blocks, block_size>>>(
                        thrust::raw_pointer_cast(sm->gaussians.positions.data()),
                        thrust::raw_pointer_cast(sm->gaussians.orientations.data()),
                        thrust::raw_pointer_cast(global_view_submap.gaussians.positions.data()) + offset,
                        thrust::raw_pointer_cast(global_view_submap.gaussians.orientations.data()) + offset,
                        submap_global,
                        n
                    );
                    
                    CUDA_CHECK_KERNEL("transformGaussians_localToGlobal_kernel");
                    
                    // Copiar scales, colors, opacities (sin transformar)
                    thrust::copy(sm->gaussians.scales.begin(), sm->gaussians.scales.begin() + n,
                                 global_view_submap.gaussians.scales.begin() + offset);
                    thrust::copy(sm->gaussians.colors.begin(), sm->gaussians.colors.begin() + n,
                                 global_view_submap.gaussians.colors.begin() + offset);
                    thrust::copy(sm->gaussians.opacities.begin(), sm->gaussians.opacities.begin() + n,
                                 global_view_submap.gaussians.opacities.begin() + offset);

                    if (current_submap && sm.get() == current_submap)
                    {
                        const int block_size = 256;
                        const int num_blocks = (n + block_size - 1) / block_size;
                        float4 *highlighted_colors = thrust::raw_pointer_cast(global_view_submap.gaussians.colors.data()) + offset;
                        tintGaussianColors_kernel<<<num_blocks, block_size>>>(
                            highlighted_colors,
                            n,
                            1.0f, 1.0f, 0.0f,
                            0.35f,
                            1.12f);
                        CUDA_CHECK_KERNEL("tintGaussianColors_kernel");
                    }
                } catch (const std::exception &e) {
                    std::cerr << "[ERROR] renderView thrust::copy failed: " << e.what() << std::endl;
                    continue;
                }

                offset += n;
            }

            global_view_submap.gaussians_count = total_count;
            submap_to_render = &global_view_submap;
        }

        if (!submap_to_render || submap_to_render->getGaussiansCount() == 0) {
            optimization_mutex_.unlock();
            return false;
        }

        if (width <= 0 || height <= 0) {
            optimization_mutex_.unlock();
            return false;
        }

        // ============================================================
        // 2. Rasterización
        // ============================================================
        Pose raster_pose = camera_pose;
        if (submap) {
            raster_pose = toSubmapLocalPose(submap, camera_pose);
        }
        rasterize(submap_to_render, raster_pose, intrinsics, width, height);

        // ============================================================
        // 3. Output
        // ============================================================
        if (rendered_rgb_gpu_.empty() || rendered_depth_gpu_.empty()) {
            optimization_mutex_.unlock();
            return false;
        }

        rendered_rgb   = rendered_rgb_gpu_;
        rendered_depth = rendered_depth_gpu_;

        optimization_mutex_.unlock();
        return true;
    }

    void GSSlam::updateCameraPoseFromImu()
    {
        Pose raw_cam_pose = computeCameraPoseFromImuState();

        // Store current_pose_ as pose LOCAL to the current submap.
        Submap* cur = getCurrentSubmap();
        if (cur) {
            current_pose_ = composePoses(invertPose(cur->getGlobalPose()), raw_cam_pose);
        } else {
            // No submap: keep as global
            current_pose_ = raw_cam_pose;
        }
    }

    Pose GSSlam::computeCameraPoseFromImuState() const
    {
        // La matriz es de la forma
        // [ px py pz qx qy qz qw ]
        Eigen::Vector3f imu_trans(static_cast<float>(P_cur_[0]),
                                  static_cast<float>(P_cur_[1]),
                                  static_cast<float>(P_cur_[2]));
        // Eigen usa la convencion (w,x,y,z)
        Eigen::Quaternionf imu_rot(static_cast<float>(P_cur_[6]),
                                   static_cast<float>(P_cur_[3]),
                                   static_cast<float>(P_cur_[4]),
                                   static_cast<float>(P_cur_[5]));
        Eigen::Vector3f cam_trans = imu_trans + imu_rot.normalized().toRotationMatrix() * t_imu_cam_.cast<float>();
        Eigen::Quaternionf cam_rot = imu_rot * q_imu_cam_.cast<float>();
        cam_rot.normalize();

        Pose cam_pose;
        cam_pose.position = make_float3(cam_trans.x(), cam_trans.y(), cam_trans.z());
        cam_pose.orientation = make_float4(cam_rot.x(), cam_rot.y(), cam_rot.z(), cam_rot.w());
        return cam_pose;
    }

    /*
    void GSSlam::initWarping(const Pose &camera_pose)
    {
        if (!intrinsics_set_ || nb_pyr_levels_ <= 0) {
            warping_cache_valid_ = false;
            return;
        }

        if (pyr_color_.empty() || pyr_color_[0].empty()) {
            warping_cache_valid_ = false;
            return;
        }

        if (pyr_intrinsics_.size() != static_cast<size_t>(nb_pyr_levels_)) {
            updateIntrinsicsPyramid();
        }

        pose_warping_ = camera_pose;

        for (int level = 0; level < nb_pyr_levels_; ++level)
        {
            const int width = pyr_color_[level].cols;
            const int height = pyr_color_[level].rows;
            if (width <= 0 || height <= 0) {
                warping_cache_valid_ = false;
                return;
            }

            rasterize(getCurrentSubmap(), camera_pose, pyr_intrinsics_[level], width, height);

            if (rendered_rgb_gpu_.empty() || rendered_depth_gpu_.empty()) {
                warping_cache_valid_ = false;
                return;
            }

            pyr_color_warping_[level] = rendered_rgb_gpu_.clone();
            pyr_depth_warping_[level] = rendered_depth_gpu_.clone();
        }

        warping_cache_valid_ = true;
    }
    */

    void GSSlam::testGaussians(const cv::Mat &rgb,
                               const cv::Mat &depth)
    {
        cudaGetLastError();

        if (!intrinsics_set_ || rgb.empty() || depth.empty()) {
            return;
        }

        std::lock_guard<std::mutex> lock(optimization_mutex_);

        if (!test_gaussians_initialized_)
        {
            initAndCopyImgs(rgb, depth);

            // Freeze the test setup to identity pose and optimize only the first frame.
            current_pose_ = Pose::Identity();
            initial_pose_imu_ = Pose::Identity();
            initial_pose_cam_ = Pose::Identity();
            has_initial_pose_ = true;

            P_cur_[0] = 0.0;
            P_cur_[1] = 0.0;
            P_cur_[2] = 0.0;
            P_cur_[3] = 0.0;
            P_cur_[4] = 0.0;
            P_cur_[5] = 0.0;
            P_cur_[6] = 1.0;

            for (int i = 0; i < 7; ++i) {
                P_prev_[i] = P_cur_[i];
            }
            for (int i = 0; i < 9; ++i) {
                VB_cur_[i] = 0.0;
                VB_prev_[i] = 0.0;
            }

            {
                Submap* cur = getCurrentSubmap();
                Pose cam_global = current_pose_;
                if (cur) cam_global = composePoses(cur->getGlobalPose(), current_pose_);
                initializeGaussiansFromRgbd(cur, cam_global);
                addKeyframe(cur);
            }

            if (!optimize_thread_.joinable()) {
                stop_optimization_.store(false);
                optimize_thread_ = std::thread(&GSSlam::optimizationLoop, this);
            }
            first_image_ = false;
            test_gaussians_initialized_ = true;

          }


        // Keep identity pose fixed in test mode while the background gaussian optimization runs.
        current_pose_ = Pose::Identity();
    }

    void GSSlam::compute(const cv::Mat &rgb,
                        const cv::Mat &depth)
    {
        cudaGetLastError();

        if (!intrinsics_set_) {
            return;
        }
        const auto frame_start = std::chrono::steady_clock::now();
        last_cpu_timings_ = GSSlamCPUTimings();

        const auto lock_wait_start = std::chrono::steady_clock::now();
        optimization_mutex_.lock();
        const auto lock_wait_end = std::chrono::steady_clock::now();
        last_cpu_timings_.lock_wait_ms = std::chrono::duration<double, std::milli>(lock_wait_end - lock_wait_start).count();
        const auto compute_start = std::chrono::steady_clock::now();
        Submap* submap = getCurrentSubmap();

        last_cpu_timings_.loop_extract_ms = loop_last_extract_ms_.load(std::memory_order_relaxed);
        last_cpu_timings_.loop_detect_ms = loop_last_detect_ms_.load(std::memory_order_relaxed);
        last_cpu_timings_.loop_verify_ms = loop_last_verify_ms_.load(std::memory_order_relaxed);
        last_cpu_timings_.loop_pgo_ms = loop_last_pgo_ms_.load(std::memory_order_relaxed);
        last_cpu_timings_.loop_total_ms = loop_last_total_ms_.load(std::memory_order_relaxed);
        last_cpu_timings_.loop_lock_hold_ms = loop_last_lock_hold_ms_.load(std::memory_order_relaxed);
        last_cpu_timings_.loop_edges = loop_last_edges_.load(std::memory_order_relaxed);

        bool new_keyframe_created = false;
        
        // ============================================================
        // PASO 1: Inicializar y copiar imágenes a GPU (con pirámides)
        // ============================================================
        initAndCopyImgs(rgb, depth);

        if (loop_diagnostics_mode_ && loop_diagnostics_executed_) {
            optimization_mutex_.unlock();
            return;
        }
        
        // ============================================================
        // PASO 3: Predicción IMU
        // ============================================================

        if (preint_->is_initialized) {
            // Bloqueamos los movimientos de submapas
            // para que la predicción IMU no se vea afectada por cambios de pose global
            std::lock_guard<std::mutex> submap_lock(submap_pose_mutex_);

            /*
            std::cout
                << "\n[TRACK BEGIN] P_cur=("
                << P_cur_[0] << ", "
                << P_cur_[1] << ", "
                << P_cur_[2] << ")"
                << std::endl;

            printPoseTrace(
                "getCameraPose(track begin)",
                getCameraPose());

            printPoseTrace(
                "current_pose_(track begin)",
                current_pose_);

            */
            // Usamos preintegracion IMU para predecir la pose actual a partir de la previa
            
            Eigen::Map<Eigen::Quaterniond> Q_cur_eig(P_cur_ + 3);
            Eigen::Map<Eigen::Vector3d> P_cur_eig(P_cur_);
            Eigen::Map<Eigen::Quaterniond> Q_prev_eig(P_prev_ + 3);
            Eigen::Map<Eigen::Vector3d> P_prev_eig(P_prev_);
            Eigen::Map<Eigen::VectorXd> VB_cur_eig(VB_cur_, 9);
            Eigen::Map<Eigen::VectorXd> VB_prev_eig(VB_prev_, 9);
            Eigen::Map<Eigen::Vector3d> V_prev_eig(VB_prev_);
            Eigen::Map<Eigen::Vector3d> V_cur_eig(VB_cur_);
            
            Eigen::Vector3d Pj, Vj;
            Eigen::Quaterniond Qj;

            const Eigen::Vector3d P_before_predict = P_cur_eig;
            const Eigen::Quaterniond Q_before_predict = Q_cur_eig.normalized();

            options_.max_num_iterations = pose_iterations_;

            /*
            std::cout
                << "[BEFORE PREDICT] P_prev=("
                << P_prev_[0] << ", "
                << P_prev_[1] << ", "
                << P_prev_[2] << ")"
                << std::endl;

            std::cout
                << "[BEFORE PREDICT] P_cur=("
                << P_cur_[0] << ", "
                << P_cur_[1] << ", "
                << P_cur_[2] << ")"
                << std::endl;

                std::cout
    << "[PREDICT INPUT] V_prev=("
    << V_prev_eig.x() << ", "
    << V_prev_eig.y() << ", "
    << V_prev_eig.z() << ")"
    << std::endl;

std::cout
    << "[PREDICT INPUT] sum_dt="
    << preint_->sum_dt
    << std::endl;
            */        

            // Predicción IMU
            const auto t_predict_start = std::chrono::steady_clock::now();
            preint_->predict(P_prev_eig, Q_prev_eig, V_prev_eig, Pj, Qj, Vj);
            const auto t_predict_end = std::chrono::steady_clock::now();
            last_cpu_timings_.imu_predict_ms = std::chrono::duration<double, std::milli>(t_predict_end - t_predict_start).count();
            //stage_predict_ms = std::chrono::duration<double, std::milli>(
            //    t_predict_end - t_predict_start).count();
            
            // Actualizar estado predicho
            P_cur_eig = Pj;
            Q_cur_eig = Qj;
            V_cur_eig = Vj;

            /*

            std::cout
                << "[AFTER PREDICT] P_cur=("
                << P_cur_[0] << ", "
                << P_cur_[1] << ", "
                << P_cur_[2] << ")"
                << std::endl;

            printPoseTrace(
                "getCameraPose(after predict)",
                getCameraPose());

                std::cout
    << "[PREDICT OUTPUT] Pj=("
    << Pj.x() << ", "
    << Pj.y() << ", "
    << Pj.z() << ")"
    << std::endl;

std::cout
    << "[PREDICT OUTPUT] Vj=("
    << Vj.x() << ", "
    << Vj.y() << ", "
    << Vj.z() << ")"
    << std::endl;

            */

                    // ============================================================
        // PASO 2: Primera imagen: Inicialización
        // ============================================================
        if (first_image_) {
            if (!has_initial_pose_) {
                optimization_mutex_.unlock();
                return;
            }
            if (loop_diagnostics_mode_ && !loop_diagnostics_executed_) {
                runLoopDiagnosticsSmokeTest(rgb, depth);
                return;
            }

            if (!submap) {
                createNewSubmap();
                submap = getCurrentSubmap();
            }



            static bool first_image_dumped = false;
            if (!first_image_dumped && !rgb.empty())
            {
                // Guardamos imagen rgb
                const std::string rgb_dump_path = "/tmp/f_vigs_slam_first_compute_rgb.png";
                if (cv::imwrite(rgb_dump_path, rgb))
                {
                    std::cerr << "[GSSlam] [DEBUG first_image] saved input RGB to "
                              << rgb_dump_path << " size=" << rgb.cols << "x" << rgb.rows
                              << " type=" << rgb.type() << std::endl;
                }
                else
                {
                    std::cerr << "[GSSlam] [WARN first_image] failed to save input RGB dump"
                              << std::endl;
                }
                // Guardamos imagen depth
                const std::string depth_dump_path = "/tmp/f_vigs_slam_first_compute_depth.png";
                if (cv::imwrite(depth_dump_path, depth))
                {
                    std::cerr << "[GSSlam] [DEBUG first_image] saved input depth to "
                              << depth_dump_path << " size=" << depth.cols << "x" << depth.rows
                              << " type=" << depth.type() << std::endl;
                }
                else
                {
                    std::cerr << "[GSSlam] [WARN first_image] failed to save input depth dump"
                              << std::endl;
                }

                first_image_dumped = true;
            }

            // Inicializar pose IMU
            P_cur_[0] = initial_pose_imu_.position.x;
            P_cur_[1] = initial_pose_imu_.position.y;
            P_cur_[2] = initial_pose_imu_.position.z;
            P_cur_[3] = initial_pose_imu_.orientation.x;  // qx
            P_cur_[4] = initial_pose_imu_.orientation.y;  // qy
            P_cur_[5] = initial_pose_imu_.orientation.z;  // qz
            P_cur_[6] = initial_pose_imu_.orientation.w;  // qw

            // Copiar a P_prev
            for (int i = 0; i < 7; ++i) {
                P_prev_[i] = P_cur_[i];
            }

            // Inicializar pose de cámara: almacenar como local al submapa actual
            if (submap) {
                current_pose_ = composePoses(invertPose(submap->getGlobalPose()), initial_pose_cam_);
            } else {
                current_pose_ = initial_pose_cam_;
            }

            // Generar gaussianas desde RGB-D
            if (submap) {
                {
                    const Pose cam_global = composePoses(submap->getGlobalPose(), current_pose_);
                    initializeGaussiansFromRgbd(submap, cam_global);
                }
                // Forzar el primer keyframe del primer submapa a identidad para que el
                // origen del submapa coincida con su primer frame.
                current_pose_ = Pose::Identity();
                submap->first_frame_pose_local = current_pose_;
                addKeyframe(submap);
                new_keyframe_created = true;
            }

            if (!optimize_thread_.joinable()) {
                stop_optimization_.store(false);
                optimize_thread_ = std::thread(&GSSlam::optimizationLoop, this);
            }

            nb_images_processed_++;
            first_image_ = false;
            optimization_mutex_.unlock();

            const auto compute_end = std::chrono::steady_clock::now();
            const auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(compute_end - compute_start).count();
            if (new_keyframe_created)
            {
                const int submap_id = submap ? static_cast<int>(submap->submap_id) : -1;
                //std::cout << "[GSSlam] New keyframe in submap #" << submap_id
                //          << " (local_keyframe_idx=" << current_keyframe_idx_ << ")" << std::endl;
                //std::cout << "[GSSlam] Time to process the keyframe: " << elapsed_ms << " milliseconds" << std::endl;
                //std::cout << "[GSSlam] nbGaussians : " << global_gaussians_count_ << std::endl;
            }

            return;
        }
            
            // ============================================================
            // PASO 5: Optimización de pose con Ceres Solver
            // ============================================================
            // Combina factores visual, IMU y prior (si existe)
            // Tracking: solo niveles coarse (omite nivel 0)
            const auto t_optimize_track_start = std::chrono::steady_clock::now();
            const Eigen::Vector3d P_before_opt_track = P_cur_eig;
            const Eigen::Quaterniond Q_before_opt_track = Q_cur_eig.normalized();
            optimizeWithCeres(1, pose_iterations_);
            const auto t_optimize_track_end = std::chrono::steady_clock::now();
            last_cpu_timings_.pose_track_ms = std::chrono::duration<double, std::milli>(t_optimize_track_end - t_optimize_track_start).count();
            
            // ============================================================
            // PASO 6: Actualizar pose de cámara desde IMU
            // ============================================================
            updateCameraPoseFromImu();

            // Aplicar actualización global antes de usar la pose para
            // transiciones de submapa, keyframes y publicación de odometría.
            /*
            if (debug_update_global_poses_ && (nb_images_processed_ > 0) &&
                (nb_images_processed_ % debug_update_period_frames_ == 0)) {
                debugLiftSecondSubmapAndUpdate(debug_update_dz_m_);
            }
            */

            // ============================================================
            // PASO 6.1: Transición de submapa por threshold (dist/rot)
            // ============================================================
            // Transition check uses the global camera pose.
            // Avoid triggering a transition in the same compute() call that
            // processed the very first image (prevents immediate split at startup).
            if (submap && nb_images_processed_ > 0 && checkSubmapTransition(getCameraPose())) {
                // Print
                std::cout << "Transitioning to new submap (current submap id: " << submap->submap_id << ")" << std::endl;
                createNewSubmap();
                submap = getCurrentSubmap();
                if (submap) {
                    // current_pose_ already local to current submap
                    submap->first_frame_pose_local = current_pose_;
                    // Actualmente inicia las gaussianas en cada submapa sin aprovechar
                    // las anteriores. Pass global camera pose to initializer.
                    {
                        Pose cam_global = composePoses(submap->getGlobalPose(), current_pose_);
                        initializeGaussiansFromRgbd(submap, cam_global);
                    }
                    addKeyframe(submap);
                    new_keyframe_created = true;
                }
            }
            
            // ============================================================
            // PASO 7: Remover outliers
            // ============================================================
            const Eigen::Vector3d P_before_outliers = P_cur_eig;
            const Eigen::Quaterniond Q_before_outliers = Q_cur_eig.normalized();
            const auto t_remove_outliers_start = std::chrono::steady_clock::now();
            removeOutliers();
            const auto t_remove_outliers_end = std::chrono::steady_clock::now();
            last_cpu_timings_.outlier_ms = std::chrono::duration<double, std::milli>(t_remove_outliers_end - t_remove_outliers_start).count();

            // ============================================================
            // PASO 8: Verificar covisibilidad para nuevo keyframe
            // ============================================================
            float covis_ratio = computeCovisibilityRatio();
            
            if (covis_ratio < covisibility_threshold_) {
                //const auto t_keyframe_start = std::chrono::steady_clock::now();
                const Eigen::Vector3d P_before_keyframe = P_cur_eig;
                const Eigen::Quaterniond Q_before_keyframe = Q_cur_eig.normalized();
                // Nuevo keyframe necesario - optimización refinada
                //std::cout << "Nuevo keyframe (covisibilidad = " << covis_ratio << ")" << std::endl;
                //keyframe_added = true;
                
                // Optimización completa con Ceres para keyframe (incluye nivel 0)
                const auto t_kf_refine_start = std::chrono::steady_clock::now();
                optimizeWithCeres(0, pose_iterations_);
                const auto t_kf_refine_end = std::chrono::steady_clock::now();
                last_cpu_timings_.pose_refine_ms = std::chrono::duration<double, std::milli>(t_kf_refine_end - t_kf_refine_start).count();
                //stage_keyframe_refine_ms = std::chrono::duration<double, std::milli>(
                //    t_kf_refine_end - t_kf_refine_start).count();
                
                // Actualizar pose de cámara nuevamente
                updateCameraPoseFromImu();
                
                // Gestión de mapa
                const auto t_map_ops_start = std::chrono::steady_clock::now();
                if (submap) {
                    prune(submap);
                    addKeyframe(submap);
                    if (current_keyframe_idx_ >= 0 &&
                        static_cast<size_t>(current_keyframe_idx_) < submap->keyframes.size()) {
                        densify(submap, submap->keyframes[static_cast<size_t>(current_keyframe_idx_)]);
                    }
                }
                const auto t_map_ops_end = std::chrono::steady_clock::now();
                last_cpu_timings_.map_ops_ms += std::chrono::duration<double, std::milli>(t_map_ops_end - t_map_ops_start).count();
                new_keyframe_created = true;
            }
            
            // ============================================================
            // PASO 9: Marginalización y actualización de estado
            // ============================================================
            const Eigen::Vector3d P_before_marg = P_cur_eig;
            const Eigen::Quaterniond Q_before_marg = Q_cur_eig.normalized();
            const auto t_marg_start = std::chrono::steady_clock::now();
            marginalization_info_.preMarginalize();
            marginalization_info_.marginalize();
            const auto t_marg_end = std::chrono::steady_clock::now();
            last_cpu_timings_.marginalization_ms = std::chrono::duration<double, std::milli>(t_marg_end - t_marg_start).count();

            // Actualizar estado previo
            Q_prev_eig = Q_cur_eig;
            P_prev_eig = P_cur_eig;
            VB_prev_eig = VB_cur_eig;
            
            // Reinicializar preintegración IMU para próximo frame
            preint_->init(last_imu_.Acc, last_imu_.Gyro,
                         VB_cur_eig.segment(3, 3), VB_cur_eig.segment(6, 3),
                         last_imu_.acc_n, last_imu_.gyr_n, 
                         last_imu_.acc_w, last_imu_.gyr_w);

        
        } // endif (imu_initialized_)
        
        nb_images_processed_++;
        if (new_keyframe_created)
        {
            const auto compute_end = std::chrono::steady_clock::now();
            const auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(compute_end - compute_start).count();
            const int submap_id = submap ? static_cast<int>(submap->submap_id) : -1;
            std::cout << "[GSSlam] New keyframe in submap #" << submap_id
                      << " (local_keyframe_idx=" << current_keyframe_idx_ << ")" << std::endl;
            std::cout << "[GSSlam] Time to process the keyframe: " << elapsed_ms << " milliseconds" << std::endl;
            std::cout << "[GSSlam] nbGaussians : " << global_gaussians_count_ << std::endl;
        }

        const auto compute_end = std::chrono::steady_clock::now();
        const auto frame_end = std::chrono::steady_clock::now();
        last_cpu_timings_.gaussian_count = static_cast<int>(submap ? submap->getGaussiansCount() : 0);
        last_cpu_timings_.total_frame_ms = std::chrono::duration<double, std::milli>(frame_end - frame_start).count();
        last_gpu_timings_.total_frame_ms = std::chrono::duration<double, std::milli>(frame_end - frame_start).count();

            constexpr size_t kMetricsBufferSize = 30;

            auto averageBuffer = [](const std::deque<double> &buffer) -> double {
                double sum = 0.0;
                size_t count = 0;
                for (double value : buffer) {
                    if (std::isfinite(value)) {
                        sum += value;
                        ++count;
                    }
                }
                return count > 0 ? (sum / static_cast<double>(count)) : std::numeric_limits<double>::quiet_NaN();
            };

            auto pushBufferedValue = [kMetricsBufferSize](std::deque<double> &buffer, double value) {
                buffer.push_back(value);
                if (buffer.size() > kMetricsBufferSize) {
                    buffer.pop_front();
                }
            };

            pushBufferedValue(track_time_buffer_, last_cpu_timings_.pose_track_ms);
            pushBufferedValue(map_time_buffer_, last_cpu_timings_.map_ops_ms);

            const double track_ms_snapshot = last_cpu_timings_.pose_track_ms;
            const double map_ms_snapshot = last_cpu_timings_.map_ops_ms;
            const bool psnr_ready = submap && intrinsics_set_ && !pyr_color_.empty() && !pyr_color_[0].empty();
            const Pose camera_pose_snapshot = getCameraPose();
            const IntrinsicParameters intrinsics_snapshot = intrinsics_;
            const int psnr_width = psnr_ready ? pyr_color_[0].cols : 0;
            const int psnr_height = psnr_ready ? pyr_color_[0].rows : 0;
            cv::Mat original_bgr_snapshot;

            if (psnr_ready)
            {
                cv::Mat original_gpu_mat;
                pyr_color_[0].download(original_gpu_mat);

                if (original_gpu_mat.channels() == 4)
                {
                    cv::cvtColor(original_gpu_mat, original_bgr_snapshot, cv::COLOR_BGRA2BGR);
                }
                else if (original_gpu_mat.channels() == 3)
                {
                    original_bgr_snapshot = original_gpu_mat.clone();
                }
            }

        optimization_mutex_.unlock();

            last_psnr_db_ = std::numeric_limits<double>::quiet_NaN();
            if (psnr_ready)
            {
                try
                {
                    cv::cuda::GpuMat rendered_rgb_gpu, rendered_depth_gpu;
                    const bool render_success = renderView(
                        submap,
                        camera_pose_snapshot,
                        intrinsics_snapshot,
                        psnr_width,
                        psnr_height,
                        rendered_rgb_gpu,
                        rendered_depth_gpu);

                    if (render_success && !rendered_rgb_gpu.empty())
                    {
                        cv::Mat rendered_rgb_host;
                        rendered_rgb_gpu.download(rendered_rgb_host);

                        if (!original_bgr_snapshot.empty() && !rendered_rgb_host.empty())
                        {
                            last_psnr_db_ = computeImagePsnrDb(original_bgr_snapshot, rendered_rgb_host);
                            if (std::isfinite(last_psnr_db_))
                            {
                                pushBufferedValue(psnr_buffer_, last_psnr_db_);
                            }
                        }
                    }
                }
                catch (const std::exception &e)
                {
                    std::cerr << "[GSSlam::compute] Exception during PSNR computation: "
                              << e.what() << std::endl;
                }
            }

            const double avg_track = averageBuffer(track_time_buffer_);
            const double avg_map = averageBuffer(map_time_buffer_);
            const double avg_psnr = averageBuffer(psnr_buffer_);

            std::ostringstream metrics_oss;
            metrics_oss << std::fixed << std::setprecision(2);
            if (frame_count_ % metrics_print_interval_frames_ == 0)
            {
                if (std::isfinite(last_psnr_db_))
                {
                    metrics_oss << "[GSSlam] █ Frame " << frame_count_
                                << " │ Track/Img: " << track_ms_snapshot << " ms (avg: "
                                << (std::isfinite(avg_track) ? avg_track : track_ms_snapshot)
                                << ") │ Map/Img: " << map_ms_snapshot << " ms (avg: "
                                << (std::isfinite(avg_map) ? avg_map : map_ms_snapshot)
                                << ") │ PSNR: " << last_psnr_db_ << " dB (avg: "
                                << (std::isfinite(avg_psnr) ? avg_psnr : last_psnr_db_) << ")";
                }
                else
                {
                    metrics_oss << "[GSSlam] █ Frame " << frame_count_
                                << " │ Track/Img: " << track_ms_snapshot << " ms (avg: "
                                << (std::isfinite(avg_track) ? avg_track : track_ms_snapshot)
                                << ") │ Map/Img: " << map_ms_snapshot << " ms (avg: "
                                << (std::isfinite(avg_map) ? avg_map : map_ms_snapshot)
                                << ") │ PSNR: n/a";
                }

                std::cout << metrics_oss.str() << std::endl;
            }

            ++frame_count_;

        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }

    void GSSlam::optimizationLoop()
    {
        while (!stop_optimization_.load())
        {
            optimization_mutex_.lock();

            Submap* opt_submap = getCurrentSubmap();
            if (!opt_submap)
            {
                optimization_mutex_.unlock();
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
                continue;
            }

            auto &opt_keyframes = opt_submap->keyframes;

            int k = 0;
            const int target_iterations = gaussian_iterations_;
            const size_t total_keyframes = opt_keyframes.size();

            if (total_keyframes == 0)
            {
                optimization_mutex_.unlock();
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
                continue;
            }

            while (k < target_iterations)
            {
                std::vector<int> selected;
                selected.reserve(4);

                int last_idx = current_keyframe_idx_;

                if (last_idx < 0 || static_cast<size_t>(last_idx) >= total_keyframes)
                {
                    last_idx = static_cast<int>(total_keyframes - 1);
                }

                // 1) Últimos dos keyframes
                add_unique(selected, last_idx);
                if (last_idx > 0)
                {
                    add_unique(selected, last_idx - 1);
                }

                // 2) Sampling
                if (selected.size() < 4 && total_keyframes > selected.size())
                {
                    std::vector<int> sampled;
                    try
                    {
                        sampled = keyframe_selector_.sample(
                            2,
                            static_cast<int>(total_keyframes),
                            keyframe_selection_config_);
                    }
                    catch (const std::invalid_argument &)
                    {
                        sampled = keyframe_selector_.sample(
                            2,
                            static_cast<int>(total_keyframes));
                    }

                    for (int idx : sampled)
                    {
                        add_unique(selected, idx);
                    }
                }

                // 3) Optimización
                int optimized = 0;

                for (int idx : selected)
                {
                    if (idx >= 0 && static_cast<size_t>(idx) < total_keyframes)
                    {
                        //std::cerr << "[GSSlam][OptimizationLoop] Optimizing keyframe idx=" << idx << std::endl;
                        optimizeGaussiansKeyframe(opt_submap, opt_keyframes[static_cast<size_t>(idx)]);
                        optimized++;
                    }
                }

                // 4) Avance por trabajo real
                k += optimized;

                // 5) Densification / pruning
                iterationsSinceDensification_ += optimized;

                if (iterationsSinceDensification_ > 400)
                {
                    Submap* submap = getCurrentSubmap();
                    if (submap) {
                        prune(submap);
                    }
                    iterationsSinceDensification_ = 0;
                }
            }

            optimization_mutex_.unlock();

            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
    }


    void GSSlam::optimizeGaussiansKeyframe(Submap* submap, const KeyframeData &keyframe)
    {
        cudaEventRecord(cuda_evt_keyframe_opt_start_);

        // ============================================================
        // 1. Early exit
        // ============================================================
        if (!submap || submap->getGaussiansCount() == 0 || keyframe.color_img.empty() || keyframe.depth_img.empty()) {
            cudaEventRecord(cuda_evt_keyframe_opt_end_);
            last_gpu_timings_.keyword_optimization_ms = getCudaEventElapsedMs(cuda_evt_keyframe_opt_start_, cuda_evt_keyframe_opt_end_);
            return;
        }

        // ============================================================
        // 23. Inicialización de Adam
        // ============================================================
        if (submap->adam_states.empty()) {
            submap->adam_states.resize(submap->max_gaussians);
            cudaMemset(thrust::raw_pointer_cast(submap->adam_states.data()),
                0,
                submap->max_gaussians * sizeof(AdamStateGaussian3D));
        }

        // ============================================================
        // 3. Parámetros de optimización
        // ============================================================
        const float lambda_iso = 0.01f;

        // ============================================================
        // 4. Optimizacion
        // ============================================================
        {
            const int width  = keyframe.getWidth();
            const int height = keyframe.getHeight();

            if (width <= 0 || height <= 0) return;

            // ========================================================
            // 5.1 Proyección de gaussianas (raster setup)
            // ========================================================
            prepareRasterization(submap, keyframe.getRelativePose(), keyframe.getIntrinsics(), width, height);

            if (last_nb_instances_ == 0) return;

            // ========================================================
            // 5.2 Construcción de buckets por tile
            // ========================================================
            int num_tiles_total = num_tiles_.x * num_tiles_.y;
            if (num_tiles_total <= 0) return;

            opt_bucket_offsets_.resize(num_tiles_total);

            perTileBucketCount<<<(num_tiles_total + 255) / 256, 256>>>(
                thrust::raw_pointer_cast(opt_bucket_offsets_.data()),
                thrust::raw_pointer_cast(tile_ranges_.data()),
                num_tiles_total);
            
            CUDA_CHECK_KERNEL("perTileBucketCount");

            thrust::inclusive_scan(
                opt_bucket_offsets_.begin(),
                opt_bucket_offsets_.end(),
                opt_bucket_offsets_.begin());

            uint32_t num_buckets = opt_bucket_offsets_.back();
            if (num_buckets == 0) return;

            // ========================================================
            // 5.3 Resize buffers de optimización
            // ========================================================
            const uint32_t num_pixels = width * height;
            const uint32_t block_size = tile_size_.x * tile_size_.y;

            opt_bucket_to_tile_.resize(num_buckets);
            opt_sampled_T_.resize(num_buckets * block_size);
            opt_sampled_ar_.resize(num_buckets * block_size);

            opt_final_T_.resize(num_pixels);
            opt_n_contrib_.resize(num_pixels);
            opt_max_contrib_.resize(num_tiles_total);

            opt_output_color_.resize(num_pixels);
            opt_output_depth_.resize(num_pixels);

            opt_color_error_.resize(num_pixels);
            opt_depth_error_.resize(num_pixels);

                const uint32_t n_gauss = submap->getGaussiansCount();
                opt_delta_gaussians_2d_.resize(n_gauss);
                opt_delta_gaussians_3d_.resize(n_gauss);

            // ========================================================
            // 5.4 Reset de gradientes 2D
            // ========================================================
            cudaMemset(thrust::raw_pointer_cast(opt_delta_gaussians_2d_.data()),
                    0,
                    n_gauss * sizeof(DeltaGaussian2D));

            // ========================================================
            // 5.5 Preparación de observaciones
            // ========================================================

            cudaTextureObject_t observed_rgb = keyframe.getColorTex().getTextureObject();
            cudaTextureObject_t observed_depth = keyframe.getDepthTex().getTextureObject();
            cudaTextureObject_t observed_normal = keyframe.getNormalTex().getTextureObject();
            (void)observed_normal;

            const float3 random_bg = bg_color_; // Por ahora no lo uso random

            // ========================================================
            // 5.6 Forward pass (render + error)
            // ========================================================
            optimizeGaussiansForwardPass<<<dim3(num_tiles_.x, num_tiles_.y),
                                        dim3(tile_size_.x, tile_size_.y)>>>(
                thrust::raw_pointer_cast(tile_ranges_.data()),
                thrust::raw_pointer_cast(gaussian_indices_.data()),
                thrust::raw_pointer_cast(positions_2d_.data()),
                thrust::raw_pointer_cast(inv_covariances_2d_.data()),
                thrust::raw_pointer_cast(p_hats_.data()),
                thrust::raw_pointer_cast(submap->gaussians.colors.data()),
                thrust::raw_pointer_cast(submap->gaussians.opacities.data()),

                thrust::raw_pointer_cast(opt_bucket_offsets_.data()),
                thrust::raw_pointer_cast(opt_bucket_to_tile_.data()),

                thrust::raw_pointer_cast(opt_sampled_T_.data()),
                thrust::raw_pointer_cast(opt_sampled_ar_.data()),

                thrust::raw_pointer_cast(opt_final_T_.data()),
                thrust::raw_pointer_cast(opt_n_contrib_.data()),
                thrust::raw_pointer_cast(opt_max_contrib_.data()),

                thrust::raw_pointer_cast(opt_output_color_.data()),
                thrust::raw_pointer_cast(opt_output_depth_.data()),

                thrust::raw_pointer_cast(opt_color_error_.data()),
                thrust::raw_pointer_cast(opt_depth_error_.data()),

                observed_rgb,
                observed_depth,
                random_bg,
                num_tiles_,
                width,
                height);

            CUDA_CHECK_KERNEL("optimizeGaussiansKeyframe::optimizeGaussiansForwardPass");

            // ========================================================
            // 5.7 Backward parcial por gaussianas (2D)
            // ========================================================
            optimizeGaussiansPerGaussianPass<<<num_buckets, 32>>>(
                thrust::raw_pointer_cast(tile_ranges_.data()),
                thrust::raw_pointer_cast(gaussian_indices_.data()),
                thrust::raw_pointer_cast(positions_2d_.data()),
                thrust::raw_pointer_cast(inv_covariances_2d_.data()),
                thrust::raw_pointer_cast(p_hats_.data()),
                thrust::raw_pointer_cast(submap->gaussians.colors.data()),
                thrust::raw_pointer_cast(submap->gaussians.opacities.data()),

                thrust::raw_pointer_cast(opt_bucket_offsets_.data()),
                thrust::raw_pointer_cast(opt_bucket_to_tile_.data()),

                thrust::raw_pointer_cast(opt_sampled_T_.data()),
                thrust::raw_pointer_cast(opt_sampled_ar_.data()),

                thrust::raw_pointer_cast(opt_n_contrib_.data()),
                thrust::raw_pointer_cast(opt_max_contrib_.data()),

                thrust::raw_pointer_cast(opt_output_color_.data()),
                thrust::raw_pointer_cast(opt_output_depth_.data()),

                thrust::raw_pointer_cast(opt_color_error_.data()),
                thrust::raw_pointer_cast(opt_depth_error_.data()),

                thrust::raw_pointer_cast(opt_delta_gaussians_2d_.data()),

                gaussian_w_depth_,
                gaussian_w_dist_,
                num_tiles_,
                width,
                height,
                static_cast<int>(num_buckets));

            CUDA_CHECK_KERNEL("optimizeGaussiansKeyframe::optimizeGaussiansPerGaussianPass");

            // ========================================================
            // 5.8 Lift 2D → 3D (gradientes en parámetros reales)
            // ========================================================
            dim3 block(256);
            dim3 grid((n_gauss + block.x - 1) / block.x);

            computeDeltaGaussians3D_kernel<<<grid, block>>>(
                thrust::raw_pointer_cast(opt_delta_gaussians_3d_.data()),

                thrust::raw_pointer_cast(submap->gaussians.positions.data()),
                thrust::raw_pointer_cast(submap->gaussians.scales.data()),
                thrust::raw_pointer_cast(submap->gaussians.orientations.data()),
                thrust::raw_pointer_cast(submap->gaussians.colors.data()),
                thrust::raw_pointer_cast(submap->gaussians.opacities.data()),

                thrust::raw_pointer_cast(opt_delta_gaussians_2d_.data()),

                keyframe.getRelativePose(),
                keyframe.getIntrinsics(),
                lambda_iso,
                n_gauss);
            
            CUDA_CHECK_KERNEL("optimizeGaussiansKeyframe::computeDeltaGaussians3D");

            // ========================================================
            // 5.9 Update con Adam
            // ========================================================
            const auto t_adam_start = std::chrono::steady_clock::now();
            updateGaussiansParametersAdam_kernel<<<grid, block>>>(
                thrust::raw_pointer_cast(submap->gaussians.positions.data()),
                thrust::raw_pointer_cast(submap->gaussians.scales.data()),
                thrust::raw_pointer_cast(submap->gaussians.orientations.data()),
                thrust::raw_pointer_cast(submap->gaussians.colors.data()),
                thrust::raw_pointer_cast(submap->gaussians.opacities.data()),
                thrust::raw_pointer_cast(submap->adam_states.data()),
                thrust::raw_pointer_cast(opt_delta_gaussians_3d_.data()),
                adam_eta_,
                adam_beta1_,
                adam_beta2_,
                adam_eps_,
                n_gauss);
            CUDA_CHECK_KERNEL("optimizeGaussiansKeyframe::updateGaussiansParametersAdam");
            cudaDeviceSynchronize();
            const auto t_adam_end = std::chrono::steady_clock::now();
            last_cpu_timings_.map_ops_ms += std::chrono::duration<double, std::milli>(t_adam_end - t_adam_start).count();

            opt_iteration_++;
        }
    
        cudaEventRecord(cuda_evt_keyframe_opt_end_);
        last_gpu_timings_.keyword_optimization_ms = getCudaEventElapsedMs(cuda_evt_keyframe_opt_start_, cuda_evt_keyframe_opt_end_);
    }
    
    void GSSlam::addKeyframe(Submap* submap)
    {
        if (!submap || pyr_color_.empty() || pyr_depth_.empty() || pyr_normals_.empty())
            return;

        if (pyr_color_[0].empty() || pyr_depth_[0].empty() || pyr_normals_[0].empty())
            return;

        const uint32_t id = static_cast<uint32_t>(submap->keyframes.size());
        const double ts = static_cast<double>(nb_images_processed_);

        // ============================================================
        // Calcular pose relativa del keyframe al submapa
        // ============================================================
        // El keyframe se almacena en coordenadas locales del submapa
        // T_keyframe_relative = T_submap_inv * T_camera_global
        
        Pose T_submap_global = submap->getGlobalPose();
        // current_pose_ is stored local to the current submap, so it's already the relative pose
        Pose T_keyframe_relative = current_pose_;

        KeyframeData kf(
            pyr_color_[0],      // CV_8UC4
            pyr_depth_[0],      // CV_32FC1
            pyr_normals_[0],    // CV_32FC4
            T_keyframe_relative,  // Calcular pose relativa del keyframe al submapa
            intrinsics_,
            id,
            ts
        );

        const bool netvlad_enabled = static_cast<bool>(netvlad_);
        // std::cout << "[GSSlam::addKeyframe] NetVLAD extractor "
        //          << (netvlad_enabled ? "active" : "inactive")
        //          << " submap=" << submap->submap_id
        //          << " kf=" << id << std::endl;

        if (netvlad_enabled) {
            cv::Mat color_cpu;
            if (!kf.color_img.empty())
            {
                kf.color_img.download(color_cpu);
                descriptor_extraction_queue_.enqueue(DescriptorExtractionTask(
                    static_cast<int>(submap->submap_id),
                    static_cast<int>(id),
                    color_cpu));

                //std::cout << "[GSSlam::addKeyframe] NetVLAD enabled; extraction enqueued"
                //          << " submap=" << submap->submap_id << " kf=" << id << std::endl;
            }
        }

        if (!submap->keyframes.empty())
        {
            const Pose &prev_pose = submap->keyframes.back().getRelativePose();
            const Pose cur_local_pose = T_keyframe_relative;
            const float dx = cur_local_pose.position.x - prev_pose.position.x;
            const float dy = cur_local_pose.position.y - prev_pose.position.y;
            const float dz = cur_local_pose.position.z - prev_pose.position.z;
            const float distance_m = std::sqrt(dx * dx + dy * dy + dz * dz);

            Eigen::Quaternionf q_prev(prev_pose.orientation.w,
                                      prev_pose.orientation.x,
                                      prev_pose.orientation.y,
                                      prev_pose.orientation.z);
            Eigen::Quaternionf q_cur(cur_local_pose.orientation.w,
                                     cur_local_pose.orientation.x,
                                     cur_local_pose.orientation.y,
                                     cur_local_pose.orientation.z);
            q_prev.normalize();
            q_cur.normalize();
            const float cos_half_angle = std::clamp(std::abs(q_prev.dot(q_cur)), 0.0f, 1.0f);
            constexpr float kRadToDeg = 57.2957795131f;
            const float rotation_deg = 2.0f * std::acos(cos_half_angle) * kRadToDeg;

            submap->accumulated_translation_uncertainty_m += 0.02f + 0.05f * distance_m;
            submap->accumulated_rotation_uncertainty_deg += 0.2f + 0.02f * rotation_deg;
        }

        if (preint_ && preint_->is_initialized && preint_->covariance.rows() >= 9)
        {
            const double cov_trace =
                preint_->covariance.block<3, 3>(0, 0).trace() +
                preint_->covariance.block<3, 3>(6, 6).trace();
            if (std::isfinite(cov_trace))
            {
                const float imu_unc = static_cast<float>(std::max(0.0, cov_trace));
                submap->accumulated_translation_uncertainty_m += 0.001f * imu_unc;
            }
        }

        kf.updateGlobalPose(submap->getGlobalPose());
        submap->keyframes.push_back(std::move(kf));
        submap->keyframe_gaussian_counts.push_back(submap->getGaussiansCount());
        keyframe_gaussian_counts_.push_back(submap ? submap->getGaussiansCount() : 0);
        current_keyframe_idx_ = static_cast<int>(submap->keyframes.size()) - 1;

        const KeyframeData &saved_kf = submap->keyframes.back();
        
        // ============================================================
        // Descriptor extraction will happen asynchronously in background thread
        // No need to append to GPU database here - it will be done by 
        // extraction completes inside the loop closure worker
        // ============================================================
        

        if (saved_kf.hasDescriptor() && submap->keyframes.size() > 1)
        {
            float new_min = submap->min_descriptor_similarity;
            const auto &d_new = saved_kf.getDescriptor();
            for (size_t i = 0; i + 1 < submap->keyframes.size(); ++i)
            {
                const auto &d_old = submap->keyframes[i].getDescriptor();
                if (d_old.empty())
                {
                    continue;
                }
                new_min = std::min(new_min, cosineSimilarityHost(d_new, d_old));
            }
            submap->min_descriptor_similarity = new_min;
            submap->has_descriptor_similarity_stats = true;
        }
        
        /*
        std::cout << "[GSSlam::addKeyframe] submap=" << submap->submap_id
                  << " kf=" << saved_kf.keyframe_id
                  << " descriptor_dim=" << saved_kf.getDescriptor().size()
                  << " min_self_sim=" << submap->min_descriptor_similarity
                  << " unc_t=" << submap->accumulated_translation_uncertainty_m
                  << " unc_r=" << submap->accumulated_rotation_uncertainty_deg
                  << std::endl;
        */
        startLoopThreadsIfNeeded();
    }

    void GSSlam::startLoopThreadsIfNeeded()
    {
        // Descriptor extraction is now handled inside loopDetectionAndClosureThread.
        // Keep the dedicated thread disabled.
        // Descriptor extraction is folded into loopDetectionAndClosureThread.
        // Use combined worker for detection+verification+PGO instead of separate threads
        // Original individual loop threads are intentionally not started here.
        // if (!loop_detection_thread_.joinable()) {
        //     stop_loop_detection_.store(false, std::memory_order_release);
        //     loop_detection_thread_ = std::thread(&GSSlam::loopDetectionThread, this);
        // }
        // if (!loop_verification_thread_.joinable()) {
        //     stop_loop_verification_.store(false, std::memory_order_release);
        //     loop_verification_thread_ = std::thread(&GSSlam::loopVerificationThread, this);
        // }
        // if (!pgo_thread_.joinable()) {
        //     stop_pgo_.store(false, std::memory_order_release);
        //     pgo_thread_ = std::thread(&GSSlam::poseGraphOptimizationThread, this);
        // }
        if (!loop_detection_and_closure_thread_.joinable()) {
            stop_loop_detection_.store(false, std::memory_order_release);
            loop_detection_and_closure_thread_ = std::thread(&GSSlam::loopDetectionAndClosureThread, this);
        }
    }
    
    void GSSlam::densify(Submap* submap, const KeyframeData &keyframe)
    {
        cudaEventRecord(cuda_evt_densify_start_);
        if (!submap) {
            return;
        }
        const int before_densify = static_cast<int>(submap->getGaussiansCount());
        if (keyframe.color_img.empty() || keyframe.depth_img.empty()) {
            return;
        }
        //std::cout << "[GSSlam] Densification start..." << std::endl;

        const int width  = keyframe.getWidth();
        const int height = keyframe.getHeight();
        const IntrinsicParameters keyframe_intrinsics = keyframe.getIntrinsics();

        // ============================================================
        // 1. Preparar rasterización
        // ============================================================
        prepareRasterization(
            submap,
            keyframe.getRelativePose(),
            keyframe_intrinsics,
            width,
            height);

        // ============================================================
        // 2. Density mask
        // ============================================================
        // M_i: mascara de regiones faltantes por insuficiencia de gaussianas.
        density_mask_.create(height, width, CV_32FC1);

        computeDensityMask_kernel<<<
            dim3(num_tiles_.x, num_tiles_.y),
            dim3(tile_size_.x, tile_size_.y)>>>(

            density_mask_.ptr<float>(),

            thrust::raw_pointer_cast(tile_ranges_.data()),
            thrust::raw_pointer_cast(gaussian_indices_.data()),
            thrust::raw_pointer_cast(positions_2d_.data()),
            thrust::raw_pointer_cast(inv_covariances_2d_.data()),
            thrust::raw_pointer_cast(p_hats_.data()),
            thrust::raw_pointer_cast(submap->gaussians.opacities.data()),

            // PASAMOS PROFUNDIDAD COMO TEXTURA
            keyframe.getDepthTex().getTextureObject(),

            num_tiles_,
            width,
            height,
            density_mask_.step / sizeof(float),
            submap->getGaussiansCount(),
            last_nb_instances_);

        CUDA_CHECK_KERNEL("densify::computeDensityMask");

        // ============================================================
        // 3. Inicializar contador
        // ============================================================
        uint32_t counter_host = submap->getGaussiansCount();

        cudaMemcpy(
            thrust::raw_pointer_cast(instance_counter_.data()),
            &counter_host,
            sizeof(uint32_t),
            cudaMemcpyHostToDevice);

        const Pose keyframe_global_pose = composePoses(submap->getGlobalPose(), keyframe.getRelativePose());

        const auto densify_pass = [&](const cv::cuda::GpuMat &mask_gpu,
                                      uint32_t sample_px,
                                      float scale_factor,
                                      const char *tag)
        {
            auto mask_tex = std::make_shared<Texture<float>>(mask_gpu);

            const int sample_w = (width + static_cast<int>(sample_px) - 1) / static_cast<int>(sample_px);
            const int sample_h = (height + static_cast<int>(sample_px) - 1) / static_cast<int>(sample_px);

            dim3 block(16, 16);
            dim3 grid(
                (sample_w + block.x - 1) / block.x,
                (sample_h + block.y - 1) / block.y);

            densifyGaussians_kernel<<<grid, block>>>(
                thrust::raw_pointer_cast(submap->gaussians.positions.data()),
                thrust::raw_pointer_cast(submap->gaussians.scales.data()),
                thrust::raw_pointer_cast(submap->gaussians.orientations.data()),
                thrust::raw_pointer_cast(submap->gaussians.colors.data()),
                thrust::raw_pointer_cast(submap->gaussians.opacities.data()),
                thrust::raw_pointer_cast(instance_counter_.data()),

                keyframe.getColorTex().getTextureObject(),
                keyframe.getDepthTex().getTextureObject(),
                keyframe.getNormalTex().getTextureObject(),
                mask_tex->getTextureObject(),

                keyframe_global_pose,
                submap->getGlobalPose(),
                keyframe_intrinsics,

                sample_px,
                sample_px,
                width,
                height,
                scale_factor,
                submap->max_gaussians
            );

            CUDA_CHECK_KERNEL(tag);
        };

        if (densification_strategy_ == RegionSamplingStrategy::VigsFusion)
        {
            density_mask_tex_ = std::make_shared<Texture<float>>(density_mask_);
            densify_pass(density_mask_,
                         static_cast<uint32_t>(gauss_init_size_px_),
                         1.0f,
                         "densify::densifyGaussians");
        }
        else
        {
            // ============================================================
            // 3. M_i + M_h + M_l
            // ============================================================
            // M_i: regiones faltantes detectadas por opacidad / densidad.
            // M_h: regiones de alta frecuencia.
            // M_l: regiones de baja frecuencia.
            // M_m: en esta implementación se materializa como la partición
            //      de M_i en dos submascaras, una para cada banda.
            cv::cuda::GpuMat missing_u8_gpu;
            cv::cuda::threshold(density_mask_, missing_u8_gpu, 0.5, 255.0, cv::THRESH_BINARY);
            missing_u8_gpu.convertTo(missing_u8_gpu, CV_8U);

            FrequencyMaskGpuPair frequency_masks_gpu;
            const bool masks_ready = buildStrategyFrequencyMasks(
                keyframe.color_img, densification_strategy_, frequency_masks_gpu, "densification");

            if (!masks_ready)
            {
                density_mask_tex_ = std::make_shared<Texture<float>>(density_mask_);
                densify_pass(density_mask_,
                             static_cast<uint32_t>(gauss_init_size_px_),
                             1.0f,
                             "densify::densifyGaussians");
            }
            else
            {
                cv::cuda::GpuMat missing_high_u8_gpu;
                cv::cuda::GpuMat missing_low_u8_gpu;
                cv::cuda::bitwise_and(missing_u8_gpu, frequency_masks_gpu.high_mask_u8_gpu, missing_high_u8_gpu);
                cv::cuda::bitwise_and(missing_u8_gpu, frequency_masks_gpu.low_mask_u8_gpu, missing_low_u8_gpu);

                if (cv::cuda::countNonZero(missing_high_u8_gpu) > 0)
                {
                    // M_i \interseccion M_h
                    cv::cuda::GpuMat missing_high_gpu;
                    missing_high_u8_gpu.convertTo(missing_high_gpu, CV_32F, 1.0 / 255.0);
                    densify_pass(missing_high_gpu,
                                 static_cast<uint32_t>(fgs_sample_high_px_),
                                 fgs_scale_high_,
                                 "densify::densifyGaussiansHigh");
                }

                if (cv::cuda::countNonZero(missing_low_u8_gpu) > 0)
                {
                    // M_i \interseccion M_l
                    cv::cuda::GpuMat missing_low_gpu;
                    missing_low_u8_gpu.convertTo(missing_low_gpu, CV_32F, 1.0 / 255.0);
                    densify_pass(missing_low_gpu,
                                 static_cast<uint32_t>(fgs_sample_low_px_),
                                 fgs_scale_low_,
                                 "densify::densifyGaussiansLow");
                }
            }
        }

        // ============================================================
        // 5. Recuperar contador
        // ============================================================
        cudaMemcpy(
            &counter_host,
            thrust::raw_pointer_cast(instance_counter_.data()),
            sizeof(uint32_t),
            cudaMemcpyDeviceToHost);

        uint32_t prev = submap->getGaussiansCount();
        submap->gaussians_count = std::min(counter_host, submap->max_gaussians);
        refreshGlobalGaussiansCount();
        last_cpu_timings_.densify_added = std::max(0, static_cast<int>(submap->getGaussiansCount()) - before_densify);

        //std::cout << "Densify: " << prev << " -> " << n_Gaussians
        //        << " (+" << (n_Gaussians - prev) << ")\n";

        cudaEventRecord(cuda_evt_densify_end_);
        last_gpu_timings_.densification_ms = getCudaEventElapsedMs(cuda_evt_densify_start_, cuda_evt_densify_end_);
    }

    struct KeepZero
    {
        __host__ __device__
        bool operator()(unsigned char s) const { 
            return s == 0; 
        }
    };
    
    void GSSlam::prune(Submap* submap)
    {
        cudaEventRecord(cuda_evt_prune_start_);
        // Keep pruning logic here
        if (!submap || submap->getGaussiansCount() == 0) {
            return;
        }
        //std::cout << "[GSSlam] Pruning start..." << std::endl;

        // ============================================================
        // 1. Parámetros
        // ============================================================
        const float alpha_threshold       = 0.05f;
        const float scale_ratio_threshold = 0.05f;

        // ============================================================
        // 2. Buffers auxiliares
        // ============================================================
        const uint32_t n_gauss = submap->getGaussiansCount();
        thrust::device_vector<unsigned char> states(n_gauss, 0);

        // Compaction output buffers must match current submap size.
        if (new_positions.size() < n_gauss)
        {
            new_positions.resize(n_gauss);
            new_scales.resize(n_gauss);
            new_orientations.resize(n_gauss);
            new_colors.resize(n_gauss);
            new_opacities.resize(n_gauss);
            new_adam_states.resize(n_gauss);
        }

        // contador de eliminadas (lo seguimos usando como en tu kernel)
        uint32_t zero = 0;
        cudaMemcpy(
            thrust::raw_pointer_cast(instance_counter_.data()),
            &zero,
            sizeof(uint32_t),
            cudaMemcpyHostToDevice);

        // ============================================================
        // 3. Kernel prune
        // ============================================================
        dim3 block(256);
        dim3 grid((n_gauss + block.x - 1) / block.x);

        pruneGaussians_kernel<<<grid, block>>>(
            thrust::raw_pointer_cast(instance_counter_.data()),
            thrust::raw_pointer_cast(states.data()),
            thrust::raw_pointer_cast(submap->gaussians.scales.data()),
            thrust::raw_pointer_cast(submap->gaussians.opacities.data()),
            alpha_threshold,
            scale_ratio_threshold,
            n_gauss);

        CUDA_CHECK_KERNEL("prune::kernel");

        // ============================================================
        // 5. Compaction
        // ============================================================
        auto zip_in = thrust::make_zip_iterator(thrust::make_tuple(
            submap->gaussians.positions.begin(),
            submap->gaussians.scales.begin(),
            submap->gaussians.orientations.begin(),
            submap->gaussians.colors.begin(),
            submap->gaussians.opacities.begin(),
            submap->adam_states.begin()
        ));

        auto zip_out = thrust::make_zip_iterator(thrust::make_tuple(
            new_positions.begin(),
            new_scales.begin(),
            new_orientations.begin(),
            new_colors.begin(),
            new_opacities.begin(),
            new_adam_states.begin()
        ));



        // Copiar solo las gaussianas válidas (estado == 0)
        auto end_it = thrust::copy_if(
            zip_in,
            zip_in + n_gauss,
            states.begin(),
            zip_out,
            KeepZero()
        );

        // Calcular cuántas gaussianas se copiaron
        uint32_t new_count = end_it - zip_out;

        submap->gaussians.positions.swap(new_positions);
        submap->gaussians.scales.swap(new_scales);
        submap->gaussians.orientations.swap(new_orientations);
        submap->gaussians.colors.swap(new_colors);
        submap->gaussians.opacities.swap(new_opacities);
        submap->adam_states.swap(new_adam_states);
        // ==============================================================
        // 6. Limpiar adam states
        // ==============================================================
        int nb_removed = static_cast<int>(n_gauss - new_count);
        if (nb_removed > 0)
        {
            cudaMemset(
                thrust::raw_pointer_cast(submap->adam_states.data()) + new_count,
                0,
                sizeof(AdamStateGaussian3D) * nb_removed
            );
        }

        // ============================================================
        // 7. Actualizar contador
        // ============================================================
        submap->gaussians_count = new_count;
        refreshGlobalGaussiansCount();
        last_cpu_timings_.prune_removed = nb_removed;

        cudaEventRecord(cuda_evt_prune_end_);
        last_gpu_timings_.pruning_ms = getCudaEventElapsedMs(cuda_evt_prune_start_, cuda_evt_prune_end_);
    }
    
    void GSSlam::removeOutliers()
    {
        //std::cout << "[GSSlam] Outlier removal start..." << std::endl;
        
        Submap* submap = getCurrentSubmap();
        if (!submap) {
            return;
        }
        
        // ============================================================
        // 1. Early exit & parámetros
        // ============================================================
        if (submap->getGaussiansCount() == 0 || pyr_depth_.empty()) {
            return;
        }

        const int width  = pyr_depth_[0].cols;
        const int height = pyr_depth_[0].rows;

        if (width <= 0 || height <= 0) {
            return;
        }

        const float outlier_threshold = 0.6f;

        // ============================================================
        // 2. Preparación de rasterización
        //    (proyección + binning en tiles)
        // ============================================================
        // current_pose_ is local to submap
        prepareRasterization(submap, current_pose_, intrinsics_, width, height);

        // ============================================================
        // 3. Buffers auxiliares (probabilidad de outlier)
        //    - outlier_prob: contribuciones inconsistentes
        //    - total_alpha: peso total acumulado
        //    - states: 0 = mantener, 0xff = eliminar
        // ============================================================
        thrust::device_vector<float> outlier_prob(submap->getGaussiansCount());
        thrust::device_vector<float> total_alpha(submap->getGaussiansCount());
        thrust::device_vector<unsigned char> states(submap->getGaussiansCount());

        thrust::fill(outlier_prob.begin(), outlier_prob.end(), 0.0f);
        thrust::fill(total_alpha.begin(), total_alpha.end(), 0.0f);
        thrust::fill(states.begin(), states.end(), 0);

        // ============================================================
        // 4. Evaluación de outliers (por píxel / tile)
        //    Acumula evidencia de inconsistencia con el depth
        // ============================================================
        dim3 block(tile_size_.x, tile_size_.y);
        dim3 grid(num_tiles_.x, num_tiles_.y);

        computeOutliers_kernel<<<grid, block>>>(
            thrust::raw_pointer_cast(outlier_prob.data()),
            thrust::raw_pointer_cast(total_alpha.data()),
            thrust::raw_pointer_cast(tile_ranges_.data()),
            thrust::raw_pointer_cast(gaussian_indices_.data()),
            thrust::raw_pointer_cast(positions_2d_.data()),
            thrust::raw_pointer_cast(inv_covariances_2d_.data()),
            thrust::raw_pointer_cast(p_hats_.data()),
            thrust::raw_pointer_cast(submap->gaussians.opacities.data()),
            pyr_depth_[0].ptr<float>(),
            pyr_depth_[0].step,
            num_tiles_,
            width,
            height,
            submap->getGaussiansCount(),
            last_nb_instances_);

        CUDA_CHECK_KERNEL("removeOutliers::computeOutliers_kernel");

        // ============================================================
        // 5. Clasificación de gaussianas
        // ============================================================
        thrust::device_vector<uint32_t> nb_removed(1, 0);

        dim3 block2(256);
        dim3 grid2((submap->getGaussiansCount() + block2.x - 1) / block2.x);

        removeOutliers_kernel<<<grid2, block2>>>(
            thrust::raw_pointer_cast(nb_removed.data()),
            thrust::raw_pointer_cast(states.data()),
            thrust::raw_pointer_cast(outlier_prob.data()),
            thrust::raw_pointer_cast(total_alpha.data()),
            outlier_threshold,
            submap->getGaussiansCount());

        CUDA_CHECK_KERNEL("removeOutliers::removeOutliers_kernel");

        // ============================================================
        // 6. Reordenamiento (compaction implícita)
        //    - states = 0   → primero (válidas)
        //    - states = 0xff → al final (outliers)
        // ============================================================
        auto zip_begin = thrust::make_zip_iterator(thrust::make_tuple(
            submap->gaussians.positions.begin(),
            submap->gaussians.scales.begin(),
            submap->gaussians.orientations.begin(),
            submap->gaussians.colors.begin(),
            submap->gaussians.opacities.begin()));

        thrust::sort_by_key(
            states.begin(),
            states.begin() + submap->getGaussiansCount(),
            zip_begin);

        // ============================================================
        // 7. Actualización del contador
        // ============================================================
        uint32_t nb_removed_host = 0;
        cudaMemcpy(&nb_removed_host,
                thrust::raw_pointer_cast(nb_removed.data()),
                sizeof(uint32_t),
                cudaMemcpyDeviceToHost);

        if (nb_removed_host > submap->getGaussiansCount()) {
            // protección básica contra corrupción
            return;
        }

        submap->gaussians_count -= nb_removed_host;
        refreshGlobalGaussiansCount();
        last_cpu_timings_.outliers_removed = static_cast<int>(nb_removed_host);

        // ============================================================
        // 8. Fin
        // ============================================================
        //std::cout << "[GSSlam] Outlier removal finished. Total gaussians: " << n_Gaussians << std::endl;
    }

    float GSSlam::computeCovisibilityRatio()
    {
        // ============================================================
        // 1. Early exit
        //    - Sin gaussianas o sin datos válidos → covisibilidad trivial
        // ============================================================
        Submap* submap = getCurrentSubmap();
        if (!submap || submap->getGaussiansCount() == 0 || submap->keyframes.empty() || pyr_color_.empty()) {
            return 1.0f;
        }
        
        const uint32_t n_gauss = submap->getGaussiansCount();

        // Selección segura del keyframe actual
        size_t keyframe_idx = static_cast<size_t>(current_keyframe_idx_);
        if (current_keyframe_idx_ < 0 || keyframe_idx >= submap->keyframes.size()) {
            keyframe_idx = submap->keyframes.size() - 1;
        }

        const KeyframeData &keyframe = submap->keyframes[keyframe_idx];

        if (keyframe.color_img.empty() || keyframe.depth_img.empty()) {
            return 1.0f;
        }

        const int frame_width  = pyr_color_[0].cols;
        const int frame_height = pyr_color_[0].rows;

        if (frame_width <= 0 || frame_height <= 0) {
            return 1.0f;
        }

    //std::cout << "[GSSlam] Computing covisibility ratio with keyframe " << keyframe_idx << "..." << std::endl;

        // ============================================================
        // 2. Inicialización de buffers
        //    - keyframeVis: visibilidad en keyframe
        //    - frameVis:    visibilidad en frame actual
        //    - visInter:    intersección
        //    - visUnion:    unión
        // ============================================================
        thrust::fill(d_keyframeVis_.begin(), d_keyframeVis_.begin() + n_gauss, 0);
        thrust::fill(d_frameVis_.begin(),    d_frameVis_.begin()    + n_gauss, 0);

        thrust::fill(d_visInter_.begin(), d_visInter_.begin() + 1, 0);
        thrust::fill(d_visUnion_.begin(), d_visUnion_.begin() + 1, 0);

        // ============================================================
        // 3. Visibilidad en el keyframe
        // ============================================================
        prepareRasterization(submap, keyframe.getRelativePose(),
                    keyframe.getIntrinsics(),
                    keyframe.getWidth(),
                    keyframe.getHeight());

        if (last_nb_instances_ == 0) {
            return 0.0f;
        }

        // Copia de posiciones proyectadas (consistencia con buffers persistentes)
        const uint32_t valid_size = static_cast<uint32_t>(n_gauss);

        if (valid_size == 0 || inv_covariances_2d_.size() < valid_size) {
            return 0.0f;
        }

        if (d_imgPositions_.size() < static_cast<size_t>(submap->max_gaussians)) {
            d_imgPositions_.resize(submap->max_gaussians);
        }

        thrust::copy(positions_2d_.begin(),
                    positions_2d_.begin() + valid_size,
                    d_imgPositions_.begin());

        // Kernel: marca gaussianas visibles en el keyframe
        computeGaussiansVisibility_kernel<<<
            dim3(num_tiles_.x, num_tiles_.y),
            dim3(tile_size_.x, tile_size_.y)
        >>>(
            thrust::raw_pointer_cast(d_keyframeVis_.data()),
            thrust::raw_pointer_cast(tile_ranges_.data()),
            thrust::raw_pointer_cast(gaussian_indices_.data()),
            thrust::raw_pointer_cast(d_imgPositions_.data()),
            thrust::raw_pointer_cast(inv_covariances_2d_.data()),
            thrust::raw_pointer_cast(submap->gaussians.opacities.data()),
            num_tiles_,
            keyframe.getWidth(),
            keyframe.getHeight()
        );

        CUDA_CHECK_KERNEL("ComputeCovisibilityRatio::keyframe visibility");

        // ============================================================
        // 4. Visibilidad en el frame actual
        // ============================================================
        // current_pose_ is local to submap
        prepareRasterization(submap, current_pose_,
                    intrinsics_,
                    frame_width,
                    frame_height);

        if (last_nb_instances_ == 0) {
            return 0.0f;
        }

        // Reutilizamos el mismo buffer de posiciones proyectadas
        thrust::copy(positions_2d_.begin(),
                    positions_2d_.begin() + valid_size,
                    d_imgPositions_.begin());

        // Kernel: marca gaussianas visibles en el frame actual
        computeGaussiansVisibility_kernel<<<
            dim3(num_tiles_.x, num_tiles_.y),
            dim3(tile_size_.x, tile_size_.y)
        >>>(
            thrust::raw_pointer_cast(d_frameVis_.data()),
            thrust::raw_pointer_cast(tile_ranges_.data()),
            thrust::raw_pointer_cast(gaussian_indices_.data()),
            thrust::raw_pointer_cast(d_imgPositions_.data()),
            thrust::raw_pointer_cast(inv_covariances_2d_.data()),
            thrust::raw_pointer_cast(submap->gaussians.opacities.data()),
            num_tiles_,
            frame_width,
            frame_height
        );
        CUDA_CHECK_KERNEL("ComputeCovisibilityRatio::frame visibility");

    

        // ============================================================
        // 5. Cómputo de covisibilidad
        //    ratio = |intersección| / |unión|
        // ============================================================
        const int block_size = 256;

        computeGaussiansCovisibility_kernel<<<
            (n_gauss + block_size - 1) / block_size,
            block_size
        >>>(
            thrust::raw_pointer_cast(d_visInter_.data()),
            thrust::raw_pointer_cast(d_visUnion_.data()),
            thrust::raw_pointer_cast(d_keyframeVis_.data()),
            thrust::raw_pointer_cast(d_frameVis_.data()),
            n_gauss
        );

        CUDA_CHECK_KERNEL("ComputeCovisibilityRatio::covisibility");

        // ============================================================
        // 6. Lectura de resultados
        // ============================================================
        const uint32_t inter = d_visInter_[0];
        const uint32_t uni   = d_visUnion_[0];

        if (uni == 0) {
            return 0.0f;
        }
    //std::cout << "[GSSlam] Covisibility: inter=" << inter << ", union=" << uni << ", ratio=" << (static_cast<float>(inter) / static_cast<float>(uni)) << std::endl;

        return inter / static_cast<float>(uni);
    }

    void GSSlam::initAndCopyImgs(const cv::Mat &rgb, const cv::Mat &depth)
    {
        cudaEventRecord(cuda_evt_imgcopy_start_);
        // ============================================================
        // 0. EARLY EXIT
        // ============================================================
        if (rgb.empty() || depth.empty()) return;
        if (nb_pyr_levels_ <= 0)
        {
            std::cerr << "[GSSlam] [ERROR initAndCopyImgs] nb_pyr_levels_ <= 0" << std::endl;
            return;
        }

       // std::cout << "[GSSlam] initAndCopyImgs start at... " << nb_images_processed_ << "..." << std::endl;

        //warping_cache_valid_ = false;

        // ============================================================
        // 2. NORMALIZACIÓN CPU
        // ============================================================
        cv::Mat rgb_bgr;
        if (rgb.type() == CV_8UC3) rgb_bgr = rgb;
        else rgb.convertTo(rgb_bgr, CV_8UC3);

        cv::Mat depth_float;
        if (depth.type() == CV_32FC1) depth_float = depth;
        else if (depth.type() == CV_16UC1)
            depth.convertTo(depth_float, CV_32FC1, depth_scale_);
        else return;

        // ============================================================
        // 3. UPLOAD GPU
        // ============================================================
        rgb_gpu_.upload(rgb_bgr);
        depth_gpu_.upload(depth_float);

        cv::cuda::cvtColor(rgb_gpu_, pyr_color_[0], cv::COLOR_BGR2BGRA);
        depth_gpu_.copyTo(pyr_depth_[0]);

        // ============================================================
        // 4. PIRÁMIDE
        // ============================================================
        for (int i = 1; i < nb_pyr_levels_; i++)
        {
            cv::cuda::pyrDown(pyr_color_[i - 1], pyr_color_[i]);
            cv::cuda::pyrDown(pyr_depth_[i - 1], pyr_depth_[i]);
        }

        // ============================================================
        // 5. GRADIENTES (DerivFilter o Sobel custom)
        // ============================================================
        if (use_deriv_filters_ && (!deriv_dx_filter_ || !deriv_dy_filter_))
        {
            deriv_dx_filter_ = cv::cuda::createDerivFilter(CV_8UC4, CV_32FC4, 1, 0, 3, true, 1.0 / 255.0);
            deriv_dy_filter_ = cv::cuda::createDerivFilter(CV_8UC4, CV_32FC4, 0, 1, 3, true, 1.0 / 255.0);
        }

        for (int i = 0; i < nb_pyr_levels_; i++)
        {
            pyr_dx_[i].create(pyr_color_[i].size(), CV_32FC4);
            pyr_dy_[i].create(pyr_color_[i].size(), CV_32FC4);

            if (use_deriv_filters_)
            {
                deriv_dx_filter_->apply(pyr_color_[i], pyr_dx_[i]);
                deriv_dy_filter_->apply(pyr_color_[i], pyr_dy_[i]);
            }
            else
            {
                dim3 block(16, 16);
                dim3 grid(
                    (pyr_color_[i].cols + block.x - 1) / block.x,
                    (pyr_color_[i].rows + block.y - 1) / block.y);

                size_t shared_bytes =
                    (block.x + 2) * (block.y + 2) * sizeof(float4);

                computeSobelRgb_kernel<<<grid, block, shared_bytes>>>(
                    reinterpret_cast<const uchar4*>(pyr_color_[i].ptr<uchar4>()),
                    pyr_color_[i].step,
                    reinterpret_cast<float4*>(pyr_dx_[i].ptr<float4>()),
                    pyr_dx_[i].step,
                    reinterpret_cast<float4*>(pyr_dy_[i].ptr<float4>()),
                    pyr_dy_[i].step,
                    pyr_color_[i].cols,
                    pyr_color_[i].rows
                );

                CUDA_CHECK_KERNEL("computeSobelRgb_kernel");
            }
        }

        // ============================================================
        // 6. TEXTURAS (COLOR + DEPTH + GRADIENTES + NORMALES)
        // ============================================================
        auto ensureSize = [&](auto &vec) {
            if (vec.size() != static_cast<size_t>(nb_pyr_levels_))
                vec.resize(nb_pyr_levels_);
        };

        //std::cout << "[GSSlam] Updating textures for " << nb_pyr_levels_ << " pyramid levels..." << std::endl;

        ensureSize(pyr_normals_);
        if (pyr_normals_.empty())
        {
            std::cerr << "[GSSlam] [ERROR initAndCopyImgs] pyr_normals_ not initialized" << std::endl;
            return;
        }

        pyr_normals_[0].create(depth_gpu_.size(), CV_32FC4);

        ensureSize(pyr_color_tex_);
        ensureSize(pyr_depth_tex_);
        ensureSize(pyr_normals_tex_);
        ensureSize(pyr_dx_tex_);
        ensureSize(pyr_dy_tex_);

        for (int i = 0; i < nb_pyr_levels_; i++)
        {
            if (pyr_color_[i].empty() || pyr_depth_[i].empty() || pyr_dx_[i].empty() || pyr_dy_[i].empty())
            {
                std::cerr << "[GSSlam] [ERROR initAndCopyImgs] Empty pyramid buffer at level " << i << std::endl;
                return;
            }

            pyr_color_tex_[i] = std::make_shared<Texture<uchar4>>(pyr_color_[i]);
            pyr_depth_tex_[i] = std::make_shared<Texture<float>>(pyr_depth_[i]);
            // Actualmente las normales solo se usan en el nivel 0 por lo que
            // dejo la pirámide con un solo nivel para ahorrar memoria y los demás vacíos,
            // pero dejo el código preparado
            if (i == 0)
            {
                pyr_normals_tex_[i] = std::make_shared<Texture<float4>>(pyr_normals_[i]);
            }
            else
            {
                pyr_normals_tex_[i].reset();
            }
            pyr_dx_tex_[i] = std::make_shared<Texture<float4>>(pyr_dx_[i]);
            pyr_dy_tex_[i] = std::make_shared<Texture<float4>>(pyr_dy_[i]);

            if (!pyr_color_tex_[i] || !pyr_depth_tex_[i] || !pyr_dx_tex_[i] || !pyr_dy_tex_[i])
            {
                std::cerr << "[GSSlam] [ERROR initAndCopyImgs] Texture allocation failed at level " << i << std::endl;
                return;
            }

            if (i == 0 && (!pyr_normals_tex_[0] || pyr_depth_tex_[0]->getTextureObject() == 0))
            {
                std::cerr << "[GSSlam] [ERROR initAndCopyImgs] Invalid normals/depth texture at level 0" << std::endl;
                return;
            }
        }

        //std::cout << "[GSSlam] Textures updated for all pyramid levels." << std::endl;

        // ============================================================
        // 7. Normales
        // ============================================================
        
        //std::cout << "[GSSlam] Computing normals for level 0..." << std::endl;

        dim3 n_block(16, 16);
        dim3 n_grid(
            (depth_gpu_.cols + n_block.x - 1) / n_block.x,
            (depth_gpu_.rows + n_block.y - 1) / n_block.y);

        computeNormalsFromDepth_kernel<<<n_grid, n_block>>>(
            pyr_depth_tex_[0]->getTextureObject(),
            pyr_normals_[0].ptr<float4>(),
            pyr_normals_[0].step,
            depth_gpu_.cols,
            depth_gpu_.rows,
            intrinsics_
        );

        CUDA_CHECK_KERNEL("computeNormalsFromDepth_kernel");

        // ============================================================
        // 7. LEGACY POINTERS
        // ============================================================
        rgb_gpu_   = pyr_color_[0];
        depth_gpu_ = pyr_depth_[0];
        
        cudaEventRecord(cuda_evt_imgcopy_end_);
        last_gpu_timings_.image_copy_ms = getCudaEventElapsedMs(cuda_evt_imgcopy_start_, cuda_evt_imgcopy_end_);
            //std::cout << "[GSSlam] initAndCopyImgs finished." << std::endl;
    }
    


    void GSSlam::processImu(double t, const ImuData &imu_data)
    {
        if (preint_->is_initialized)
        {
            const double dt = t - last_imu_time_;
            preint_->add_imu(dt, imu_data.Acc, imu_data.Gyro);
        }
        else
        {
            Eigen::Map<Eigen::VectorXd> VB_cur_eig(VB_cur_, 9);
            preint_->init(imu_data.Acc, imu_data.Gyro,
                          VB_cur_eig.segment(3, 3),
                          VB_cur_eig.segment(6, 3),
                          imu_data.acc_n, imu_data.gyr_n,
                          imu_data.acc_w, imu_data.gyr_w);
        }

        last_imu_ = imu_data;
        last_imu_time_ = t;
    }
    
    void GSSlam::initializeImu(const ImuData &imu_data)
    {
        if (!preint_) {
            return;
        }
        
        // Guardar datos del IMU
        last_imu_ = imu_data;
        
        Eigen::Map<Eigen::VectorXd> VB_cur_eig(VB_cur_, 9);
        preint_->init(imu_data.Acc, imu_data.Gyro,
                     VB_cur_eig.segment(3, 3), VB_cur_eig.segment(6, 3),
                     imu_data.acc_n, imu_data.gyr_n,
                     imu_data.acc_w, imu_data.gyr_w);

        last_imu_time_ = -1.0;
        
        // Marcar como inicializada
        imu_initialized_ = true;
    }

    void GSSlam::setImuBias(const Eigen::Vector3d &b_a,
                            const Eigen::Vector3d &b_g)
    {
        Eigen::Map<Eigen::Vector3d> BA_prev_eig(VB_prev_ + 3);
        Eigen::Map<Eigen::Vector3d> BG_prev_eig(VB_prev_ + 6);
        Eigen::Map<Eigen::Vector3d> BA_cur_eig(VB_cur_ + 3);
        Eigen::Map<Eigen::Vector3d> BG_cur_eig(VB_cur_ + 6);

        BA_cur_eig = BA_prev_eig = b_a;
        BG_cur_eig = BG_prev_eig = b_g;
    }

    // Equivalente a optimizePoseGNCeres
    void GSSlam::computeRgbdPoseJacobians(
        Eigen::Matrix<double, 6, 6> &JtJ,
        Eigen::Vector<double, 6> &Jtr,
        int level,
        const Eigen::Vector3d &P_imu,
        const Eigen::Quaterniond &Q_imu)
    {
        // ============================================================
        // 1. IMU → CÁMARA
        // ============================================================
        Eigen::Vector3d P_cam = P_imu + Q_imu * t_imu_cam_;
        Eigen::Quaterniond Q_cam = Q_imu * q_imu_cam_;

        Pose camera_pose;
        camera_pose.position = make_float3(
            static_cast<float>(P_cam.x()),
            static_cast<float>(P_cam.y()),
            static_cast<float>(P_cam.z()));

        Eigen::Quaternionf Q_cam_f = Q_cam.cast<float>();
        camera_pose.orientation = make_float4(
            Q_cam_f.x(), Q_cam_f.y(), Q_cam_f.z(), Q_cam_f.w());

        // ============================================================
        // 2. NIVEL PIRÁMIDE
        // ============================================================
        int pyr_level = std::min(level, nb_pyr_levels_ - 1);

        int width  = pyr_color_[pyr_level].cols;
        int height = pyr_color_[pyr_level].rows;

        if (pyr_intrinsics_.size() != static_cast<size_t>(nb_pyr_levels_)) {
            updateIntrinsicsPyramid();
        }

        IntrinsicParameters level_intrinsics = pyr_intrinsics_[pyr_level];

        // ============================================================
        // 3. RASTERIZACIÓN
        // ============================================================
        // por el momento anulo warping

        // Get current submap
        Submap* submap = getCurrentSubmap();
        if (!submap) {
            JtJ.setZero();
            Jtr.setZero();
            return;
        }
        
        prepareRasterization(submap, toSubmapLocalPose(submap, camera_pose), level_intrinsics, width, height);

        // ============================================================
        // 4. EARLY EXIT
        // ============================================================
        JtJ.setZero();
        Jtr.setZero();

        if (submap->getGaussiansCount() == 0) return;
        if (pyr_color_.empty() || pyr_depth_.empty()) return;

        int num_tiles_x = num_tiles_.x;
        int num_tiles_y = num_tiles_.y;
        int num_tiles_total = num_tiles_x * num_tiles_y;

        if (num_tiles_total == 0) return;

        // ============================================================
        // 5. BUFFER DE SALIDA GLOBAL
        // ============================================================
        thrust::device_vector<PoseOptimizationRgbdData> posedata(1);

        cudaMemset(
            thrust::raw_pointer_cast(posedata.data()),
            0,
            sizeof(PoseOptimizationRgbdData));

        dim3 block(tile_size_.x, tile_size_.y);
        dim3 grid(num_tiles_x, num_tiles_y);

        // ============================================================
        // 6. TEXTURAS
        // ============================================================
        cudaTextureObject_t color_tex = pyr_color_tex_[pyr_level]->getTextureObject();
        cudaTextureObject_t depth_tex = pyr_depth_tex_[pyr_level]->getTextureObject();
        cudaTextureObject_t dx_tex    = pyr_dx_tex_[pyr_level]->getTextureObject();
        cudaTextureObject_t dy_tex    = pyr_dy_tex_[pyr_level]->getTextureObject();

        // ============================================================
        // 7. KERNEL
        // ============================================================

        getRgbdPoseJacobians_fast<<<grid, block>>>(
            thrust::raw_pointer_cast(posedata.data()),

            // rasterización
            thrust::raw_pointer_cast(tile_ranges_.data()),
            thrust::raw_pointer_cast(gaussian_indices_.data()),
            thrust::raw_pointer_cast(positions_2d_.data()),
            thrust::raw_pointer_cast(inv_covariances_2d_.data()),
            thrust::raw_pointer_cast(p_hats_.data()),
            thrust::raw_pointer_cast(submap->gaussians.colors.data()),
            thrust::raw_pointer_cast(submap->gaussians.opacities.data()),

            // TEXTURAS
            color_tex,
            depth_tex,
            dx_tex,
            dy_tex,

            // parámetros
            camera_pose,
            level_intrinsics,
            bg_color_,
            pose_alpha_thresh_,
            pose_color_thresh_,
            pose_depth_thresh_,

            width,
            height,
            num_tiles_x,
            num_tiles_y
        );

        // ============================================================
        // 8. DESCARGA GLOBAL
        // ============================================================
        PoseOptimizationRgbdData total;
        thrust::copy(posedata.begin(), posedata.end(), &total);

        int idx = 0;
        for (int i = 0; i < 6; ++i)
        {
            for (int j = i; j < 6; ++j)
            {
                double value = static_cast<double>(total.JtJ[idx++]);
                JtJ(i, j) = value;
                JtJ(j, i) = value;
            }
            Jtr(i) = -static_cast<double>(total.Jtr[i]);
        }

        // ============================================================
        // 9. CAM -> IMU
        // ============================================================
        Eigen::Matrix3d R_imu_cam = q_imu_cam_.toRotationMatrix();
        Eigen::Matrix3d R_imu = Q_imu.toRotationMatrix();
        Eigen::Matrix3d P_imu_cam_skew = skewSymmetric(t_imu_cam_);

        Eigen::Matrix<double, 6, 6> J_cam_imu = Eigen::Matrix<double, 6, 6>::Zero();
        J_cam_imu.block<3, 3>(0, 0) = Eigen::Matrix3d::Identity();
        J_cam_imu.block<3, 3>(0, 3) = -R_imu * P_imu_cam_skew;
        J_cam_imu.block<3, 3>(3, 3) = R_imu_cam.transpose();

        JtJ = J_cam_imu.transpose() * JtJ * J_cam_imu;
        Jtr = J_cam_imu.transpose() * Jtr;
    }

    void GSSlam::optimizeWithCeres(int min_pyr_level, int max_iterations)
    {
        // ============================================================
        // PASO 1: Agregamos el residuo imu
        // ============================================================
        if (preint_ && preint_->is_initialized && !imu_residual_added_)
        {
            imu_cost_ = new ImuCostFunction(preint_shared_,
                                            imu_reprop_ba_thresh_,
                                            imu_reprop_bg_thresh_);

            imu_residual_block_id_ = problem_.AddResidualBlock(
                imu_cost_, nullptr,
                P_prev_, VB_prev_,
                P_cur_, VB_cur_);

            imu_residual_added_ = true;
        }

        // ============================================================
        // PASO 2: Configuramos el solver
        // ============================================================
        options_.max_num_iterations =
            (max_iterations > 0) ? max_iterations : pose_iterations_;

        min_pyr_level = std::max(0, std::min(min_pyr_level, nb_pyr_levels_ - 1));

        // ============================================================
        // PASO 3: Optimizamos la piramide multi-nivel
        // ============================================================
        for (int level = nb_pyr_levels_ - 1; level >= min_pyr_level; --level)
        {
            // actualizar costo visual al nivel
            visual_cost_->update(level);

            // resolver
            ceres::Solve(options_, &problem_, &summary_);
        }
    }

} // namespace f_vigs_slam
