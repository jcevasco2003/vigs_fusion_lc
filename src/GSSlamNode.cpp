#include <f_vigs_slam/GSSlamNode.hpp>
#include "rclcpp/rclcpp.hpp"
#include <f_vigs_slam/GSSlam.cuh>
#include <f_vigs_slam/GaussianSplattingViewer.hpp>
#include <f_vigs_slam/RepresentationClasses.hpp>
#include <f_vigs_slam/MetricsHelper.hpp>
#include <cv_bridge/cv_bridge.hpp>
#include <sensor_msgs/image_encodings.hpp>
#include <functional>
#include <string>
#include <limits>
#include <algorithm>
#include <cctype>
#include <cmath>
#include <chrono>
#include <sstream>
#include <iomanip>
#include <deque>
#include <tf2/time.hpp>

// Implementamos toda la logica del nodo

namespace f_vigs_slam
{
    namespace
    {
        std::string normalizeFrameId(std::string frame_id)
        {
            while (!frame_id.empty() && frame_id.front() == '/')
            {
                frame_id.erase(frame_id.begin());
            }
            return frame_id;
        }

        std::string formatVector3(const Eigen::Vector3d &v)
        {
            std::ostringstream oss;
            oss << std::fixed << std::setprecision(4)
                << '[' << v.x() << ' ' << v.y() << ' ' << v.z() << ']';
            return oss.str();
        }

        std::string formatMatrix3(const Eigen::Matrix3d &m)
        {
            std::ostringstream oss;
            oss << std::fixed << std::setprecision(4)
                << '['
                << m(0,0) << ' ' << m(0,1) << ' ' << m(0,2) << ';'
                << ' ' << m(1,0) << ' ' << m(1,1) << ' ' << m(1,2) << ';'
                << ' ' << m(2,0) << ' ' << m(2,1) << ' ' << m(2,2)
                << ']';
            return oss.str();
        }

        std::string formatQuaternion(const Eigen::Quaterniond &q)
        {
            std::ostringstream oss;
            oss << std::fixed << std::setprecision(4)
                << '[' << q.x() << ' ' << q.y() << ' ' << q.z() << ' ' << q.w() << ']';
            return oss.str();
        }
    } // namespace

    struct GSSlamNode::Impl
    {
        GSSlam gs_core_;
        std::shared_ptr<GaussianSplattingViewer> viewer_;
        IntrinsicParameters intrinsics;
        ImuData imu_data_;
        Preintegration preint_;
        Pose odom_pose_init_imu_;
        Pose odom_pose_init_cam_;
        // Cached extrinsics as Pose to avoid rebuilding/inverting repeatedly
        Pose T_imu_cam_pose_ = Pose::Identity(); // maps cam -> imu
        Pose T_cam_imu_pose_ = Pose::Identity(); // maps imu -> cam (inverse)
    };

    GSSlamNode::GSSlamNode(const rclcpp::NodeOptions & options)
        : Node("gs_slam_node", options),
          impl_(std::make_unique<Impl>())
        {
            RCLCPP_INFO(this->get_logger(), "GS_Node has been started.");

            tf_buffer_ = std::make_unique<tf2_ros::Buffer>(this->get_clock());
            tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);
            tf_broadcaster_ = std::make_unique<tf2_ros::TransformBroadcaster>(this);

            // Declaramos parametros y topicos del nodo, luego los leemos
            this->declare_parameter<std::string>("imu_topic", "imu");
            this->declare_parameter<std::string>("depth_topic", "image_depth");
            this->declare_parameter<std::string>("color_topic", "image_color");
            this->declare_parameter<std::string>("camera_info_topic", "camera_info");
            this->declare_parameter<std::string>("depth_camera_info_topic", "");
            this->declare_parameter<std::string>("depth_input_topic", "");
            this->declare_parameter<std::string>("depth_registration_target_frame", "");
            this->declare_parameter<std::string>("depth_registration_source_frame", "");
            this->declare_parameter<bool>("use_depth_registration", false);
            this->declare_parameter<std::string>("imu_preint_topic", "imu_preint");
            this->declare_parameter<std::string>("imu_frame_id_fallback", "");
            this->declare_parameter<std::string>("world_frame_id", "world");
            this->declare_parameter<std::string>("tf_mode", "static");
            this->declare_parameter<double>("tf_lookup_timeout_s", 0.1);
            this->declare_parameter<std::string>("color_transport", "raw");
            this->declare_parameter<std::string>("depth_transport", "raw");

            this->declare_parameter<double>("acc_n", 0.1);
            this->declare_parameter<double>("gyr_n", 0.01);
            this->declare_parameter<double>("acc_w", 0.001);
            this->declare_parameter<double>("gyr_w", 0.0001);
            this->declare_parameter<bool>("publish_pointcloud", true);
            this->declare_parameter<bool>("publish_reconstructed_images", true);
            this->declare_parameter<bool>("visualize_current_submap", false);
            this->declare_parameter<bool>("test_gaussians_only", false);
            this->declare_parameter<bool>("viewer", true);
            this->declare_parameter<int>("metrics_print_interval_frames", 10);

            auto imu_topic = this->get_parameter("imu_topic").as_string();
            auto depth_topic = this->get_parameter("depth_topic").as_string();
            auto color_topic = this->get_parameter("color_topic").as_string();
            auto camera_info_topic = this->get_parameter("camera_info_topic").as_string();
            auto depth_camera_info_topic = this->get_parameter("depth_camera_info_topic").as_string();
            auto depth_input_topic = this->get_parameter("depth_input_topic").as_string();
            auto depth_registration_target_frame = this->get_parameter("depth_registration_target_frame").as_string();
            auto depth_registration_source_frame = this->get_parameter("depth_registration_source_frame").as_string();
            const bool use_depth_registration = this->get_parameter("use_depth_registration").as_bool();
            auto imu_preint_topic = this->get_parameter("imu_preint_topic").as_string();
            imu_frame_id_fallback_ = this->get_parameter("imu_frame_id_fallback").as_string();
            auto color_transport = this->get_parameter("color_transport").as_string();
            auto depth_transport = this->get_parameter("depth_transport").as_string();
            tf_mode_ = this->get_parameter("tf_mode").as_string();
            if (tf_mode_ != "static" && tf_mode_ != "dynamic") {
                RCLCPP_WARN(this->get_logger(),
                            "Invalid tf_mode='%s'. Falling back to 'static'",
                            tf_mode_.c_str());
                tf_mode_ = "static";
            }
            tf_lookup_timeout_s_ = std::max(0.0, this->get_parameter("tf_lookup_timeout_s").as_double());
            RCLCPP_INFO(this->get_logger(), "TF mode: %s (lookup timeout %.3fs)",
                        tf_mode_.c_str(), tf_lookup_timeout_s_);
            if (imu_topic.empty()) {
                RCLCPP_WARN(this->get_logger(),
                            "imu_topic is empty; IMU integration will receive no data");
            }
            if (imu_preint_topic.empty()) {
                RCLCPP_WARN(this->get_logger(),
                            "imu_preint_topic is empty; preintegrated IMU diagnostic subscription will receive no data");
            }
            RCLCPP_INFO(this->get_logger(), "Using imu_preint_topic '%s' for IMU integration subscription", imu_preint_topic.c_str());
            RCLCPP_INFO(this->get_logger(), "Using imu_topic '%s' for diagnostic IMU subscription", imu_topic.c_str());
            if (!imu_frame_id_fallback_.empty()) {
                RCLCPP_INFO(this->get_logger(), "Using imu_frame_id_fallback='%s' when IMU messages have empty frame_id", imu_frame_id_fallback_.c_str());
            }
            // image_transport soporta raw y compressed, no advertencia necesaria
            world_frame_id_ = this->get_parameter("world_frame_id").as_string();
            // By convention: 'world' denotes the reality frame and 'map' denotes the
            // estimated pose frame. If the configured world_frame_id_ equals "world",
            // interpret it as the reality frame and publish the estimation in 'map'.
            if (world_frame_id_ == "world") {
                publish_frame_id_ = "map";
                RCLCPP_INFO(this->get_logger(), "Interpreting world_frame_id '%s' as reality frame; publishing estimation in '%s'",
                            world_frame_id_.c_str(), publish_frame_id_.c_str());
            } else {
                publish_frame_id_ = world_frame_id_;
            }

            if (use_depth_registration)
            {
                RCLCPP_INFO(this->get_logger(),
                            "Depth registration enabled: input='%s' output='%s' target_frame='%s' source_frame='%s' depth_info='%s'",
                            depth_input_topic.c_str(),
                            depth_topic.c_str(),
                            depth_registration_target_frame.c_str(),
                            depth_registration_source_frame.c_str(),
                            depth_camera_info_topic.c_str());
            }

            this->get_parameter<double>("acc_n", impl_->imu_data_.acc_n);
            this->get_parameter<double>("gyr_n", impl_->imu_data_.gyr_n);
            this->get_parameter<double>("acc_w", impl_->imu_data_.acc_w);
            this->get_parameter<double>("gyr_w", impl_->imu_data_.gyr_w);
            publish_pointcloud_ = this->get_parameter("publish_pointcloud").as_bool();
            publish_reconstructed_images_ =
                this->get_parameter("publish_reconstructed_images").as_bool();
            visualize_current_submap_ = this->get_parameter("visualize_current_submap").as_bool();
            test_gaussians_only_ = this->get_parameter("test_gaussians_only").as_bool();
            const bool enable_viewer = this->get_parameter("viewer").as_bool();
            const auto metrics_print_interval_frames_param =
                this->get_parameter("metrics_print_interval_frames").as_int();
            metrics_print_interval_frames_ = static_cast<int>(
                std::max<int64_t>(1, metrics_print_interval_frames_param));
            impl_->gs_core_.setMetricsPrintIntervalFrames(metrics_print_interval_frames_);
            RCLCPP_INFO(this->get_logger(), "visualize_current_submap=%s",
                        visualize_current_submap_ ? "true" : "false");
            if (test_gaussians_only_)
            {
                RCLCPP_WARN(this->get_logger(),
                            "test_gaussians_only=true: compute IMU/Ceres pipeline disabled. Using first RGBD frame with identity pose and gaussian-only optimization.");
            }

            this->declare_parameter<int>("gauss_init_size_px", 7);
            this->declare_parameter<int>("downsample_factor", 1);
            this->declare_parameter<double>("gauss_init_scale", 0.01);
            this->declare_parameter<double>("depth_scale", 1.0);
            this->declare_parameter<int>("pose_iterations", 4);
            this->declare_parameter<int>("gaussian_iterations", 10);
            this->declare_parameter<double>("eta_pose", 0.01);
            this->declare_parameter<double>("eta_gaussian", 0.002);
            this->declare_parameter<double>("adam_eta", 1e-4);
            this->declare_parameter<double>("adam_beta1", 0.9);
            this->declare_parameter<double>("adam_beta2", 0.999);
            this->declare_parameter<double>("adam_epsilon", 1e-8);
            this->declare_parameter<double>("w_depth", 1.0);
            this->declare_parameter<double>("w_dist", 0.1);
            this->declare_parameter<double>("covisibility_threshold", 0.95);
            this->declare_parameter<double>("submap_dist_threshold_m", 0.5);
            this->declare_parameter<double>("submap_rot_threshold_deg", 50.0);
            this->declare_parameter<double>("imu_reprop_ba_thresh", 0.10);
            this->declare_parameter<double>("imu_reprop_bg_thresh", 0.01);
            this->declare_parameter<double>("pose_alpha_threshold", 0.1);
            this->declare_parameter<double>("pose_color_threshold", 0.2);
            this->declare_parameter<double>("pose_depth_threshold", 0.4);
            this->declare_parameter<bool>("use_deriv_filters", true);
            this->declare_parameter<int>("imu_init_samples", 100);
            this->declare_parameter<double>("imu_dt_warn_max_s", 0.05);
            this->declare_parameter<double>("imu_acc_norm_min", 1.0);
            this->declare_parameter<double>("imu_acc_norm_max", 30.0);
            this->declare_parameter<double>("imu_gyro_norm_max", 10.0);
            this->declare_parameter<double>("diag_state_jump_pos_thresh_m", 0.20);
            this->declare_parameter<double>("diag_state_jump_rot_thresh_deg", 5.0);
            this->declare_parameter<double>("diag_proc_time_domain_abs_limit_s", 10.0);
            
            this->declare_parameter<std::string>("keyframe_selection_method", "beta_binomial");
            this->declare_parameter<double>("gumbel_alpha", 0.7);
            this->declare_parameter<double>("gumbel_beta", 2.0);
            this->declare_parameter<double>("beta_binomial_alpha", 0.7);
            this->declare_parameter<double>("beta_binomial_beta", 2.0);
            this->declare_parameter<double>("exponential_lambda", 1.0);
            this->declare_parameter<int>("sliding_window_window_size", -1);
            this->declare_parameter<std::string>("slam_profile", "vigs_fusion");
            this->declare_parameter<std::string>("initialization_strategy", "vigs_fusion");
            this->declare_parameter<std::string>("densification_strategy", "vigs_fusion");
            this->declare_parameter<bool>("frequency_guided_densification", false);
            this->declare_parameter<bool>("frequency_guided_initialization", false);
            this->declare_parameter<int>("strategy_sample_high_px", 2);
            this->declare_parameter<int>("strategy_sample_low_px", 6);
            this->declare_parameter<double>("strategy_scale_high", 0.8);
            this->declare_parameter<double>("strategy_scale_low", 1.0);
            this->declare_parameter<double>("strategy_highpass_sigma_ratio", 0.06);
            this->declare_parameter<double>("canny_threshold1", 50.0);
            this->declare_parameter<double>("canny_threshold2", 150.0);

            gauss_init_size_px_ = this->get_parameter("gauss_init_size_px").as_int();
            downsample_factor_ = static_cast<int>(this->get_parameter("downsample_factor").as_int());
            gauss_init_scale_ = this->get_parameter("gauss_init_scale").as_double();
            depth_scale_ = this->get_parameter("depth_scale").as_double();

            if (downsample_factor_ < 1 || (downsample_factor_ > 1 && (downsample_factor_ % 2) != 0))
            {
                RCLCPP_WARN(this->get_logger(),
                            "Invalid downsample_factor=%d. It must be 1 or an even integer. Falling back to 1.",
                            downsample_factor_);
                downsample_factor_ = 1;
            }
            RCLCPP_INFO(this->get_logger(), "Using downsample_factor=%d", downsample_factor_);

            impl_->gs_core_.setGaussInitSizePx(gauss_init_size_px_);
            impl_->gs_core_.setGaussInitScale(static_cast<float>(gauss_init_scale_));
            impl_->gs_core_.setDepthScale(static_cast<float>(depth_scale_));
            
            // Configurar parámetros de optimización
            impl_->gs_core_.setPoseIterations(this->get_parameter("pose_iterations").as_int());
            gaussian_iterations_ = this->get_parameter("gaussian_iterations").as_int();
            impl_->gs_core_.setGaussianIterations(gaussian_iterations_);
            impl_->gs_core_.setEtaPose(static_cast<float>(this->get_parameter("eta_pose").as_double()));
            impl_->gs_core_.setEtaGaussian(static_cast<float>(this->get_parameter("eta_gaussian").as_double()));
            impl_->gs_core_.setAdamParameters(
                static_cast<float>(this->get_parameter("adam_eta").as_double()),
                static_cast<float>(this->get_parameter("adam_beta1").as_double()),
                static_cast<float>(this->get_parameter("adam_beta2").as_double()),
                static_cast<float>(this->get_parameter("adam_epsilon").as_double()));
            impl_->gs_core_.setGaussianErrorWeights(
                static_cast<float>(this->get_parameter("w_depth").as_double()),
                static_cast<float>(this->get_parameter("w_dist").as_double()));
            impl_->gs_core_.setCovisibilityThreshold(
                static_cast<float>(this->get_parameter("covisibility_threshold").as_double()));
            // Leer thresholds de transición de submapa y setear en GSSlam
            const double submap_dist_threshold_m = this->get_parameter("submap_dist_threshold_m").as_double();
            const double submap_rot_threshold_deg = this->get_parameter("submap_rot_threshold_deg").as_double();
            impl_->gs_core_.setSubmapTransitionThresholds(
                static_cast<float>(submap_dist_threshold_m),
                static_cast<float>(submap_rot_threshold_deg));
            impl_->gs_core_.setImuRepropagationThresholds(
                this->get_parameter("imu_reprop_ba_thresh").as_double(),
                this->get_parameter("imu_reprop_bg_thresh").as_double());
            impl_->gs_core_.setPoseResidualThresholds(
                static_cast<float>(this->get_parameter("pose_alpha_threshold").as_double()),
                static_cast<float>(this->get_parameter("pose_color_threshold").as_double()),
                static_cast<float>(this->get_parameter("pose_depth_threshold").as_double()));

            this->declare_parameter<int>("loop.self_similarity_percentile", 50);
            this->declare_parameter<int>("loop.min_votes_for_loop_closure", 2);
            this->declare_parameter<int>("loop.loop_min_submap_difference", 1);
            this->declare_parameter<int>("loop.max_descriptor_batch_size", 1024);
            this->declare_parameter<double>("loop.loop_verify_max_distance", 1.0);
            this->declare_parameter<double>("loop.loop_verify_max_rotation", 15.0);
            this->declare_parameter<double>("loop.registration.loop_confidence_threshold", 0.5);
            this->declare_parameter<double>("loop.registration.min_similarity_floor", 0.55);
            this->declare_parameter<int>("loop.registration.max_submaps_to_compare", 10);
            this->declare_parameter<int>("loop.registration.max_keyframes_per_submap", 12);
            this->declare_parameter<int>("loop.registration.frustum_overlap_min_visible_keyframes", 1);
            this->declare_parameter<int>("loop.registration.min_submap_gap", 2);
            this->declare_parameter<double>("loop.registration.imu_max_anchor_distance_m", 8.0);
            this->declare_parameter<double>("loop.registration.imu_max_anchor_rotation_deg", 135.0);
            this->declare_parameter<double>("loop.registration.imu_max_uncertainty_score", 30.0);
            this->declare_parameter<double>("loop.registration.geometric_overlap_threshold", 0.2);
            this->declare_parameter<int>("loop.open3d.max_points", 20000);
            this->declare_parameter<int>("loop.open3d.min_points", 30);
            this->declare_parameter<double>("loop.open3d.voxel_size_m", 0.03);
            this->declare_parameter<int>("loop.open3d.min_downsampled_points", 20);
            this->declare_parameter<double>("loop.open3d.normal_radius_scale", 2.0);
            this->declare_parameter<int>("loop.open3d.normal_max_nn", 30);
            this->declare_parameter<double>("loop.open3d.fpfh_radius_scale", 5.0);
            this->declare_parameter<int>("loop.open3d.fpfh_max_nn", 100);
            this->declare_parameter<double>("loop.open3d.ransac_distance_scale", 1.5);
            this->declare_parameter<int>("loop.open3d.ransac_n", 4);
            this->declare_parameter<int>("loop.open3d.ransac_max_iteration", 4000000);
            this->declare_parameter<double>("loop.open3d.ransac_confidence", 0.999);
            this->declare_parameter<double>("loop.open3d.icp_threshold_scale", 0.8);
            this->declare_parameter<double>("loop.open3d.icp_min_distance_m", 1.5);
            this->declare_parameter<int>("loop.open3d.icp_max_iteration", 30);
            this->declare_parameter<double>("loop.open3d.pgo_max_translation_apply_m", 0.5);
            this->declare_parameter<double>("loop.open3d.pgo_max_rotation_apply_deg", 10.0);
            this->declare_parameter<bool>("loop.apply_pgo_updates", false);
            this->declare_parameter<std::string>("loop.pgo_backend", "ceres");
            this->declare_parameter<bool>("loop.diagnostics_mode", false);
            this->declare_parameter<std::string>("GPU_EXP_EVALUATION", "DEFAULT");

            int loop_self_similarity_percentile = this->get_parameter("loop.self_similarity_percentile").as_int();
            if (loop_self_similarity_percentile <= 0 || loop_self_similarity_percentile > 100) {
                RCLCPP_WARN(this->get_logger(),
                            "Invalid loop.self_similarity_percentile=%d; forcing to 50",
                            loop_self_similarity_percentile);
                loop_self_similarity_percentile = 50;
            }
            int loop_min_votes_for_loop_closure = this->get_parameter("loop.min_votes_for_loop_closure").as_int();
            if (this->get_parameter("loop.diagnostics_mode").as_bool())
            {
                loop_min_votes_for_loop_closure = 1;
            }
            const int loop_min_submap_difference = this->get_parameter("loop.loop_min_submap_difference").as_int();
            const int loop_max_descriptor_batch_size = this->get_parameter("loop.max_descriptor_batch_size").as_int();
            const double loop_verify_max_distance = this->get_parameter("loop.loop_verify_max_distance").as_double();
            const double loop_verify_max_rotation = this->get_parameter("loop.loop_verify_max_rotation").as_double();
            std::string gpu_exp_evaluation = this->get_parameter("GPU_EXP_EVALUATION").as_string();
            std::transform(gpu_exp_evaluation.begin(), gpu_exp_evaluation.end(), gpu_exp_evaluation.begin(),
                           [](unsigned char c) { return static_cast<char>(std::toupper(c)); });
            if (gpu_exp_evaluation != "DEFAULT" && gpu_exp_evaluation != "TAYLOR") {
                RCLCPP_WARN(this->get_logger(),
                            "Invalid GPU_EXP_EVALUATION='%s'. Falling back to 'DEFAULT'",
                            gpu_exp_evaluation.c_str());
                gpu_exp_evaluation = "DEFAULT";
            }

            impl_->gs_core_.setLoopClosureParameters(
                static_cast<float>(loop_self_similarity_percentile),
                loop_min_votes_for_loop_closure,
                loop_min_submap_difference,
                loop_max_descriptor_batch_size);
            impl_->gs_core_.setLoopVerifyThresholds(
                static_cast<float>(loop_verify_max_distance),
                static_cast<float>(loop_verify_max_rotation));

            LoopClosureConfig loop_cfg;
            loop_cfg.loop_confidence_threshold = static_cast<float>(this->get_parameter("loop.registration.loop_confidence_threshold").as_double());
            loop_cfg.min_similarity_floor = static_cast<float>(this->get_parameter("loop.registration.min_similarity_floor").as_double());
            loop_cfg.max_submaps_to_compare = this->get_parameter("loop.registration.max_submaps_to_compare").as_int();
            loop_cfg.max_keyframes_per_submap = this->get_parameter("loop.registration.max_keyframes_per_submap").as_int();
            loop_cfg.frustum_overlap_min_visible_keyframes = static_cast<int>(std::max<int64_t>(1, this->get_parameter("loop.registration.frustum_overlap_min_visible_keyframes").as_int()));
            loop_cfg.min_submap_gap = this->get_parameter("loop.registration.min_submap_gap").as_int();
            loop_cfg.imu_max_anchor_distance_m = static_cast<float>(this->get_parameter("loop.registration.imu_max_anchor_distance_m").as_double());
            loop_cfg.imu_max_anchor_rotation_deg = static_cast<float>(this->get_parameter("loop.registration.imu_max_anchor_rotation_deg").as_double());
            loop_cfg.imu_max_uncertainty_score = static_cast<float>(this->get_parameter("loop.registration.imu_max_uncertainty_score").as_double());
            loop_cfg.geometric_overlap_threshold = static_cast<float>(this->get_parameter("loop.registration.geometric_overlap_threshold").as_double());
            loop_cfg.open3d_max_points = static_cast<size_t>(std::max<int>(1, this->get_parameter("loop.open3d.max_points").as_int()));
            loop_cfg.open3d_min_points = static_cast<size_t>(std::max<int>(1, this->get_parameter("loop.open3d.min_points").as_int()));
            loop_cfg.open3d_voxel_size_m = this->get_parameter("loop.open3d.voxel_size_m").as_double();
            loop_cfg.open3d_min_downsampled_points = static_cast<size_t>(std::max<int>(1, this->get_parameter("loop.open3d.min_downsampled_points").as_int()));
            loop_cfg.open3d_normal_radius_scale = this->get_parameter("loop.open3d.normal_radius_scale").as_double();
            loop_cfg.open3d_normal_max_nn = this->get_parameter("loop.open3d.normal_max_nn").as_int();
            loop_cfg.open3d_fpfh_radius_scale = this->get_parameter("loop.open3d.fpfh_radius_scale").as_double();
            loop_cfg.open3d_fpfh_max_nn = this->get_parameter("loop.open3d.fpfh_max_nn").as_int();
            loop_cfg.open3d_ransac_distance_scale = this->get_parameter("loop.open3d.ransac_distance_scale").as_double();
            loop_cfg.open3d_ransac_n = this->get_parameter("loop.open3d.ransac_n").as_int();
            loop_cfg.open3d_ransac_max_iteration = this->get_parameter("loop.open3d.ransac_max_iteration").as_int();
            loop_cfg.open3d_ransac_confidence = this->get_parameter("loop.open3d.ransac_confidence").as_double();
            loop_cfg.open3d_icp_threshold_scale = this->get_parameter("loop.open3d.icp_threshold_scale").as_double();
            loop_cfg.open3d_icp_min_distance_m = this->get_parameter("loop.open3d.icp_min_distance_m").as_double();
            loop_cfg.open3d_icp_max_iteration = this->get_parameter("loop.open3d.icp_max_iteration").as_int();
            loop_cfg.pgo_max_translation_apply_m = static_cast<float>(this->get_parameter("loop.open3d.pgo_max_translation_apply_m").as_double());
            loop_cfg.pgo_max_rotation_apply_deg = static_cast<float>(this->get_parameter("loop.open3d.pgo_max_rotation_apply_deg").as_double());
            loop_cfg.apply_pgo_updates = this->get_parameter("loop.apply_pgo_updates").as_bool();
            loop_cfg.pgo_backend = this->get_parameter("loop.pgo_backend").as_string();

            impl_->gs_core_.setLoopClosureModuleConfig(loop_cfg);
            impl_->gs_core_.setGpuExpEvaluation(gpu_exp_evaluation);

            const bool loop_diagnostics_mode = this->get_parameter("loop.diagnostics_mode").as_bool();
            impl_->gs_core_.setLoopDiagnosticsMode(loop_diagnostics_mode);

            RCLCPP_INFO(this->get_logger(),
                        "Loop params: p=%d min_votes=%d min_submap_diff=%d batch=%d verify_dist=%.2f verify_rot=%.2f apply_pgo_updates=%s",
                        loop_self_similarity_percentile,
                        loop_min_votes_for_loop_closure,
                        loop_min_submap_difference,
                        loop_max_descriptor_batch_size,
                        loop_verify_max_distance,
                        loop_verify_max_rotation,
                        loop_cfg.apply_pgo_updates ? "true" : "false");
            RCLCPP_INFO(this->get_logger(),
                        "Open3D params: voxel=%.3fm min_points=%zu min_ds=%zu normal_scale=%.2f normal_nn=%d fpfh_scale=%.2f fpfh_nn=%d ransac_scale=%.2f ransac_n=%d ransac_iter=%d ransac_conf=%.3f icp_scale=%.2f icp_min=%.3f icp_iter=%d pgo_max_trans=%.3f pgo_max_rot=%.3f",
                        loop_cfg.open3d_voxel_size_m,
                        loop_cfg.open3d_min_points,
                        loop_cfg.open3d_min_downsampled_points,
                        loop_cfg.open3d_normal_radius_scale,
                        loop_cfg.open3d_normal_max_nn,
                        loop_cfg.open3d_fpfh_radius_scale,
                        loop_cfg.open3d_fpfh_max_nn,
                        loop_cfg.open3d_ransac_distance_scale,
                        loop_cfg.open3d_ransac_n,
                        loop_cfg.open3d_ransac_max_iteration,
                        loop_cfg.open3d_ransac_confidence,
                        loop_cfg.open3d_icp_threshold_scale,
                        loop_cfg.open3d_icp_min_distance_m,
                        loop_cfg.open3d_icp_max_iteration,
                        loop_cfg.pgo_max_translation_apply_m,
                        loop_cfg.pgo_max_rotation_apply_deg);
            RCLCPP_INFO(this->get_logger(),
                        "GPU_EXP_EVALUATION=%s",
                        gpu_exp_evaluation.c_str());

            RCLCPP_INFO(this->get_logger(),
                        "Submap transition thresholds: dist=%.3fm rot=%.2fdeg",
                        impl_->gs_core_.getSubmapDistanceThresholdM(),
                        impl_->gs_core_.getSubmapRotationThresholdDeg());
            const bool use_deriv_filters = this->get_parameter("use_deriv_filters").as_bool();
            impl_->gs_core_.setUseDerivFilters(use_deriv_filters);
            RCLCPP_INFO(this->get_logger(),
                        "Gradient backend: %s",
                        use_deriv_filters ? "DerivFilter (default)" : "Custom Sobel");

            

            std::string slam_profile = this->get_parameter("slam_profile").as_string();
            std::string initialization_strategy = this->get_parameter("initialization_strategy").as_string();
            std::string densification_strategy = this->get_parameter("densification_strategy").as_string();
            std::transform(slam_profile.begin(), slam_profile.end(), slam_profile.begin(),
                           [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
            std::transform(initialization_strategy.begin(), initialization_strategy.end(), initialization_strategy.begin(),
                           [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
            std::transform(densification_strategy.begin(), densification_strategy.end(), densification_strategy.begin(),
                           [](unsigned char c) { return static_cast<char>(std::tolower(c)); });

            const bool frequency_guided_densification =
                this->get_parameter("frequency_guided_densification").as_bool();
            const bool frequency_guided_initialization =
                this->get_parameter("frequency_guided_initialization").as_bool();

            if (slam_profile == "fgs")
            {
                initialization_strategy = "fgs";
                densification_strategy = "fgs";
            }
            else if (slam_profile == "sobel")
            {
                initialization_strategy = "sobel";
                densification_strategy = "sobel";
            }
            else if (slam_profile == "laplacian")
            {
                initialization_strategy = "laplacian";
                densification_strategy = "laplacian";
            }
            else if (slam_profile == "canny")
            {
                initialization_strategy = "canny";
                densification_strategy = "canny";
            }
            else if (slam_profile == "original" || slam_profile == "vigs_fusion")
            {
                initialization_strategy = "vigs_fusion";
                densification_strategy = "vigs_fusion";
            }
            else if (slam_profile == "custom")
            {
                // Keep per-stage strategies as configured.
            }
            else
            {
                RCLCPP_WARN(this->get_logger(),
                            "Invalid slam_profile='%s'. Valid: vigs_fusion, fgs, sobel, laplacian, canny. Falling back to 'vigs_fusion'",
                            slam_profile.c_str());
                slam_profile = "vigs_fusion";
                initialization_strategy = "vigs_fusion";
                densification_strategy = "vigs_fusion";
            }

            if (frequency_guided_initialization)
            {
                initialization_strategy = "fgs";
            }
            if (frequency_guided_densification)
            {
                densification_strategy = "fgs";
            }

            int strategy_sample_high_px = this->get_parameter("strategy_sample_high_px").as_int();
            int strategy_sample_low_px = this->get_parameter("strategy_sample_low_px").as_int();
            double strategy_scale_high = this->get_parameter("strategy_scale_high").as_double();
            double strategy_scale_low = this->get_parameter("strategy_scale_low").as_double();
            double strategy_highpass_sigma_ratio =
                this->get_parameter("strategy_highpass_sigma_ratio").as_double();

            impl_->gs_core_.setInitializationStrategy(initialization_strategy);
            impl_->gs_core_.setDensificationStrategy(densification_strategy);
            impl_->gs_core_.setFrequencyGuidedSampling(
                strategy_sample_high_px,
                strategy_sample_low_px);
            impl_->gs_core_.setFrequencyGuidedScaleFactors(
                static_cast<float>(strategy_scale_high),
                static_cast<float>(strategy_scale_low));
            impl_->gs_core_.setFrequencyGuidedHighpassSigmaRatio(
                static_cast<float>(strategy_highpass_sigma_ratio));

            double canny_threshold1 = this->get_parameter("canny_threshold1").as_double();
            double canny_threshold2 = this->get_parameter("canny_threshold2").as_double();
            impl_->gs_core_.setCannyThresholds(canny_threshold1, canny_threshold2);

            RCLCPP_INFO(this->get_logger(),
                        "SLAM profile=%s | init_strategy=%s densify_strategy=%s high_px=%d low_px=%d high_scale=%.3f low_scale=%.3f sigma_ratio=%.4f",
                        slam_profile.c_str(),
                        impl_->gs_core_.getInitializationStrategyName().c_str(),
                        impl_->gs_core_.getDensificationStrategyName().c_str(),
                        std::max(1, strategy_sample_high_px),
                        std::max(std::max(1, strategy_sample_high_px), strategy_sample_low_px),
                        std::max(1e-5, strategy_scale_high),
                        std::max(1e-5, strategy_scale_low),
                        std::clamp(strategy_highpass_sigma_ratio, 1e-4, 1.0));

            imu_init_samples_ = std::max(
                1,
                static_cast<int>(this->get_parameter("imu_init_samples").as_int()));
            imu_dt_warn_max_s_ = std::max(1e-6, this->get_parameter("imu_dt_warn_max_s").as_double());
            imu_acc_norm_min_ = std::max(0.0, this->get_parameter("imu_acc_norm_min").as_double());
            imu_acc_norm_max_ = std::max(imu_acc_norm_min_, this->get_parameter("imu_acc_norm_max").as_double());
            imu_gyro_norm_max_ = std::max(0.0, this->get_parameter("imu_gyro_norm_max").as_double());
            diag_state_jump_pos_thresh_m_ = std::max(0.0, this->get_parameter("diag_state_jump_pos_thresh_m").as_double());
            diag_state_jump_rot_thresh_deg_ = std::max(0.0, this->get_parameter("diag_state_jump_rot_thresh_deg").as_double());
            diag_proc_time_domain_abs_limit_s_ =
                std::max(1.0, this->get_parameter("diag_proc_time_domain_abs_limit_s").as_double());

            KeyframeSelectionConfig keyframe_selection_config;
            keyframe_selection_config.method = this->get_parameter("keyframe_selection_method").as_string();
            if (keyframe_selection_config.method != "gumbel" &&
                keyframe_selection_config.method != "beta_binomial" &&
                keyframe_selection_config.method != "uniform" &&
                keyframe_selection_config.method != "exponential" &&
                keyframe_selection_config.method != "sliding_window")
            {
                RCLCPP_WARN(this->get_logger(),
                            "Invalid keyframe_selection_method='%s'. Falling back to 'beta_binomial'",
                            keyframe_selection_config.method.c_str());
                keyframe_selection_config.method = "beta_binomial";
            }

            keyframe_selection_config.gumbel_alpha =
                static_cast<float>(this->get_parameter("gumbel_alpha").as_double());
            keyframe_selection_config.gumbel_beta =
                static_cast<float>(this->get_parameter("gumbel_beta").as_double());
            keyframe_selection_config.beta_binomial_alpha =
                static_cast<float>(this->get_parameter("beta_binomial_alpha").as_double());
            keyframe_selection_config.beta_binomial_beta =
                static_cast<float>(this->get_parameter("beta_binomial_beta").as_double());
            keyframe_selection_config.exponential_lambda =
                static_cast<float>(this->get_parameter("exponential_lambda").as_double());
            keyframe_selection_config.sliding_window_window_size =
                this->get_parameter("sliding_window_window_size").as_int();

            if (!std::isfinite(keyframe_selection_config.gumbel_alpha) ||
                keyframe_selection_config.gumbel_alpha <= 0.0f)
            {
                RCLCPP_WARN(this->get_logger(), "Invalid gumbel_alpha. Falling back to 0.7");
                keyframe_selection_config.gumbel_alpha = 0.7f;
            }
            if (!std::isfinite(keyframe_selection_config.gumbel_beta) ||
                keyframe_selection_config.gumbel_beta <= 0.0f)
            {
                RCLCPP_WARN(this->get_logger(), "Invalid gumbel_beta. Falling back to 2.0");
                keyframe_selection_config.gumbel_beta = 2.0f;
            }
            if (!std::isfinite(keyframe_selection_config.beta_binomial_alpha) ||
                keyframe_selection_config.beta_binomial_alpha <= 0.0f)
            {
                RCLCPP_WARN(this->get_logger(), "Invalid beta_binomial_alpha. Falling back to 0.7");
                keyframe_selection_config.beta_binomial_alpha = 0.7f;
            }
            if (!std::isfinite(keyframe_selection_config.beta_binomial_beta) ||
                keyframe_selection_config.beta_binomial_beta <= 0.0f)
            {
                RCLCPP_WARN(this->get_logger(), "Invalid beta_binomial_beta. Falling back to 2.0");
                keyframe_selection_config.beta_binomial_beta = 2.0f;
            }
            if (!std::isfinite(keyframe_selection_config.exponential_lambda) ||
                keyframe_selection_config.exponential_lambda <= 0.0f)
            {
                RCLCPP_WARN(this->get_logger(), "Invalid exponential_lambda. Falling back to 1.0");
                keyframe_selection_config.exponential_lambda = 1.0f;
            }
            if (keyframe_selection_config.sliding_window_window_size == 0 ||
                keyframe_selection_config.sliding_window_window_size < -1)
            {
                RCLCPP_WARN(this->get_logger(),
                            "Invalid sliding_window_window_size=%d. Using -1 (total_kfs/2)",
                            keyframe_selection_config.sliding_window_window_size);
                keyframe_selection_config.sliding_window_window_size = -1;
            }

            impl_->gs_core_.setKeyframeSelectionConfig(keyframe_selection_config);
            RCLCPP_INFO(this->get_logger(),
                        "Keyframe selection: method=%s gumbel(a=%.3f,b=%.3f) beta_binomial(a=%.3f,b=%.3f) exponential(lambda=%.3f) sliding_window(size=%d)",
                        keyframe_selection_config.method.c_str(),
                        keyframe_selection_config.gumbel_alpha,
                        keyframe_selection_config.gumbel_beta,
                        keyframe_selection_config.beta_binomial_alpha,
                        keyframe_selection_config.beta_binomial_beta,
                        keyframe_selection_config.exponential_lambda,
                        keyframe_selection_config.sliding_window_window_size);

            // Subscribimos los nodos a los topicos
            imu_sub_ = this->create_subscription<sensor_msgs::msg::Imu>(
                imu_preint_topic, 1000,
                std::bind(&GSSlamNode::imuCallback, this, std::placeholders::_1));

            imu_preint_sub_ = this->create_subscription<sensor_msgs::msg::Imu>(
                imu_topic, 1000,
                std::bind(&GSSlamNode::imuPreintegratedCallback, this, std::placeholders::_1));

            color_diag_sub_ = this->create_subscription<sensor_msgs::msg::Image>(
                color_topic,
                rclcpp::SensorDataQoS(),
                std::bind(&GSSlamNode::colorDiagCallback, this, std::placeholders::_1));

            depth_diag_sub_ = this->create_subscription<sensor_msgs::msg::Image>(
                depth_topic,
                rclcpp::SensorDataQoS(),
                std::bind(&GSSlamNode::depthDiagCallback, this, std::placeholders::_1));

            RCLCPP_INFO(this->get_logger(),
                        "Diagnostic image subscriptions active: color='%s' depth='%s'",
                        color_topic.c_str(), depth_topic.c_str());


            depth_sub_ = std::make_shared<image_transport::SubscriberFilter>();
            color_sub_ = std::make_shared<image_transport::SubscriberFilter>();
            depth_sub_->subscribe(
                this,
                depth_topic,
                depth_transport,
                rmw_qos_profile_sensor_data
            );
            color_sub_->subscribe(
                this,
                color_topic,
                color_transport,
                rmw_qos_profile_sensor_data
            );

            // Para el algoritmo es necesario que las imagenes de color y profundidad
            // esten sincronizadas
            rgbd_sync_ = std::make_shared<message_filters::Synchronizer<RGBDSyncPolicy>>(
                RGBDSyncPolicy(2), *color_sub_, *depth_sub_);
            rgbd_sync_->registerCallback(
                std::bind(&GSSlamNode::rgbdCallback, this, std::placeholders::_1, std::placeholders::_2));

            camera_info_sub_ = this->create_subscription<sensor_msgs::msg::CameraInfo>(
                camera_info_topic, 1,
                std::bind(&GSSlamNode::cameraInfoCallback, this, std::placeholders::_1));


            // Crear publishers de odometría
            odom_pub_ = this->create_publisher<nav_msgs::msg::Odometry>("odom", 10);
            odom_imu_pub_ = this->create_publisher<nav_msgs::msg::Odometry>("odom_imu", 10);
            
            // Crear publishers para métricas de SLAM
            track_time_pub_ = this->create_publisher<std_msgs::msg::Float32>("slam_metrics/track_time", 10);
            map_time_pub_ = this->create_publisher<std_msgs::msg::Float32>("slam_metrics/map_time", 10);
            
            // Crear publisher de PointCloud2 para visualización de gaussianas
            // Usar QoS RELIABLE + VOLATILE para compatibilidad con RViz2
            if (publish_pointcloud_)
            {
                auto pointcloud_qos = rclcpp::QoS(rclcpp::KeepLast(10));
                pointcloud_qos.reliability(rclcpp::ReliabilityPolicy::Reliable);
                pointcloud_qos.durability(rclcpp::DurabilityPolicy::Volatile);
                pointcloud_pub_ = this->create_publisher<sensor_msgs::msg::PointCloud2>(
                    "gaussian_pointcloud",
                    pointcloud_qos);

                this->declare_parameter<double>("pointcloud_publish_rate_hz", 1.0);
                pointcloud_publish_rate_hz_ = std::max(
                    0.1,
                    this->get_parameter("pointcloud_publish_rate_hz").as_double());

                const auto pointcloud_period = std::chrono::duration_cast<std::chrono::nanoseconds>(
                    std::chrono::duration<double>(1.0 / pointcloud_publish_rate_hz_));
                pointcloud_timer_ = this->create_wall_timer(
                    pointcloud_period,
                    std::bind(&GSSlamNode::pointCloudTimerCallback, this));
            }
            else
            {
                RCLCPP_INFO(this->get_logger(),
                            "PointCloud publication disabled by parameter 'publish_pointcloud'");
            }

            // Publisher para imagen reconstruida
            if (publish_reconstructed_images_)
            {
                reconstructed_image_pub_ =
                    this->create_publisher<sensor_msgs::msg::Image>(
                        "reconstructed_image",
                        rclcpp::SensorDataQoS());

                this->declare_parameter<double>("reconstructed_image_rate_hz", 10.0);
                const double reconstructed_image_rate_hz =
                    this->get_parameter("reconstructed_image_rate_hz").as_double();
                const double safe_rate_hz = std::max(0.1, reconstructed_image_rate_hz);
                const auto period = std::chrono::duration_cast<std::chrono::nanoseconds>(
                    std::chrono::duration<double>(1.0 / safe_rate_hz));
                reconstructed_image_timer_ = this->create_wall_timer(
                    period,
                    std::bind(&GSSlamNode::reconstructedImageTimerCallback, this));
            }
            else
            {
                RCLCPP_INFO(this->get_logger(),
                            "Reconstructed image publication disabled by parameter 'publish_reconstructed_images'");
            }

            
            
            // Configurar frame_id de mensajes de odometría (publicar en la frame de estimación)
            odom_msg_.header.frame_id = publish_frame_id_;
            odom_imu_msg_.header.frame_id = publish_frame_id_;

            RCLCPP_INFO(this->get_logger(), "Odometry publishers created on topics 'odom' and 'odom_imu'");
            if (publish_pointcloud_)
            {
                RCLCPP_INFO(this->get_logger(), "PointCloud2 publisher created on topic 'gaussian_pointcloud'");
            }
            if (publish_reconstructed_images_)
            {
                RCLCPP_INFO(this->get_logger(), "Reconstructed image publisher created on topic 'reconstructed_image'");
            }

            if (enable_viewer)
            {
                impl_->viewer_ = std::make_shared<GaussianSplattingViewer>(impl_->gs_core_, visualize_current_submap_);
                impl_->viewer_->startThread();
                RCLCPP_INFO(this->get_logger(), "Gaussian viewer enabled (parameter 'viewer'=true)");
            }
            else
            {
                RCLCPP_INFO(this->get_logger(), "Gaussian viewer disabled (parameter 'viewer'=false)");
            }

            RCLCPP_INFO(this->get_logger(), "--------------------------------------------------");
            RCLCPP_INFO(this->get_logger(), "GS SLAM NODE finished initializing");
            RCLCPP_INFO(this->get_logger(), "--------------------------------------------------");
        }

    GSSlamNode::~GSSlamNode()
    {
    }

    // Callbacks
    Eigen::Matrix3d GSSlamNode::computeGravityAlignment(const Eigen::Vector3d& acc) const
    {
        if (acc.norm() < 1e-6) {
            return Eigen::Matrix3d::Identity();
        }
        const Eigen::Vector3d ng1 = acc.normalized();
        const Eigen::Vector3d ng2(0.0, 0.0, 1.0);

        Eigen::Matrix3d R0 = Eigen::Quaterniond::FromTwoVectors(ng1, ng2).toRotationMatrix();

        const Eigen::Vector3d n = R0.col(0);
        const double yaw = std::atan2(n.y(), n.x());
        const double cy = std::cos(-yaw);
        const double sy = std::sin(-yaw);
        Eigen::Matrix3d Rz;
        Rz << cy, -sy, 0.0,
              sy,  cy, 0.0,
              0.0, 0.0, 1.0;

        return Rz * R0;
    }

    void GSSlamNode::cameraInfoCallback(const sensor_msgs::msg::CameraInfo::ConstSharedPtr msg)
    {
    // Intrínsecas
    // [fx 0 cx]
    // [0 fy cy]
    // [0 0 1]
    IntrinsicParameters base_intrinsics;
    base_intrinsics.f = float2{static_cast<float>(msg->k[0]), static_cast<float>(msg->k[4])};
    base_intrinsics.c = float2{static_cast<float>(msg->k[2]), static_cast<float>(msg->k[5])};

    IntrinsicParameters scaled_intrinsics = base_intrinsics;
    const float ds = static_cast<float>(downsample_factor_);
    scaled_intrinsics.f.x /= ds;
    scaled_intrinsics.f.y /= ds;
    scaled_intrinsics.c.x /= ds;
    scaled_intrinsics.c.y /= ds;
    impl_->intrinsics = scaled_intrinsics;
    impl_->gs_core_.setIntrinsics(impl_->intrinsics);

    if (!hasIntrinsics){
        RCLCPP_INFO(this->get_logger(), "[CAM] Intrinsics loaded: fx=%.3f fy=%.3f cx=%.3f cy=%.3f", 
        scaled_intrinsics.f.x, scaled_intrinsics.f.y, scaled_intrinsics.c.x, scaled_intrinsics.c.y);

        hasIntrinsics = true;
    }
    
    // Extrínsecas
    // Traslación t y rotación q
    // Expresadas desde el marco desde la cámara rgb al marco de la IMU (cam -> imu)
    const std::string cam_frame = normalizeFrameId(msg->header.frame_id);
    const std::string& imu_frame = imu_frame_id_;

    camera_frame_id_ = cam_frame;

    if (imu_frame.empty())
    {
        RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 1000,
                             "[EXTR] IMU frame_id not set yet, cannot load extrinsics.");
        return;
    }

    try
    {
        const auto tf = tf_buffer_->lookupTransform(
            imu_frame, cam_frame, tf2::TimePointZero);

        tf2::fromMsg(tf.transform.rotation, q_imu_cam_);
        tf2::fromMsg(tf.transform.translation, t_imu_cam_);

        // Set core extrinsics (Eigen) as before
        impl_->gs_core_.setCamToImuExtrinsics(t_imu_cam_, q_imu_cam_);

        // Also cache as Pose in Impl for fast reuse in the node
        impl_->T_imu_cam_pose_.position = make_float3(static_cast<float>(t_imu_cam_.x()),
                                                     static_cast<float>(t_imu_cam_.y()),
                                                     static_cast<float>(t_imu_cam_.z()));
        impl_->T_imu_cam_pose_.orientation = make_float4(static_cast<float>(q_imu_cam_.x()),
                                                         static_cast<float>(q_imu_cam_.y()),
                                                         static_cast<float>(q_imu_cam_.z()),
                                                         static_cast<float>(q_imu_cam_.w()));
        impl_->T_cam_imu_pose_ = invertPose(impl_->T_imu_cam_pose_);

        if (!hasExtrinsics){
            const Eigen::Quaterniond q_imu_cam(q_imu_cam_);
            const Eigen::Matrix3d R_imu_cam = q_imu_cam.normalized().toRotationMatrix();
            RCLCPP_INFO(this->get_logger(),
                        "[EXTR] Loaded T_imu_cam | t=[%.3f %.3f %.3f] q=[%.3f %.3f %.3f %.3f] (%s <- %s)",
                        t_imu_cam_.x(), t_imu_cam_.y(), t_imu_cam_.z(),
                        q_imu_cam_.x(), q_imu_cam_.y(), q_imu_cam_.z(), q_imu_cam_.w(),
                        imu_frame.c_str(), cam_frame.c_str());
            RCLCPP_INFO(this->get_logger(),
                        "[EXTR] T_imu_cam rotation matrix=%s | |t|=%.4f | |q|=%.6f",
                        formatMatrix3(R_imu_cam).c_str(),
                        t_imu_cam_.norm(),
                        q_imu_cam.norm());
            hasExtrinsics = true;
        }
    }
    catch (const tf2::TransformException& ex)
    {
        RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
                             "[EXTR] TF unavailable (%s <- %s): %s",
                             imu_frame.c_str(), cam_frame.c_str(), ex.what());

        return;
    }

    hasCameraInfo = true;
    }

    void GSSlamNode::imuCallback(const sensor_msgs::msg::Imu::SharedPtr msg)
    {
        //last_imu_callback_time_ = msg->header.stamp;
        imu_frame_id_ = normalizeFrameId(msg->header.frame_id);
        if (imu_frame_id_.empty() && !imu_frame_id_fallback_.empty()) {
            imu_frame_id_ = normalizeFrameId(imu_frame_id_fallback_);
        }
        auto stamp_imu = rclcpp::Time(msg->header.stamp, RCL_ROS_TIME);

        if (!hasCameraInfo) {
            RCLCPP_INFO_ONCE(this->get_logger(), "Waiting for camera info and TF to be available before processing IMU");
            return;
        }

        // Lectura de aceleracion y velocidad angular
        Eigen::Vector3d raw_acc(msg->linear_acceleration.x,
                                msg->linear_acceleration.y,
                                msg->linear_acceleration.z);

        Eigen::Vector3d raw_gyro(msg->angular_velocity.x,
                                 msg->angular_velocity.y,
                                 msg->angular_velocity.z);


        // Acumulamos las primeras N muestras para estimar sesgos 
        if (nb_init_imu_ < imu_init_samples_) {
            avg_acc_ += raw_acc;
            avg_gyro_ += raw_gyro;
            nb_init_imu_++;
            last_imu_stamp_ = stamp_imu;
        }

        double dt = (stamp_imu - last_imu_stamp_).seconds();
        last_imu_stamp_ = stamp_imu;

        if (dt > imu_dt_warn_max_s_) {
            RCLCPP_WARN(this->get_logger(),
                        "Large dt between IMU messages: %.3f s.",
                        dt);
        }

        // Actualizamos datos IMU actuales
        impl_->imu_data_.Acc = raw_acc;
        impl_->imu_data_.Gyro = raw_gyro;

        if (nb_init_imu_ >= imu_init_samples_ && impl_->gs_core_.getIsInitialized())
        {
            // Inicialización del sistema:
            // Estimamos la orientación inicial en el periodo de inmovilidad del robot
            // Entonces la unica aceleracion medida deberia ser la gravedad
            if (!imuInitialized) {
                avg_acc_ /= static_cast<double>(nb_init_imu_);
                avg_gyro_ /= static_cast<double>(nb_init_imu_);

                Eigen::Matrix3d R0 = computeGravityAlignment(avg_acc_);
                Eigen::Vector3d gravity(0.0, 0.0, 9.81);
                const Eigen::Vector3d aligned_acc = R0 * avg_acc_;
                const Eigen::Vector3d expected_acc = R0.inverse() * gravity;
                const double acc_norm = avg_acc_.norm();
                const double aligned_norm = aligned_acc.norm();
                const double gravity_error_norm = (avg_acc_ - expected_acc).norm();
                const double cos_to_z = std::clamp(aligned_acc.normalized().dot(Eigen::Vector3d::UnitZ()), -1.0, 1.0);
                const double gravity_alignment_deg = std::acos(cos_to_z) * 180.0 / M_PI;

                // Estimacion del bias de aceleracion usando la gravedad
                acc_bias_ = avg_acc_ - R0.inverse() * gravity;

                // Expresamos la pose de los objetos en el estandar (cuerpo -> mundo)

                // Pose inicial IMU: traslación cero + orientación inicial
                Eigen::Quaterniond q_init_imu(R0);
                
                impl_->odom_pose_init_imu_.position = make_float3(0.0f, 0.0f, 0.0f);
                impl_->odom_pose_init_imu_.orientation.x = static_cast<float>(q_init_imu.x());
                impl_->odom_pose_init_imu_.orientation.y = static_cast<float>(q_init_imu.y());
                impl_->odom_pose_init_imu_.orientation.z = static_cast<float>(q_init_imu.z());
                impl_->odom_pose_init_imu_.orientation.w = static_cast<float>(q_init_imu.w());

                // Pose inicial cámara: aplicar extrínsecos (tenemos los de cam a imu)
                Eigen::Quaterniond q_init_cam = q_init_imu * q_imu_cam_; // q_w_cam = q_w_imu * q_imu_cam
                Eigen::Vector3d t_init_cam = q_init_imu * t_imu_cam_; // t_w_cam = q_w_imu * t_imu_cam + t_w_imu (t_w_imu=0)

                impl_->odom_pose_init_cam_.position.x = static_cast<float>(t_init_cam.x());
                impl_->odom_pose_init_cam_.position.y = static_cast<float>(t_init_cam.y());
                impl_->odom_pose_init_cam_.position.z = static_cast<float>(t_init_cam.z());
                impl_->odom_pose_init_cam_.orientation.x = static_cast<float>(q_init_cam.x());
                impl_->odom_pose_init_cam_.orientation.y = static_cast<float>(q_init_cam.y());
                impl_->odom_pose_init_cam_.orientation.z = static_cast<float>(q_init_cam.z());
                impl_->odom_pose_init_cam_.orientation.w = static_cast<float>(q_init_cam.w());

                // Inicializar preintegración en GSSlam y en el nodo
                impl_->gs_core_.initialize(impl_->odom_pose_init_imu_, impl_->odom_pose_init_cam_);
                impl_->gs_core_.setImuBias(acc_bias_, avg_gyro_);

                impl_->preint_.init(impl_->imu_data_.Acc, impl_->imu_data_.Gyro, acc_bias_, avg_gyro_,
                             impl_->imu_data_.acc_n, impl_->imu_data_.gyr_n, impl_->imu_data_.acc_w, impl_->imu_data_.gyr_w);

                imuInitialized = true;

                // --------------------------------------------------------------------------------------
                // DIAGNOSTICOS 
                RCLCPP_INFO(this->get_logger(), "IMU initialized:");
                RCLCPP_INFO(this->get_logger(), "  avg_acc = [%.4f %.4f %.4f] m/s²",
                            avg_acc_.x(), avg_acc_.y(), avg_acc_.z());
                RCLCPP_INFO(this->get_logger(), "  avg_acc_norm = %.4f m/s² | aligned_acc = %s | aligned_norm = %.4f",
                            acc_norm,
                            formatVector3(aligned_acc).c_str(),
                            aligned_norm);
                RCLCPP_INFO(this->get_logger(), "  gravity_alignment_error = %.6f m/s² | gravity_alignment_deg = %.3f deg",
                            gravity_error_norm,
                            gravity_alignment_deg);
                RCLCPP_INFO(this->get_logger(), "  R0(gravity align) = %s", formatMatrix3(R0).c_str());
                RCLCPP_INFO(this->get_logger(), "  q_init_imu = %s", formatQuaternion(q_init_imu).c_str());
                RCLCPP_INFO(this->get_logger(), "  q_init_cam = %s", formatQuaternion(q_init_cam).c_str());
                RCLCPP_INFO(this->get_logger(), "  t_init_cam = %s | |t_init_cam|=%.4f", formatVector3(t_init_cam).c_str(), t_init_cam.norm());
                RCLCPP_INFO(this->get_logger(), "  bias_acc = [%.4f %.4f %.4f] m/s²",
                            acc_bias_.x(), acc_bias_.y(), acc_bias_.z());
                RCLCPP_INFO(this->get_logger(), "  avg_gyro = [%.4f %.4f %.4f] rad/s",
                            avg_gyro_.x(), avg_gyro_.y(), avg_gyro_.z());
                RCLCPP_INFO(this->get_logger(), "  Preintegration initialized in GSSlam core");

                RCLCPP_INFO(this->get_logger(),
                            "  frame_ids: imu='%s' cam='%s' | q_imu_cam_norm=%.6f",
                            imu_frame_id_.c_str(),
                            camera_frame_id_.c_str(),
                            q_imu_cam_.norm());

                // --------------------------------------------------------------------------------------
            }
            else
            {

                impl_->preint_.add_imu(dt, impl_->imu_data_.Acc, impl_->imu_data_.Gyro);

                const double *pose = impl_->gs_core_.getImuPose();
                const double *velocity = impl_->gs_core_.getImuVelocity();

                if (pose && velocity)
                {
                    Eigen::Map<const Eigen::Vector3d> P(pose);
                    Eigen::Map<const Eigen::Quaterniond> Q(pose + 3);
                    Eigen::Map<const Eigen::Vector3d> V(velocity);

                    Eigen::Vector3d pos_imu, vel_imu;
                    Eigen::Quaterniond rot_imu;

                    impl_->preint_.predict(P, Q, V, pos_imu, rot_imu, vel_imu);

                    odom_imu_msg_.header.stamp = msg->header.stamp;
                    odom_imu_msg_.child_frame_id = msg->header.frame_id;

                    odom_imu_msg_.pose.pose.position.x = pos_imu.x();
                    odom_imu_msg_.pose.pose.position.y = pos_imu.y();
                    odom_imu_msg_.pose.pose.position.z = pos_imu.z();

                    odom_imu_msg_.pose.pose.orientation.x = rot_imu.x();
                    odom_imu_msg_.pose.pose.orientation.y = rot_imu.y();
                    odom_imu_msg_.pose.pose.orientation.z = rot_imu.z();
                    odom_imu_msg_.pose.pose.orientation.w = rot_imu.w();

                    odom_imu_msg_.twist.twist.linear.x = vel_imu.x();
                    odom_imu_msg_.twist.twist.linear.y = vel_imu.y();
                    odom_imu_msg_.twist.twist.linear.z = vel_imu.z();

                    odom_imu_pub_->publish(odom_imu_msg_);
                }

                sensor_msgs::msg::Imu::SharedPtr imu_cache(new sensor_msgs::msg::Imu);
                imu_cache->header = msg->header;
                imu_cache->angular_velocity.x = impl_->imu_data_.Gyro.x();
                imu_cache->angular_velocity.y = impl_->imu_data_.Gyro.y();
                imu_cache->angular_velocity.z = impl_->imu_data_.Gyro.z();
                imu_cache->linear_acceleration.x = impl_->imu_data_.Acc.x();
                imu_cache->linear_acceleration.y = impl_->imu_data_.Acc.y();
                imu_cache->linear_acceleration.z = impl_->imu_data_.Acc.z();

                imu_cache_preint_.add(imu_cache);

                syncRgbdImu();
            }
        }
    }

    void GSSlamNode::imuPreintegratedCallback(const sensor_msgs::msg::Imu::SharedPtr msg)
    {
        RCLCPP_DEBUG(this->get_logger(), "IMU preintegrated received: frame_id=%s ts=%u.%u",
                     msg->header.frame_id.c_str(), msg->header.stamp.sec, msg->header.stamp.nanosec);
    }

    void GSSlamNode::colorDiagCallback(const sensor_msgs::msg::Image::SharedPtr msg)
    {
        last_color_diag_time_ = this->now();
        RCLCPP_INFO_ONCE(this->get_logger(),
                         "[DIAG color] first image received: frame_id='%s' stamp=%u.%u size=%ux%u encoding='%s' step=%u",
                         msg->header.frame_id.c_str(),
                         msg->header.stamp.sec,
                         msg->header.stamp.nanosec,
                         msg->width,
                         msg->height,
                         msg->encoding.c_str(),
                         msg->step);
    }

    void GSSlamNode::depthDiagCallback(const sensor_msgs::msg::Image::SharedPtr msg)
    {
        last_depth_diag_time_ = this->now();
        RCLCPP_INFO_ONCE(this->get_logger(),
                         "[DIAG depth] first image received: frame_id='%s' stamp=%u.%u size=%ux%u encoding='%s' step=%u",
                         msg->header.frame_id.c_str(),
                         msg->header.stamp.sec,
                         msg->header.stamp.nanosec,
                         msg->width,
                         msg->height,
                         msg->encoding.c_str(),
                         msg->step);
    }


    void GSSlamNode::rgbdCallback(const std::shared_ptr<const sensor_msgs::msg::Image>& color,
                                  const std::shared_ptr<const sensor_msgs::msg::Image>& depth)
    {
        const bool ready_for_rgbd = test_gaussians_only_
            ? (hasIntrinsics && impl_->gs_core_.getIsInitialized())
            : (hasCameraInfo && impl_->gs_core_.getIsInitialized() && imuInitialized);

        if (!ready_for_rgbd) {
            RCLCPP_DEBUG_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
                                  "Skipping RGBD callback: hasIntrinsics=%d hasCameraInfo=%d imuInitialized=%d GSCoreInitialized=%d test_gaussians_only=%d",
                                  static_cast<int>(hasIntrinsics),
                                  static_cast<int>(hasCameraInfo),
                                  static_cast<int>(imuInitialized),
                                  static_cast<int>(impl_->gs_core_.getIsInitialized()),
                                  static_cast<int>(test_gaussians_only_));
            return;
        }

        if (color->width == 0 || color->height == 0 || depth->width == 0 || depth->height == 0)
        {
            RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
                                 "Skipping RGBD callback due invalid header dimensions: color=%ux%u depth=%ux%u",
                                 color->width, color->height, depth->width, depth->height);
            return;
        }

        if (color->width != depth->width || color->height != depth->height)
        {
            RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
                                 "RGBD callback size mismatch in headers: color=%ux%u depth=%ux%u",
                                 color->width, color->height, depth->width, depth->height);
        }

        // Guardamos tiempos de llegada
        const rclcpp::Time rgb_stamp(color->header.stamp);
        last_rgbd_callback_time_ = this->now();

        RCLCPP_INFO_ONCE(this->get_logger(),
                         "[RGBD sync] first synchronized pair received: color frame_id='%s' stamp=%u.%u size=%ux%u encoding='%s' | depth frame_id='%s' stamp=%u.%u size=%ux%u encoding='%s'",
                         color->header.frame_id.c_str(),
                         color->header.stamp.sec,
                         color->header.stamp.nanosec,
                         color->width,
                         color->height,
                         color->encoding.c_str(),
                         depth->header.frame_id.c_str(),
                         depth->header.stamp.sec,
                         depth->header.stamp.nanosec,
                         depth->width,
                         depth->height,
                         depth->encoding.c_str());

        RCLCPP_DEBUG_THROTTLE(this->get_logger(), *this->get_clock(), 1000,
            "RGBD received: color %ux%u | depth %ux%u, stamp=%ld",
            color->width, color->height,
            depth->width, depth->height,
            rgb_stamp.nanoseconds());

        // Guardamos las imagenes recibidas
        try
        {
            RCLCPP_INFO_ONCE(this->get_logger(),
                             "[RGBD sync] converting color encoding '%s' to bgr8 and depth encoding '%s'",
                             color->encoding.c_str(),
                             depth->encoding.c_str());

            if (depth->encoding == sensor_msgs::image_encodings::TYPE_16UC1)
            {
                cv::Mat depth_mm = cv_bridge::toCvShare(
                    depth, sensor_msgs::image_encodings::TYPE_16UC1)->image;
                depth_mm.convertTo(depthImg, CV_32FC1, 0.001);
            }
            else if (depth->encoding == sensor_msgs::image_encodings::TYPE_32FC1)
            {
                depthImg = cv_bridge::toCvShare(
                    depth, sensor_msgs::image_encodings::TYPE_32FC1)->image;
            }
            else
            {
                RCLCPP_ERROR(this->get_logger(),
                             "unsupported depth encoding '%s' (expected 16UC1 or 32FC1)",
                             depth->encoding.c_str());
                return;
            }
        }
        catch (cv_bridge::Exception &e)
        {
            RCLCPP_ERROR(this->get_logger(),
                         "could not convert depth image with encoding '%s'", depth->encoding.c_str());
            return;
        }

        try
        {
            rgbImg = cv_bridge::toCvShare(color, "bgr8")->image;
        }
        catch (cv_bridge::Exception &e)
        {
            RCLCPP_ERROR(this->get_logger(),
                         "could not convert color image with encoding '%s'.", color->encoding.c_str());
                return;
        }

        if (rgbImg.empty() || depthImg.empty())
        {
            RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
                                 "Skipping RGBD callback due empty OpenCV image after conversion");
            return;
        }

        if (rgbImg.cols != depthImg.cols || rgbImg.rows != depthImg.rows)
        {
            RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
                                 "RGBD callback OpenCV size mismatch: color=%dx%d depth=%dx%d",
                                 rgbImg.cols, rgbImg.rows, depthImg.cols, depthImg.rows);
        }

        cv::Mat proc_rgb = rgbImg;
        cv::Mat proc_depth = depthImg;
        if (downsample_factor_ > 1)
        {
            proc_rgb = rgbImg.clone();
            proc_depth = depthImg.clone();
            int ds = downsample_factor_;
            while (ds > 1)
            {
                cv::pyrDown(proc_rgb, proc_rgb);
                cv::pyrDown(proc_depth, proc_depth);
                ds /= 2;
            }
        }

        if (proc_rgb.cols != proc_depth.cols || proc_rgb.rows != proc_depth.rows)
        {
            RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
                                 "RGBD callback processed size mismatch: color=%dx%d depth=%dx%d",
                                 proc_rgb.cols, proc_rgb.rows, proc_depth.cols, proc_depth.rows);
        }

        last_rgb_stamp_ = rgb_stamp;
        last_rgb_msg_ = color;
        local_depth_img = proc_depth.clone();
        local_rgb_img = proc_rgb.clone();

        syncRgbdImu();
    }

    void GSSlamNode::syncRgbdImu()
    {
        RCLCPP_DEBUG(this->get_logger(), "syncRgbdImu called");
        const auto t_sync_start = std::chrono::steady_clock::now();

        if (isProcessing){
            return;
        }

        if (last_rgb_stamp_ <= last_processed_rgbd_stamp_) {
            return;
        }

        if (test_gaussians_only_)
        {
            isProcessing = true;
            last_processed_rgbd_stamp_ = last_rgb_stamp_;

            const auto t_compute_start = std::chrono::steady_clock::now();

            try
            {
                impl_->gs_core_.testGaussians(local_rgb_img, local_depth_img);
            }
            catch (const std::exception &e)
            {
                RCLCPP_ERROR(this->get_logger(), "Exception in GSCore testGaussians: %s", e.what());
                isProcessing = false;
                return;
            }

            // In test_gaussians_only_ mode we must keep the published pose fixed (identity).
            odom_msg_.header.stamp = last_rgb_stamp_;
            odom_msg_.child_frame_id = imu_frame_id_.empty() ? camera_frame_id_ : imu_frame_id_;

            // Identity pose: position = 0, orientation = (0,0,0,1)
            odom_msg_.pose.pose.position.x = 0.0;
            odom_msg_.pose.pose.position.y = 0.0;
            odom_msg_.pose.pose.position.z = 0.0;
            odom_msg_.pose.pose.orientation.x = 0.0;
            odom_msg_.pose.pose.orientation.y = 0.0;
            odom_msg_.pose.pose.orientation.z = 0.0;
            odom_msg_.pose.pose.orientation.w = 1.0;

            // Zero velocities in test mode
            odom_msg_.twist.twist.linear.x = 0.0;
            odom_msg_.twist.twist.linear.y = 0.0;
            odom_msg_.twist.twist.linear.z = 0.0;
            odom_msg_.twist.twist.angular.x = 0.0;
            odom_msg_.twist.twist.angular.y = 0.0;
            odom_msg_.twist.twist.angular.z = 0.0;

            RCLCPP_DEBUG(this->get_logger(), "test_gaussians_only_: publishing identity odom");
            odom_pub_->publish(odom_msg_);

            isProcessing = false;
            return;
        }

        rclcpp::Time last_imu_time(imu_cache_preint_.getLatestTime(), RCL_ROS_TIME);
        if (last_imu_time < last_rgb_stamp_) {
            return;
        }

        auto imu_interval = imu_cache_preint_.getInterval(last_processed_rgbd_stamp_, last_rgb_stamp_);

        if (imu_interval.empty()) {
            return;
        }

        // ==================================================================================
       
        // Integramos la IMU

        const auto t_imu_integration_start = std::chrono::steady_clock::now();

        for (const auto &m : imu_interval)
        {
            ImuData imu_data = impl_->imu_data_;

            imu_data.Acc = Eigen::Vector3d(m->linear_acceleration.x,
                                           m->linear_acceleration.y,
                                           m->linear_acceleration.z);

            imu_data.Gyro = Eigen::Vector3d(m->angular_velocity.x,
                                            m->angular_velocity.y,
                                            m->angular_velocity.z);

            const double t = rclcpp::Time(m->header.stamp, RCL_ROS_TIME).seconds();
            impl_->gs_core_.processImu(t, imu_data);
        }

        const auto t_imu_integration_end = std::chrono::steady_clock::now();

        // ==================================================================================

        last_processed_rgbd_stamp_ = last_rgb_stamp_;

        // Usamos el core para procesar y optimizar (usando su metodo compute)

        isProcessing = true;

        const auto t_compute_start = std::chrono::steady_clock::now();

        try{
            impl_->gs_core_.compute(local_rgb_img, local_depth_img);
        }
        catch (const std::exception& e)
        {            
            RCLCPP_ERROR(this->get_logger(), "Exception in GSCore compute: %s", e.what());
            isProcessing = false;
            return;
        }

        const auto t_compute_end = std::chrono::steady_clock::now();
        

        // ===================================================================================

        // Obtenemos el estado IMU actualizado del core para publicar la odometría

        const double *velocity = impl_->gs_core_.getImuVelocity();

        // Use the canonical global camera pose from the core.
        const Submap *current_submap = impl_->gs_core_.getCurrentSubmap();
        const Pose cam_pose_global = impl_->gs_core_.getCameraPose();
        if (!current_submap)
        {
            RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
                "[ODOM] No current submap. Publishing pose as identity.");
        }
        else if (!current_submap->pose_cache_valid)
        {
            RCLCPP_WARN(this->get_logger(),
                "[ODOM] CRITICAL: Current submap cache is INVALID. "
                "getGlobalPose() will return Identity. odometry will be incorrect!");
        }

        // Publish IMU pose (world->imu) as odom to match VIGS-Fusion
        odom_msg_.header.stamp = last_rgb_stamp_;
        odom_msg_.child_frame_id = imu_frame_id_.empty() ? camera_frame_id_ : imu_frame_id_;

        // The TF lookup returned T_imu_cam (cam -> imu). We need T_cam_imu when composing
        // world IMU pose from camera global pose: T_world_imu = T_world_cam * T_cam_imu
        const Pose imu_pose_global = composePoses(cam_pose_global, impl_->T_cam_imu_pose_);

        odom_msg_.pose.pose.position.x = imu_pose_global.position.x;
        odom_msg_.pose.pose.position.y = imu_pose_global.position.y;
        odom_msg_.pose.pose.position.z = imu_pose_global.position.z;

        // Normalize quaternion if necessary
        const float qn_cam = std::sqrt(
            imu_pose_global.orientation.x * imu_pose_global.orientation.x +
            imu_pose_global.orientation.y * imu_pose_global.orientation.y +
            imu_pose_global.orientation.z * imu_pose_global.orientation.z +
            imu_pose_global.orientation.w * imu_pose_global.orientation.w);
        const float inv_qn_cam = (qn_cam > 1e-12f) ? (1.0f / qn_cam) : 1.0f;

        odom_msg_.pose.pose.orientation.x = imu_pose_global.orientation.x * inv_qn_cam;
        odom_msg_.pose.pose.orientation.y = imu_pose_global.orientation.y * inv_qn_cam;
        odom_msg_.pose.pose.orientation.z = imu_pose_global.orientation.z * inv_qn_cam;
        odom_msg_.pose.pose.orientation.w = imu_pose_global.orientation.w * inv_qn_cam;

        if (velocity)
        {
            Eigen::Map<const Eigen::Vector3d> V(velocity);
            Eigen::Map<const Eigen::Vector3d> R(velocity + 3);

            odom_msg_.twist.twist.linear.x = V.x();
            odom_msg_.twist.twist.linear.y = V.y();
            odom_msg_.twist.twist.linear.z = V.z();

            odom_msg_.twist.twist.angular.x = R.x();
            odom_msg_.twist.twist.angular.y = R.y();
            odom_msg_.twist.twist.angular.z = R.z();
        }
        else
        {
            RCLCPP_WARN(this->get_logger(), "No valid pose/velocity from GSCore to publish odometry");
        }

        odom_pub_->publish(odom_msg_);

        // Normalize quaternion
        const float qn_imu = std::sqrt(
            imu_pose_global.orientation.x * imu_pose_global.orientation.x +
            imu_pose_global.orientation.y * imu_pose_global.orientation.y +
            imu_pose_global.orientation.z * imu_pose_global.orientation.z +
            imu_pose_global.orientation.w * imu_pose_global.orientation.w);
        const float inv_qn_imu = (qn_imu > 1e-12f) ? (1.0f / qn_imu) : 1.0f;

        odom_imu_msg_.header.stamp = last_rgb_stamp_;
        odom_imu_msg_.child_frame_id = imu_frame_id_;

        odom_imu_msg_.pose.pose.position.x = imu_pose_global.position.x;
        odom_imu_msg_.pose.pose.position.y = imu_pose_global.position.y;
        odom_imu_msg_.pose.pose.position.z = imu_pose_global.position.z;

        odom_imu_msg_.pose.pose.orientation.x = imu_pose_global.orientation.x * inv_qn_imu;
        odom_imu_msg_.pose.pose.orientation.y = imu_pose_global.orientation.y * inv_qn_imu;
        odom_imu_msg_.pose.pose.orientation.z = imu_pose_global.orientation.z * inv_qn_imu;
        odom_imu_msg_.pose.pose.orientation.w = imu_pose_global.orientation.w * inv_qn_imu;

        odom_imu_pub_->publish(odom_imu_msg_);

        // ===================================================================================
        // Publicar métricas de tiempo de tracking y mapping
        // ===================================================================================
        
        const auto& cpu_timings = impl_->gs_core_.getLastCPUTimings();
        
        std_msgs::msg::Float32 track_time_msg;
        track_time_msg.data = cpu_timings.pose_track_ms;
        track_time_pub_->publish(track_time_msg);
        last_track_time_ms_ = cpu_timings.pose_track_ms;
        
        std_msgs::msg::Float32 map_time_msg;
        map_time_msg.data = cpu_timings.map_ops_ms;
        map_time_pub_->publish(map_time_msg);
        last_map_time_ms_ = cpu_timings.map_ops_ms;
        
        // ===== ACUMULAR MÉTRICAS EN BUFFERS =====
        track_time_buffer_.push_back(cpu_timings.pose_track_ms);
        if (track_time_buffer_.size() > METRICS_BUFFER_SIZE) {
            track_time_buffer_.pop_front();
        }
        
        map_time_buffer_.push_back(cpu_timings.map_ops_ms);
        if (map_time_buffer_.size() > METRICS_BUFFER_SIZE) {
            map_time_buffer_.pop_front();
        }
        
        frame_count_++;
        
        // ===== IMPRIMIR MÉTRICAS EN TERMINAL =====
        // Cada metrics_print_interval_frames_ frames mostrar resumen
        if (frame_count_ % static_cast<uint64_t>(metrics_print_interval_frames_) == 0) {
            double avg_track = 0.0, avg_map = 0.0;
            double min_track = 1e6, max_track = -1e6;
            double min_map = 1e6, max_map = -1e6;
            
            for (double t : track_time_buffer_) {
                avg_track += t;
                min_track = std::min(min_track, t);
                max_track = std::max(max_track, t);
            }
            avg_track /= std::max(1UL, track_time_buffer_.size());
            
            for (double m : map_time_buffer_) {
                avg_map += m;
                min_map = std::min(min_map, m);
                max_map = std::max(max_map, m);
            }
            avg_map /= std::max(1UL, map_time_buffer_.size());
            
            RCLCPP_INFO(this->get_logger(),
                "█ Frame %lu │ Track/Img: %.2f ms (avg: %.2f, min: %.2f, max: %.2f) │ Map/Img: %.2f ms (avg: %.2f, min: %.2f, max: %.2f)",
                frame_count_,
                cpu_timings.pose_track_ms, avg_track, min_track, max_track,
                cpu_timings.map_ops_ms, avg_map, min_map, max_map);
        } else {
            // En frames intermedios mostrar solo línea corta
            RCLCPP_DEBUG(this->get_logger(),
                "Frame %lu: Track/Img=%.2f ms, Map/Img=%.2f ms",
                frame_count_,
                cpu_timings.pose_track_ms,
                cpu_timings.map_ops_ms);
        }

        // ===================================================================================

        // Integramos el IMU restante

        auto remaining_imu = imu_cache_preint_.getInterval(last_processed_rgbd_stamp_, rclcpp::Time(imu_cache_preint_.getLatestTime(), RCL_ROS_TIME));

        if (!remaining_imu.empty())
        {
            auto first = remaining_imu.front();

            Eigen::Vector3d acc(first->linear_acceleration.x,
                                first->linear_acceleration.y,
                                first->linear_acceleration.z);

            Eigen::Vector3d gyr(first->angular_velocity.x,
                                first->angular_velocity.y,
                                first->angular_velocity.z);

            Eigen::Vector3d ba, bg;
            impl_->gs_core_.getBiases(ba, bg);

            rclcpp::Time t_imu(first->header.stamp, RCL_ROS_TIME);

            impl_->preint_.init(acc, gyr, ba, bg,
                                impl_->imu_data_.acc_n,
                                impl_->imu_data_.gyr_n,
                                impl_->imu_data_.acc_w,
                                impl_->imu_data_.gyr_w);

            for (const auto &m : remaining_imu)
            {
                Eigen::Vector3d acc_i(m->linear_acceleration.x,
                                    m->linear_acceleration.y,
                                    m->linear_acceleration.z);

                Eigen::Vector3d gyr_i(m->angular_velocity.x,
                                    m->angular_velocity.y,
                                    m->angular_velocity.z);

                impl_->preint_.add_imu(
                    (rclcpp::Time(m->header.stamp, RCL_ROS_TIME) - t_imu).seconds(),
                    acc_i, gyr_i);

                t_imu = rclcpp::Time(m->header.stamp, RCL_ROS_TIME);
            }
        }

        // ==================================================================================

        // Imprimimos los timers

        const auto t_sync_end = std::chrono::steady_clock::now();

        auto ms = [](auto d) {
            return std::chrono::duration<double, std::milli>(d).count();
        };

        RCLCPP_DEBUG_THROTTLE(this->get_logger(), *this->get_clock(), 5000,
                      "syncRgbdImu: total=%.2f ms | imu_integration=%.2f ms | compute=%.2f ms | remaining_imu=%.2f ms",
                      ms(t_sync_end - t_sync_start),
                      ms(t_imu_integration_end - t_imu_integration_start),
                      ms(t_compute_end - t_compute_start),
                      ms(t_sync_end - t_compute_end));

        // GPU Timing metrics
        const auto &gpu_timings = impl_->gs_core_.getLastGPUTimings();
        const bool has_gpu_timings =
            (gpu_timings.prepare_rasterization_ms > 0.0f) ||
            (gpu_timings.rasterize_ms > 0.0f) ||
            (gpu_timings.pyramid_build_ms > 0.0f) ||
            (gpu_timings.image_copy_ms > 0.0f) ||
            (gpu_timings.keyword_optimization_ms > 0.0f) ||
            (gpu_timings.densification_ms > 0.0f) ||
            (gpu_timings.pruning_ms > 0.0f) ||
            (gpu_timings.total_frame_ms > 0.0f);

        //const auto &cpu_timings = impl_->gs_core_.getLastCPUTimings();
        const bool has_cpu_timings =
            (cpu_timings.lock_wait_ms > 0.0f) ||
            (cpu_timings.imu_predict_ms > 0.0f) ||
            (cpu_timings.pose_track_ms > 0.0f) ||
            (cpu_timings.pose_refine_ms > 0.0f) ||
            (cpu_timings.outlier_ms > 0.0f) ||
            (cpu_timings.map_ops_ms > 0.0f) ||
            (cpu_timings.loop_extract_ms > 0.0f) ||
            (cpu_timings.loop_detect_ms > 0.0f) ||
            (cpu_timings.loop_verify_ms > 0.0f) ||
            (cpu_timings.loop_pgo_ms > 0.0f) ||
            (cpu_timings.loop_total_ms > 0.0f) ||
            (cpu_timings.marginalization_ms > 0.0f) ||
            (cpu_timings.total_frame_ms > 0.0f);

        if (has_gpu_timings || has_cpu_timings)
        {
            const float gpu_sum_ms =
                gpu_timings.pyramid_build_ms +
                gpu_timings.prepare_rasterization_ms +
                gpu_timings.rasterize_ms +
                gpu_timings.keyword_optimization_ms +
                gpu_timings.densification_ms +
                gpu_timings.pruning_ms +
                gpu_timings.image_copy_ms;

            const float cpu_sum_ms =
                cpu_timings.lock_wait_ms +
                cpu_timings.imu_predict_ms +
                cpu_timings.pose_track_ms +
                cpu_timings.pose_refine_ms +
                cpu_timings.outlier_ms +
                cpu_timings.map_ops_ms +
                cpu_timings.marginalization_ms;

            const float loop_sum_ms =
                cpu_timings.loop_extract_ms +
                cpu_timings.loop_detect_ms +
                cpu_timings.loop_verify_ms +
                cpu_timings.loop_pgo_ms;

            RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), 7000,
                          "Timing Table:\n"
                          "  [GPU]\n"
                          "    pyramid_build_ms         : %.2f\n"
                          "    prepare_rasterization_ms : %.2f\n"
                          "    rasterize_ms             : %.2f\n"
                          "    keyframe_optimization_ms : %.2f\n"
                          "    densification_ms         : %.2f\n"
                          "    pruning_ms               : %.2f\n"
                          "    image_copy_ms            : %.2f\n"
                          "    sum_ms                   : %.2f\n"
                          "    total_frame_ms           : %.2f\n"
                          "  [CPU]\n"
                          "    lock_wait_ms             : %.2f\n"
                          "    imu_predict_ms           : %.2f\n"
                          "    pose_track_ms            : %.2f\n"
                          "    pose_refine_ms           : %.2f\n"
                          "    outlier_ms               : %.2f\n"
                          "    map_ops_ms               : %.2f\n"
                          "    loop_extract_ms          : %.2f\n"
                          "    loop_detect_ms           : %.2f\n"
                          "    loop_verify_ms           : %.2f\n"
                          "    loop_pgo_ms              : %.2f\n"
                          "    loop_total_ms            : %.2f\n"
                          "    loop_sum_ms              : %.2f\n"
                          "    marginalization_ms       : %.2f\n"
                          "    sum_ms                   : %.2f\n"
                          "    total_frame_ms           : %.2f\n"
                          "  [CNT]\n"
                          "    gaussian_count           : %d\n"
                          "    densify_added            : %d\n"
                          "    prune_removed            : %d\n"
                          "    outliers_removed         : %d",
                          gpu_timings.pyramid_build_ms,
                          gpu_timings.prepare_rasterization_ms,
                          gpu_timings.rasterize_ms,
                          gpu_timings.keyword_optimization_ms,
                          gpu_timings.densification_ms,
                          gpu_timings.pruning_ms,
                          gpu_timings.image_copy_ms,
                          gpu_sum_ms,
                          gpu_timings.total_frame_ms,
                          cpu_timings.lock_wait_ms,
                          cpu_timings.imu_predict_ms,
                          cpu_timings.pose_track_ms,
                          cpu_timings.pose_refine_ms,
                          cpu_timings.outlier_ms,
                          cpu_timings.map_ops_ms,
                          cpu_timings.loop_extract_ms,
                          cpu_timings.loop_detect_ms,
                          cpu_timings.loop_verify_ms,
                          cpu_timings.loop_pgo_ms,
                          cpu_timings.loop_total_ms,
                          loop_sum_ms,
                          cpu_timings.marginalization_ms,
                          cpu_sum_ms,
                          cpu_timings.total_frame_ms,
                          cpu_timings.gaussian_count,
                          cpu_timings.densify_added,
                          cpu_timings.prune_removed,
                          cpu_timings.outliers_removed);
        }

        // ==================================================================================

        // Fin del pipeline
        isProcessing = false;
    }

    void GSSlamNode::reconstructedImageTimerCallback()
    {
        if (!publish_reconstructed_images_)
        {
            return;
        }

        const bool ready_for_render = test_gaussians_only_
            ? (hasIntrinsics && impl_->gs_core_.getIsInitialized())
            : (imuInitialized && hasCameraInfo && impl_->gs_core_.getIsInitialized());

        if (!ready_for_render)
        {
            return;
        }

        if (isProcessing)
        {
            return;
        }

        publishReconstructedImage();
    }

    void GSSlamNode::pointCloudTimerCallback()
    {
        if (!publish_pointcloud_)
        {
            return;
        }

        const bool ready_for_pointcloud = test_gaussians_only_
            ? (hasIntrinsics && impl_->gs_core_.getIsInitialized())
            : (hasCameraInfo && impl_->gs_core_.getIsInitialized() && imuInitialized);

        if (!ready_for_pointcloud)
        {
            return;
        }

        if (isProcessing)
        {
            return;
        }

        const rclcpp::Time stamp =
            (last_rgb_stamp_.nanoseconds() > 0) ? last_rgb_stamp_ : this->now();
        publishGaussiansAsPointCloud(stamp);
    }

    

    void GSSlamNode::publishReconstructedImage()
    {
        if (!publish_reconstructed_images_)
        {
            return;
        }

        // No renderizar si nadie escucha
        if (!reconstructed_image_pub_ || reconstructed_image_pub_->get_subscription_count() == 0)
        {
            return;
        }

        const Pose cam_pose = impl_->gs_core_.getCameraPose();

        sensor_msgs::msg::Image::ConstSharedPtr rgb_msg;
        {
            std::lock_guard<std::mutex> lock(shared_data_mutex_);
            rgb_msg = last_rgb_msg_;
        }

        // Usar tamaño de la última imagen recibida
        if (!rgb_msg)
        {
            RCLCPP_DEBUG(this->get_logger(),
            "No last RGB message available, cannot determine image size for reconstructing");
            return;
        }

        const int ds = std::max(1, downsample_factor_);
        const int width  = std::max(1, static_cast<int>(rgb_msg->width) / ds);
        const int height = std::max(1, static_cast<int>(rgb_msg->height) / ds);

        cv::cuda::GpuMat rgb_gpu, depth_gpu;

        const auto t_render_start = std::chrono::steady_clock::now();
        bool ok = impl_->gs_core_.renderView(
            impl_->gs_core_.getCurrentSubmap(),
            cam_pose,
            impl_->gs_core_.getIntrinsics(),
            width,
            height,
            rgb_gpu,
            depth_gpu);
        const auto t_render_end = std::chrono::steady_clock::now();
        const double render_view_ms = std::chrono::duration<double, std::milli>(
            t_render_end - t_render_start).count();
        
        if (!ok || rgb_gpu.empty())
        {
            RCLCPP_DEBUG(this->get_logger(),
            "RenderView failed or RGB GPU is empty");
            return;
        }
        cv::Mat rgb_host;
        const auto t_download_start = std::chrono::steady_clock::now();
        rgb_gpu.download(rgb_host);
        const auto t_download_end = std::chrono::steady_clock::now();
        const double render_download_ms = std::chrono::duration<double, std::milli>(
            t_download_end - t_download_start).count();

        if (rgb_host.empty()){
            RCLCPP_DEBUG(this->get_logger(), "Failed to download RGB GPU data to host");
            return;
        }

        double psnr_db = std::numeric_limits<double>::quiet_NaN();
        try
        {
            const auto rgb_cv = cv_bridge::toCvShare(rgb_msg, sensor_msgs::image_encodings::BGR8);
            psnr_db = computeImagePsnrDb(rgb_cv->image, rgb_host);
        }
        catch (const cv_bridge::Exception &e)
        {
            RCLCPP_WARN(this->get_logger(),
                        "PSNR could not be computed from reconstructed image: %s",
                        e.what());
        }

        std_msgs::msg::Header header;
        header.stamp = rgb_msg->header.stamp;
        header.frame_id = world_frame_id_;

        auto msg =
            cv_bridge::CvImage(header, "bgr8", rgb_host).toImageMsg();

        const auto t_publish_start = std::chrono::steady_clock::now();
        reconstructed_image_pub_->publish(*msg);
        const auto t_publish_end = std::chrono::steady_clock::now();
        const double render_publish_ms = std::chrono::duration<double, std::milli>(
            t_publish_end - t_publish_start).count();

        last_render_view_ms_ = render_view_ms;
        last_render_download_ms_ = render_download_ms;
        last_render_publish_ms_ = render_publish_ms;
        last_render_total_ms_ = render_view_ms + render_download_ms + render_publish_ms;
        last_render_stamp_ = header.stamp;

        if (std::isfinite(psnr_db))
        {
            RCLCPP_INFO(this->get_logger(),
                        "Reconstructed image published | PSNR: %.2f dB | render=%.2f ms | download=%.2f ms | publish=%.2f ms",
                        psnr_db,
                        render_view_ms,
                        render_download_ms,
                        render_publish_ms);
        }
        else
        {
            RCLCPP_INFO(this->get_logger(),
                        "Reconstructed image published | PSNR: inf dB | render=%.2f ms | download=%.2f ms | publish=%.2f ms",
                        render_view_ms,
                        render_download_ms,
                        render_publish_ms);
        }
    }


    void GSSlamNode::publishGaussiansAsPointCloud(const rclcpp::Time &stamp)
    {
        // ============================================================
        // 1. Early exit
        // ============================================================
        if (!publish_pointcloud_) return;

        if (!pointcloud_pub_ || pointcloud_pub_->get_subscription_count() == 0)
            return;

        const rclcpp::Time now = this->now();

        if (last_pointcloud_publish_time_.nanoseconds() > 0)
        {
            const double elapsed = (now - last_pointcloud_publish_time_).seconds();

            // Evita frecuencia excesiva o división por cero
            const double min_period = 1.0 / std::max(0.1, pointcloud_publish_rate_hz_);

            if (elapsed < min_period) {
                return;
            }
        }

        // ============================================================
        // 2. Buffers CPU
        // ============================================================
        static std::vector<float4> pos_host;
        static std::vector<float4> col_host;

        // Capacidad máxima esperada (ajustable)
        const uint32_t max_expected = 1'000'000;

        if (pos_host.size() < max_expected)
        {
            pos_host.resize(max_expected);
            col_host.resize(max_expected);
        }

        // ============================================================
        // 3. Obtener datos desde el core (GPU -> CPU)
        // ============================================================
        uint32_t n = impl_->gs_core_.getGaussianDataGlobal(
            pos_host.data(),
            col_host.data(),
            pos_host.size());

        if (n == 0)
        {
            RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 5000,
                                "No gaussians to publish in point cloud");
            return;
        }

        // ============================================================
        // 4. Mensaje
        // ============================================================
        sensor_msgs::msg::PointCloud2 cloud_msg;

        cloud_msg.header.stamp = stamp;              // tiempo de los datos
        cloud_msg.header.frame_id = world_frame_id_; // frame de referencia

        cloud_msg.height = 1;        // nube no organizada
        cloud_msg.width = n;         // número de puntos
        cloud_msg.is_dense = true;   // no hay NaNs
        cloud_msg.is_bigendian = false;

        sensor_msgs::PointCloud2Modifier modifier(cloud_msg);
        modifier.setPointCloud2Fields(4,
            "x", 1, sensor_msgs::msg::PointField::FLOAT32,
            "y", 1, sensor_msgs::msg::PointField::FLOAT32,
            "z", 1, sensor_msgs::msg::PointField::FLOAT32,
            "rgb", 1, sensor_msgs::msg::PointField::FLOAT32);

        modifier.resize(n);

        // ============================================================
        // 5. Llenamos el mensaje
        // ============================================================
        uint8_t* data_ptr = cloud_msg.data.data();
        const size_t step = cloud_msg.point_step; // bytes por punto

        for (uint32_t i = 0; i < n; ++i)
        {
            // Offset del punto i dentro del buffer
            float* ptr = reinterpret_cast<float*>(data_ptr + i * step);

            const float4& p = pos_host[i]; // posición
            const float4& c = col_host[i]; // color en formato BGR

            // ----------------------------
            // Posicion
            // ----------------------------
            ptr[0] = p.x;
            ptr[1] = p.y;
            ptr[2] = p.z;

            // ----------------------------
            // Color (BGR -> RGB)
            // ----------------------------
            uint8_t r = static_cast<uint8_t>(std::clamp(c.z * 255.0f, 0.0f, 255.0f));
            uint8_t g = static_cast<uint8_t>(std::clamp(c.y * 255.0f, 0.0f, 255.0f));
            uint8_t b = static_cast<uint8_t>(std::clamp(c.x * 255.0f, 0.0f, 255.0f));

            // Formato 0xRRGGBB
            uint32_t rgb = (r << 16) | (g << 8) | b;

            float rgb_f;
            std::memcpy(&rgb_f, &rgb, sizeof(float));

            ptr[3] = rgb_f;
        }

        // ============================================================
        // 6. Publicación
        // ============================================================
        pointcloud_pub_->publish(cloud_msg);

        last_pointcloud_publish_time_ = now;

        RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), 1000,
                            "Published PointCloud2 with %u gaussians", n);
    }

}
