#pragma once

#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/imu.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "sensor_msgs/msg/camera_info.hpp"
#include "sensor_msgs/msg/point_cloud2.hpp"
#include "sensor_msgs/point_cloud2_iterator.hpp"
#include "nav_msgs/msg/odometry.hpp"
#include "std_msgs/msg/float32.hpp"
#include "image_transport/subscriber_filter.hpp"
#include "message_filters/synchronizer.hpp"
#include "message_filters/sync_policies/approximate_time.hpp"
#include "message_filters/cache.hpp"
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>
#include <tf2_ros/transform_broadcaster.h>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include "tf2_geometry_msgs/tf2_geometry_msgs.hpp"
#include "tf2_eigen/tf2_eigen.hpp"
#include <memory>
#include <vector>
#include <atomic>
#include <condition_variable>
#include <mutex>
#include <thread>
#include <Eigen/Dense>
#include <opencv2/opencv.hpp>

// Definimos la clase que se encarga de la logica del nodo en si mismo

namespace f_vigs_slam
{
    class GSSlam;
    struct IntrinsicParameters;
    struct ImuData;
    struct Pose;
    class Preintegration;

    class GSSlamNode : public rclcpp::Node
    {
    public:
        explicit GSSlamNode(const rclcpp::NodeOptions & options = rclcpp::NodeOptions());
        ~GSSlamNode();

    protected:
        struct Impl;
        std::unique_ptr<Impl> impl_;

        std::atomic<bool> hasIntrinsics{false};
        std::atomic<bool> hasExtrinsics{false};
        std::atomic<bool> hasImu{false};
        std::atomic<bool> imuInitialized{false};
        std::atomic<bool> isProcessing{false};
        std::atomic<bool> hasCameraInfo{false};
        cv::Mat rgbImg, depthImg;
        cv::Mat local_rgb_img, local_depth_img;
        int nb_init_imu_ = 0;
        Eigen::Vector3d avg_acc_ = Eigen::Vector3d::Zero();
        Eigen::Vector3d avg_gyro_ = Eigen::Vector3d::Zero();
        Eigen::Vector3d acc_bias_ = Eigen::Vector3d::Zero();
        Eigen::Quaterniond q_imu_cam_ = Eigen::Quaterniond::Identity();
        Eigen::Vector3d t_imu_cam_ = Eigen::Vector3d::Zero();
        int gaussian_iterations_ = 10;
        double depth_scale_ = 1.0;
        int downsample_factor_ = 1;
        int gauss_init_size_px_ = 7;
        double gauss_init_scale_ = 0.01;
        rclcpp::Time last_imu_stamp_{0, 0, RCL_ROS_TIME};
        rclcpp::Time last_integrated_imu_stamp_{0, 0, RCL_ROS_TIME};
        rclcpp::Time last_processed_rgbd_stamp_{0, 0, RCL_ROS_TIME};
        rclcpp::Time last_rgb_stamp_{0, 0, RCL_ROS_TIME};
        sensor_msgs::msg::Image::ConstSharedPtr last_rgb_msg_;

        std::unique_ptr<tf2_ros::Buffer> tf_buffer_;
        std::shared_ptr<tf2_ros::TransformListener> tf_listener_{nullptr};
        std::unique_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;
        std::string imu_frame_id_;
        std::string imu_frame_id_fallback_;
        std::string camera_frame_id_;
        std::string tf_mode_;
        double tf_lookup_timeout_s_ = 0.1;
        bool tf_static_cached_ = false;
        geometry_msgs::msg::TransformStamped tf_static_cached_msg_;

        // Dedicated sync worker (callbacks only notify; worker executes sync).
        std::thread sync_worker_;
        std::mutex worker_mutex_;
        std::condition_variable worker_cv_;
        bool worker_stop_ = false;
        bool sync_requested_ = false;

        // Protect shared callback data read by the sync worker.
        std::mutex shared_data_mutex_;
        std::mutex imu_cache_mutex_;

        // Suscripciones a los topicos de ros2
        rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr imu_sub_;
        rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr imu_preint_sub_;
        rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr color_diag_sub_;
        rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr depth_diag_sub_;
        std::shared_ptr<image_transport::SubscriberFilter> depth_sub_;
        std::shared_ptr<image_transport::SubscriberFilter> color_sub_;
        using RGBDSyncPolicy = message_filters::sync_policies::ApproximateTime<sensor_msgs::msg::Image, sensor_msgs::msg::Image>;
        std::shared_ptr<message_filters::Synchronizer<RGBDSyncPolicy>> rgbd_sync_;
        rclcpp::Subscription<sensor_msgs::msg::CameraInfo>::SharedPtr camera_info_sub_;
        message_filters::Cache<sensor_msgs::msg::Imu> imu_cache_preint_{1000};

        // Publishers de odometría y nube de puntos
        rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr odom_pub_;
        rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr odom_imu_pub_;
        rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pointcloud_pub_;
        rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr reconstructed_image_pub_;
        
        // Publishers para métricas de SLAM
        rclcpp::Publisher<std_msgs::msg::Float32>::SharedPtr track_time_pub_;
        rclcpp::Publisher<std_msgs::msg::Float32>::SharedPtr map_time_pub_;
        
        rclcpp::TimerBase::SharedPtr pointcloud_timer_;
        rclcpp::TimerBase::SharedPtr reconstructed_image_timer_;
        
        // Mensaje y frame para odometría
        nav_msgs::msg::Odometry odom_msg_;
        nav_msgs::msg::Odometry odom_imu_msg_;
        std::string world_frame_id_;
        std::string publish_frame_id_;

        // Guardamos los tiempos de los ultimos callbacks recibidos
        rclcpp::Time last_imu_callback_time_{0, 0, RCL_CLOCK_UNINITIALIZED};
        rclcpp::Time last_rgbd_callback_time_{0, 0, RCL_CLOCK_UNINITIALIZED};
        rclcpp::Time last_color_diag_time_{0, 0, RCL_CLOCK_UNINITIALIZED};
        rclcpp::Time last_depth_diag_time_{0, 0, RCL_CLOCK_UNINITIALIZED};
        rclcpp::Time last_timer_block_log_time_{0, 0, RCL_ROS_TIME};
        rclcpp::Time last_render_stamp_{0, 0, RCL_ROS_TIME};
        rclcpp::Time last_pointcloud_publish_time_{0, 0, RCL_ROS_TIME};
        double last_render_view_ms_ = 0.0;
        double last_render_download_ms_ = 0.0;
        double last_render_publish_ms_ = 0.0;
        double last_render_total_ms_ = 0.0;
        double last_track_time_ms_ = 0.0;  // Tiempo de tracking (odometría/pose estimation)
        double last_map_time_ms_ = 0.0;     // Tiempo de mapping (optimización de gaussianas)
        
        // Contadores y buffers para métricas en terminal
        uint64_t frame_count_ = 0;
        std::deque<double> track_time_buffer_;  // Buffer circular para promedio móvil
        std::deque<double> map_time_buffer_;
        static constexpr size_t METRICS_BUFFER_SIZE = 30;
        int metrics_print_interval_frames_ = 10;  // Frecuencia configurable de impresión en frames
        
        double last_print_track_time_ms_ = 0.0;
        double last_print_map_time_ms_ = 0.0;
        
        double pointcloud_publish_rate_hz_ = 1.0;
        bool publish_pointcloud_ = true;
        bool publish_reconstructed_images_ = true;
        bool visualize_current_submap_ = false;
        bool test_gaussians_only_ = false;
        int imu_init_samples_ = 100;
        double imu_dt_warn_max_s_ = 0.05;
        double imu_acc_norm_min_ = 1.0;
        double imu_acc_norm_max_ = 30.0;
        double imu_gyro_norm_max_ = 10.0;
        double diag_state_jump_pos_thresh_m_ = 0.20;
        double diag_state_jump_rot_thresh_deg_ = 5.0;
        double diag_proc_time_domain_abs_limit_s_ = 10.0;
        
        

        // Callbacks para manejo de mensajes
        void imuCallback(const sensor_msgs::msg::Imu::SharedPtr msg);
        void rgbdCallback(const std::shared_ptr<const sensor_msgs::msg::Image>& color,
                  const std::shared_ptr<const sensor_msgs::msg::Image>& depth);
        void cameraInfoCallback(const sensor_msgs::msg::CameraInfo::ConstSharedPtr msg);
        void imuPreintegratedCallback(const sensor_msgs::msg::Imu::SharedPtr msg);
        void colorDiagCallback(const sensor_msgs::msg::Image::SharedPtr msg);
        void depthDiagCallback(const sensor_msgs::msg::Image::SharedPtr msg);
        
        // IMU processing helpers
        void syncRgbdImu();
        void syncWorkerLoop();
        void requestSyncWorker();
        
        Eigen::Matrix3d computeGravityAlignment(const Eigen::Vector3d& acc) const;
        bool updateImuCameraExtrinsics(const rclcpp::Time &stamp);

        // PointCloud publishing
        void publishGaussiansAsPointCloud(const rclcpp::Time &stamp);
        void pointCloudTimerCallback();

        // Image publishing
        void publishReconstructedImage();
        void reconstructedImageTimerCallback();

    };
} // namespace f_vigs_slam

