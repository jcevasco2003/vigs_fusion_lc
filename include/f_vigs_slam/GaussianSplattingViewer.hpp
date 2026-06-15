#pragma once

#include <atomic>
#include <filesystem>
#include <thread>

#include <Eigen/Dense>
#include <opencv2/core/cuda.hpp>
#include <opencv2/highgui.hpp>

#include <f_vigs_slam/GSSlam.cuh>

namespace f_vigs_slam
{
    class GaussianSplattingViewer
    {
    public:
        explicit GaussianSplattingViewer(GSSlam &gs_slam, bool visualize_current_submap = false);
        ~GaussianSplattingViewer();

        void startThread();

    private:
        static void mouseCallbackStatic(int event, int x, int y, int flags, void *userdata);
        void mouseCallback(int event, int x, int y, int flags);
        void keyCallback(int key);
        void renderLoop();
        void render();
        void resetView();
        bool saveCurrentView();
        std::string makeScreenshotPath() const;

        GSSlam &gs_slam_;
        bool visualize_current_submap_ = false;

        cv::Mat rendered_rgb_;
        cv::Mat rendered_depth_;
        cv::Mat current_display_frame_;
        cv::cuda::GpuMat rendered_rgb_gpu_;
        cv::cuda::GpuMat rendered_depth_gpu_;

        IntrinsicParameters camera_intrinsics_;
        Pose camera_pose_;

        int width_ = 848;
        int height_ = 480;
        double fov_ = 60.0;

        bool follow_ = false;
        bool show_keyframe_poses_ = true;

        Eigen::Vector3f camera_view_position_;
        double yaw_ = 0.0;
        double pitch_ = 0.2;
        Eigen::Vector3f focal_point_;
        double distance_ = 3.0;

        int prev_mouse_x_ = 0;
        int prev_mouse_y_ = 0;

        std::thread render_thread_;
        std::atomic<bool> stop_requested_{false};

        enum RenderType
        {
            RENDER_TYPE_RGB = 0,
            RENDER_TYPE_DEPTH = 1,
            RENDER_TYPE_BLOBS = 2,
            RENDER_TYPE_NUM
        } render_type_ = RENDER_TYPE_RGB;
    };
} // namespace f_vigs_slam
