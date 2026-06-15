#include <f_vigs_slam/GaussianSplattingViewer.hpp>

#include <chrono>
#include <cmath>
#include <ctime>
#include <iomanip>
#include <iostream>
#include <limits>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <sstream>

namespace f_vigs_slam
{
    GaussianSplattingViewer::GaussianSplattingViewer(GSSlam &gs_slam, bool visualize_current_submap)
        : gs_slam_(gs_slam),
          visualize_current_submap_(visualize_current_submap)
    {
        resetView();
    }

    GaussianSplattingViewer::~GaussianSplattingViewer()
    {
        stop_requested_.store(true);
        if (render_thread_.joinable())
        {
            render_thread_.join();
        }
    }

    void GaussianSplattingViewer::resetView()
    {
        focal_point_ = Eigen::Vector3f(2.0f, 0.0f, 0.0f);
        distance_ = 3.0;
        camera_view_position_ = Eigen::Vector3f(-3.0f, 0.0f, 1.0f);
        yaw_ = 0.0;
        pitch_ = 0.2;
        fov_ = 60.0;
    }

    void GaussianSplattingViewer::mouseCallbackStatic(int event, int x, int y, int flags, void *userdata)
    {
        auto *viewer = static_cast<GaussianSplattingViewer *>(userdata);
        viewer->mouseCallback(event, x, y, flags);
    }

    void GaussianSplattingViewer::mouseCallback(int event, int x, int y, int flags)
    {
        if (event == cv::EVENT_LBUTTONDOWN || event == cv::EVENT_RBUTTONDOWN || event == cv::EVENT_MBUTTONDOWN)
        {
            prev_mouse_x_ = x;
            prev_mouse_y_ = y;
            return;
        }

        if (event == cv::EVENT_MOUSEMOVE && (flags & cv::EVENT_FLAG_LBUTTON))
        {
            const int dx = x - prev_mouse_x_;
            const int dy = y - prev_mouse_y_;

            yaw_ -= static_cast<double>(dx) * 0.01;
            pitch_ += static_cast<double>(dy) * 0.01;

            prev_mouse_x_ = x;
            prev_mouse_y_ = y;
            return;
        }

        if (event == cv::EVENT_MOUSEMOVE && (flags & cv::EVENT_FLAG_RBUTTON))
        {
            const int dx = x - prev_mouse_x_;
            const int dy = y - prev_mouse_y_;

            const Eigen::Quaternionf orientation =
                Eigen::AngleAxisf(static_cast<float>(yaw_), Eigen::Vector3f::UnitZ()) *
                Eigen::AngleAxisf(static_cast<float>(pitch_), Eigen::Vector3f::UnitY());

            camera_view_position_ += orientation * Eigen::Vector3f(0.0f,
                                                                    static_cast<float>(dx) * 0.01f,
                                                                    static_cast<float>(dy) * 0.01f);

            prev_mouse_x_ = x;
            prev_mouse_y_ = y;
            return;
        }

        if (event == cv::EVENT_MOUSEWHEEL)
        {
            const int delta = cv::getMouseWheelDelta(flags);
            const Eigen::Quaternionf orientation =
                Eigen::AngleAxisf(static_cast<float>(yaw_), Eigen::Vector3f::UnitZ()) *
                Eigen::AngleAxisf(static_cast<float>(pitch_), Eigen::Vector3f::UnitY());
            camera_view_position_ += orientation * Eigen::Vector3f(-static_cast<float>(delta) * 0.1f, 0.0f, 0.0f);
        }
    }

    void GaussianSplattingViewer::keyCallback(int key)
    {
        if (key == static_cast<int>(' '))
        {
            render_type_ = static_cast<RenderType>((render_type_ + 1) % RENDER_TYPE_NUM);
        }
        else if (key == static_cast<int>('r'))
        {
            resetView();
        }
        else if (key == static_cast<int>('f'))
        {
            resetView();
            follow_ = !follow_;
        }
        else if (key == static_cast<int>('k'))
        {
            show_keyframe_poses_ = !show_keyframe_poses_;
            std::cout << "[Viewer] Keyframe poses " << (show_keyframe_poses_ ? "ON" : "OFF") << std::endl;
        }
        else if (key == static_cast<int>('p'))
        {
            saveCurrentView();
        }
        else if (key == 82 || key == 84)
        {
            const float d = (key == 82) ? 0.1f : -0.1f;
            const Eigen::Quaternionf orientation =
                Eigen::AngleAxisf(static_cast<float>(yaw_), Eigen::Vector3f::UnitZ()) *
                Eigen::AngleAxisf(static_cast<float>(pitch_), Eigen::Vector3f::UnitY());
            camera_view_position_ += orientation * Eigen::Vector3f(d, 0.0f, 0.0f);
        }
        else if (key == 81 || key == 83)
        {
            const float d = (key == 81) ? 0.1f : -0.1f;
            const Eigen::Quaternionf orientation =
                Eigen::AngleAxisf(static_cast<float>(yaw_), Eigen::Vector3f::UnitZ()) *
                Eigen::AngleAxisf(static_cast<float>(pitch_), Eigen::Vector3f::UnitY());
            camera_view_position_ += orientation * Eigen::Vector3f(0.0f, d, 0.0f);
        }
        else if (key == static_cast<int>('q'))
        {
            stop_requested_.store(true);
        }
    }

    void GaussianSplattingViewer::startThread()
    {
        if (render_thread_.joinable())
        {
            return;
        }
        stop_requested_.store(false);
        render_thread_ = std::thread(&GaussianSplattingViewer::renderLoop, this);
    }

    void GaussianSplattingViewer::renderLoop()
    {
        const std::string window_name = "F-VIGS Gaussian Viewer";
        cv::namedWindow(window_name, cv::WINDOW_AUTOSIZE);
        cv::setMouseCallback(window_name, mouseCallbackStatic, this);
        std::cout << "[Viewer] Controls: K toggle keyframe poses, P save screenshot, SPACE switch render mode, R reset, F follow, Q quit" << std::endl;

        while (!stop_requested_.load())
        {
            render();

            const int key = cv::waitKey(30);
            if (key >= 0)
            {
                keyCallback(key);
            }
        }

        cv::destroyWindow(window_name);
    }

    void GaussianSplattingViewer::render()
    {
        Eigen::Map<Eigen::Quaternionf> camera_orientation(reinterpret_cast<float *>(&camera_pose_.orientation));
        const Eigen::Quaternionf pitch_yaw =
            Eigen::AngleAxisf(static_cast<float>(yaw_), Eigen::Vector3f::UnitZ()) *
            Eigen::AngleAxisf(static_cast<float>(pitch_), Eigen::Vector3f::UnitY());
        camera_orientation = pitch_yaw * Eigen::Quaternionf(-0.5f, 0.5f, -0.5f, 0.5f);

        Eigen::Map<Eigen::Vector3f> camera_position(reinterpret_cast<float *>(&camera_pose_.position));
        camera_position = camera_view_position_;

        camera_intrinsics_.f.x = static_cast<float>(width_ / (2.0 * std::tan(fov_ * M_PI / 360.0)));
        camera_intrinsics_.f.y = camera_intrinsics_.f.x;
        camera_intrinsics_.c.x = static_cast<float>(width_ / 2.0);
        camera_intrinsics_.c.y = static_cast<float>(height_ / 2.0);

        if (!gs_slam_.renderView(nullptr, camera_pose_, camera_intrinsics_, width_, height_, rendered_rgb_gpu_, rendered_depth_gpu_, visualize_current_submap_))
        {
            return;
        }

        if (render_type_ == RENDER_TYPE_DEPTH)
        {
            rendered_depth_gpu_.download(rendered_depth_);
            cv::Mat depth_display;
            rendered_depth_.convertTo(depth_display, CV_8UC1, 255.0 * 0.15f);
            current_display_frame_ = depth_display;
            cv::imshow("F-VIGS Gaussian Viewer", current_display_frame_);
            return;
        }

        // RGB/BLOBS fallback: use RGB base image and draw overlays.
        rendered_rgb_gpu_.download(rendered_rgb_);

        // Overlay: keyframe poses, path, orientation, and submap starts
        const auto keyframe_poses = gs_slam_.getAllKeyframeGlobalPoses();
        const auto submap_starts = gs_slam_.getSubmapFirstFrameGlobalPoses();

        // Debug: if viewer sees more than one submap at startup, print details
            const auto submap_info = gs_slam_.getSubmapDebugInfo();

            // Debug: if viewer sees more than one submap at startup, print details
            /*
            if (submap_info.size() > 1) {
                std::cout << "[Viewer] Detected submaps=" << submap_info.size()
                          << " keyframes_total=" << keyframe_poses.size() << std::endl;
                for (size_t si = 0; si < submap_info.size(); ++si) {
                    const auto &info = submap_info[si];
                    const auto &p = info.first_frame_global_pose.position;
                    std::cout << "  submap[" << si << "] id=" << info.submap_id
                              << " gaussians=" << info.gaussian_count
                              << " keyframes=" << info.keyframe_count
                              << " pos=(" << p.x << "," << p.y << "," << p.z << ")"
                              << std::endl;
                }
            }
            */

        // Helper: project world point to image using current camera_pose_ and intrinsics
        auto projectWorldPoint = [&](const Eigen::Vector3f &pw_v, cv::Point &out) -> bool {
            // camera_pose_ is the viewer camera in world coords
            Eigen::Vector3f cam_pos = Eigen::Map<const Eigen::Vector3f>(reinterpret_cast<const float *>(&camera_pose_.position));
            Eigen::Quaternionf cam_q = Eigen::Map<const Eigen::Quaternionf>(reinterpret_cast<const float *>(&camera_pose_.orientation));

            Eigen::Vector3f rel = cam_q.conjugate() * (pw_v - cam_pos);
            if (rel.z() <= 0.001f) return false;

            float u = camera_intrinsics_.f.x * (rel.x() / rel.z()) + camera_intrinsics_.c.x;
            float v = camera_intrinsics_.f.y * (rel.y() / rel.z()) + camera_intrinsics_.c.y;

            if (u < 0 || u >= rendered_rgb_.cols || v < 0 || v >= rendered_rgb_.rows) return false;
            out = cv::Point(static_cast<int>(std::round(u)), static_cast<int>(std::round(v)));
            return true;
        };

        auto posePosition = [](const Pose &p) -> Eigen::Vector3f {
            return Eigen::Vector3f(p.position.x, p.position.y, p.position.z);
        };

        auto poseOrientation = [](const Pose &p) -> Eigen::Quaternionf {
            // Pose stores quaternion as (x, y, z, w)
            return Eigen::Quaternionf(p.orientation.w, p.orientation.x, p.orientation.y, p.orientation.z).normalized();
        };

        if (show_keyframe_poses_)
        {
            // Draw path (connect keyframes sequentially)
            cv::Point prev_pt;
            bool has_prev = false;
            for (size_t i = 0; i < keyframe_poses.size(); ++i)
            {
                const Eigen::Vector3f kf_pos = posePosition(keyframe_poses[i]);
                cv::Point pt;
                if (projectWorldPoint(kf_pos, pt)) {
                    // draw keyframe as small filled circle (blue)
                    cv::circle(rendered_rgb_, pt, 3, cv::Scalar(255, 0, 0), -1);

                    // Draw keyframe viewing direction as arrow (orange).
                    // Convention: camera forward axis is local +Z.
                    const Eigen::Quaternionf kf_q = poseOrientation(keyframe_poses[i]);
                    const Eigen::Vector3f kf_forward_world = kf_q * Eigen::Vector3f::UnitZ();
                    const Eigen::Vector3f kf_tip_world = kf_pos + 0.20f * kf_forward_world;
                    cv::Point pt_tip;
                    if (projectWorldPoint(kf_tip_world, pt_tip)) {
                        cv::arrowedLine(rendered_rgb_, pt, pt_tip, cv::Scalar(0, 165, 255), 1, cv::LINE_AA, 0, 0.25);
                    }

                    if (has_prev) {
                        cv::line(rendered_rgb_, prev_pt, pt, cv::Scalar(200, 200, 0), 1);
                    }
                    prev_pt = pt;
                    has_prev = true;
                }
            }

            // Draw submap start markers (red cross) and label them with submap id above the cross
            for (size_t si = 0; si < submap_info.size(); ++si)
            {
                const auto &info = submap_info[si];
                const Eigen::Vector3f sf_pos = posePosition(info.first_frame_global_pose);
                cv::Point pt;
                if (projectWorldPoint(sf_pos, pt)) {
                    cv::drawMarker(rendered_rgb_, pt, cv::Scalar(0, 0, 255), cv::MARKER_CROSS, 12, 2);

                    // Draw submap id above the cross
                    const std::string label = std::to_string(info.submap_id);
                    int baseline = 0;
                    const double font_scale = 0.5;
                    const int thickness = 1;
                    const int font_face = cv::FONT_HERSHEY_SIMPLEX;
                    const cv::Size text_size = cv::getTextSize(label, font_face, font_scale, thickness, &baseline);
                    // Position text centered horizontally, slightly above the marker
                    cv::Point text_org(pt.x - text_size.width / 2, pt.y - 10);
                    // Clamp text position inside image
                    text_org.x = std::max(0, std::min(text_org.x, rendered_rgb_.cols - text_size.width));
                    text_org.y = std::max(text_size.height, std::min(text_org.y, rendered_rgb_.rows - 1));

                    // Draw a thin black outline for readability
                    cv::putText(rendered_rgb_, label, text_org, font_face, font_scale, cv::Scalar(0, 0, 0), thickness + 2, cv::LINE_AA);
                    cv::putText(rendered_rgb_, label, text_org, font_face, font_scale, cv::Scalar(255, 255, 255), thickness, cv::LINE_AA);
                }
            }
        }

        current_display_frame_ = rendered_rgb_;
        cv::imshow("F-VIGS Gaussian Viewer", current_display_frame_);
    }

    bool GaussianSplattingViewer::saveCurrentView()
    {
        if (current_display_frame_.empty())
        {
            std::cout << "[Viewer] No frame available to save yet" << std::endl;
            return false;
        }

        const std::string output_path = makeScreenshotPath();
        if (!cv::imwrite(output_path, current_display_frame_))
        {
            std::cout << "[Viewer] Failed to save screenshot to " << output_path << std::endl;
            return false;
        }

        std::cout << "[Viewer] Screenshot saved to " << output_path << std::endl;
        return true;
    }

    std::string GaussianSplattingViewer::makeScreenshotPath() const
    {
        const auto now = std::chrono::system_clock::now();
        const auto now_time = std::chrono::system_clock::to_time_t(now);
        const auto millis = std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()) % 1000;

        std::tm time_info{};
        localtime_r(&now_time, &time_info);

        std::ostringstream filename;
        filename << "f_vigs_viewer_" << std::put_time(&time_info, "%Y%m%d_%H%M%S")
                 << '_' << std::setw(3) << std::setfill('0') << millis.count() << ".png";

        const std::filesystem::path snapshot_dir = std::filesystem::current_path() / "f_vigs_slam_viewer_snapshots";
        std::filesystem::create_directories(snapshot_dir);

        return (snapshot_dir / filename.str()).string();
    }
} // namespace f_vigs_slam
