#include "f_vigs_slam/NetVLADWrapper.hpp"

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <array>
#include <chrono>
#include <cerrno>
#include <cstring>
#include <iostream>
#include <sstream>
#include <thread>

namespace f_vigs_slam
{
    namespace
    {
        std::string readEnvOrDefault(const char *name, const std::string &fallback)
        {
            const char *value = std::getenv(name);
            if (!value || value[0] == '\0')
            {
                return fallback;
            }
            return std::string(value);
        }

        int readEnvIntOrDefault(const char *name, int fallback)
        {
            const char *value = std::getenv(name);
            if (!value || value[0] == '\0')
            {
                return fallback;
            }

            try
            {
                return std::stoi(value);
            }
            catch (...)
            {
                return fallback;
            }
        }
    } // namespace

    NetVLADWrapper::NetVLADWrapper()
        : socket_config_(resolveSocketConfig())
    {
        const char *desc_env = std::getenv("F_VIGS_NETVLAD_DESC_SIZE");
        if (desc_env)
        {
            const int requested = std::atoi(desc_env);
            if (requested > 0 && requested <= 4096)
            {
                descriptor_size_ = requested;
            }
        }

        if (verbose_logs_)
        {
            std::cerr << "[NetVLADWrapper] socket client ready host=" << socket_config_.host
                      << " port=" << socket_config_.port
                      << " expected_dim=" << descriptor_size_ << std::endl;
        }
    }

    NetVLADWrapper::~NetVLADWrapper()
    {
        closeSocket();
    }

    NetVLADWrapper::SocketConfig NetVLADWrapper::resolveSocketConfig() const
    {
        SocketConfig cfg;
        cfg.host = readEnvOrDefault("F_VIGS_NETVLAD_HOST", "127.0.0.1");
        cfg.port = readEnvIntOrDefault("F_VIGS_NETVLAD_PORT", 5000);
        cfg.connect_retries = readEnvIntOrDefault("F_VIGS_NETVLAD_CONNECT_RETRIES", 20);
        cfg.connect_delay_ms = readEnvIntOrDefault("F_VIGS_NETVLAD_CONNECT_DELAY_MS", 250);
        return cfg;
    }

    bool NetVLADWrapper::connectSocket()
    {
        closeSocket();

        for (int attempt = 1; attempt <= socket_config_.connect_retries; ++attempt)
        {
            socket_fd_ = ::socket(AF_INET, SOCK_STREAM, 0);
            if (socket_fd_ < 0)
            {
                if (verbose_logs_)
                {
                    std::cerr << "[NetVLADWrapper] socket() failed: " << std::strerror(errno) << std::endl;
                }
                return false;
            }

            sockaddr_in server_addr{};
            server_addr.sin_family = AF_INET;
            server_addr.sin_port = htons(static_cast<uint16_t>(socket_config_.port));

            if (::inet_pton(AF_INET, socket_config_.host.c_str(), &server_addr.sin_addr) <= 0)
            {
                if (verbose_logs_)
                {
                    std::cerr << "[NetVLADWrapper] inet_pton() failed for host=" << socket_config_.host
                              << std::endl;
                }
                closeSocket();
                return false;
            }

            if (::connect(socket_fd_, reinterpret_cast<sockaddr *>(&server_addr), sizeof(server_addr)) == 0)
            {
                socket_ready_ = true;
                if (verbose_logs_)
                {
                    std::cerr << "[NetVLADWrapper] connected to NetVLAD server at "
                              << socket_config_.host << ':' << socket_config_.port << std::endl;
                }
                return true;
            }

            if (verbose_logs_)
            {
                std::cerr << "[NetVLADWrapper] connect attempt " << attempt << '/'
                          << socket_config_.connect_retries << " failed: "
                          << std::strerror(errno) << std::endl;
            }

            closeSocket();
            std::this_thread::sleep_for(std::chrono::milliseconds(socket_config_.connect_delay_ms));
        }

        return false;
    }

    void NetVLADWrapper::closeSocket()
    {
        if (socket_fd_ >= 0)
        {
            ::close(socket_fd_);
            socket_fd_ = -1;
        }
        socket_ready_ = false;
    }

    bool NetVLADWrapper::ensureConnected()
    {
        if (socket_ready_ && socket_fd_ >= 0)
        {
            return true;
        }

        return connectSocket();
    }

    bool NetVLADWrapper::sendAll(const void *data, size_t size)
    {
        const auto *ptr = static_cast<const uint8_t *>(data);
        size_t total_sent = 0;

        while (total_sent < size)
        {
            const ssize_t sent = ::send(socket_fd_, ptr + total_sent, size - total_sent, 0);
            if (sent < 0)
            {
                if (errno == EINTR)
                {
                    continue;
                }
                return false;
            }
            if (sent == 0)
            {
                return false;
            }
            total_sent += static_cast<size_t>(sent);
        }

        return true;
    }

    bool NetVLADWrapper::recvAll(void *data, size_t size)
    {
        auto *ptr = static_cast<uint8_t *>(data);
        size_t total_recv = 0;

        while (total_recv < size)
        {
            const ssize_t received = ::recv(socket_fd_, ptr + total_recv, size - total_recv, 0);
            if (received < 0)
            {
                if (errno == EINTR)
                {
                    continue;
                }
                return false;
            }
            if (received == 0)
            {
                return false;
            }
            total_recv += static_cast<size_t>(received);
        }

        return true;
    }

    std::vector<float> NetVLADWrapper::extractDescriptor(const cv::Mat &rgb_image)
    {
        std::vector<float> descriptor;
        if (rgb_image.empty())
        {
            if (verbose_logs_)
            {
                std::cerr << "[NetVLADWrapper] empty input image" << std::endl;
            }
            return descriptor;
        }

        cv::Mat bgr_image;
        if (rgb_image.type() == CV_8UC4)
        {
            cv::cvtColor(rgb_image, bgr_image, cv::COLOR_BGRA2BGR);
        }
        else if (rgb_image.type() == CV_8UC3)
        {
            bgr_image = rgb_image;
        }
        else if (rgb_image.type() == CV_8UC1)
        {
            cv::cvtColor(rgb_image, bgr_image, cv::COLOR_GRAY2BGR);
        }
        else
        {
            cv::Mat converted;
            rgb_image.convertTo(converted, CV_8UC3);
            bgr_image = converted;
        }

        std::vector<uchar> jpeg_buffer;
        if (!cv::imencode(".jpg", bgr_image, jpeg_buffer))
        {
            if (verbose_logs_)
            {
                std::cerr << "[NetVLADWrapper] cv::imencode(.jpg) failed" << std::endl;
            }
            return descriptor;
        }

        std::lock_guard<std::mutex> socket_lock(socket_mutex_);
        if (!ensureConnected())
        {
            if (verbose_logs_)
            {
                std::cerr << "[NetVLADWrapper] unable to connect to NetVLAD server" << std::endl;
            }
            return descriptor;
        }

        const uint32_t image_size = static_cast<uint32_t>(jpeg_buffer.size());
        const uint32_t image_size_net = htonl(image_size);

        if (!sendAll(&image_size_net, sizeof(image_size_net)) ||
            !sendAll(jpeg_buffer.data(), jpeg_buffer.size()))
        {
            if (verbose_logs_)
            {
                std::cerr << "[NetVLADWrapper] failed to send image to server" << std::endl;
            }
            closeSocket();
            return descriptor;
        }

        uint32_t desc_size_net = 0;
        if (!recvAll(&desc_size_net, sizeof(desc_size_net)))
        {
            if (verbose_logs_)
            {
                std::cerr << "[NetVLADWrapper] failed to receive descriptor size" << std::endl;
            }
            closeSocket();
            return descriptor;
        }

        const uint32_t desc_size_bytes = ntohl(desc_size_net);
        if (desc_size_bytes == 0 || (desc_size_bytes % sizeof(float)) != 0)
        {
            if (verbose_logs_)
            {
                std::cerr << "[NetVLADWrapper] invalid descriptor byte size=" << desc_size_bytes << std::endl;
            }
            closeSocket();
            return descriptor;
        }

        std::vector<uint8_t> desc_bytes(desc_size_bytes);
        if (!recvAll(desc_bytes.data(), desc_bytes.size()))
        {
            if (verbose_logs_)
            {
                std::cerr << "[NetVLADWrapper] failed to receive descriptor payload" << std::endl;
            }
            closeSocket();
            return descriptor;
        }

        descriptor.resize(desc_size_bytes / sizeof(float));
        std::memcpy(descriptor.data(), desc_bytes.data(), desc_size_bytes);
        normalizeDescriptor(descriptor);
        ++extraction_count_;

        if (verbose_logs_)
        {
            //std::cerr << "[NetVLADWrapper] descriptor received #" << extraction_count_
            //          << " floats=" << descriptor.size()
            //          << " bytes=" << desc_size_bytes << std::endl;
        }

        if (descriptor_size_ > 0 && descriptor.size() != static_cast<size_t>(descriptor_size_) && verbose_logs_)
        {
            std::cerr << "[NetVLADWrapper] warning: expected " << descriptor_size_
                      << " floats, got " << descriptor.size() << std::endl;
        }

        return descriptor;
    }

    std::vector<NetVLADWrapper::LoopCandidate> NetVLADWrapper::searchCandidates(
        const std::vector<float> &query_descriptor,
        int k,
        float min_similarity_thresh)
    {
        std::vector<LoopCandidate> candidates;
        if (query_descriptor.empty() || k <= 0)
        {
            return candidates;
        }

        std::lock_guard<std::mutex> lock(descriptor_mutex_);
        for (const auto &submap_pair : descriptor_db_)
        {
            for (const auto &kf_pair : submap_pair.second)
            {
                const float similarity = cosineSimilarity(query_descriptor, kf_pair.second);
                if (similarity < min_similarity_thresh)
                {
                    continue;
                }

                candidates.push_back({submap_pair.first, kf_pair.first, similarity});
            }
        }

        std::sort(candidates.begin(), candidates.end(),
                  [](const LoopCandidate &lhs, const LoopCandidate &rhs)
                  {
                      return lhs.similarity > rhs.similarity;
                  });

        if (static_cast<int>(candidates.size()) > k)
        {
            candidates.resize(static_cast<size_t>(k));
        }

        return candidates;
    }

    std::vector<NetVLADWrapper::LoopCandidate> NetVLADWrapper::filterByTemporal(
        const std::vector<LoopCandidate> &candidates,
        int current_submap_id,
        int current_keyframe_idx,
        int temporal_window)
    {
        std::vector<LoopCandidate> filtered;
        filtered.reserve(candidates.size());

        for (const LoopCandidate &cand : candidates)
        {
            if (cand.submap_id == current_submap_id &&
                std::abs(cand.keyframe_idx - current_keyframe_idx) <= temporal_window)
            {
                continue;
            }
            filtered.push_back(cand);
        }

        return filtered;
    }

    std::vector<NetVLADWrapper::LoopCandidate> NetVLADWrapper::filterByDistance(
        const std::vector<LoopCandidate> &candidates,
        const std::vector<Pose> &submap_poses,
        float distance_threshold)
    {
        std::vector<LoopCandidate> filtered;
        filtered.reserve(candidates.size());

        for (const LoopCandidate &cand : candidates)
        {
            if (cand.submap_id < 0)
            {
                continue;
            }

            if (submap_poses.empty())
            {
                continue;
            }

            const size_t candidate_idx = static_cast<size_t>(cand.submap_id);
            if (candidate_idx >= submap_poses.size())
            {
                filtered.push_back(cand);
                continue;
            }

            const Pose &query_pose = submap_poses.back();
            const Pose &candidate_pose = submap_poses[candidate_idx];
            const float dx = query_pose.position.x - candidate_pose.position.x;
            const float dy = query_pose.position.y - candidate_pose.position.y;
            const float dz = query_pose.position.z - candidate_pose.position.z;
            const float dist = std::sqrt(dx * dx + dy * dy + dz * dz);

            if (dist > distance_threshold)
            {
                filtered.push_back(cand);
            }
        }

        return filtered;
    }

    void NetVLADWrapper::addDescriptor(int submap_id, int keyframe_idx, const std::vector<float> &descriptor)
    {
        if (descriptor.empty())
        {
            return;
        }

        std::vector<float> normalized_desc = descriptor;
        normalizeDescriptor(normalized_desc);

        std::lock_guard<std::mutex> lock(descriptor_mutex_);
        descriptor_db_[submap_id][keyframe_idx] = normalized_desc;
    }

    void NetVLADWrapper::clear()
    {
        std::lock_guard<std::mutex> lock(descriptor_mutex_);
        descriptor_db_.clear();
    }

    bool NetVLADWrapper::getDescriptor(int submap_id, int keyframe_idx, std::vector<float> &descriptor_out) const
    {
        std::lock_guard<std::mutex> lock(descriptor_mutex_);
        const auto submap_it = descriptor_db_.find(submap_id);
        if (submap_it == descriptor_db_.end())
        {
            return false;
        }

        const auto kf_it = submap_it->second.find(keyframe_idx);
        if (kf_it == submap_it->second.end())
        {
            return false;
        }

        descriptor_out = kf_it->second;
        return true;
    }

    float NetVLADWrapper::cosineSimilarity(const std::vector<float> &desc1,
                                           const std::vector<float> &desc2)
    {
        if (desc1.size() != desc2.size() || desc1.empty())
        {
            return 0.0f;
        }

        float dot_product = 0.0f;
        float norm1 = 0.0f;
        float norm2 = 0.0f;

        for (size_t i = 0; i < desc1.size(); ++i)
        {
            dot_product += desc1[i] * desc2[i];
            norm1 += desc1[i] * desc1[i];
            norm2 += desc2[i] * desc2[i];
        }

        const float denom = std::sqrt(norm1) * std::sqrt(norm2);
        if (denom < 1e-9f)
        {
            return 0.0f;
        }

        return dot_product / denom;
    }

    void NetVLADWrapper::normalizeDescriptor(std::vector<float> &descriptor)
    {
        float norm = 0.0f;
        for (float value : descriptor)
        {
            norm += value * value;
        }

        norm = std::sqrt(norm);
        if (norm < 1e-9f)
        {
            return;
        }

        for (float &value : descriptor)
        {
            value /= norm;
        }
    }

} // namespace f_vigs_slam
