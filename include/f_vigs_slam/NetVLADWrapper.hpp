#pragma once

#include <arpa/inet.h>
#include <cstdint>
#include <mutex>
#include <netinet/in.h>
#include <string>
#include <sys/socket.h>
#include <unistd.h>
#include <vector>
#include <unordered_map>
#include <cmath>
#include <algorithm>
#include <opencv2/core.hpp>
#include "f_vigs_slam/RepresentationClasses.hpp"

namespace f_vigs_slam
{
    /**
     * @brief NetVLADWrapper
     * Interfaz para extracción de descriptores globales y búsqueda de candidatos de loop closure.
     * El extractor se apoya en la implementación pre-entrenada de NetVLAD de hloc.
     */
    class NetVLADWrapper
    {
    public:
        NetVLADWrapper();
        ~NetVLADWrapper();

        // Bloquea hasta que el servidor NetVLAD acepte la conexión TCP.
        bool waitForServerReady();

        // ===== DESCRIPTOR EXTRACTION =====
        /**
         * Extrae un descriptor global de una imagen RGB.
         * Usa hloc/NetVLAD (VGG16-NetVLAD-Pitts30K) y retorna descriptor normalizado.
         */
        std::vector<float> extractDescriptor(const cv::Mat &rgb_image);

        // Devuelve la dimensionalidad esperada del descriptor NetVLAD de hloc.
        inline int descriptorSize() const { return descriptor_size_; }

        // ===== LOOP CANDIDATE STRUCTURES =====
        
        struct LoopCandidate
        {
            int submap_id;
            int keyframe_idx;
            float similarity;  // [0, 1], cosine similarity
        };

        // ===== CANDIDATE SEARCH =====
        /**
         * Busca Top-K candidatos más similares en la base histórica.
         * Usa búsqueda lineal + similitud coseno.
         */
        std::vector<LoopCandidate> searchCandidates(
            const std::vector<float> &query_descriptor,
            int k = 5,
            float min_similarity_thresh = 0.3f);

        // ===== TEMPORAL FILTERING =====
        /**
         * Filtra candidatos demasiado cercanos en tiempo.
         * Excluye keyframes en rango [current - temporal_window, current].
         */
        std::vector<LoopCandidate> filterByTemporal(
            const std::vector<LoopCandidate> &candidates,
            int current_submap_id,
            int current_keyframe_idx,
            int temporal_window = 10);

        // ===== GEOMETRIC FILTERING =====
        /**
         * Filtra candidatos muy cercanos espacialmente.
         * Excluye loops si distancia Euclidea < distance_threshold.
         */
        std::vector<LoopCandidate> filterByDistance(
            const std::vector<LoopCandidate> &candidates,
            const std::vector<Pose> &submap_poses,
            float distance_threshold = 0.2f);

        // ===== DESCRIPTOR MANAGEMENT =====
        /**
         * Agrega un descriptor a la base de datos indexado por submap_id y keyframe_idx.
         */
        void addDescriptor(int submap_id, int keyframe_idx, const std::vector<float> &descriptor);

        /**
         * Limpia completamente la base de datos de descriptores.
         */
        void clear();

        // Consulta thread-safe para recuperar descriptor ya almacenado.
        bool getDescriptor(int submap_id, int keyframe_idx, std::vector<float> &descriptor_out) const;

    private:
        struct SocketConfig
        {
            std::string host = "127.0.0.1";
            int port = 5000;
            int connect_retries = 120;
            int connect_delay_ms = 500;
        };

        // ===== SOCKET BRIDGE =====
        SocketConfig resolveSocketConfig() const;
        bool ensureConnected();
        bool connectSocket();
        void closeSocket();
        bool sendAll(const void *data, size_t size);
        bool recvAll(void *data, size_t size);

        // ===== DESCRIPTOR DATABASE =====
        // Base de datos: [submap_id][keyframe_idx] -> descriptor
        std::unordered_map<int, std::unordered_map<int, std::vector<float>>> descriptor_db_;
        mutable std::mutex descriptor_mutex_;
        mutable std::mutex socket_mutex_;

        int descriptor_size_ = 4096;
        SocketConfig socket_config_;
        int socket_fd_ = -1;
        bool socket_ready_ = false;
        bool verbose_logs_ = true;
        size_t extraction_count_ = 0;

        // ===== HELPER FUNCTIONS =====
        /**
         * Calcula similitud coseno entre dos descriptores (ambos deben estar normalizados).
         */
        float cosineSimilarity(const std::vector<float> &desc1, const std::vector<float> &desc2);

        /**
         * Normaliza un descriptor a norma L2 = 1.
         */
        void normalizeDescriptor(std::vector<float> &descriptor);
    };

} // namespace f_vigs_slam
