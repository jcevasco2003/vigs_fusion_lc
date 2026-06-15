#include "f_vigs_slam/LoopClosureModule.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <limits>
#include <numeric>
#include <sstream>
#include <string>
#include <unordered_map>
#include <opencv2/core.hpp>
#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <thrust/device_vector.h>
#include <thrust/copy.h>
#include <thrust/host_vector.h>
#include <ceres/ceres.h>
#include <open3d/Open3D.h>
#include "f_vigs_slam/GSCudaKernels.cuh"
#include "f_vigs_slam/LoopClosureModule_ceres_helpers.hpp"

namespace f_vigs_slam
{
    namespace
    {
        std::string poseToString(const Pose &pose)
        {
            std::ostringstream oss;
            oss << "pos=(" << pose.position.x << ", " << pose.position.y << ", " << pose.position.z << ")"
                << " quat=(" << pose.orientation.x << ", " << pose.orientation.y << ", " << pose.orientation.z << ", " << pose.orientation.w << ")";
            return oss.str();
        }

        float cosineSimilarity(const std::vector<float> &a, const std::vector<float> &b)
        {
            if (a.size() != b.size() || a.empty())
            {
                return 0.0f;
            }

            float dot = 0.0f;
            float norm_a = 0.0f;
            float norm_b = 0.0f;
            for (size_t i = 0; i < a.size(); ++i)
            {
                dot += a[i] * b[i];
                norm_a += a[i] * a[i];
                norm_b += b[i] * b[i];
            }

            const float denom = std::sqrt(norm_a) * std::sqrt(norm_b);
            if (denom < 1e-9f)
            {
                return 0.0f;
            }
            return dot / denom;
        }

        float percentile(std::vector<float> values, float p)
        {
            if (values.empty())
            {
                return 0.0f;
            }
            const float pc = std::clamp(p, 0.0f, 100.0f) / 100.0f;
            const size_t idx = static_cast<size_t>(pc * static_cast<float>(values.size() - 1));
            std::nth_element(values.begin(), values.begin() + static_cast<std::ptrdiff_t>(idx), values.end());
            return values[idx];
        }

        float submapSelfSimilarityThreshold(const Submap &submap, float percentile_value)
        {
            if (submap.keyframes.size() < 2)
            {
                return 0.0f;
            }

            std::vector<float> sims;
            sims.reserve((submap.keyframes.size() * (submap.keyframes.size() - 1)) / 2);
            for (size_t i = 0; i < submap.keyframes.size(); ++i)
            {
                const auto &d_i = submap.keyframes[i].getDescriptor();
                if (d_i.empty())
                {
                    continue;
                }
                for (size_t j = i + 1; j < submap.keyframes.size(); ++j)
                {
                    const auto &d_j = submap.keyframes[j].getDescriptor();
                    if (d_j.empty())
                    {
                        continue;
                    }
                    sims.push_back(cosineSimilarity(d_i, d_j));
                }
            }

            return percentile(std::move(sims), percentile_value);
        }

        float submapMinSimilarity(const Submap &submap)
        {
            if (submap.keyframes.size() < 2)
            {
                return 1.0f;
            }

            float min_sim = 1.0f;
            bool has_pair = false;
            for (size_t i = 0; i < submap.keyframes.size(); ++i)
            {
                const auto &d_i = submap.keyframes[i].getDescriptor();
                if (d_i.empty())
                {
                    continue;
                }
                for (size_t j = i + 1; j < submap.keyframes.size(); ++j)
                {
                    const auto &d_j = submap.keyframes[j].getDescriptor();
                    if (d_j.empty())
                    {
                        continue;
                    }
                    min_sim = std::min(min_sim, cosineSimilarity(d_i, d_j));
                    has_pair = true;
                }
            }

            return has_pair ? min_sim : 1.0f;
        }

        float crossSimilarityPercentile(const Submap &a,
                                        const Submap &b,
                                        float percentile_value,
                                        int max_keyframes)
        {
            if (a.keyframes.empty() || b.keyframes.empty())
            {
                return 0.0f;
            }

            const size_t a_start = (a.keyframes.size() > static_cast<size_t>(max_keyframes))
                ? a.keyframes.size() - static_cast<size_t>(max_keyframes)
                : 0;
            const size_t b_start = (b.keyframes.size() > static_cast<size_t>(max_keyframes))
                ? b.keyframes.size() - static_cast<size_t>(max_keyframes)
                : 0;

            std::vector<float> sims;
            sims.reserve(static_cast<size_t>(max_keyframes * max_keyframes));
            for (size_t i = a_start; i < a.keyframes.size(); ++i)
            {
                const auto &d_i = a.keyframes[i].getDescriptor();
                if (d_i.empty())
                {
                    continue;
                }
                for (size_t j = b_start; j < b.keyframes.size(); ++j)
                {
                    const auto &d_j = b.keyframes[j].getDescriptor();
                    if (d_j.empty())
                    {
                        continue;
                    }
                    sims.push_back(cosineSimilarity(d_i, d_j));
                }
            }

            return percentile(std::move(sims), percentile_value);
        }

        // ================= MATRIZ DE INFORMACION PARA LOOP EDGES =================
        // Esta funcion construye la matriz de informacion 6x6 (inversa de covarianza)
        // de una arista de loop closure.
        // Orden de variables: [tx, ty, tz, rx, ry, rz].
        // Idea: mayor confianza/overlap/inliers y menor residual/incertidumbre IMU
        // -> mayor peso en la optimizacion.
        
        Eigen::Matrix<float,6,6> computeInformationMatrix(const Submap* src,
                                                          const Submap* tgt,
                                                          float visual_similarity,
                                                          float overlap,
                                                          float mean_residual,
                                                          int num_inliers,
                                                          float confidence)
        {
            // AHORA MISMO NO SE USA

            //


            Eigen::Matrix<float,6,6> info = Eigen::Matrix<float,6,6>::Zero();

            float sim_term = (visual_similarity - 0.5f) * 2.0f;
            sim_term = std::clamp(sim_term, 0.0f, 1.0f);

            // Normalizacion/clamp de entradas para evitar valores fuera de rango.
            float conf = std::clamp(confidence, 0.0f, 1.0f);
            float ov = std::clamp(overlap, 0.0f, 1.0f);
            float res_term = std::exp(-mean_residual); // decays with residual
            float inlier_term = std::sqrt(std::max(1.0f, static_cast<float>(num_inliers))); // grows with inliers
            inlier_term /= 10.0f;
            inlier_term = std::clamp(inlier_term, 0.2f, 3.0f);

            // Incertidumbre acumulada de ambos submapas (proxy de fiabilidad inercial).
            float imu_trans_unc = 0.0f;
            float imu_rot_unc_deg = 0.0f;
            if (src) {
                imu_trans_unc += src->accumulated_translation_uncertainty_m;
                imu_rot_unc_deg += src->accumulated_rotation_uncertainty_deg;
            }
            if (tgt) {
                imu_trans_unc += tgt->accumulated_translation_uncertainty_m;
                imu_rot_unc_deg += tgt->accumulated_rotation_uncertainty_deg;
            }
            
            // Relacion inversa: menor incertidumbre IMU -> mayor peso de informacion.
            float imu_trans_prior = 1.0f / (1.0f + 0.01f * imu_trans_unc);  // stronger prior on translation
            float imu_rot_prior = 1.0f / (1.0f + 0.0001f * imu_rot_unc_deg); // stronger prior on rotation

            // Escalas base (ajustables): translacion y rotacion.
            const float base_trans_scale = 800.0f;  // translational information weight (increased for stronger prior)
            const float base_rot_scale = 1000.0f;    // rotational information weight (increased for stronger prior)

            std::cout << "-------------------------------------------------------- computeInformationMatrix ----" << std::endl;

            std::cout
                << "conf=" << conf
                << " sim_term=" << sim_term
                << " ov=" << ov
                << " res_term=" << res_term
                << " inlier_term=" << inlier_term
                << " imu_trans_prior=" << imu_trans_prior
                << " imu_rot_prior=" << imu_rot_prior
                << std::endl;
        
            std::cout << "-------------------------------------------------------- computeInformationMatrix END ----" << std::endl;

            // Pesos diagonales principales para bloques translacional y rotacional.
            float omega_trans = base_trans_scale * conf * sim_term * ov * res_term * inlier_term * imu_trans_prior;
            float omega_rot = base_rot_scale * conf * sim_term * (0.5f + 0.5f * res_term) * imu_rot_prior;

            // Saturacion para estabilidad numerica en la factorizacion posterior.
            omega_trans = std::clamp(omega_trans, 1e-2f, 1e3f);
            omega_rot = std::clamp(omega_rot, 1e-3f, 1e3f);

            // Bloques diagonales: penalizan error puro de traslacion y rotacion.
            info.block<3,3>(0,0) = Eigen::Matrix3f::Identity() * omega_trans;
            info.block<3,3>(3,3) = Eigen::Matrix3f::Identity() * omega_rot;

            // Terminos de acoplamiento traslacion-rotacion (off-diagonal).
            // Representan correlacion entre errores de posicion y orientacion en SE(3).
            float coupling_strength = 0.1f * conf * res_term * imu_trans_prior * imu_rot_prior;
            coupling_strength = std::clamp(coupling_strength, 0.0f, 10.0f);
            
            // Se usa un acoplamiento diagonal pequeno para no volver mal condicionada la matriz.
            Eigen::Matrix<float,3,3> coupling = Eigen::Matrix<float,3,3>::Zero();
            for (int i = 0; i < 3; ++i) {
                coupling(i, i) = coupling_strength * 0.05f;
            }
            
            info.block<3,3>(0,3) = coupling;  // translation-rotation coupling
            info.block<3,3>(3,0) = coupling;  // rotation-translation coupling (symmetric)

            // TEST: meto esto en 0
            info.block<3,3>(0,3).setZero();
            info.block<3,3>(3,0).setZero();

            return info;
        }

        // ================= MATRIZ DE INFORMACION PARA EDGES ODOM/IMU =================
        // Esta matriz se usa en aristas consecutivas del grafo (i -> i+1).
        // Su objetivo es mantener fuerte consistencia local con la odometria/IMU.
        Eigen::Matrix<float,6,6> computeIMUPriorInformationMatrix(const Submap* src,
                                                                  const Submap* tgt)
        {
            Eigen::Matrix<float,6,6> info = Eigen::Matrix<float,6,6>::Zero();

            // Se acumula incertidumbre de ambos nodos extremos de la arista.
            float imu_trans_unc = 0.0f;
            float imu_rot_unc_deg = 0.0f;
            if (src) {
                imu_trans_unc += src->accumulated_translation_uncertainty_m;
                imu_rot_unc_deg += src->accumulated_rotation_uncertainty_deg;
            }
            if (tgt) {
                imu_trans_unc += tgt->accumulated_translation_uncertainty_m;
                imu_rot_unc_deg += tgt->accumulated_rotation_uncertainty_deg;
            }

            // Prior IMU mas fuerte que loop closure para estabilizar la cadena local.
            const float imu_base_trans_scale = 10.0f;
            const float imu_base_rot_scale = 1000.0f;

            // Menor incertidumbre -> mayor informacion.
            float imu_trans_confidence = 1.0f / (1.0f + 0.3f * imu_trans_unc);
            float imu_rot_confidence = 1.0f / (1.0f + 0.05f * imu_rot_unc_deg);

            float omega_trans = imu_base_trans_scale * imu_trans_confidence;
            float omega_rot = imu_base_rot_scale * imu_rot_confidence;

            // Limites para evitar pesos extremos.
            omega_trans = std::clamp(omega_trans, 1.0f, 1e4f);
            omega_rot = std::clamp(omega_rot, 0.1f, 1e4f);

            // Se asume independencia entre ejes (matriz diagonal).
            info.block<3,3>(0,0) = Eigen::Matrix3f::Identity() * omega_trans;
            info.block<3,3>(3,3) = Eigen::Matrix3f::Identity() * omega_rot;

            return info;
        }

        // ================= UTILIDADES DE TRANSFORMACIONES SE(3) =================
        // Conversores Pose <-> Isometry y operaciones de composicion/inversion.
        // Convencion: Pose representa transformaciones 3D con quaternion (x,y,z,w).
        Eigen::Isometry3f poseToIso(const Pose &pose)
        {
            Eigen::Quaternionf q(pose.orientation.w,
                                 pose.orientation.x,
                                 pose.orientation.y,
                                 pose.orientation.z);
            q.normalize();

            Eigen::Isometry3f T = Eigen::Isometry3f::Identity();
            T.linear() = q.toRotationMatrix();
            T.translation() = Eigen::Vector3f(pose.position.x, pose.position.y, pose.position.z);
            return T;
        }

        Pose isoToPose(const Eigen::Isometry3f &T)
        {
            Eigen::Quaternionf q(T.linear());
            q.normalize();
            return Pose(
                make_float3(T.translation().x(), T.translation().y(), T.translation().z()),
                make_float4(q.x(), q.y(), q.z(), q.w()));
        }

        Pose composePose(const Pose &a, const Pose &b)
        {
            // Composicion: aplica primero 'a' y luego 'b'.
            return isoToPose(poseToIso(a) * poseToIso(b));
        }

        Pose inversePose(const Pose &a)
        {
            // Inversa rigid body: pasa de T a T^{-1}.
            return isoToPose(poseToIso(a).inverse());
        }
        // Use helpers from LoopClosureModule.hpp: poseTranslationError / poseRotationDeg

        Pose blendPose(const Pose &current, const Pose &target, float alpha)
        {
            const float a = std::clamp(alpha, 0.0f, 1.0f);
            Eigen::Quaternionf q_cur(current.orientation.w,
                                     current.orientation.x,
                                     current.orientation.y,
                                     current.orientation.z);
            Eigen::Quaternionf q_tgt(target.orientation.w,
                                     target.orientation.x,
                                     target.orientation.y,
                                     target.orientation.z);
            q_cur.normalize();
            q_tgt.normalize();
            Eigen::Quaternionf q_out = q_cur.slerp(a, q_tgt);
            q_out.normalize();

            Pose out;
            out.position = make_float3(
                current.position.x + a * (target.position.x - current.position.x),
                current.position.y + a * (target.position.y - current.position.y),
                current.position.z + a * (target.position.z - current.position.z));
            out.orientation = make_float4(q_out.x(), q_out.y(), q_out.z(), q_out.w());
            return out;
        }

        std::vector<Eigen::Vector3f> backprojectDepthImage(
            const KeyframeData &keyframe,
            size_t max_points)
        {
            std::vector<Eigen::Vector3f> points;
            if (keyframe.depth_img.empty() || max_points == 0)
            {
                return points;
            }

            const int width = keyframe.depth_img.cols;
            const int height = keyframe.depth_img.rows;
            if (width <= 0 || height <= 0)
            {
                return points;
            }

            const float fx = keyframe.intrinsics.f.x;
            const float fy = keyframe.intrinsics.f.y;
            const float cx = keyframe.intrinsics.c.x;
            const float cy = keyframe.intrinsics.c.y;
            if (fx <= 1e-6f || fy <= 1e-6f)
            {
                return points;
            }

            const size_t total_pixels = static_cast<size_t>(width) * static_cast<size_t>(height);
            const size_t target_points = std::max<size_t>(1, max_points);
            const int stride = std::max(
                1,
                static_cast<int>(std::sqrt(static_cast<double>(total_pixels) / static_cast<double>(target_points))));

            thrust::device_vector<float3> device_points(target_points);
            thrust::device_vector<uint32_t> device_count(1, 0u);

            const dim3 block(16, 16);
            const dim3 grid(
                static_cast<unsigned int>((width + static_cast<int>(block.x) - 1) / static_cast<int>(block.x)),
                static_cast<unsigned int>((height + static_cast<int>(block.y) - 1) / static_cast<int>(block.y)));

            backprojectDepthImage_kernel<<<grid, block>>>(
                thrust::raw_pointer_cast(device_points.data()),
                thrust::raw_pointer_cast(device_count.data()),
                static_cast<uint32_t>(target_points),
                keyframe.depth_img.ptr<float>(),
                keyframe.depth_img.step,
                width,
                height,
                keyframe.intrinsics,
                static_cast<uint32_t>(stride));

            const cudaError_t launch_error = cudaPeekAtLastError();
            if (launch_error != cudaSuccess)
            {
                std::cout << "[LoopClosureModule] backprojectDepthImage CUDA launch failed: "
                          << cudaGetErrorString(launch_error) << std::endl;
                return points;
            }

            const cudaError_t sync_error = cudaDeviceSynchronize();
            if (sync_error != cudaSuccess)
            {
                std::cout << "[LoopClosureModule] backprojectDepthImage CUDA sync failed: "
                          << cudaGetErrorString(sync_error) << std::endl;
                return points;
            }

            uint32_t count = 0u;
            thrust::copy(device_count.begin(), device_count.end(), &count);
            count = std::min<uint32_t>(count, static_cast<uint32_t>(target_points));

            thrust::host_vector<float3> host_points(count);
            if (count > 0u)
            {
                thrust::copy_n(device_points.begin(), count, host_points.begin());
            }

            points.reserve(count);
            for (uint32_t i = 0; i < count; ++i)
            {
                const float3 &p = host_points[i];
                if (std::isfinite(p.x) && std::isfinite(p.y) && std::isfinite(p.z))
                {
                    points.emplace_back(p.x, p.y, p.z);
                }
            }

            return points;
        }

        Eigen::Matrix3f averageRotations(
            const std::vector<Eigen::Matrix3f> &rotations,
            const std::vector<float> &weights)
        {
            Eigen::Matrix3f mean = Eigen::Matrix3f::Zero();
            if (rotations.empty() || rotations.size() != weights.size())
            {
                return Eigen::Matrix3f::Identity();
            }

            float total_weight = 0.0f;
            for (size_t i = 0; i < rotations.size(); ++i)
            {
                if (!rotations[i].allFinite())
                {
                    continue;
                }

                const float w = std::max(0.0f, weights[i]);
                mean += w * rotations[i];
                total_weight += w;
            }

            if (total_weight <= 1e-6f)
            {
                return Eigen::Matrix3f::Identity();
            }

            mean /= total_weight;
            Eigen::JacobiSVD<Eigen::Matrix3f> svd(mean, Eigen::ComputeFullU | Eigen::ComputeFullV);
            Eigen::Matrix3f R = svd.matrixU() * svd.matrixV().transpose();
            if (R.determinant() < 0.0f)
            {
                Eigen::Matrix3f U = svd.matrixU();
                U.col(2) *= -1.0f;
                R = U * svd.matrixV().transpose();
            }
            return R;
        }

        struct PairRegistration
        {
            Pose relative_pose;
            float pair_similarity = 0.0f;
            float residual = std::numeric_limits<float>::infinity();
            float weight = 0.0f;
            int inliers = 0;
        };

        std::vector<Eigen::Vector3f> extractGaussianMeans(const Submap *submap, size_t max_points)
        {
            std::vector<Eigen::Vector3f> points;
            if (!submap || submap->gaussians_count == 0)
            {
                return points;
            }

            const size_t n = std::min(static_cast<size_t>(submap->gaussians_count), max_points);
            thrust::host_vector<float4> host_positions(n);
            thrust::copy_n(submap->gaussians.positions.begin(), static_cast<std::ptrdiff_t>(n), host_positions.begin());

            points.reserve(n);
            for (size_t i = 0; i < n; ++i)
            {
                const float4 &p = host_positions[i];
                if (std::isfinite(p.x) && std::isfinite(p.y) && std::isfinite(p.z))
                {
                    points.emplace_back(p.x, p.y, p.z);
                }
            }
            return points;
        }

        Eigen::Vector3f centroidOf(const std::vector<Eigen::Vector3f> &pts)
        {
            if (pts.empty())
            {
                return Eigen::Vector3f::Zero();
            }

            Eigen::Vector3f c = Eigen::Vector3f::Zero();
            for (const auto &p : pts)
            {
                c += p;
            }
            c /= static_cast<float>(pts.size());
            return c;
        }

        struct ReciprocalMatch
        {
            int src_idx = -1;
            int tgt_idx = -1;
            float dist = std::numeric_limits<float>::infinity();
        };

        std::vector<Eigen::Vector3f> transformPoints(
            const std::vector<Eigen::Vector3f> &pts,
            const Eigen::Isometry3f &T)
        {
            std::vector<Eigen::Vector3f> out;
            out.reserve(pts.size());
            for (const auto &p : pts)
            {
                out.emplace_back(T * p);
            }
            return out;
        }

        std::vector<ReciprocalMatch> computeReciprocalMatches(
            const std::vector<Eigen::Vector3f> &src_pts,
            const std::vector<Eigen::Vector3f> &tgt_pts,
            bool apply_distance_gate,
            float max_distance,
            size_t max_pairs)
        {
            std::vector<ReciprocalMatch> matches;
            if (src_pts.empty() || tgt_pts.empty())
            {
                return matches;
            }

            const float max_dist_sq = max_distance * max_distance;

            std::vector<int> src_to_tgt(src_pts.size(), -1);
            std::vector<float> src_to_tgt_dist_sq(src_pts.size(), std::numeric_limits<float>::infinity());
            for (size_t i = 0; i < src_pts.size(); ++i)
            {
                for (size_t j = 0; j < tgt_pts.size(); ++j)
                {
                    const float d2 = (src_pts[i] - tgt_pts[j]).squaredNorm();
                    if (d2 < src_to_tgt_dist_sq[i])
                    {
                        src_to_tgt_dist_sq[i] = d2;
                        src_to_tgt[i] = static_cast<int>(j);
                    }
                }
            }

            std::vector<int> tgt_to_src(tgt_pts.size(), -1);
            std::vector<float> tgt_to_src_dist_sq(tgt_pts.size(), std::numeric_limits<float>::infinity());
            for (size_t j = 0; j < tgt_pts.size(); ++j)
            {
                for (size_t i = 0; i < src_pts.size(); ++i)
                {
                    const float d2 = (tgt_pts[j] - src_pts[i]).squaredNorm();
                    if (d2 < tgt_to_src_dist_sq[j])
                    {
                        tgt_to_src_dist_sq[j] = d2;
                        tgt_to_src[j] = static_cast<int>(i);
                    }
                }
            }

            matches.reserve(std::min(src_pts.size(), tgt_pts.size()));
            for (size_t i = 0; i < src_pts.size(); ++i)
            {
                const int j = src_to_tgt[i];
                if (j < 0)
                {
                    continue;
                }
                if (tgt_to_src[static_cast<size_t>(j)] != static_cast<int>(i))
                {
                    continue;
                }

                const float d2 = src_to_tgt_dist_sq[i];
                if (apply_distance_gate && d2 > max_dist_sq)
                {
                    continue;
                }

                ReciprocalMatch m;
                m.src_idx = static_cast<int>(i);
                m.tgt_idx = j;
                m.dist = std::sqrt(std::max(0.0f, d2));
                matches.push_back(m);
            }

            std::sort(matches.begin(), matches.end(),
                      [](const ReciprocalMatch &a, const ReciprocalMatch &b)
                      {
                          return a.dist < b.dist;
                      });

            if (matches.size() > max_pairs)
            {
                matches.resize(max_pairs);
            }

            return matches;
        }

        bool isFinitePose(const Pose &p)
        {
            return std::isfinite(p.position.x) &&
                   std::isfinite(p.position.y) &&
                   std::isfinite(p.position.z) &&
                   std::isfinite(p.orientation.x) &&
                   std::isfinite(p.orientation.y) &&
                   std::isfinite(p.orientation.z) &&
                   std::isfinite(p.orientation.w);
        }

        // Diagnostic structures for Ceres logging
        struct ResidualDescriptor {
            ceres::CostFunction* cost_function = nullptr; // not owning
            std::vector<double*> parameter_blocks; // pointers to parameter block memory
            Eigen::Matrix<double,6,6> sqrt_info = Eigen::Matrix<double,6,6>::Identity();
            int residual_size = 6;
            std::string label;
            int source_submap_id = -1;
            int target_submap_id = -1;
        };

        class CeresIterationLogger : public ceres::IterationCallback {
        public:
            CeresIterationLogger(const std::vector<ResidualDescriptor> &residuals,
                                 std::vector<double> &global_poses_ref,
                                 int num_poses)
                : residuals_(residuals), global_poses_(global_poses_ref), num_poses_(num_poses)
            {
                prev_params_ = global_poses_; // snapshot
            }

            ceres::CallbackReturnType operator()(const ceres::IterationSummary& summary) override {
                // compute param step (cur - prev)
                double max_trans_step = 0.0;
                double max_rot_step_deg = 0.0;
                for (int p = 0; p < num_poses_; ++p) {
                    const size_t off = static_cast<size_t>(p) * 7;
                    double dx = global_poses_[off+0] - prev_params_[off+0];
                    double dy = global_poses_[off+1] - prev_params_[off+1];
                    double dz = global_poses_[off+2] - prev_params_[off+2];
                    double trans = std::sqrt(dx*dx + dy*dy + dz*dz);
                    if (trans > max_trans_step) max_trans_step = trans;
                    Eigen::Quaterniond q_prev(prev_params_[off+6], prev_params_[off+3], prev_params_[off+4], prev_params_[off+5]);
                    Eigen::Quaterniond q_cur(global_poses_[off+6], global_poses_[off+3], global_poses_[off+4], global_poses_[off+5]);
                    q_prev.normalize(); q_cur.normalize();
                    Eigen::Quaterniond dq = q_prev.conjugate() * q_cur;
                    double angle = 2.0 * std::acos(std::min(1.0, std::abs(dq.w())));
                    double angle_deg = angle * 180.0 / M_PI;
                    if (angle_deg > max_rot_step_deg) max_rot_step_deg = angle_deg;
                }
                /*
                std::cerr << "[PGO][CERES][iter " << summary.iteration << "]"
                          << " num_poses=" << num_poses_
                          << " num_residuals=" << residuals_.size()
                          << " cost=" << summary.cost
                          << " step_max_trans_m=" << max_trans_step
                          << " step_max_rot_deg=" << max_rot_step_deg
                          << " objective_change=" << summary.cost_change
                          << " gradient_max_norm=" << summary.gradient_max_norm
                          << " step_norm=" << summary.step_norm
                          << " trust_region_radius=" << summary.trust_region_radius
                          << " linear_solver_iterations=" << summary.linear_solver_iterations
                          << std::endl;
                
                for (int p = 0; p < num_poses_; ++p) {
                    const size_t off = static_cast<size_t>(p) * 7;
                    std::cerr << "[PGO][CERES][pose " << p << "]"
                              << " pos=(" << global_poses_[off+0] << ", " << global_poses_[off+1] << ", " << global_poses_[off+2] << ")"
                              << " quat=(" << global_poses_[off+3] << ", " << global_poses_[off+4] << ", " << global_poses_[off+5] << ", " << global_poses_[off+6] << ")"
                              << std::endl;
                }
           
                std::cerr << "[PGO][CERES][iter " << summary.iteration << "][counts]"
                          << " param_blocks=" << (num_poses_ * 1)
                          << " residual_blocks=" << residuals_.size()
                          << " pose_blocks_fixed=" << 1
                          << std::endl;
                */

                // Per-residual diagnostics
                for (size_t ri = 0; ri < residuals_.size(); ++ri) {
                    const ResidualDescriptor &rd = residuals_[ri];
                    std::cerr << "[PGO][CERES][res " << ri << "]"
                              << " label=" << rd.label
                              << " source_submap_id=" << rd.source_submap_id
                              << " target_submap_id=" << rd.target_submap_id
                              << " residual_size=" << rd.residual_size
                              << " param_blocks=" << rd.parameter_blocks.size()
                              << std::endl;
                    // prepare parameter pointers
                    std::vector<double*> params(rd.parameter_blocks.size());
                    for (size_t k = 0; k < rd.parameter_blocks.size(); ++k) params[k] = rd.parameter_blocks[k];

                    std::vector<double> residuals(rd.residual_size);
                    std::vector<std::vector<double>> jac_storage(params.size());
                    std::vector<double*> jac_ptrs;
                    jac_ptrs.resize(params.size());
                    for (size_t k = 0; k < params.size(); ++k) {
                        jac_storage[k].assign(static_cast<size_t>(rd.residual_size * 7), 0.0);
                        jac_ptrs[k] = jac_storage[k].data();
                    }

                    bool ok = rd.cost_function->Evaluate(params.data(), residuals.data(), jac_ptrs.data());
                    if (!ok) {
                        std::cerr << "[PGO][CERES][res " << ri << "][reject] reason=evaluate_failed"
                                  << " label=" << rd.label
                                  << " source_submap_id=" << rd.source_submap_id
                                  << " target_submap_id=" << rd.target_submap_id
                                  << std::endl;
                        continue;
                    }

                    Eigen::VectorXd rvec = Eigen::Map<Eigen::VectorXd>(residuals.data(), rd.residual_size);
                    Eigen::VectorXd sr = rd.sqrt_info * rvec; // scaled residual
                    double sr_norm = sr.norm();

                    std::ostringstream ss;
                    ss << "[PGO][CERES][res " << ri << "] raw_norm=" << rvec.norm()
                              << " scaled_norm=" << sr_norm
                              << " label=" << rd.label
                              << " source_submap_id=" << rd.source_submap_id
                              << " target_submap_id=" << rd.target_submap_id
                              << " residual_size=" << rd.residual_size
                              << " param_blocks=" << params.size();

                    // per-parameter-block gradient norm = || J_k^T * (S*r) ||
                    for (size_t k = 0; k < params.size(); ++k) {
                        // J_k is (residual_size x 7) stored row-major in jac_storage[k]
                        Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, 7, Eigen::RowMajor>> Jk(jac_storage[k].data(), rd.residual_size, 7);
                        Eigen::Matrix<double, 7, 1> g = Jk.transpose() * sr;
                        double gn = g.norm();
                        ss << " blk=" << k << " grad_norm=" << gn;
                    }
                    std::cerr << ss.str() << std::endl;
                }

                // snapshot params
                prev_params_ = global_poses_;

                return ceres::SOLVER_CONTINUE;
            }

        private:
            const std::vector<ResidualDescriptor> &residuals_;
            std::vector<double> &global_poses_;
            std::vector<double> prev_params_;
            int num_poses_;
        };
    } // namespace

    LoopClosureModule::LoopClosureModule()
    {
        std::cout << "[LoopClosureModule] Loop module initialized" << std::endl;
    }

    LoopClosureModule::~LoopClosureModule() = default;

    void LoopClosureModule::setConfiguration(const LoopClosureConfig &config)
    {
        config_ = config;
    }

    const LoopClosureConfig &LoopClosureModule::getConfiguration() const
    {
        return config_; 
    }

    std::vector<LoopEdge> LoopClosureModule::detectLoops(
        const std::vector<std::shared_ptr<Submap>>& submaps,
        float geometric_overlap_threshold,
        float self_similarity_percentile,
        int max_candidates_per_query)
    {
        std::vector<LoopEdge> loop_edges;

        if (submaps.size() < 2)
        {
            return loop_edges;
        }

        const size_t query_index = submaps.size() - 1;
        Submap *query = submaps[query_index].get();
        if (!query || query->keyframes.empty() || !query->keyframes.back().hasDescriptor())
        {
            return loop_edges;
        }

        query->self_similarity_percentile_score = submapSelfSimilarityThreshold(*query, self_similarity_percentile);
        query->min_descriptor_similarity = submapMinSimilarity(*query);
        query->has_descriptor_similarity_stats = true;

        const size_t begin_idx = (query_index > static_cast<size_t>(config_.max_submaps_to_compare))
            ? query_index - static_cast<size_t>(config_.max_submaps_to_compare)
            : 0;

        for (size_t i = begin_idx; i < query_index; ++i)
        {
            Submap *target = submaps[i].get();
            if (!target || target->keyframes.empty())
            {
                continue;
            }

            if (query->submap_id <= target->submap_id + static_cast<uint32_t>(config_.min_submap_gap))
            {
                continue;
            }

            if (!target->has_descriptor_similarity_stats)
            {
                target->self_similarity_percentile_score = submapSelfSimilarityThreshold(*target, self_similarity_percentile);
                target->min_descriptor_similarity = submapMinSimilarity(*target);
                target->has_descriptor_similarity_stats = true;
            }

            const float s_cross = crossSimilarityPercentile(*query, *target, self_similarity_percentile, config_.max_keyframes_per_submap);
            const float gate = std::min(query->self_similarity_percentile_score,
                                        target->self_similarity_percentile_score);

            std::cout << "[LoopDetect] i=" << query->submap_id
                      << " j=" << target->submap_id
                      << " s_self_i=" << query->self_similarity_percentile_score
                      << " s_self_j=" << target->self_similarity_percentile_score
                      << " s_cross=" << s_cross
                      << " min_i=" << query->min_descriptor_similarity
                      << " min_j=" << target->min_descriptor_similarity
                      << std::endl;

            if (!(s_cross > gate) || s_cross < config_.min_similarity_floor)
            {
                continue;
            }

            const Pose query_global_pose = query->getGlobalPose();
            const Pose target_global_pose = target->getGlobalPose();
            const float anchor_dist = poseDistance(query_global_pose, target_global_pose);
            const float anchor_rot = poseRotationDeg(query_global_pose, target_global_pose);
            const float imu_unc_score = query->accumulated_translation_uncertainty_m +
                                        target->accumulated_translation_uncertainty_m +
                                        0.02f * (query->accumulated_rotation_uncertainty_deg +
                                                 target->accumulated_rotation_uncertainty_deg);

            if (anchor_dist > config_.imu_max_anchor_distance_m ||
                anchor_rot > config_.imu_max_anchor_rotation_deg ||
                imu_unc_score > config_.imu_max_uncertainty_score)
            {
                std::cout << "[LoopDetect] rejected imu dist=" << anchor_dist
                          << " rot=" << anchor_rot
                          << " unc=" << imu_unc_score << std::endl;
                continue;
            }

            const float overlap = computeOverlapRatio(query, target);
            std::cout << "[LoopDetect] overlap=" << overlap << std::endl;
            if (overlap <= std::max(0.2f, geometric_overlap_threshold))
            {
                continue;
            }

            Pose relative_pose;
            float confidence = 0.0f;
            int inliers = 0;
            float mean_residual = 0.0f;

            // Open3D is the only registration backend used in production.
            try {
                const Pose open3d_init = composePose(inversePose(query_global_pose), target_global_pose);
                if (!registerSubmapsOpen3D(query, target, s_cross, open3d_init, relative_pose, confidence, inliers, mean_residual))
                {
                    continue;
                }
            } catch (...) {
                continue;
            }

            if (confidence < config_.loop_confidence_threshold)
            {
                continue;
            }

            LoopEdge edge;
            edge.source_submap_id = static_cast<int>(query->submap_id);
            edge.target_submap_id = static_cast<int>(target->submap_id);
            edge.relative_pose = relative_pose;
            edge.confidence = confidence;
            edge.visual_similarity = s_cross;
            edge.source_uncertainty = query->accumulated_translation_uncertainty_m +
                                      0.02f * query->accumulated_rotation_uncertainty_deg;
            edge.target_uncertainty = target->accumulated_translation_uncertainty_m +
                                      0.02f * target->accumulated_rotation_uncertainty_deg;
            edge.num_inliers = inliers;
            // compute and attach information matrix
            edge.information_matrix = computeInformationMatrix(query, target, s_cross, overlap, mean_residual, inliers, confidence);
            loop_edges.push_back(edge);
        }

        std::sort(loop_edges.begin(), loop_edges.end(),
                  [](const LoopEdge &a, const LoopEdge &b)
                  {
                      return a.confidence > b.confidence;
                  });

        if (loop_edges.size() > static_cast<size_t>(std::max(1, max_candidates_per_query)))
        {
            loop_edges.resize(static_cast<size_t>(std::max(1, max_candidates_per_query)));
        }

        return loop_edges;
    }

    inline Pose vectorToPose(const double* v)
    {
        Pose p;

        // translation
        p.position.x = static_cast<float>(v[0]);
        p.position.y = static_cast<float>(v[1]);
        p.position.z = static_cast<float>(v[2]);

        // quaternion (IMPORTANT: x,y,z,w order)
        p.orientation.x = static_cast<float>(v[3]);
        p.orientation.y = static_cast<float>(v[4]);
        p.orientation.z = static_cast<float>(v[5]);
        p.orientation.w = static_cast<float>(v[6]);

        return p;
    }

    LoopClosureModule::PGOResult LoopClosureModule::optimizePoseGraph(
        const std::vector<std::shared_ptr<Submap>>& submaps,
        const std::vector<LoopEdge>& loop_edges,
        int max_iterations)
    {
        // ================= BACKEND OPEN3D PARA PGO =================
        // Se arma un PoseGraph con:
        // 1) nodos = poses globales actuales,
        // 2) aristas odometricas (seguras),
        // 3) aristas de loop (uncertain=true),
        // y luego se ejecuta la optimizacion global de Open3D.
        /*if (config_.pgo_backend == "open3d") {
            using namespace open3d;
            PGOResult result;
            if (submaps.empty()) return result;

            std::cout << "[PGO][OPEN3D][start]"
                      << " submaps=" << submaps.size()
                      << " loop_edges=" << loop_edges.size()
                      << " max_iterations=" << max_iterations
                      << std::endl;

            pipelines::registration::PoseGraph pose_graph;
            // Create nodes with current global pose
            for (size_t i = 0; i < submaps.size(); ++i) {
                const Pose p = submaps[i]->getGlobalPose();
                Eigen::Quaterniond qd(p.orientation.w, p.orientation.x, p.orientation.y, p.orientation.z);
                qd.normalize();
                Eigen::Matrix4d T = Eigen::Matrix4d::Identity();
                T.block<3,3>(0,0) = qd.toRotationMatrix();
                T.block<3,1>(0,3) = Eigen::Vector3d(p.position.x, p.position.y, p.position.z);
                pose_graph.nodes_.push_back(pipelines::registration::PoseGraphNode(T));
                result.corrected_submap_poses.push_back(p); // placeholder

                std::cout << "[PGO][OPEN3D][node_init]"
                          << " node=" << i
                          << " submap_id=" << submaps[i]->submap_id
                          << " pose=" << poseToString(p)
                          << std::endl;
            }

            // Aristas odometricas: conectan submapas consecutivos y se marcan como no inciertas.
            for (size_t i = 0; i + 1 < submaps.size(); ++i) {
                const Pose rel = submaps[i+1]->T_relative;

                std::cout   << "[PGO][ODOM_MEAS_CHECK]"
                            << " i=" << i
                            << " Ti=" << poseToString(submaps[i]->getGlobalPose())
                            << " Tj=" << poseToString(submaps[i+1]->getGlobalPose())
                            << " rel_meas=" << poseToString(rel)
                            << std::endl;

                Eigen::Matrix4d T = Eigen::Matrix4d::Identity();
                Eigen::Quaterniond qd(rel.orientation.w, rel.orientation.x, rel.orientation.y, rel.orientation.z);
                qd.normalize();
                T.block<3,3>(0,0) = qd.toRotationMatrix();
                T.block<3,1>(0,3) = Eigen::Vector3d(rel.position.x, rel.position.y, rel.position.z);
                Eigen::Matrix<double,6,6> info = computeIMUPriorInformationMatrix(submaps[i].get(), submaps[i+1].get()).cast<double>();
                // Ensure symmetric
                info = 0.5 * (info + info.transpose());
                std::cout << "[PGO][OPEN3D][edge_odom]"
                          << " src_node=" << i
                          << " tgt_node=" << (i + 1)
                          << " src_submap_id=" << submaps[i]->submap_id
                          << " tgt_submap_id=" << submaps[i+1]->submap_id
                          << " meas=" << poseToString(rel)
                          << " info_diag=("
                          << info(0,0) << ", " << info(1,1) << ", " << info(2,2) << ", "
                          << info(3,3) << ", " << info(4,4) << ", " << info(5,5) << ")"
                          << " info_finite=" << (info.allFinite() ? 1 : 0)
                          << std::endl;
                pose_graph.edges_.push_back(pipelines::registration::PoseGraphEdge(static_cast<int>(i), static_cast<int>(i+1), T, info, false));
            }

            // Aristas de loop: restricciones de cierre de lazo, se marcan como inciertas.
            for (const auto &edge : loop_edges) {
                auto it_i = std::find_if(submaps.begin(), submaps.end(), [&](const std::shared_ptr<Submap> &s){ return static_cast<int>(s->submap_id) == edge.source_submap_id; });
                auto it_j = std::find_if(submaps.begin(), submaps.end(), [&](const std::shared_ptr<Submap> &s){ return static_cast<int>(s->submap_id) == edge.target_submap_id; });
                if (it_i == submaps.end() || it_j == submaps.end()) continue;
                const int idx_i = static_cast<int>(std::distance(submaps.begin(), it_i));
                const int idx_j = static_cast<int>(std::distance(submaps.begin(), it_j));
                Eigen::Matrix4d T = Eigen::Matrix4d::Identity();
                Eigen::Quaterniond qd(edge.relative_pose.orientation.w,
                                     edge.relative_pose.orientation.x,
                                     edge.relative_pose.orientation.y,
                                     edge.relative_pose.orientation.z);
                qd.normalize();
                T.block<3,3>(0,0) = qd.toRotationMatrix();
                T.block<3,1>(0,3) = Eigen::Vector3d(edge.relative_pose.position.x, edge.relative_pose.position.y, edge.relative_pose.position.z);
                Eigen::Matrix<double,6,6> info = edge.information_matrix.cast<double>();
                info = 0.5 * (info + info.transpose());
                // Atenuacion por confianza para no sobredimensionar loops debiles.
                const double gain = std::clamp(static_cast<double>(edge.confidence), 0.2, 0.95);
                info *= gain;
                std::cout << "[PGO][OPEN3D][edge_loop]"
                          << " src_node=" << idx_i
                          << " tgt_node=" << idx_j
                          << " src_submap_id=" << edge.source_submap_id
                          << " tgt_submap_id=" << edge.target_submap_id
                          << " confidence=" << edge.confidence
                          << " gain=" << gain
                          << " inliers=" << edge.num_inliers
                          << " similarity=" << edge.visual_similarity
                          << " meas=" << poseToString(edge.relative_pose)
                          << " info_diag=("
                          << info(0,0) << ", " << info(1,1) << ", " << info(2,2) << ", "
                          << info(3,3) << ", " << info(4,4) << ", " << info(5,5) << ")"
                          << " info_finite=" << (info.allFinite() ? 1 : 0)
                          << std::endl;
                pose_graph.edges_.push_back(pipelines::registration::PoseGraphEdge(
                    idx_i, idx_j, T, info, true, edge.confidence));
            }

            std::cout << "[PGO][OPEN3D][graph_built]"
                      << " nodes=" << pose_graph.nodes_.size()
                      << " edges=" << pose_graph.edges_.size()
                      << " odom_edges=" << (submaps.size() > 0 ? submaps.size() - 1 : 0)
                      << " loop_edges=" << loop_edges.size()
                      << std::endl;

            // Pre-solve per-edge predicted vs measured diagnostics
            std::cout << "[PGO][OPEN3D][pre_edge_deltas] start" << std::endl;
            for (size_t ei = 0; ei < pose_graph.edges_.size(); ++ei) {
                const auto &e = pose_graph.edges_[ei];
                const int src = e.source_node_id_;
                const int tgt = e.target_node_id_;
                if (src < 0 || tgt < 0 || src >= static_cast<int>(pose_graph.nodes_.size()) || tgt >= static_cast<int>(pose_graph.nodes_.size())) {
                    std::cout << "[PGO][OPEN3D][edge_pre] idx=" << ei << " invalid_node_ids src=" << src << " tgt=" << tgt << std::endl;
                    continue;
                }
                const Eigen::Matrix4d &Tsrc = pose_graph.nodes_[static_cast<size_t>(src)].pose_;
                const Eigen::Matrix4d &Ttgt = pose_graph.nodes_[static_cast<size_t>(tgt)].pose_;
                const Eigen::Matrix4d Tij_pred = Tsrc.inverse() * Ttgt;
                const Eigen::Matrix4d Delta = Tij_pred * e.transformation_.inverse();
                const Eigen::Vector3d dtrans = Delta.block<3,1>(0,3);
                const double delta_trans_m = dtrans.norm();
                Eigen::Matrix3d Rdelta = Delta.block<3,3>(0,0);
                Eigen::Quaterniond qd(Rdelta);
                qd.normalize();
                const double delta_rot_rad = 2.0 * std::acos(std::min(1.0, std::abs(qd.w())));
                const double delta_rot_deg = delta_rot_rad * 180.0 / M_PI;
                // information diagonal (if present)
                Eigen::Matrix<double,6,6> info_diag = Eigen::Matrix<double,6,6>::Zero();
                bool info_ok = false;
                try {
                    info_diag = e.information_;
                    info_ok = true;
                } catch (...) {
                    // fallthrough - some Open3D versions may have different field names; ignore
                    info_ok = false;
                }
                std::cout << "[PGO][OPEN3D][edge_pre] idx=" << ei
                          << " src=" << src << " tgt=" << tgt
                          << " delta_trans_m=" << delta_trans_m
                          << " delta_rot_deg=" << delta_rot_deg
                          << " uncertain=" << (e.uncertain_ ? 1 : 0)
                          << " info_diag=(";
                if (info_ok) {
                    std::cout << info_diag(0,0) << ", " << info_diag(1,1) << ", " << info_diag(2,2) << ", "
                              << info_diag(3,3) << ", " << info_diag(4,4) << ", " << info_diag(5,5);
                } else {
                    std::cout << "n/a";
                }
                std::cout << ")" << std::endl;
            }
            std::cout << "[PGO][OPEN3D][pre_edge_deltas] end" << std::endl;

            // Run Open3D global optimization
            pipelines::registration::GlobalOptimizationOption option;
            option.max_correspondence_distance_ = std::max(1e-6, config_.open3d_icp_min_distance_m);
            option.edge_prune_threshold_ = 0.25;
            option.reference_node_ = 0;
            pipelines::registration::GlobalOptimizationLevenbergMarquardt method;
            pipelines::registration::GlobalOptimizationConvergenceCriteria criteria;
            std::cout << "[PGO][OPEN3D][solve]"
                      << " max_correspondence_distance=" << option.max_correspondence_distance_
                      << " edge_prune_threshold=" << option.edge_prune_threshold_
                      << " reference_node=" << option.reference_node_
                      << std::endl;
            try {
                pipelines::registration::GlobalOptimization(pose_graph, method, criteria, option);
            } catch (const std::exception &e) {
                std::cout << "[LoopClosureModule::optimizePoseGraph][open3d] global optimization failed: " << e.what() << std::endl;
                return result;
            }

            std::cout << "[PGO][OPEN3D][solve_done]"
                      << " nodes=" << pose_graph.nodes_.size()
                      << " edges=" << pose_graph.edges_.size()
                      << std::endl;

            // Post-solve per-edge diagnostics using optimized pose_graph.nodes_
            std::cout << "[PGO][OPEN3D][post_edge_deltas] start" << std::endl;
            for (size_t ei = 0; ei < pose_graph.edges_.size(); ++ei) {
                const auto &e = pose_graph.edges_[ei];
                const int src = e.source_node_id_;
                const int tgt = e.target_node_id_;
                if (src < 0 || tgt < 0 || src >= static_cast<int>(pose_graph.nodes_.size()) || tgt >= static_cast<int>(pose_graph.nodes_.size())) {
                    std::cout << "[PGO][OPEN3D][edge_post] idx=" << ei << " invalid_node_ids src=" << src << " tgt=" << tgt << std::endl;
                    continue;
                }
                const Eigen::Matrix4d &Tsrc = pose_graph.nodes_[static_cast<size_t>(src)].pose_;
                const Eigen::Matrix4d &Ttgt = pose_graph.nodes_[static_cast<size_t>(tgt)].pose_;
                const Eigen::Matrix4d Tij_pred = Tsrc.inverse() * Ttgt;
                const Eigen::Matrix4d Delta = Tij_pred * e.transformation_.inverse();
                const Eigen::Vector3d dtrans = Delta.block<3,1>(0,3);
                const double delta_trans_m = dtrans.norm();
                Eigen::Matrix3d Rdelta = Delta.block<3,3>(0,0);
                Eigen::Quaterniond qd(Rdelta);
                qd.normalize();
                const double delta_rot_rad = 2.0 * std::acos(std::min(1.0, std::abs(qd.w())));
                const double delta_rot_deg = delta_rot_rad * 180.0 / M_PI;
                Eigen::Matrix<double,6,6> info_diag = Eigen::Matrix<double,6,6>::Zero();
                bool info_ok = false;
                try {
                    info_diag = e.information_;
                    info_ok = true;
                } catch (...) {
                    info_ok = false;
                }
                std::cout << "[PGO][OPEN3D][edge_post] idx=" << ei
                          << " src=" << src << " tgt=" << tgt
                          << " delta_trans_m=" << delta_trans_m
                          << " delta_rot_deg=" << delta_rot_deg
                          << " uncertain=" << (e.uncertain_ ? 1 : 0)
                          << " info_diag=(";
                if (info_ok) {
                    std::cout << info_diag(0,0) << ", " << info_diag(1,1) << ", " << info_diag(2,2) << ", "
                              << info_diag(3,3) << ", " << info_diag(4,4) << ", " << info_diag(5,5);
                } else {
                    std::cout << "n/a";
                }
                std::cout << ")" << std::endl;
            }
            std::cout << "[PGO][OPEN3D][post_edge_deltas] end" << std::endl;

            // Extract optimized poses
            for (size_t i = 0; i < submaps.size(); ++i) {
                const Eigen::Matrix4d &T = pose_graph.nodes_[i].pose_;
                Eigen::Matrix3d R = T.block<3,3>(0,0);
                Eigen::Quaterniond q(R);
                q.normalize();
                Pose p;
                p.position.x = static_cast<float>(T(0,3));
                p.position.y = static_cast<float>(T(1,3));
                p.position.z = static_cast<float>(T(2,3));
                p.orientation.x = static_cast<float>(q.x());
                p.orientation.y = static_cast<float>(q.y());
                p.orientation.z = static_cast<float>(q.z());
                p.orientation.w = static_cast<float>(q.w());
                result.corrected_submap_poses[i] = p;

                std::cout << "[PGO][OPEN3D][node_opt]"
                          << " node=" << i
                          << " submap_id=" << submaps[i]->submap_id
                          << " pose=" << poseToString(p)
                          << " delta_from_init_trans_m=" << translationError(submaps[i]->getGlobalPose(), p)
                          << " delta_from_init_rot_deg=" << poseRotationDeg(submaps[i]->getGlobalPose(), p)
                          << std::endl;
            }
            std::cout << "[PGO][OPEN3D][summary]"
                      << " converged=1"
                      << " residual_error=0"
                      << " note=no_direct_cost_from_open3d"
                      << std::endl;
            result.converged = true;
            result.residual_error = 0.0f; // no direct cost reported from Open3D API here
            return result;
        }*/
        PGOResult result;
        if (submaps.empty())
        {
            return result;
        }

        /*
        std::cout << "[LoopClosureModule::optimizePoseGraph][start]"
                  << " submaps=" << submaps.size()
                  << " loop_edges=" << loop_edges.size()
                  << " max_iterations=" << max_iterations
                  << std::endl;
        */
        // Guardamos la indexacion a partir del submap_id 
        std::unordered_map<int, size_t> id_to_index;
        id_to_index.reserve(submaps.size());
        for (size_t i = 0; i < submaps.size(); ++i)
        {
            id_to_index[static_cast<int>(submaps[i]->submap_id)] = i;
            result.corrected_submap_poses.push_back(submaps[i]->getGlobalPose());
        }

        const std::vector<Pose> initial_global_poses = result.corrected_submap_poses;

        // ================= BACKEND CERES PARA PGO =================
        // Parametrizacion: cada nodo es una pose global (tx,ty,tz,qx,qy,qz,qw).
        const size_t N = submaps.size();
        std::vector<double> global_poses;
        global_poses.resize(N * 7);
        for (size_t i = 0; i < N; ++i) {
            Pose p = submaps[i]->getGlobalPose();
            global_poses[i*7 + 0] = static_cast<double>(p.position.x);
            global_poses[i*7 + 1] = static_cast<double>(p.position.y);
            global_poses[i*7 + 2] = static_cast<double>(p.position.z);
            global_poses[i*7 + 3] = static_cast<double>(p.orientation.x);
            global_poses[i*7 + 4] = static_cast<double>(p.orientation.y);
            global_poses[i*7 + 5] = static_cast<double>(p.orientation.z);
            global_poses[i*7 + 6] = static_cast<double>(p.orientation.w);
            //std::cout << "[PGO][INIT][pose " << i << "]"
            //          << " submap_id=" << submaps[i]->submap_id
            //          << " global_pose=" << poseToString(p)
            //          << std::endl;
        }

        ceres::Problem problem;
        // Armado del problema
        for (size_t i = 0; i < N; ++i) {
            problem.AddParameterBlock(&global_poses[i*7], 7, new PoseLocalParameterization());
        }
        // Fijamos el primer nodo para que haya un marco global
        problem.SetParameterBlockConstant(&global_poses[0]);

        // Convierte matriz de informacion a su raiz (sqrt information) para pesar residuales.
        // Si LLT falla por condicionamiento, se regulariza diagonalmente y se usa fallback.
        auto computeSqrtInfo = [&](const Eigen::Matrix<double,6,6> &info)->Eigen::Matrix<double,6,6>{
            Eigen::Matrix<double,6,6> reg = info;
            // jitter diagonal if needed
            for (int k=0;k<6;++k) reg(k,k) += 1e-9;
            Eigen::LLT<Eigen::Matrix<double,6,6>> llt(reg);
            if (llt.info() != Eigen::Success) {
                // try regularize more
                Eigen::Matrix<double,6,6> reg2 = info;
                for (int k=0;k<6;++k) reg2(k,k) += 1e-6;
                Eigen::LLT<Eigen::Matrix<double,6,6>> llt2(reg2);
                if (llt2.info() != Eigen::Success) {
                    // fallback to diag
                    Eigen::Matrix<double,6,6> D = Eigen::Matrix<double,6,6>::Zero();
                    for (int k=0;k<6;++k) D(k,k) = std::sqrt(std::max(1e-6, static_cast<double>(info(k,k))));
                    return D;
                }
                return llt2.matrixL().transpose();
            }
            return llt.matrixL().transpose();
        };

        // Residuales odometricos: medida = T_relative del submapa hijo respecto al padre.
        // Estos bordes anclan fuertemente la geometria local usando prior IMU.
        std::vector<ResidualDescriptor> residuals_info;
        auto log_relative_convention = [&](const char *tag,
                                           const Pose &T_meas,
                                           const Pose &Ti,
                                           const Pose &Tj,
                                           int source_submap_id,
                                           int target_submap_id)
        {
            const Pose T_pred_ij = composePose(inversePose(Ti), Tj);
            const Pose T_pred_ji = composePose(inversePose(Tj), Ti);
            const Pose T_err_ij = composePose(inversePose(T_meas), T_pred_ij);
            const Pose T_err_ji = composePose(inversePose(T_meas), T_pred_ji);
            /*
            std::cout << "[PGO][CONVENTION][" << tag << "]"
                      << " source_submap_id=" << source_submap_id
                      << " target_submap_id=" << target_submap_id
                      << " T_meas=pos=(" << T_meas.position.x << ", " << T_meas.position.y << ", " << T_meas.position.z << ")"
                      << " quat=(" << T_meas.orientation.x << ", " << T_meas.orientation.y << ", " << T_meas.orientation.z << ", " << T_meas.orientation.w << ")"
                      << " T_pred_ij=pos=(" << T_pred_ij.position.x << ", " << T_pred_ij.position.y << ", " << T_pred_ij.position.z << ")"
                      << " quat=(" << T_pred_ij.orientation.x << ", " << T_pred_ij.orientation.y << ", " << T_pred_ij.orientation.z << ", " << T_pred_ij.orientation.w << ")"
                      << " T_err_ij=pos=(" << T_err_ij.position.x << ", " << T_err_ij.position.y << ", " << T_err_ij.position.z << ")"
                      << " quat=(" << T_err_ij.orientation.x << ", " << T_err_ij.orientation.y << ", " << T_err_ij.orientation.z << ", " << T_err_ij.orientation.w << ")"
                      << " T_pred_ji=pos=(" << T_pred_ji.position.x << ", " << T_pred_ji.position.y << ", " << T_pred_ji.position.z << ")"
                      << " quat=(" << T_pred_ji.orientation.x << ", " << T_pred_ji.orientation.y << ", " << T_pred_ji.orientation.z << ", " << T_pred_ji.orientation.w << ")"
                      << " T_err_ji=pos=(" << T_err_ji.position.x << ", " << T_err_ji.position.y << ", " << T_err_ji.position.z << ")"
                      << " quat=(" << T_err_ji.orientation.x << ", " << T_err_ji.orientation.y << ", " << T_err_ji.orientation.z << ", " << T_err_ji.orientation.w << ")"
                      << " err_ij_trans_m=" << translationError(T_meas, T_pred_ij)
                      << " err_ij_rot_deg=" << poseRotationDeg(T_meas, T_pred_ij)
                      << " err_ji_trans_m=" << translationError(T_meas, T_pred_ji)
                      << " err_ji_rot_deg=" << poseRotationDeg(T_meas, T_pred_ji)
                      << std::endl;
            */
        };
        for (size_t i = 0; i + 1 < N; ++i) {
            Pose rel = submaps[i+1]->T_relative; // Pose de i+1 relativa a i
            // IMU Prior
            Eigen::Matrix<float,6,6> info_f = computeIMUPriorInformationMatrix(submaps[i].get(), submaps[i+1].get());
            Eigen::Matrix<double,6,6> info = info_f.cast<double>();
            // Simetrica por las dudas
            info = 0.5 * (info + info.transpose());
            Eigen::Matrix<double,6,6> sqrtI = computeSqrtInfo(info);
            // Residual entre la prediccion actual inv(T_i) * T_j y la medida relativa.
            ceres::CostFunction* cost = new ceres::AutoDiffCostFunction<PoseGraphEdgeAutoDiff,6,7,7>(
                new PoseGraphEdgeAutoDiff(rel, sqrtI));
            // Mostrar diagnosticos de error
            /*
            {
                const Pose Ti = vectorToPose(&global_poses[i*7]);
                const Pose Tj = vectorToPose(&global_poses[(i+1)*7]);
                Pose T_pred = composePose(inversePose(Ti), Tj);
                Pose T_err = composePose(inversePose(rel), T_pred);

                std::cout   << "[PGO][ODOM_CHECK_CERES]"
                            << " pred=" << poseToString(T_pred)
                            << " meas=" << poseToString(rel)
                            << " err=" << poseToString(T_err)
                            << " trans_err=" << translationError(rel, T_pred)
                            << " rot_err=" << poseRotationDeg(rel, T_pred)
                            << std::endl;

                log_relative_convention("odom", rel, Ti, Tj, static_cast<int>(submaps[i]->submap_id), static_cast<int>(submaps[i+1]->submap_id));
                std::cout << "[PGO][RES_ODOM][i=" << i << "] T_meas=pos=(" << rel.position.x << ", " << rel.position.y << ", " << rel.position.z << ") quat=(" 
                          << rel.orientation.x << ", " << rel.orientation.y << ", " << rel.orientation.z << ", " << rel.orientation.w << ")";
                std::cout << " T_pred=pos=(" << T_pred.position.x << ", " << T_pred.position.y << ", " << T_pred.position.z << ") quat=(" 
                          << T_pred.orientation.x << ", " << T_pred.orientation.y << ", " << T_pred.orientation.z << ", " << T_pred.orientation.w << ")";
                std::cout << " T_error=pos=(" << T_err.position.x << ", " << T_err.position.y << ", " << T_err.position.z << ") quat=(" 
                          << T_err.orientation.x << ", " << T_err.orientation.y << ", " << T_err.orientation.z << ", " << T_err.orientation.w << ")" << std::endl;
                
            }
            */
            // Guardamos el descriptor para dignosticos
            {
                ResidualDescriptor rd;
                rd.cost_function = cost;
                rd.parameter_blocks = { &global_poses[i*7], &global_poses[(i+1)*7] };
                // Revision
                for (double* pb : rd.parameter_blocks) {
                    if (pb < global_poses.data() || pb >= (global_poses.data() + global_poses.size())) {
                        std::cout << "[LoopClosureModule::optimizePoseGraph][warn] param_block_pointer_out_of_range" 
                                  << " idx=" << i << " ptr=" << static_cast<const void*>(pb) << std::endl;
                    }
                }
                rd.sqrt_info = sqrtI;
                rd.residual_size = 6;
                rd.label = "odom";
                rd.source_submap_id = static_cast<int>(submaps[i]->submap_id);
                rd.target_submap_id = static_cast<int>(submaps[i+1]->submap_id);
                residuals_info.push_back(rd);
            }
            problem.AddResidualBlock(cost, nullptr, &global_poses[i*7], &global_poses[(i+1)*7]);
        }

        // Residuales de loop closure: usan Huber para robustecer ante outliers de registro.
        const double loop_huber_delta = 3.0; // mayor delta = mas robusto ante outliers, menor delta = mas estricto con loops malos
        size_t loop_edges_added = 0;
        size_t loop_edges_ignored = 0;
        for (const auto &edge : loop_edges) {
            const auto it_i = id_to_index.find(edge.source_submap_id);
            const auto it_j = id_to_index.find(edge.target_submap_id);
            if (it_i == id_to_index.end() || it_j == id_to_index.end()) {
                ++loop_edges_ignored;
                std::cout << "[LoopClosureModule::optimizePoseGraph][edge][reject] reason=missing_submap_id"
                          << " source_submap_id=" << edge.source_submap_id
                          << " target_submap_id=" << edge.target_submap_id
                          << std::endl;
                continue;
            }
            size_t ii = it_i->second;
            size_t jj = it_j->second;
            // Matriz de informacion de la arista de loop (estimada en registro).
            Eigen::Matrix<double,6,6> infod = edge.information_matrix.cast<double>();
            // ensure symmetric
            infod = 0.5 * (infod + infod.transpose());
            // Se atenúa por confianza para reducir impacto de loops poco confiables.
            const double gain = std::clamp(static_cast<double>(edge.confidence), 0.2, 0.999);
            infod *= gain;
            const bool infod_finite = infod.allFinite();
            const double info_diag_min = infod.diagonal().minCoeff();
            const double info_diag_max = infod.diagonal().maxCoeff();
            std::cout << "[LoopClosureModule::optimizePoseGraph][edge]"
                      << " source_submap_id=" << edge.source_submap_id
                      << " target_submap_id=" << edge.target_submap_id
                      << " confidence=" << edge.confidence
                      << " visual_similarity=" << edge.visual_similarity
                      << " num_inliers=" << edge.num_inliers
                      << " info_diag_min=" << info_diag_min
                      << " info_diag_max=" << info_diag_max
                      << " info_finite=" << (infod_finite ? 1 : 0)
                      << " relative_pose_translation_m=" << std::sqrt(edge.relative_pose.position.x * edge.relative_pose.position.x +
                                                                       edge.relative_pose.position.y * edge.relative_pose.position.y +
                                                                       edge.relative_pose.position.z * edge.relative_pose.position.z)
                      << std::endl;

            //std::cout
            //            << "[PGO][LOOP_INFO]"
            //            << " src=" << edge.source_submap_id
            //            << " tgt=" << edge.target_submap_id
            //            << "\n"
            //            << infod
            //            << std::endl;
            if (!infod_finite)
            {
                std::cout << "[LoopClosureModule::optimizePoseGraph][edge][warn] non_finite_information_matrix"
                          << " source_submap_id=" << edge.source_submap_id
                          << " target_submap_id=" << edge.target_submap_id << std::endl;
            }
            Eigen::Matrix<double,6,6> sqrtI = computeSqrtInfo(infod);

            std::cout << "\n=== LOOP INFO ===\n";
            std::cout << infod << std::endl;

            std::cout << "\n=== LOOP SQRT INFO ===\n";
            std::cout << sqrtI << std::endl;

            //std::cout
            //    << "[PGO][LOOP_SQRT_INFO]"
            //    << " src=" << edge.source_submap_id
            //    << " tgt=" << edge.target_submap_id
            //    << " diag="
            //    << sqrtI.diagonal().transpose()
            //    << std::endl;
            /*
            std::cout
                << "[LOOP EDGE]"
                << " src=" << source_id
                << " tgt=" << target_id
                << "\n";
            
            std::cout
                << "meas_t = "
                << relative_pose.position.x << " "
                << relative_pose.position.y << " "
                << relative_pose.position.z << "\n";
            */
            std::cout
                << "sqrt_info diag = "
                << sqrtI.diagonal().transpose()
                << "\n";

            ceres::CostFunction* cost = new ceres::AutoDiffCostFunction<PoseGraphEdgeAutoDiff,6,7,7>(
                new PoseGraphEdgeAutoDiff(edge.relative_pose, sqrtI));
            // compute and log T_error = T_meas^{-1} * (inv(Ti) * Tj) using current globals
            {
                const Pose Ti = vectorToPose(&global_poses[ii*7]);
                const Pose Tj = vectorToPose(&global_poses[jj*7]);
                Pose T_pred = composePose(inversePose(Ti), Tj);
                Pose T_err = composePose(inversePose(edge.relative_pose), T_pred);
                log_relative_convention("loop", edge.relative_pose, vectorToPose(&global_poses[ii*7]), vectorToPose(&global_poses[jj*7]), edge.source_submap_id, edge.target_submap_id);
                std::cout << "[PGO][RES_LOOP][src=" << edge.source_submap_id << " tgt=" << edge.target_submap_id << "]"
                          << " T_meas=pos=(" << edge.relative_pose.position.x << ", " << edge.relative_pose.position.y << ", " << edge.relative_pose.position.z << ")"
                          << " quat=(" << edge.relative_pose.orientation.x << ", " << edge.relative_pose.orientation.y << ", " << edge.relative_pose.orientation.z << ", " << edge.relative_pose.orientation.w << ")";
                std::cout << " T_pred=pos=(" << T_pred.position.x << ", " << T_pred.position.y << ", " << T_pred.position.z << ")"
                          << " quat=(" << T_pred.orientation.x << ", " << T_pred.orientation.y << ", " << T_pred.orientation.z << ", " << T_pred.orientation.w << ")";
                std::cout << " T_error=pos=(" << T_err.position.x << ", " << T_err.position.y << ", " << T_err.position.z << ")"
                          << " quat=(" << T_err.orientation.x << ", " << T_err.orientation.y << ", " << T_err.orientation.z << ", " << T_err.orientation.w << ")" << std::endl;
            }
            // Registro auxiliar para diagnostico de residuales por iteracion.
            {
                ResidualDescriptor rd;
                rd.cost_function = cost;
                rd.parameter_blocks = { &global_poses[ii*7], &global_poses[jj*7] };
                // sanity: ensure parameter block pointers reference the live global_poses buffer
                for (double* pb : rd.parameter_blocks) {
                    if (pb < global_poses.data() || pb >= (global_poses.data() + global_poses.size())) {
                        std::cout << "[LoopClosureModule::optimizePoseGraph][warn] param_block_pointer_out_of_range" 
                                  << " src=" << edge.source_submap_id << " tgt=" << edge.target_submap_id 
                                  << " ptr=" << static_cast<const void*>(pb) << std::endl;
                    }
                }
                rd.sqrt_info = sqrtI;
                rd.residual_size = 6;
                rd.label = "loop";
                rd.source_submap_id = edge.source_submap_id;
                rd.target_submap_id = edge.target_submap_id;
                residuals_info.push_back(rd);
            }
            problem.AddResidualBlock(cost, new ceres::HuberLoss(loop_huber_delta), &global_poses[ii*7], &global_poses[jj*7]);
            ++loop_edges_added;
        }

        const auto solve_start = std::chrono::steady_clock::now();
        // Solve
        ceres::Solver::Options options;
        options.linear_solver_type = ceres::DENSE_QR;
        options.max_num_iterations = std::max(1, std::min(max_iterations, 50));
        options.minimizer_progress_to_stdout = false;
        // Attach iteration logger to dump per-residual diagnostics
        //CeresIterationLogger iter_logger(residuals_info, global_poses, static_cast<int>(N));
        //options.callbacks.push_back(&iter_logger);
        ceres::Solver::Summary summary;
        ceres::Solve(options, &problem, &summary);
        std::cout << summary.FullReport() << std::endl;
        const auto solve_end = std::chrono::steady_clock::now();

        // Copy back results
        for (size_t i=0;i<N;++i) {
            Pose p;
            p.position.x = static_cast<float>(global_poses[i*7 + 0]);
            p.position.y = static_cast<float>(global_poses[i*7 + 1]);
            p.position.z = static_cast<float>(global_poses[i*7 + 2]);
            p.orientation.x = static_cast<float>(global_poses[i*7 + 3]);
            p.orientation.y = static_cast<float>(global_poses[i*7 + 4]);
            p.orientation.z = static_cast<float>(global_poses[i*7 + 5]);
            p.orientation.w = static_cast<float>(global_poses[i*7 + 6]);
            result.corrected_submap_poses[i] = p;
            std::cout << "[PGO][FINAL][pose " << i << "]"
                      << " submap_id=" << submaps[i]->submap_id
                      << " global_pose=" << poseToString(p)
                      << " delta_from_init_trans_m=" << translationError(initial_global_poses[i], p)
                      << " delta_from_init_rot_deg=" << poseRotationDeg(initial_global_poses[i], p)
                      << std::endl;
        }

        // Post-PGO edge consistency diagnostics: compare each measurement against the
        // optimized relative pose implied by the solved global poses.
        for (size_t i = 0; i + 1 < N; ++i)
        {
            const Pose Ti_opt = result.corrected_submap_poses[i];
            const Pose Tj_opt = result.corrected_submap_poses[i + 1];
            const Pose T_rel_opt = composePose(inversePose(Ti_opt), Tj_opt);
            // La Tf medida la guardamos en el modelo como T_relative del hijo
            const Pose T_meas = submaps[i + 1]->T_relative;
            const Pose T_err_post = composePose(inversePose(T_meas), T_rel_opt);
            const float meas_vs_opt_trans = translationError(T_meas, T_rel_opt);
            const float meas_vs_opt_rot = poseRotationDeg(T_meas, T_rel_opt);
            //log_relative_convention("post_odom", T_meas, Ti_opt, Tj_opt, static_cast<int>(submaps[i]->submap_id), static_cast<int>(submaps[i+1]->submap_id));
            /*
            std::cout << "[PGO][POST_ODOM][i=" << i << "]"
                      << " T_meas=pos=(" << T_meas.position.x << ", " << T_meas.position.y << ", " << T_meas.position.z << ")"
                      << " quat=(" << T_meas.orientation.x << ", " << T_meas.orientation.y << ", " << T_meas.orientation.z << ", " << T_meas.orientation.w << ")"
                      << " T_rel_opt=pos=(" << T_rel_opt.position.x << ", " << T_rel_opt.position.y << ", " << T_rel_opt.position.z << ")"
                      << " quat=(" << T_rel_opt.orientation.x << ", " << T_rel_opt.orientation.y << ", " << T_rel_opt.orientation.z << ", " << T_rel_opt.orientation.w << ")"
                      << " T_error_post=pos=(" << T_err_post.position.x << ", " << T_err_post.position.y << ", " << T_err_post.position.z << ")"
                      << " quat=(" << T_err_post.orientation.x << ", " << T_err_post.orientation.y << ", " << T_err_post.orientation.z << ", " << T_err_post.orientation.w << ")"
                      << " meas_vs_opt_translation_m=" << meas_vs_opt_trans
                      << " meas_vs_opt_rotation_deg=" << meas_vs_opt_rot
                      << std::endl;
            */
        }

        for (const auto &edge : loop_edges)
        {
            const auto it_i = id_to_index.find(edge.source_submap_id);
            const auto it_j = id_to_index.find(edge.target_submap_id);
            if (it_i == id_to_index.end() || it_j == id_to_index.end())
            {
                continue;
            }

            const Pose Ti_opt = result.corrected_submap_poses[it_i->second];
            const Pose Tj_opt = result.corrected_submap_poses[it_j->second];
            const Pose T_rel_opt = composePose(inversePose(Ti_opt), Tj_opt);
            const Pose T_meas = edge.relative_pose;
            const Pose T_err_post = composePose(inversePose(T_meas), T_rel_opt);
            const float meas_vs_opt_trans = translationError(T_meas, T_rel_opt);
            const float meas_vs_opt_rot = poseRotationDeg(T_meas, T_rel_opt);
            //log_relative_convention("post_loop", T_meas, Ti_opt, Tj_opt, edge.source_submap_id, edge.target_submap_id);
            /*
            std::cout << "[PGO][POST_LOOP][src=" << edge.source_submap_id << " tgt=" << edge.target_submap_id << "]"
                      << " T_meas=pos=(" << T_meas.position.x << ", " << T_meas.position.y << ", " << T_meas.position.z << ")"
                      << " quat=(" << T_meas.orientation.x << ", " << T_meas.orientation.y << ", " << T_meas.orientation.z << ", " << T_meas.orientation.w << ")"
                      << " T_rel_opt=pos=(" << T_rel_opt.position.x << ", " << T_rel_opt.position.y << ", " << T_rel_opt.position.z << ")"
                      << " quat=(" << T_rel_opt.orientation.x << ", " << T_rel_opt.orientation.y << ", " << T_rel_opt.orientation.z << ", " << T_rel_opt.orientation.w << ")"
                      << " T_error_post=pos=(" << T_err_post.position.x << ", " << T_err_post.position.y << ", " << T_err_post.position.z << ")"
                      << " quat=(" << T_err_post.orientation.x << ", " << T_err_post.orientation.y << ", " << T_err_post.orientation.z << ", " << T_err_post.orientation.w << ")"
                      << " meas_vs_opt_translation_m=" << meas_vs_opt_trans
                      << " meas_vs_opt_rotation_deg=" << meas_vs_opt_rot
                      << std::endl;
            */
        }

        // update uncertainties based on loop edges (same as before)
        for (const auto &edge : loop_edges)
        {
            const auto it_i = id_to_index.find(edge.source_submap_id);
            const auto it_j = id_to_index.find(edge.target_submap_id);
            if (it_i == id_to_index.end() || it_j == id_to_index.end())
            {
                continue;
            }

            Submap *s_i = submaps[it_i->second].get();
            Submap *s_j = submaps[it_j->second].get();
            if (!s_i || !s_j)
            {
                continue;
            }

            const float gain = std::clamp(edge.confidence, 0.2f, 0.95f);
            const float factor = 1.0f - 0.35f * gain;

            s_i->accumulated_translation_uncertainty_m *= factor;
            s_j->accumulated_translation_uncertainty_m *= factor;
            s_i->accumulated_rotation_uncertainty_deg *= factor;
            s_j->accumulated_rotation_uncertainty_deg *= factor;
        }

        float max_translation_shift = 0.0f;
        float max_rotation_shift = 0.0f;
        float sum_translation_shift = 0.0f;
        float sum_rotation_shift = 0.0f;
        size_t changed_nodes = 0;
        for (size_t i = 0; i < result.corrected_submap_poses.size(); ++i)
        {
            const float dpos = translationError(initial_global_poses[i], result.corrected_submap_poses[i]);
            const float drot = poseRotationDeg(initial_global_poses[i], result.corrected_submap_poses[i]);
            if (dpos > 1e-6f || drot > 1e-4f)
            {
                ++changed_nodes;
            }
            max_translation_shift = std::max(max_translation_shift, dpos);
            max_rotation_shift = std::max(max_rotation_shift, drot);
            sum_translation_shift += dpos;
            sum_rotation_shift += drot;
        }

        result.converged = summary.IsSolutionUsable();
        result.residual_error = static_cast<float>(summary.final_cost);
        const bool all_poses_finite = std::all_of(
            result.corrected_submap_poses.begin(),
            result.corrected_submap_poses.end(),
            [](const Pose &pose)
            {
                return std::isfinite(pose.position.x) && std::isfinite(pose.position.y) && std::isfinite(pose.position.z) &&
                       std::isfinite(pose.orientation.x) && std::isfinite(pose.orientation.y) && std::isfinite(pose.orientation.z) &&
                       std::isfinite(pose.orientation.w);
            });
        const auto solve_ms = std::chrono::duration<double, std::milli>(solve_end - solve_start).count();
        const double cost_delta = summary.final_cost - summary.initial_cost;
        const char *trend = (summary.final_cost <= summary.initial_cost) ? "improved" : "worsened";
        std::cout << "[LoopClosureModule::optimizePoseGraph][timing]"
                  << " submaps=" << N
                  << " edges=" << loop_edges.size()
                  << " edges_added=" << loop_edges_added
                  << " edges_ignored=" << loop_edges_ignored
                  << " iterations=" << summary.iterations.size()
                  << " initial_cost=" << summary.initial_cost
                  << " final_cost=" << summary.final_cost
                  << " cost_delta=" << cost_delta
                  << " trend=" << trend
                  << " changed_nodes=" << changed_nodes
                  << " mean_translation_shift=" << (result.corrected_submap_poses.empty() ? 0.0f : sum_translation_shift / static_cast<float>(result.corrected_submap_poses.size()))
                  << " mean_rotation_shift_deg=" << (result.corrected_submap_poses.empty() ? 0.0f : sum_rotation_shift / static_cast<float>(result.corrected_submap_poses.size()))
                  << " max_translation_shift=" << max_translation_shift
                  << " max_rotation_shift_deg=" << max_rotation_shift
                  << " converged=" << (summary.IsSolutionUsable() ? 1 : 0)
                  << " all_poses_finite=" << (all_poses_finite ? 1 : 0)
                  << " solve_ms=" << solve_ms
                  << std::endl;
        return result;
    }

    bool LoopClosureModule::registerSubmaps(
        const Submap* source,
        const Submap* target,
        float visual_similarity,
        Pose& estimated_T,
        float& confidence,
        int& num_inliers,
        float& mean_residual_out)
    {
        if (!source || !target)
        {
            std::cout << "[LoopClosureModule::registerSubmaps][reject] reason=null_input"
                      << " source_ptr=" << source
                      << " target_ptr=" << target
                      << " visual_similarity=" << visual_similarity
                      << std::endl;
            return false;
        }

        const Pose anchor_guess = composePose(inversePose(source->getGlobalPose()), target->getGlobalPose());
        const float anchor_dist = poseDistance(source->getGlobalPose(), target->getGlobalPose());
        const float anchor_rot = poseRotationDeg(source->getGlobalPose(), target->getGlobalPose());
        const float overlap = computeOverlapRatio(source, target);
        const std::vector<Eigen::Vector3f> src_pts = extractGaussianMeans(source, 1500);
        const std::vector<Eigen::Vector3f> tgt_pts = extractGaussianMeans(target, 1500);

        std::cout << "[LoopClosureModule::registerSubmaps] Start"
                  << " source=" << source->submap_id
                  << " target=" << target->submap_id
                  << " vis=" << visual_similarity
                  << " anchor_dist=" << anchor_dist
                  << " anchor_rot=" << anchor_rot
                  << " overlap=" << overlap
                  << " anchor_guess=" << poseToString(anchor_guess)
                  << " source_global=" << poseToString(source->getGlobalPose())
                  << " target_global=" << poseToString(target->getGlobalPose())
                  << " source_pts=" << src_pts.size()
                  << " target_pts=" << tgt_pts.size()
                  << std::endl;

        if (src_pts.size() < 50 || tgt_pts.size() < 50)
        {
            std::cout << "[LoopClosureModule::registerSubmaps] Rejected: not enough points"
                      << " source_pts=" << src_pts.size()
                      << " target_pts=" << tgt_pts.size()
                      << " source_id=" << source->submap_id
                      << " target_id=" << target->submap_id
                      << " anchor_guess=" << poseToString(anchor_guess)
                      << std::endl;
            return false;
        }

        auto try_register = [&](float corr_dist,
                                const char* tag,
                                Pose& out_estimated_T,
                                float& out_confidence,
                                int& out_num_inliers,
                                float& out_mean_residual) -> bool
        {
            Eigen::Isometry3f T_cur = poseToIso(anchor_guess);
            constexpr int kIters = 3;
            constexpr size_t kMaxPairs = 800;
            std::vector<ReciprocalMatch> matches;

            for (int iter = 0; iter < kIters; ++iter)
            {
                const std::vector<Eigen::Vector3f> src_transformed = transformPoints(src_pts, T_cur);
                matches = computeReciprocalMatches(src_transformed, tgt_pts, true, corr_dist, kMaxPairs);
                if (matches.size() < 6)
                {
                    break;
                }

                Eigen::Matrix<float, 3, Eigen::Dynamic> src_mat(3, static_cast<int>(matches.size()));
                Eigen::Matrix<float, 3, Eigen::Dynamic> tgt_mat(3, static_cast<int>(matches.size()));
                for (size_t k = 0; k < matches.size(); ++k)
                {
                    src_mat.col(static_cast<int>(k)) = src_transformed[static_cast<size_t>(matches[k].src_idx)];
                    tgt_mat.col(static_cast<int>(k)) = tgt_pts[static_cast<size_t>(matches[k].tgt_idx)];
                }

                const Eigen::Matrix4f delta = Eigen::umeyama(src_mat, tgt_mat, false);
                if (!delta.allFinite())
                {
                    break;
                }

                T_cur = Eigen::Isometry3f(delta) * T_cur;
            }

            const std::vector<Eigen::Vector3f> src_final = transformPoints(src_pts, T_cur);
            matches = computeReciprocalMatches(src_final, tgt_pts, true, corr_dist, kMaxPairs);
            std::cout << "[LoopClosureModule::registerSubmaps] Attempt"
                      << " tag=" << tag
                      << " corr_dist=" << corr_dist
                      << " matches=" << matches.size()
                      << " current_estimate=" << poseToString(out_estimated_T)
                      << std::endl;
            if (matches.empty())
            {
                out_confidence = 0.0f;
                out_num_inliers = 0;
                out_mean_residual = std::numeric_limits<float>::infinity();
                std::cout << "[LoopClosureModule::registerSubmaps] Rejected(" << tag << "): no matches"
                          << " corr_dist=" << corr_dist
                          << " source_pts=" << src_pts.size()
                          << " target_pts=" << tgt_pts.size()
                          << std::endl;
                return false;
            }

            float residual_sum = 0.0f;
            for (const auto &m : matches)
            {
                residual_sum += m.dist;
            }
            const float mean_residual = residual_sum / static_cast<float>(matches.size());
            out_mean_residual = mean_residual;
            const float inlier_ratio = std::clamp(
                static_cast<float>(matches.size()) / static_cast<float>(std::max<size_t>(1, std::min(src_pts.size(), tgt_pts.size()))),
                0.0f,
                1.0f);

            out_estimated_T = isoToPose(T_cur);
            const bool transform_finite = isFinitePose(out_estimated_T);

            const bool valid_registration =
                transform_finite &&
                std::isfinite(mean_residual) &&
                (matches.size() >= 6) &&
                (mean_residual < 5.0f);

            if (!valid_registration)
            {
                out_confidence = 0.0f;
                out_num_inliers = static_cast<int>(matches.size());
                std::cout << "[LoopClosureModule::registerSubmaps] Rejected(" << tag << "): invalid registration"
                          << " transform_finite=" << transform_finite
                          << " matches=" << matches.size()
                          << " mean_residual=" << mean_residual
                          << " estimated_T=" << poseToString(out_estimated_T)
                          << std::endl;
                return false;
            }

            const float residual_term = std::exp(-std::max(0.0f, mean_residual));
            const float quality = std::clamp(
                0.50f * std::clamp(visual_similarity, 0.0f, 1.0f) +
                0.25f * overlap +
                0.15f * inlier_ratio +
                0.10f * residual_term,
                0.0f,
                1.0f);

            out_confidence = 0.50f + 0.50f * quality;
            out_num_inliers = static_cast<int>(matches.size());

            std::cout << "[LoopClosureModule::registerSubmaps] Accepted"
                      << " tag=" << tag
                      << " corr_dist=" << corr_dist
                      << " matches=" << matches.size()
                      << " mean_residual=" << mean_residual
                      << " confidence=" << out_confidence
                      << " estimated_T=" << poseToString(out_estimated_T)
                      << " inlier_ratio=" << inlier_ratio
                      << " residual_term=" << residual_term
                      << " overlap=" << overlap
                      << std::endl;
            return out_confidence > 0.0f;
        };

        const float vis = std::clamp(visual_similarity, 0.0f, 1.0f);
        const float adaptive_radius = std::clamp(0.40f + 0.45f * vis, 0.25f, 0.90f);

        // First try the nominal correspondence radius.
        if (try_register(0.25f, "nominal", estimated_T, confidence, num_inliers, mean_residual_out))
        {
            return true;
        }

        // Retry with a looser radius in case the anchor guess is slightly off.
        if (try_register(0.40f, "fallback", estimated_T, confidence, num_inliers, mean_residual_out))
        {
            return true;
        }

        // High-visual-similarity loops can still be valid even with larger pose drift.
        return try_register(adaptive_radius, "adaptive", estimated_T, confidence, num_inliers, mean_residual_out);
    }

    bool LoopClosureModule::registerSubmapsByOverlap(
        const Submap* source,
        const Submap* target,
        float visual_similarity,
        Pose& estimated_T,
        float& confidence,
        int& num_inliers,
        float& mean_residual_out)
    {
        if (!source || !target)
        {
            std::cout << "[LoopClosureModule::registerSubmapsByOverlap][reject] reason=null_input"
                      << " source_ptr=" << source
                      << " target_ptr=" << target
                      << " visual_similarity=" << visual_similarity
                      << std::endl;
            return false;
        }

        const float vis = std::clamp(visual_similarity, 0.0f, 1.0f);
        const float overlap = computeOverlapRatio(source, target);
        const Pose anchor_guess = composePose(inversePose(source->getGlobalPose()), target->getGlobalPose());

        std::cout << "[LoopClosureModule::registerSubmapsByOverlap] Start"
                  << " source=" << source->submap_id
                  << " target=" << target->submap_id
                  << " vis=" << vis
                  << " overlap=" << overlap
                  << " anchor_guess=" << poseToString(anchor_guess)
                  << " source_global=" << poseToString(source->getGlobalPose())
                  << " target_global=" << poseToString(target->getGlobalPose())
                  << std::endl;

        if (source->keyframes.empty() || target->keyframes.empty())
        {
            std::cout << "[LoopClosureModule::registerSubmapsByOverlap][reject] reason=empty_keyframes"
                      << " source_keyframes=" << source->keyframes.size()
                      << " target_keyframes=" << target->keyframes.size()
                      << std::endl;
            return false;
        }

        if (overlap < 0.05f && vis < config_.min_similarity_floor)
        {
            std::cout << "[LoopClosureModule::registerSubmapsByOverlap][reject] reason=overlap_below_threshold"
                      << " overlap=" << overlap
                      << " vis=" << vis
                      << " min_similarity_floor=" << config_.min_similarity_floor
                      << " anchor_guess=" << poseToString(anchor_guess)
                      << std::endl;
            return false;
        }

        if (overlap < 0.05f)
        {
            std::cout << "[LoopClosureModule::registerSubmapsByOverlap] Warning: weak geometric overlap, proceeding with visual fallback"
                      << " overlap=" << overlap
                      << " vis=" << vis << std::endl;
        }

        std::vector<PairRegistration> pair_registrations;
        pair_registrations.reserve(8);

        std::vector<std::tuple<float, size_t, size_t>> ranked_pairs;
        ranked_pairs.reserve(source->keyframes.size() * target->keyframes.size());

        for (size_t i = 0; i < source->keyframes.size(); ++i)
        {
            const auto &kf_src = source->keyframes[i];
            if (!kf_src.hasDescriptor())
            {
                continue;
            }

            for (size_t j = 0; j < target->keyframes.size(); ++j)
            {
                const auto &kf_tgt = target->keyframes[j];
                if (!kf_tgt.hasDescriptor())
                {
                    continue;
                }

                const float sim = cosineSimilarity(kf_src.getDescriptor(), kf_tgt.getDescriptor());
                if (sim > 0.0f)
                {
                    ranked_pairs.emplace_back(sim, i, j);
                }
            }
        }

        if (ranked_pairs.empty())
        {
            std::cout << "[LoopClosureModule::registerSubmapsByOverlap][reject] reason=no_descriptor_pairs"
                      << " source_keyframes=" << source->keyframes.size()
                      << " target_keyframes=" << target->keyframes.size()
                      << std::endl;
            return false;
        }

        std::sort(ranked_pairs.begin(), ranked_pairs.end(),
                  [](const auto &a, const auto &b)
                  {
                      return std::get<0>(a) > std::get<0>(b);
                  });

        const size_t max_candidate_pairs = std::min<size_t>(8, ranked_pairs.size());
        const float min_pair_similarity = std::max(0.20f, 0.65f * vis);
        const float overlap_factor = 1.0f - std::clamp(overlap, 0.0f, 1.0f);
        const float coarse_radius = std::clamp(0.25f + 0.45f * overlap_factor + 0.20f * (1.0f - vis), 0.20f, 1.00f);
        const float final_radius = std::clamp(coarse_radius * 0.85f, 0.15f, 0.90f);

        for (size_t idx = 0; idx < max_candidate_pairs; ++idx)
        {
            const float pair_similarity = std::get<0>(ranked_pairs[idx]);
            if (pair_similarity < min_pair_similarity)
            {
                continue;
            }

            const size_t src_idx = std::get<1>(ranked_pairs[idx]);
            const size_t tgt_idx = std::get<2>(ranked_pairs[idx]);
            const KeyframeData &kf_src = source->keyframes[src_idx];
            const KeyframeData &kf_tgt = target->keyframes[tgt_idx];

            const std::vector<Eigen::Vector3f> src_pts = backprojectDepthImage(kf_src, 1800);
            const std::vector<Eigen::Vector3f> tgt_pts = backprojectDepthImage(kf_tgt, 1800);

            if (src_pts.size() < 80 || tgt_pts.size() < 80)
            {
                continue;
            }

            Eigen::Isometry3f T_cur = poseToIso(composePose(inversePose(kf_src.getGlobalPose()), kf_tgt.getGlobalPose()));
            if (!T_cur.matrix().allFinite())
            {
                T_cur = poseToIso(anchor_guess);
            }

            constexpr int kIters = 4;
            constexpr size_t kMaxPairs = 700;
            bool pair_valid = false;
            std::vector<ReciprocalMatch> matches;

            for (int iter = 0; iter < kIters; ++iter)
            {
                const std::vector<Eigen::Vector3f> src_transformed = transformPoints(src_pts, T_cur);
                matches = computeReciprocalMatches(src_transformed, tgt_pts, true, coarse_radius, kMaxPairs);

                if (matches.size() < 6)
                {
                    break;
                }

                Eigen::Matrix<float, 3, Eigen::Dynamic> src_mat(3, static_cast<int>(matches.size()));
                Eigen::Matrix<float, 3, Eigen::Dynamic> tgt_mat(3, static_cast<int>(matches.size()));
                for (size_t k = 0; k < matches.size(); ++k)
                {
                    src_mat.col(static_cast<int>(k)) = src_transformed[static_cast<size_t>(matches[k].src_idx)];
                    tgt_mat.col(static_cast<int>(k)) = tgt_pts[static_cast<size_t>(matches[k].tgt_idx)];
                }

                const Eigen::Matrix4f delta = Eigen::umeyama(src_mat, tgt_mat, false);
                if (!delta.allFinite())
                {
                    break;
                }

                T_cur = Eigen::Isometry3f(delta) * T_cur;
                pair_valid = true;
            }

            const std::vector<Eigen::Vector3f> src_final = transformPoints(src_pts, T_cur);
            matches = computeReciprocalMatches(src_final, tgt_pts, true, final_radius, kMaxPairs);

            if (matches.size() < 6)
            {
                continue;
            }

            float residual_sum = 0.0f;
            for (const auto &m : matches)
            {
                residual_sum += m.dist;
            }

            const float mean_residual = residual_sum / static_cast<float>(matches.size());
            if (!std::isfinite(mean_residual) || mean_residual > 5.0f)
            {
                continue;
            }

            const float inlier_ratio = std::clamp(
                static_cast<float>(matches.size()) / static_cast<float>(std::max<size_t>(1, std::min(src_pts.size(), tgt_pts.size()))),
                0.0f,
                1.0f);
            const float residual_term = std::exp(-std::max(0.0f, mean_residual));
            const float pair_weight = std::clamp(
                0.55f * pair_similarity +
                0.20f * vis +
                0.15f * overlap +
                0.10f * residual_term,
                0.0f,
                1.0f);

            if (!pair_valid || pair_weight <= 0.0f)
            {
                continue;
            }

            PairRegistration reg;
            reg.relative_pose = isoToPose(T_cur);
            reg.pair_similarity = pair_similarity;
            reg.residual = mean_residual;
            reg.weight = pair_weight * std::max(0.1f, inlier_ratio);
            reg.inliers = static_cast<int>(matches.size());
            pair_registrations.push_back(reg);
        }
        // Attempt Open3D coarse-to-fine registration as an alternative when pair-based fusion fails
        Pose open3d_pose;
        float open3d_conf = 0.0f;
        int open3d_inliers = 0;
        float open3d_mean = 0.0f;
        if (pair_registrations.empty()) {
            try {
                if (registerSubmapsOpen3D(source, target, visual_similarity, anchor_guess, open3d_pose, open3d_conf, open3d_inliers, open3d_mean)) {
                    std::cout << "[LoopClosureModule::registerSubmapsByOverlap][fallback_open3d]"
                              << " source=" << source->submap_id
                              << " target=" << target->submap_id
                              << " confidence=" << open3d_conf
                              << " inliers=" << open3d_inliers
                              << " mean_residual=" << open3d_mean
                              << " pose=" << poseToString(open3d_pose)
                              << std::endl;
                    estimated_T = open3d_pose;
                    confidence = open3d_conf;
                    num_inliers = open3d_inliers;
                    mean_residual_out = open3d_mean;
                    return confidence > 0.0f;
                }
            } catch (...) { /* ignore */ }
        }
        if (pair_registrations.empty())
        {
            confidence = 0.0f;
            num_inliers = 0;
            mean_residual_out = std::numeric_limits<float>::infinity();
            std::cout << "[LoopClosureModule::registerSubmapsByOverlap][reject] reason=no_valid_pair_registrations"
                      << " source=" << source->submap_id
                      << " target=" << target->submap_id
                      << " ranked_pairs=" << ranked_pairs.size()
                      << " overlap=" << overlap
                      << " vis=" << vis
                      << std::endl;
            return false;
        }

        std::vector<Eigen::Matrix3f> rotations;
        std::vector<Eigen::Vector3f> translations;
        std::vector<float> weights;
        rotations.reserve(pair_registrations.size());
        translations.reserve(pair_registrations.size());
        weights.reserve(pair_registrations.size());

        float weighted_residual_sum = 0.0f;
        float total_weight = 0.0f;
        int total_inliers = 0;
        for (const auto &reg : pair_registrations)
        {
            const Eigen::Isometry3f T = poseToIso(reg.relative_pose);
            rotations.push_back(T.linear());
            translations.push_back(T.translation());
            weights.push_back(reg.weight);
            weighted_residual_sum += reg.weight * reg.residual;
            total_weight += reg.weight;
            total_inliers += reg.inliers;
        }

        if (total_weight <= 1e-6f)
        {
            confidence = 0.0f;
            num_inliers = 0;
            mean_residual_out = std::numeric_limits<float>::infinity();
            std::cout << "[LoopClosureModule::registerSubmapsByOverlap][reject] reason=zero_total_weight"
                      << " source=" << source->submap_id
                      << " target=" << target->submap_id
                      << " total_weight=" << total_weight
                      << " pair_registrations=" << pair_registrations.size()
                      << std::endl;
            return false;
        }

        const Eigen::Matrix3f fused_rotation = averageRotations(rotations, weights);
        Eigen::Vector3f fused_translation = Eigen::Vector3f::Zero();
        for (size_t i = 0; i < translations.size(); ++i)
        {
            fused_translation += std::max(0.0f, weights[i]) * translations[i];
        }
        fused_translation /= total_weight;

        Pose fused_pose;
        fused_pose.position = make_float3(fused_translation.x(), fused_translation.y(), fused_translation.z());
        Eigen::Quaternionf fused_q(fused_rotation);
        fused_q.normalize();
        fused_pose.orientation = make_float4(fused_q.x(), fused_q.y(), fused_q.z(), fused_q.w());

        estimated_T = fused_pose;
        const bool transform_finite = isFinitePose(estimated_T);
        const float mean_residual = weighted_residual_sum / total_weight;
        mean_residual_out = mean_residual;

        const float avg_pair_similarity = std::accumulate(
            pair_registrations.begin(),
            pair_registrations.end(),
            0.0f,
            [](float acc, const PairRegistration &reg)
            {
                return acc + reg.pair_similarity;
            }) / static_cast<float>(pair_registrations.size());

        const float inlier_ratio = std::clamp(
            static_cast<float>(total_inliers) / static_cast<float>(std::max<size_t>(1, pair_registrations.size() * 150)),
            0.0f,
            1.0f);

        const bool valid_registration =
            transform_finite &&
            std::isfinite(mean_residual) &&
            (pair_registrations.size() >= 1) &&
            (mean_residual < 5.0f);

        if (!valid_registration)
        {
            confidence = 0.0f;
            num_inliers = total_inliers;
            std::cout << "[LoopClosureModule::registerSubmapsByOverlap][reject] reason=invalid_registration"
                      << " transform_finite=" << transform_finite
                      << " pairs=" << pair_registrations.size()
                      << " mean_residual=" << mean_residual
                      << " estimated_T=" << poseToString(estimated_T)
                      << " overlap=" << overlap
                      << " vis=" << vis
                      << std::endl;
            return false;
        }

        const float residual_term = std::exp(-std::max(0.0f, mean_residual));
        const float quality = std::clamp(
            0.30f * vis +
            0.25f * overlap +
            0.25f * avg_pair_similarity +
            0.10f * inlier_ratio +
            0.10f * residual_term,
            0.0f,
            1.0f);

        confidence = 0.50f + 0.50f * quality;
        num_inliers = total_inliers;

        std::cout << "[LoopClosureModule::registerSubmapsByOverlap] Accepted"
                  << " pairs=" << pair_registrations.size()
                  << " inliers=" << total_inliers
                  << " mean_residual=" << mean_residual
                  << " confidence=" << confidence
                  << " estimated_T=" << poseToString(estimated_T)
                  << " avg_pair_similarity=" << avg_pair_similarity
                  << " inlier_ratio=" << inlier_ratio
                  << " residual_term=" << residual_term
                  << " overlap=" << overlap
                  << std::endl;

        return confidence > 0.0f;
    }

    float LoopClosureModule::computeOverlapRatio(const Submap* submap1, const Submap* submap2)
    {
        if (!submap1 || !submap2)
        {
            std::cout << "[LoopClosureModule::computeOverlapRatio][reject] reason=null_input"
                      << " submap1_ptr=" << submap1
                      << " submap2_ptr=" << submap2
                      << std::endl;
            return 0.0f;
        }

        const std::vector<Eigen::Vector3f> src_pts = extractGaussianMeans(submap1, 1200);
        const std::vector<Eigen::Vector3f> tgt_pts = extractGaussianMeans(submap2, 1200);
        if (src_pts.size() < 50 || tgt_pts.size() < 50)
        {
            std::cout << "[LoopClosureModule::computeOverlapRatio][reject] reason=not_enough_points"
                      << " submap1=" << submap1->submap_id
                      << " submap2=" << submap2->submap_id
                      << " src_pts=" << src_pts.size()
                      << " tgt_pts=" << tgt_pts.size()
                      << std::endl;
            return 0.0f;
        }

        const Pose init_relative = composePose(inversePose(submap1->getGlobalPose()), submap2->getGlobalPose());
        const std::vector<Eigen::Vector3f> src_transformed = transformPoints(src_pts, poseToIso(init_relative));
        const std::vector<ReciprocalMatch> putative = computeReciprocalMatches(
            src_transformed,
            tgt_pts,
            false,
            0.0f,
            1200);

        if (putative.empty())
        {
            std::cout << "[LoopClosureModule::computeOverlapRatio][reject] reason=no_putative_matches"
                      << " submap1=" << submap1->submap_id
                      << " submap2=" << submap2->submap_id
                      << " init_relative=" << poseToString(init_relative)
                      << " src_pts=" << src_pts.size()
                      << " tgt_pts=" << tgt_pts.size()
                      << std::endl;
            return 0.0f;
        }

        constexpr float kTau1 = 0.2f;
        size_t inliers = 0;
        for (const auto &m : putative)
        {
            if (m.dist < kTau1)
            {
                ++inliers;
            }
        }

        const float overlap = std::clamp(
            static_cast<float>(inliers) / static_cast<float>(putative.size()),
            0.0f,
            1.0f);
        std::cout << "[LoopClosureModule::computeOverlapRatio][ok]"
                  << " submap1=" << submap1->submap_id
                  << " submap2=" << submap2->submap_id
                  << " init_relative=" << poseToString(init_relative)
                  << " putative_matches=" << putative.size()
                  << " inliers=" << inliers
                  << " overlap=" << overlap
                  << std::endl;
        return overlap;
    }

} // namespace f_vigs_slam
