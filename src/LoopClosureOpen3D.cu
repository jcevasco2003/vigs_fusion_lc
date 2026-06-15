#include "f_vigs_slam/LoopClosureModule.hpp"
#include "f_vigs_slam/GSCudaKernels.cuh"
#include <open3d/Open3D.h>
#include <algorithm>
#include <cmath>
#include <chrono>
#include <functional>
#include <cuda_runtime.h>
#include "f_vigs_slam/CudaMathOperations.cuh"
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/gather.h>
#include <thrust/copy.h>
#include <thrust/sort.h>
#include <thrust/transform.h>
#include <thrust/fill.h>
#include <thrust/functional.h>
#include <thrust/sequence.h>
#include <thrust/iterator/counting_iterator.h>
#include <iostream>
#include <sstream>
#include <optional>

namespace f_vigs_slam
{
    namespace
    {
        Eigen::Matrix<float, 6, 6> buildLoopInformationMatrix(
            const Submap *src,
            const Submap *tgt,
            float visual_similarity,
            float overlap,
            float mean_residual,
            int num_inliers,
            float confidence)
        {
            /*
            Eigen::Matrix<float, 6, 6> info = Eigen::Matrix<float, 6, 6>::Zero();

            const float conf = std::clamp(confidence, 0.0f, 1.0f);
            const float ov = std::clamp(overlap, 0.0f, 1.0f);
            const float res_term = std::exp(-std::max(0.0f, mean_residual));
            const float inlier_term = std::clamp(static_cast<float>(num_inliers) / 100.0f, 0.0f, 1.0f);
            const float vis_term = std::clamp(visual_similarity, 0.0f, 1.0f);

            float imu_trans_unc = 0.0f;
            float imu_rot_unc_deg = 0.0f;
            if (src)
            {
                imu_trans_unc += src->accumulated_translation_uncertainty_m;
                imu_rot_unc_deg += src->accumulated_rotation_uncertainty_deg;
            }
            if (tgt)
            {
                imu_trans_unc += tgt->accumulated_translation_uncertainty_m;
                imu_rot_unc_deg += tgt->accumulated_rotation_uncertainty_deg;
            }

            const float imu_trans_prior = 1.0f / (1.0f + 0.5f * imu_trans_unc);
            const float imu_rot_prior = 1.0f / (1.0f + 0.02f * imu_rot_unc_deg);

            float omega_trans = 50.0f * conf * ov * res_term * inlier_term * std::max(0.1f, vis_term) * imu_trans_prior;
            float omega_rot = 20.0f * conf * (0.5f + 0.5f * res_term) * (0.5f + 0.5f * vis_term) * imu_rot_prior;

            omega_trans = std::clamp(omega_trans, 1e-2f, 1e3f);
            omega_rot = std::clamp(omega_rot, 1e-3f, 1e3f);

            info.block<3, 3>(0, 0) = Eigen::Matrix3f::Identity() * omega_trans;
            info.block<3, 3>(3, 3) = Eigen::Matrix3f::Identity() * omega_rot;

            const float coupling_strength = std::clamp(0.05f * conf * res_term * imu_trans_prior * imu_rot_prior, 0.0f, 0.2f);
            for (int i = 0; i < 3; ++i)
            {
                info(i, 3 + i) = coupling_strength;
                info(3 + i, i) = coupling_strength;
            }

            return info;
            */
                Eigen::Matrix<float, 6, 6> info = Eigen::Matrix<float, 6, 6>::Zero();

            // -----------------------------
            // 1. Normalización robusta
            // -----------------------------
            const float conf = std::clamp(confidence, 0.0f, 1.0f);
            const float ov   = std::clamp(overlap, 0.0f, 1.0f);
            const float vis  = std::clamp(visual_similarity, 0.0f, 1.0f);

            // residual: usar saturación, no exp directo agresivo
            const float res = mean_residual;
            const float res_term = 1.0f / (1.0f + res);

            // inliers: mejor saturación suave (log-like)
            const float inlier_term = std::tanh(num_inliers / 80.0f);

            // -----------------------------
            // 2. IMU priors (ok, pero suavizados)
            // -----------------------------
            float imu_t = 0.0f;
            float imu_r = 0.0f;

            if (src) {
                imu_t += src->accumulated_translation_uncertainty_m;
                imu_r += src->accumulated_rotation_uncertainty_deg;
            }
            if (tgt) {
                imu_t += tgt->accumulated_translation_uncertainty_m;
                imu_r += tgt->accumulated_rotation_uncertainty_deg;
            }

            const float imu_t_prior = std::exp(-0.5f * imu_t);
            const float imu_r_prior = std::exp(-0.05f * imu_r);

            // -----------------------------
            // 3. Separar "geometría" de "observación"
            // -----------------------------
            const float geom_score =
                conf * ov * res_term * inlier_term;

            const float obs_score =
                0.7f * geom_score + 0.3f * vis;

            // -----------------------------
            // 4. Escala base controlada (MUY importante)
            // -----------------------------
            float omega_trans = 2000.0f * obs_score * imu_t_prior;
            float omega_rot   = 100.0f * (0.5f + 0.5f * obs_score) * imu_r_prior;

            // -----------------------------
            // 5. Clamp más agresivo (evita explosiones)
            // -----------------------------
            omega_trans = std::clamp(omega_trans, 0.5f, 200.0f);
            omega_rot   = std::clamp(omega_rot, 0.1f, 100.0f);

            // -----------------------------
            // 6. Bloques diagonales
            // -----------------------------
            info.block<3, 3>(0, 0) = Eigen::Matrix3f::Identity() * omega_trans;
            info.block<3, 3>(3, 3) = Eigen::Matrix3f::Identity() * omega_rot;

            // -----------------------------
            // 7. Coupling MUCHO más conservador
            // -----------------------------
            const float coupling =
                0.01f * obs_score * imu_t_prior * imu_r_prior;

            for (int i = 0; i < 3; ++i)
            {
                info(i, 3 + i) = coupling;
                info(3 + i, i) = coupling;
            }

            return info;

        }

        void ensurePositiveQuaternionW(Eigen::Quaterniond &q)
        {
            if (q.w() < 0.0)
            {
                q.coeffs() *= -1.0;
            }
        }

        Pose eigenMatrixToPose(const Eigen::Matrix4d &T)
        {
            Pose pose = Pose::Identity();
            pose.position.x = static_cast<float>(T(0, 3));
            pose.position.y = static_cast<float>(T(1, 3));
            pose.position.z = static_cast<float>(T(2, 3));

            Eigen::Matrix3d R = T.block<3, 3>(0, 0);
            Eigen::Quaterniond q(R);
            q.normalize();
            ensurePositiveQuaternionW(q);
            pose.orientation.x = static_cast<float>(q.x());
            pose.orientation.y = static_cast<float>(q.y());
            pose.orientation.z = static_cast<float>(q.z());
            pose.orientation.w = static_cast<float>(q.w());
            return pose;
        }

        std::string poseToString(const Pose &pose)
        {
            std::ostringstream oss;
            oss << "pos=(" << pose.position.x << ", " << pose.position.y << ", " << pose.position.z << ")"
                << " quat=(" << pose.orientation.x << ", " << pose.orientation.y << ", " << pose.orientation.z << ", " << pose.orientation.w << ")";
            return oss.str();
        }

        // Use helpers from LoopClosureModule.hpp: poseTranslationError / poseRotationDeg

        Eigen::Matrix4d poseToEigenMatrix(const Pose &pose)
        {
            Eigen::Quaterniond q(
                static_cast<double>(pose.orientation.w),
                static_cast<double>(pose.orientation.x),
                static_cast<double>(pose.orientation.y),
                static_cast<double>(pose.orientation.z));
            q.normalize();

            Eigen::Matrix4d T = Eigen::Matrix4d::Identity();
            T.block<3, 3>(0, 0) = q.toRotationMatrix();
            T(0, 3) = static_cast<double>(pose.position.x);
            T(1, 3) = static_cast<double>(pose.position.y);
            T(2, 3) = static_cast<double>(pose.position.z);
            return T;
        }

        double se3LogNorm(const Eigen::Matrix4d &T)
        {
            const Eigen::Vector3d t = T.block<3, 1>(0, 3);
            Eigen::Quaterniond q(T.block<3, 3>(0, 0));
            q.normalize();
            const Eigen::AngleAxisd aa(q);
            const double rot_norm = std::abs(aa.angle());
            return std::sqrt(t.squaredNorm() + rot_norm * rot_norm);
        }

        void logPairwiseTransformDiagnostics(
            const char *tag,
            int source_submap_id,
            int target_submap_id,
            const Pose &T_icp,
            const Pose &T_graph_source_to_target)
        {
            const Eigen::Matrix4d M_icp = poseToEigenMatrix(T_icp);
            const Eigen::Matrix4d M_icp_inv = M_icp.inverse();
            const Eigen::Matrix4d M_graph_st = poseToEigenMatrix(T_graph_source_to_target);
            const Eigen::Matrix4d M_graph_ts = M_graph_st.inverse();

            const Pose T_icp_inverse = eigenMatrixToPose(M_icp_inv);
            const Pose T_graph_target_to_source = eigenMatrixToPose(M_graph_ts);

            const double n_icp_inv_graph_st = se3LogNorm(M_icp_inv * M_graph_st);
            const double n_icp_graph_st = se3LogNorm(M_icp * M_graph_st);
            const double n_icp_inv_graph_ts = se3LogNorm(M_icp_inv * M_graph_ts);
            const double n_icp_graph_ts = se3LogNorm(M_icp * M_graph_ts);

            std::cout << "[LoopClosureModule::registerSubmapsOpen3D][pairwise][" << tag << "]"
                      << " source_submap_id=" << source_submap_id
                      << " target_submap_id=" << target_submap_id
                      << " T_icp=" << poseToString(T_icp)
                      << " T_icp_inverse=" << poseToString(T_icp_inverse)
                      << " T_graph_source_to_target=" << poseToString(T_graph_source_to_target)
                      << " T_graph_target_to_source=" << poseToString(T_graph_target_to_source)
                      << " norm_log(T_icp^{-1} * T_graph_source_to_target)=" << n_icp_inv_graph_st
                      << " norm_log(T_icp * T_graph_source_to_target)=" << n_icp_graph_st
                      << " norm_log(T_icp^{-1} * T_graph_target_to_source)=" << n_icp_inv_graph_ts
                      << " norm_log(T_icp * T_graph_target_to_source)=" << n_icp_graph_ts
                      << std::endl;
        }

        struct FrustumVisibleCloud
        {
            thrust::device_vector<float> packed_xyz;
            size_t total_gaussians = 0;
            size_t visible_gaussians = 0;
            size_t discarded_gaussians = 0;
            size_t discarded_non_finite = 0;
            size_t discarded_behind_camera = 0;
            size_t discarded_out_of_bounds = 0;
            float visibility_ratio = 0.0f;
        };

        struct CountPredicate
        {
            uint32_t minv;
            __host__ __device__ bool operator()(uint32_t count) const
            {
                return count >= minv;
            }
        };

        __global__ void countFrustumVisibility_kernel(
            uint32_t *visibility_counts,
            uint32_t *reject_counts,
            const float4 *positions_local,
            Pose source_global_pose,
            Pose camera_pose,
            IntrinsicParameters intrinsics,
            float min_depth,
            uint32_t width,
            uint32_t height,
            uint32_t n_gaussians)
        {
            const uint32_t idx = static_cast<uint32_t>(blockIdx.x * blockDim.x + threadIdx.x);
            if (idx >= n_gaussians)
            {
                return;
            }

            const float4 p_local4 = positions_local[idx];
            if (!isfinite(p_local4.x) || !isfinite(p_local4.y) || !isfinite(p_local4.z))
            {
                atomicAdd(&reject_counts[0], 1u);
                return;
            }

            const float3 p_local = make_float3(p_local4.x, p_local4.y, p_local4.z);
            const float3 p_world = rotateByQuaternion(source_global_pose.orientation, p_local) + source_global_pose.position;
            const float3 p_cam = rotateByQuaternionInverse(camera_pose.orientation, p_world - camera_pose.position);

            if (!isfinite(p_cam.x) || !isfinite(p_cam.y) || !isfinite(p_cam.z) || p_cam.z <= min_depth)
            {
                if (!isfinite(p_cam.x) || !isfinite(p_cam.y) || !isfinite(p_cam.z))
                {
                    atomicAdd(&reject_counts[0], 1u);
                }
                else
                {
                    atomicAdd(&reject_counts[1], 1u);
                }
                return;
            }

            const float inv_z = 1.0f / p_cam.z;
            const float u = intrinsics.f.x * p_cam.x * inv_z + intrinsics.c.x;
            const float v = intrinsics.f.y * p_cam.y * inv_z + intrinsics.c.y;

            if (u < 0.0f || u >= static_cast<float>(width) ||
                v < 0.0f || v >= static_cast<float>(height))
            {
                atomicAdd(&reject_counts[2], 1u);
                return;
            }

            atomicAdd(&reject_counts[3], 1u);
            atomicAdd(&visibility_counts[idx], 1u);
        }

        __global__ void packVisibilityBitset_kernel(
            uint32_t *visibility_bits,
            const uint32_t *visibility_counts,
            uint32_t min_visible_keyframes,
            uint32_t n_gaussians)
        {
            const uint32_t idx = static_cast<uint32_t>(blockIdx.x * blockDim.x + threadIdx.x);
            if (idx >= n_gaussians)
            {
                return;
            }

            if (visibility_counts[idx] >= min_visible_keyframes)
            {
                atomicOr(&visibility_bits[idx >> 5], 1u << (idx & 31u));
            }
        }

        __global__ void transformFloat4ToFloat4_kernel(
            const float4 *in_positions,
            float4 *out_positions,
            Pose global_pose,
            uint32_t n_points)
        {
            const uint32_t idx = static_cast<uint32_t>(blockIdx.x * blockDim.x + threadIdx.x);
            if (idx >= n_points) return;

            float4 p4 = in_positions[idx];
            float3 p = make_float3(p4.x, p4.y, p4.z);
            float3 gw = rotateByQuaternion(global_pose.orientation, p) + global_pose.position;
            out_positions[idx] = make_float4(gw.x, gw.y, gw.z, p4.w);
        }

        __global__ void transformAndPackSelectedPositions_kernel(
            const float4 *selected_positions_local,
            float *packed_xyz_global,
            Pose source_global_pose,
            uint32_t n_points)
        {
            const uint32_t idx = static_cast<uint32_t>(blockIdx.x * blockDim.x + threadIdx.x);
            if (idx >= n_points)
            {
                return;
            }

            const float4 p_local4 = selected_positions_local[idx];
            const float3 p_local = make_float3(p_local4.x, p_local4.y, p_local4.z);
            const float3 p_global = rotateByQuaternion(source_global_pose.orientation, p_local) + source_global_pose.position;

            const uint32_t base = idx * 3u;
            packed_xyz_global[base + 0] = p_global.x;
            packed_xyz_global[base + 1] = p_global.y;
            packed_xyz_global[base + 2] = p_global.z;
        }

        __global__ void packSelectedPositionsLocal_kernel(
            const float4 *selected_positions_local,
            float *packed_xyz_local,
            uint32_t n_points)
        {
            const uint32_t idx = static_cast<uint32_t>(blockIdx.x * blockDim.x + threadIdx.x);
            if (idx >= n_points)
            {
                return;
            }

            const float4 p_local4 = selected_positions_local[idx];
            const uint32_t base = idx * 3u;
            packed_xyz_local[base + 0] = p_local4.x;
            packed_xyz_local[base + 1] = p_local4.y;
            packed_xyz_local[base + 2] = p_local4.z;
        }

        std::vector<size_t> selectKeyframesForFrustum(const std::vector<KeyframeData> &keyframes, size_t max_keyframes)
        {
            std::vector<size_t> indices;
            if (keyframes.empty())
            {
                return indices;
            }

            const size_t cap = std::max<size_t>(1, std::min(max_keyframes, keyframes.size()));
            indices.reserve(cap);

            if (cap == keyframes.size())
            {
                for (size_t i = 0; i < keyframes.size(); ++i)
                {
                    indices.push_back(i);
                }
                return indices;
            }

            if (cap == 1)
            {
                indices.push_back(keyframes.size() - 1);
                return indices;
            }

            const double step = static_cast<double>(keyframes.size() - 1) / static_cast<double>(cap - 1);
            size_t last_index = keyframes.size();
            for (size_t i = 0; i < cap; ++i)
            {
                size_t idx = static_cast<size_t>(std::round(static_cast<double>(i) * step));
                idx = std::min(idx, keyframes.size() - 1);
                if (idx != last_index)
                {
                    indices.push_back(idx);
                    last_index = idx;
                }
            }

            if (!indices.empty() && indices.back() != keyframes.size() - 1)
            {
                indices.push_back(keyframes.size() - 1);
            }

            std::sort(indices.begin(), indices.end());
            indices.erase(std::unique(indices.begin(), indices.end()), indices.end());
            return indices;
        }

        FrustumVisibleCloud collectVisibleGaussianCloudFromFrustums(
            const Submap *source,
            const std::vector<KeyframeData> &observer_keyframes,
            size_t max_gaussians,
            size_t max_keyframes,
            int min_visible_keyframes)
        {
            FrustumVisibleCloud result;

            // Modo diagnostico temporal: se deja la ruta de frustum desactivada por comentario.
            // Cuando se quiera restaurar, esta funcion volviera a usar observer_keyframes,
            // max_keyframes y min_visible_keyframes para filtrar por visibilidad real.
            (void)observer_keyframes;
            (void)max_keyframes;
            (void)min_visible_keyframes;

            if (!source || source->gaussians_count == 0)
            {
                return result;
            }

            const size_t total_gaussians = std::min(static_cast<size_t>(source->gaussians_count), max_gaussians);
            if (total_gaussians == 0)
            {
                return result;
            }

            result.total_gaussians = total_gaussians;
            /*
            const std::vector<size_t> keyframe_indices = selectKeyframesForFrustum(observer_keyframes, max_keyframes);
            if (keyframe_indices.empty())
            {
                return result;
            }

            thrust::device_vector<uint32_t> visibility_counts(total_gaussians, 0u);
            thrust::device_vector<uint32_t> visibility_bits((total_gaussians + 31u) / 32u, 0u);
            thrust::device_vector<uint32_t> reject_counts(4u, 0u);

            uint32_t *d_counts = thrust::raw_pointer_cast(visibility_counts.data());
            uint32_t *d_bits = thrust::raw_pointer_cast(visibility_bits.data());
            uint32_t *d_reject_counts = thrust::raw_pointer_cast(reject_counts.data());
            const Pose source_global_pose = source->getGlobalPose();

            const dim3 block(256);
            const dim3 grid(static_cast<uint32_t>((total_gaussians + block.x - 1) / block.x));

            for (size_t k = 0; k < keyframe_indices.size(); ++k)
            {
                const KeyframeData &kf = observer_keyframes[keyframe_indices[k]];
                if (kf.getWidth() <= 0 || kf.getHeight() <= 0)
                {
                    continue;
                }

                const uint32_t width = static_cast<uint32_t>(kf.getWidth());
                const uint32_t height = static_cast<uint32_t>(kf.getHeight());
                const Pose cam_global = kf.getGlobalPose();

                countFrustumVisibility_kernel<<<grid, block>>>(
                    d_counts,
                    d_reject_counts,
                    thrust::raw_pointer_cast(source->gaussians.positions.data()),
                    source_global_pose,
                    cam_global,
                    kf.getIntrinsics(),
                    0.4f,
                    width,
                    height,
                    static_cast<uint32_t>(total_gaussians));
            }
            */

            // Temporalmente desactivado: elegimos todas las gaussianas del submapa.
            // Esto evita que el frustum cambie la nube usada por ICP durante el diagnóstico.
            const size_t selected_count = total_gaussians;
            result.visible_gaussians = selected_count;
            result.discarded_gaussians = 0;
            result.discarded_non_finite = 0;
            result.discarded_behind_camera = 0;
            result.discarded_out_of_bounds = 0;
            result.visibility_ratio = 1.0f;

            std::cout << "[LoopClosureModule::registerSubmapsOpen3D][frustum_select]"
                      << " source_submap=" << source->submap_id
                      << " total_gaussians=" << result.total_gaussians
                      << " kept_gaussians=" << result.visible_gaussians
                      << " discarded_gaussians=" << result.discarded_gaussians
                      << " discarded_non_finite=" << result.discarded_non_finite
                      << " discarded_behind_camera=" << result.discarded_behind_camera
                      << " discarded_out_of_bounds=" << result.discarded_out_of_bounds
                      << " visibility_ratio=" << result.visibility_ratio
                      << std::endl;

            if (selected_count == 0)
            {
                return result;
            }

            const dim3 tr_block(256);
            const dim3 tr_grid(static_cast<uint32_t>((selected_count + tr_block.x - 1) / tr_block.x));
            result.packed_xyz.resize(selected_count * 3u);
            packSelectedPositionsLocal_kernel<<<tr_grid, tr_block>>>(
                thrust::raw_pointer_cast(source->gaussians.positions.data()),
                thrust::raw_pointer_cast(result.packed_xyz.data()),
                static_cast<uint32_t>(selected_count));

            cudaError_t t_err = cudaGetLastError();
            if (t_err != cudaSuccess) {
                std::cout << "[LoopClosureModule::registerSubmapsOpen3D][reject] reason=pack_selected_local_failed"
                          << " error=" << cudaGetErrorString(t_err)
                          << std::endl;
                return result;
            }

            const cudaError_t pack_sync_error = cudaDeviceSynchronize();
            if (pack_sync_error != cudaSuccess)
            {
                std::cout << "[LoopClosureModule::registerSubmapsOpen3D][reject] reason=pack_selected_positions_sync_failed"
                          << " error=" << cudaGetErrorString(pack_sync_error)
                          << std::endl;
                result.packed_xyz.clear();
                result.visible_gaussians = 0;
                result.visibility_ratio = 0.0f;
                return result;
            }

            return result;
        }

    }

    // Coarse-to-fine registration using Open3D tensor APIs:
    // - Build point clouds from Gaussian means
    // - Downsample on CUDA
    // - Estimate normals on CUDA
    // - Refine directly with Point-to-Plane ICP from the caller pose
    bool LoopClosureModule::registerSubmapsOpen3D(
        const Submap* source,
        const Submap* target,
        float visual_similarity,
        const Pose& init_guess,
        Pose& estimated_T,
        float& confidence,
        int& num_inliers,
        float& mean_residual)
    {
        using namespace open3d;
        if (!source || !target)
        {
            std::cout << "[LoopClosureModule::registerSubmapsOpen3D][reject] reason=null_input"
                      << " source_ptr=" << source
                      << " target_ptr=" << target
                      << " visual_similarity=" << visual_similarity
                      << std::endl;
            return false;
        }
        const LoopClosureConfig cfg = getConfiguration();
        const auto total_start = std::chrono::steady_clock::now();

        const size_t max_pts = std::max<size_t>(1, cfg.open3d_max_points);
        const size_t keyframe_cap = std::max<size_t>(1, static_cast<size_t>(cfg.max_keyframes_per_submap));
        const int min_visible_kfs = std::max(1, cfg.frustum_overlap_min_visible_keyframes);

        const auto source_visible = collectVisibleGaussianCloudFromFrustums(
            source,
            target->keyframes,
            max_pts,
            keyframe_cap,
            min_visible_kfs);

        const auto target_visible = collectVisibleGaussianCloudFromFrustums(
            target,
            source->keyframes,
            max_pts,
            keyframe_cap,
            min_visible_kfs);

        const size_t src_n = source_visible.packed_xyz.size() / 3u;
        const size_t tgt_n = target_visible.packed_xyz.size() / 3u;
        const Pose src_global = source->getGlobalPose();
        const Pose tgt_global = target->getGlobalPose();
        // Convencion canonica del archivo:
        // - ICP trabaja con una transformacion source -> target.
        // - LoopEdge::relative_pose tambien se guarda como source -> target.
        // Por eso la semilla inicial y la salida del ICP se expresan en esa misma direccion.
        const Pose expected_relative_guess = composePoses(invertPose(src_global), tgt_global);
        const float init_guess_translation_error = poseTranslationError(init_guess, expected_relative_guess);
        const float init_guess_rotation_error = poseRotationDeg(init_guess, expected_relative_guess);

        std::cout << "[LoopClosureModule::registerSubmapsOpen3D][init_guess_check]"
              << " source=" << source->submap_id
              << " target=" << target->submap_id
              << " expected=" << poseToString(expected_relative_guess)
              << " init_guess=" << poseToString(init_guess)
              << " translation_error_m=" << init_guess_translation_error
              << " rotation_error_deg=" << init_guess_rotation_error
              << std::endl;

        std::cout << "[LoopClosureModule::registerSubmapsOpen3D][frustum]"
                  << " source=" << source->submap_id
                  << " target=" << target->submap_id
                  << " source_global=" << poseToString(src_global)
                  << " target_global=" << poseToString(tgt_global)
                  << " init_guess_relative=" << poseToString(init_guess)
                  << " src_total=" << source_visible.total_gaussians
                  << " src_kept=" << src_n
                  << " src_discarded=" << source_visible.discarded_gaussians
                  << " src_non_finite=" << source_visible.discarded_non_finite
                  << " src_behind_camera=" << source_visible.discarded_behind_camera
                  << " src_out_of_bounds=" << source_visible.discarded_out_of_bounds
                  << " tgt_total=" << target_visible.total_gaussians
                  << " tgt_kept=" << tgt_n
                  << " tgt_discarded=" << target_visible.discarded_gaussians
                  << " tgt_non_finite=" << target_visible.discarded_non_finite
                  << " tgt_behind_camera=" << target_visible.discarded_behind_camera
                  << " tgt_out_of_bounds=" << target_visible.discarded_out_of_bounds
                  << " src_ratio=" << source_visible.visibility_ratio
                  << " tgt_ratio=" << target_visible.visibility_ratio
                  << " min_visible_kfs=" << min_visible_kfs
                  << std::endl;

        if (src_n < cfg.open3d_min_points || tgt_n < cfg.open3d_min_points)
        {
            std::cout << "[LoopClosureModule::registerSubmapsOpen3D][reject] reason=not_enough_points"
                      << " source=" << source->submap_id
                      << " target=" << target->submap_id
                      << " src_n=" << src_n
                      << " tgt_n=" << tgt_n
                      << " min_points=" << cfg.open3d_min_points
                      << std::endl;
            return false;
        }

        const auto copy_start = std::chrono::steady_clock::now();
        const open3d::core::Device cuda_device("CUDA:0");

        auto buildTensorPointCloud = [&](const FrustumVisibleCloud &visible) -> open3d::t::geometry::PointCloud
        {
            const size_t n_points = visible.packed_xyz.size() / 3u;
            open3d::t::geometry::PointCloud tensor_pc(cuda_device);
            if (n_points == 0)
            {
                return tensor_pc;
            }

            // allocate tensor on CUDA device
            open3d::core::Tensor positions = open3d::core::Tensor::Empty(
                {static_cast<int64_t>(n_points), 3},
                open3d::core::Float32,
                cuda_device);

            // packed_xyz contains source/target-local coordinates for ICP.
            open3d::core::MemoryManager::Memcpy(
                positions.GetDataPtr(),
                cuda_device,
                thrust::raw_pointer_cast(visible.packed_xyz.data()),
                cuda_device,
                n_points * 3u * sizeof(float));

            tensor_pc.SetPointPositions(positions);
            return tensor_pc;
        };

        open3d::t::geometry::PointCloud src_tensor_pc = buildTensorPointCloud(source_visible);
        open3d::t::geometry::PointCloud tgt_tensor_pc = buildTensorPointCloud(target_visible);

        const auto copy_end = std::chrono::steady_clock::now();

        // Downsample for features on CUDA tensor point clouds.
        const double voxel_size = std::max(1e-6, cfg.open3d_voxel_size_m);
        const auto downsample_start = std::chrono::steady_clock::now();
        src_tensor_pc = src_tensor_pc.VoxelDownSample(voxel_size);
        tgt_tensor_pc = tgt_tensor_pc.VoxelDownSample(voxel_size);

        const size_t src_ds_n = static_cast<size_t>(src_tensor_pc.GetPointPositions().GetLength());
        const size_t tgt_ds_n = static_cast<size_t>(tgt_tensor_pc.GetPointPositions().GetLength());
        if (src_ds_n < cfg.open3d_min_downsampled_points || tgt_ds_n < cfg.open3d_min_downsampled_points)
        {
            std::cout << "[LoopClosureModule::registerSubmapsOpen3D][reject] reason=not_enough_downsampled_points"
                      << " source=" << source->submap_id
                      << " target=" << target->submap_id
                      << " src_ds_n=" << src_ds_n
                      << " tgt_ds_n=" << tgt_ds_n
                      << " min_downsampled_points=" << cfg.open3d_min_downsampled_points
                      << std::endl;
            return false;
        }
        const auto downsample_end = std::chrono::steady_clock::now();

        const double normal_radius = voxel_size * std::max(1.0, cfg.open3d_normal_radius_scale);
        const int normal_max_nn = std::max(1, cfg.open3d_normal_max_nn);

        // Try to estimate normals on the tensor (GPU) point clouds first.
        const auto tensor_normals_start = std::chrono::steady_clock::now();
        try {
            src_tensor_pc.EstimateNormals(std::optional<int>(normal_max_nn), std::optional<double>(normal_radius));
            tgt_tensor_pc.EstimateNormals(std::optional<int>(normal_max_nn), std::optional<double>(normal_radius));
            const auto tensor_normals_end = std::chrono::steady_clock::now();
            const double tensor_normals_elapsed = std::chrono::duration<double>(tensor_normals_end - tensor_normals_start).count();
            std::cout << "[LoopClosureModule::registerSubmapsOpen3D][info] tensor EstimateNormals elapsed=" << tensor_normals_elapsed << "s" << std::endl;
            // Diagnostic checks: ensure normals were produced and are finite
            try {
                using core::Device;
                const int64_t src_n = static_cast<int64_t>(src_tensor_pc.GetPointPositions().GetLength());
                const int64_t tgt_n = static_cast<int64_t>(tgt_tensor_pc.GetPointPositions().GetLength());

                auto src_normals = src_tensor_pc.GetPointNormals().To(Device("CPU:0"));
                auto tgt_normals = tgt_tensor_pc.GetPointNormals().To(Device("CPU:0"));
                auto src_positions = src_tensor_pc.GetPointPositions().To(Device("CPU:0"));
                auto tgt_positions = tgt_tensor_pc.GetPointPositions().To(Device("CPU:0"));

                const float *src_norm_ptr = static_cast<const float*>(src_normals.GetDataPtr());
                const float *tgt_norm_ptr = static_cast<const float*>(tgt_normals.GetDataPtr());
                const float *src_pos_ptr = static_cast<const float*>(src_positions.GetDataPtr());
                const float *tgt_pos_ptr = static_cast<const float*>(tgt_positions.GetDataPtr());

                const int64_t check_n_src = std::min<int64_t>(src_n, 10);
                const int64_t check_n_tgt = std::min<int64_t>(tgt_n, 10);
                bool src_norms_ok = src_normals.NumElements() >= src_n * 3;
                bool tgt_norms_ok = tgt_normals.NumElements() >= tgt_n * 3;

                auto check_finite = [](const float *ptr, int64_t count)->bool{
                    for (int64_t i = 0; i < count; ++i) {
                        if (!std::isfinite(ptr[i])) return false;
                    }
                    return true;
                };

                bool src_sample_finite = src_norms_ok ? check_finite(src_norm_ptr, check_n_src * 3) : false;
                bool tgt_sample_finite = tgt_norms_ok ? check_finite(tgt_norm_ptr, check_n_tgt * 3) : false;

                // compute simple centroids for a small sample to help debugging
                auto compute_centroid = [](const float *pos_ptr, int64_t n)->std::array<double,3>{
                    std::array<double,3> c = {0.0,0.0,0.0};
                    if (n <= 0) return c;
                    for (int64_t i = 0; i < n; ++i) {
                        c[0] += pos_ptr[i*3 + 0];
                        c[1] += pos_ptr[i*3 + 1];
                        c[2] += pos_ptr[i*3 + 2];
                    }
                    c[0] /= static_cast<double>(n);
                    c[1] /= static_cast<double>(n);
                    c[2] /= static_cast<double>(n);
                    return c;
                };

                auto src_cent = compute_centroid(src_pos_ptr, std::min<int64_t>(src_n, 100));
                auto tgt_cent = compute_centroid(tgt_pos_ptr, std::min<int64_t>(tgt_n, 100));

                std::cout << "[LoopClosureModule::registerSubmapsOpen3D][diag] src_points=" << src_n << " tgt_points=" << tgt_n
                          << " src_norms_ok=" << src_norms_ok << " tgt_norms_ok=" << tgt_norms_ok
                          << " src_sample_finite=" << src_sample_finite << " tgt_sample_finite=" << tgt_sample_finite
                          << " src_centroid=(" << src_cent[0] << "," << src_cent[1] << "," << src_cent[2] << ")"
                          << " tgt_centroid=(" << tgt_cent[0] << "," << tgt_cent[1] << "," << tgt_cent[2] << ")"
                          << std::endl;
            }
            catch (const std::exception &e) {
                std::cout << "[LoopClosureModule::registerSubmapsOpen3D][warn] normals diagnostic failed: " << e.what() << std::endl;
            }
        }
        catch (const std::exception &e) {
            std::cout << "[LoopClosureModule::registerSubmapsOpen3D][warn] tensor EstimateNormals failed: " << e.what() << std::endl;
        }

        // Keep the whole registration path in the tensor API, starting from the caller pose.
        // If anything in the GPU path fails, stop here instead of falling back to legacy APIs.
        try
        {
            const bool cuda_available = !open3d::core::Device::GetAvailableCUDADevices().empty();
            std::cout << "[LoopClosureModule::registerSubmapsOpen3D][info] Open3D CUDA available="
                      << (cuda_available ? "yes" : "no")
                      << " src_tensor_device=" << src_tensor_pc.GetDevice().ToString()
                      << " tgt_tensor_device=" << tgt_tensor_pc.GetDevice().ToString()
                      << std::endl;

            const auto icp_start = std::chrono::steady_clock::now();
            const double icp_threshold = std::max(cfg.open3d_icp_min_distance_m,
                                                 voxel_size * std::max(1.0, cfg.open3d_icp_threshold_scale));
            // Clouds are in local frames; ICP returns source_local -> target_local.
            const Eigen::Matrix4d init_guess_eig = poseToEigenMatrix(init_guess);
            core::Tensor init_tf = open3d::core::eigen_converter::EigenMatrixToTensor(init_guess_eig);

            std::cout << "[LoopClosureModule::registerSubmapsOpen3D][icp_input]"
                      << " source=" << source->submap_id
                      << " target=" << target->submap_id
                      << " init_guess_relative=" << poseToString(init_guess)
                      << " src_global=" << poseToString(src_global)
                      << " tgt_global=" << poseToString(tgt_global)
                      << " icp_threshold=" << icp_threshold
                      << " src_points=" << src_ds_n
                      << " tgt_points=" << tgt_ds_n
                      << std::endl;

            t::pipelines::registration::ICPConvergenceCriteria icp_criteria;
            icp_criteria.max_iteration_ = cfg.open3d_icp_max_iteration;
            t::pipelines::registration::RegistrationResult icp_result =
                t::pipelines::registration::ICP(
                    src_tensor_pc,
                    tgt_tensor_pc,
                    icp_threshold,
                    init_tf,
                    t::pipelines::registration::TransformationEstimationPointToPlane(),
                    icp_criteria,
                    -1.0);
            const auto icp_end = std::chrono::steady_clock::now();

            core::Tensor tf_cpu = icp_result.transformation_.To(core::Device("CPU:0"));
            Eigen::Matrix4d T = open3d::core::eigen_converter::TensorToEigenMatrixXd(tf_cpu);

            const Pose current_relative_pose = eigenMatrixToPose(init_guess_eig);
            const Pose icp_relative_pose = eigenMatrixToPose(T);
            const Eigen::Matrix4d delta_T = T * init_guess_eig.inverse();
            const Pose delta_pose = eigenMatrixToPose(delta_T);
            // ICP devuelve directamente source -> target, que es la misma convencion
            // que usa LoopEdge::relative_pose y el resto del pipeline.
            const Pose predicted_target_global = composePoses(src_global, icp_relative_pose);
            const Pose target_global_residual = composePoses(invertPose(tgt_global), predicted_target_global);
            const Pose relative_delta = composePoses(invertPose(init_guess), icp_relative_pose);

            const float target_translation_error = poseTranslationError(predicted_target_global, tgt_global);
            const float target_rotation_error = poseRotationErrorDeg(predicted_target_global, tgt_global);
            const float relative_translation_error = poseTranslationError(icp_relative_pose, init_guess);
            const float relative_rotation_error = poseRotationDeg(icp_relative_pose, init_guess);

            const Eigen::Matrix4d Tsrc = poseToEigenMatrix(src_global);
            const Eigen::Matrix4d Ttgt = poseToEigenMatrix(tgt_global);
            const Eigen::Matrix4d Ticp = T;
            const Eigen::Matrix4d Tpred = Tsrc * Ticp;
            const Eigen::Matrix4d TpredInv = Tsrc * Ticp.inverse();
            const auto matrix_error = [&](const Eigen::Matrix4d &Ta, const Eigen::Matrix4d &Tb)
            {
                const Pose pa = eigenMatrixToPose(Ta);
                const Pose pb = eigenMatrixToPose(Tb);
                return std::pair<float, float>{poseTranslationError(pa, pb), poseRotationErrorDeg(pa, pb)};
            };
            const auto [error_pred_trans_m, error_pred_rot_deg] = matrix_error(Tpred, Ttgt);
            const auto [error_pred_inv_trans_m, error_pred_inv_rot_deg] = matrix_error(TpredInv, Ttgt);

            std::cout << "[LoopClosureModule::registerSubmapsOpen3D][pose]"
                      << " source=" << source->submap_id
                      << " target=" << target->submap_id
                      << " T_current_relative=" << poseToString(current_relative_pose)
                      << " T_icp_relative(source_to_target)=" << poseToString(icp_relative_pose)
                      << " delta_icp_minus_current=" << poseToString(delta_pose)
                      << " relative_delta_vs_guess=" << poseToString(relative_delta)
                      << " predicted_target_global=" << poseToString(predicted_target_global)
                      << " target_global_residual=" << poseToString(target_global_residual)
                      << " target_residual_translation_m=" << target_translation_error
                      << " target_residual_rotation_deg=" << target_rotation_error
                      << " relative_residual_translation_m=" << relative_translation_error
                      << " relative_residual_rotation_deg=" << relative_rotation_error
                      << " Tsrc=" << poseToString(src_global)
                      << " Ttgt=" << poseToString(tgt_global)
                      << " T_icp_relative(source_to_target)=" << poseToString(icp_relative_pose)
                      << " error_Tpred_vs_Ttgt_trans_m=" << error_pred_trans_m
                      << " error_Tpred_vs_Ttgt_rot_deg=" << error_pred_rot_deg
                      << std::endl;

            if (icp_result.fitness_ <= 0.0 || !T.allFinite())
            {
                std::cout << "[LoopClosureModule::registerSubmapsOpen3D][reject] reason=tensor_icp_invalid"
                          << " source=" << source->submap_id
                          << " target=" << target->submap_id
                          << " fitness=" << icp_result.fitness_
                          << " transform_finite=" << T.allFinite()
                          << " icp_threshold=" << icp_threshold
                          << " init_guess=" << poseToString(init_guess)
                          << std::endl;
                return false;
            }

            const float max_allowed_translation_error = std::max(1.5f, 2.0f * static_cast<float>(cfg.open3d_icp_min_distance_m));
            const float max_allowed_rotation_error_deg = 25.0f;
            if (target_translation_error > max_allowed_translation_error || target_rotation_error > max_allowed_rotation_error_deg)
            {
                std::cout << "[LoopClosureModule::registerSubmapsOpen3D][reject] reason=icp_diverged"
                          << " source=" << source->submap_id
                          << " target=" << target->submap_id
                          << " target_residual_translation_m=" << target_translation_error
                          << " target_residual_rotation_deg=" << target_rotation_error
                          << " max_allowed_translation_error=" << max_allowed_translation_error
                          << " max_allowed_rotation_error_deg=" << max_allowed_rotation_error_deg
                          << " init_guess=" << poseToString(init_guess)
                          << " returned_edge_pose=" << poseToString(icp_relative_pose)
                          << " predicted_target_global=" << poseToString(predicted_target_global)
                          << " target_global=" << poseToString(tgt_global)
                          << " icp_relative_pose=" << poseToString(icp_relative_pose)
                          << " relative_delta_vs_guess=" << poseToString(relative_delta)
                          << " target_global_residual=" << poseToString(target_global_residual)
                          << std::endl;
                return false;
            }

            // La arista final se guarda con la misma convencion que ICP: source -> target.
            // eigenMatrixToPose ya deja w >= 0, asi que no hay que invertir ni re-normalizar aqui.
            estimated_T = icp_relative_pose;

            /*logPairwiseTransformDiagnostics(
                "icp_vs_graph",
                static_cast<int>(source->submap_id),
                static_cast<int>(target->submap_id),
                estimated_T,
                estimated_T);
            */
            mean_residual = static_cast<float>(icp_result.inlier_rmse_);
            num_inliers = static_cast<int>(std::llround(icp_result.fitness_ * static_cast<double>(tgt_tensor_pc.GetPointPositions().GetLength())));

            const double frustum_support = std::clamp(
                0.5 * static_cast<double>(source_visible.visibility_ratio) +
                0.5 * static_cast<double>(target_visible.visibility_ratio),
                0.0,
                1.0);
            confidence = static_cast<float>(std::clamp(
                0.55 * icp_result.fitness_ +
                0.45 * frustum_support,
                0.0,
                1.0));

            const auto total_end = std::chrono::steady_clock::now();
            const auto ms = [](const auto &start, const auto &end)
            {
                return std::chrono::duration<double, std::milli>(end - start).count();
            };

            std::cout << "[LoopClosureModule::registerSubmapsOpen3D][timing]"
                      << " source=" << source->submap_id
                      << " target=" << target->submap_id
                      << " copy_ms=" << ms(copy_start, copy_end)
                      << " downsample_ms=" << ms(downsample_start, downsample_end)
                      << " normals_ms=" << ms(tensor_normals_start, icp_start)
                      << " icp_ms=" << ms(icp_start, icp_end)
                      << " total_ms=" << ms(total_start, total_end)
                      << " frustum_src_ratio=" << source_visible.visibility_ratio
                      << " frustum_tgt_ratio=" << target_visible.visibility_ratio
                      << " inliers=" << num_inliers
                      << " fitness=" << icp_result.fitness_
                      << " mean_residual=" << mean_residual
                      << " confidence=" << confidence
                      << " target_residual_translation_m=" << target_translation_error
                      << " target_residual_rotation_deg=" << target_rotation_error
                      << " relative_residual_translation_m=" << relative_translation_error
                      << " relative_residual_rotation_deg=" << relative_rotation_error
                      << " returned_edge_pose=" << poseToString(estimated_T)
                      << std::endl;

            return confidence > 0.0f;
        }
        catch (const std::exception &e)
        {
            std::cout << "[LoopClosureModule::registerSubmapsOpen3D][reject] reason=tensor_pipeline_failed"
                      << " error=" << e.what()
                      << " source=" << source->submap_id
                      << " target=" << target->submap_id
                      << " init_guess=" << poseToString(init_guess)
                      << " src_points=" << src_ds_n
                      << " tgt_points=" << tgt_ds_n
                      << std::endl;
            return false;
        }
    }

    bool LoopClosureModule::registerSubmapsOpen3D(
        const Submap* source,
        const Submap* target,
        float visual_similarity,
        const Pose& init_guess,
        LoopEdge& edge_out,
        float& mean_residual)
    {
        Pose estimated_T;
        float confidence = 0.0f;
        int num_inliers = 0;

        if (!registerSubmapsOpen3D(source, target, visual_similarity, init_guess, estimated_T, confidence, num_inliers, mean_residual))
        {
            return false;
        }

        edge_out.source_submap_id = static_cast<int>(source->submap_id);
        edge_out.target_submap_id = static_cast<int>(target->submap_id);
        edge_out.relative_pose = estimated_T;
        edge_out.confidence = confidence;
        edge_out.visual_similarity = visual_similarity;
        edge_out.source_uncertainty = source->accumulated_translation_uncertainty_m + 0.02f * source->accumulated_rotation_uncertainty_deg;
        edge_out.target_uncertainty = target->accumulated_translation_uncertainty_m + 0.02f * target->accumulated_rotation_uncertainty_deg;
        edge_out.num_inliers = num_inliers;

        const float overlap = computeOverlapRatio(source, target);
        edge_out.information_matrix = buildLoopInformationMatrix(
            source,
            target,
            visual_similarity,
            overlap,
            mean_residual,
            num_inliers,
            confidence);

        return true;
    }

        // Reuse rasterization pipelines: project -> hash -> ranges -> visibility
        // We declare externally-defined kernels in GSCudaKernels.cuh and call
        // the same sequence used by GSSlam::prepareRasterization()

} // namespace f_vigs_slam
