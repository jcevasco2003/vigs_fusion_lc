// ============================================================
// Trabajador combinado de cierre de bucles
// Secuencia principal: extracción -> similitud -> verificación -> registro -> PGO
// ============================================================

#include <f_vigs_slam/GSSlam.cuh>
#include <f_vigs_slam/NetVLADWrapper.hpp>
#include <f_vigs_slam/LoopClosureModule.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <limits>
#include <unordered_map>
#include <utility>
#include <vector>
#include <thread>

#include <cublas_v2.h>
#include <thrust/copy.h>
#include <thrust/device_vector.h>

namespace f_vigs_slam
{

    float computeSubmapSelfSimilarityGPU(
            const std::vector<uint32_t> &descriptor_indices,
            const thrust::device_vector<float> &descriptor_database_gpu,
            int descriptor_dim,
            float percentile,
            cudaStream_t stream);

    thrust::device_vector<float> computeSimilarityMatrixGPU(
        const thrust::device_vector<float> &left_rows,
        int left_count,
        const thrust::device_vector<float> &right_rows,
        int right_count,
        int dim,
        cudaStream_t stream);

    namespace
    {
        std::string poseToString(const Pose &pose)
        {
            std::ostringstream oss;
            oss << "pos=(" << pose.position.x << ", " << pose.position.y << ", " << pose.position.z << ")"
                << " quat=(" << pose.orientation.x << ", " << pose.orientation.y << ", " << pose.orientation.z << ", " << pose.orientation.w << ")";
            return oss.str();
        }
    }

    struct CandidateStats
    {
        int votes = 0;
        float best_similarity = 0.0f;
    };

    // Hilo principal de cierre de bucles:
    // 1) recibe el evento de submapa cerrado,
    // 2) extrae descriptores faltantes,
    // 3) calcula similitudes y candidatos,
    // 4) verifica los candidatos con Open3D
    // 5) y aplica PGO si corresponde.
    void GSSlam::loopDetectionAndClosureThread()
    {


        std::cout << "[loopDetectionAndClosureThread] started" << std::endl;

        int loop_cycle = 0;
        std::vector<LoopEdge> batch_edges;
        batch_edges.reserve(8);

        int period = 0;

        while (!stop_loop_detection_.load(std::memory_order_acquire))
        {
            
            if (getDebugUpdateGlobalPoses() && (period % 2000 == 0) && false)
            {
                std::cout << "[loopDetectionAndClosureThread][debug] applying periodic global pose update by lifting submap 1" << std::endl;
                optimization_mutex_.lock();
                debugLiftSecondSubmapAndUpdate(getDebugUpdateDzM());
                optimization_mutex_.unlock();
            }

            period++;


            // Entrada: esperar un evento de submapa cerrado para disparar el ciclo de cierre.
            const auto cycle_start = std::chrono::steady_clock::now();
            bool did_work = false;

            // Cierre: si hay un submapa cerrado pendiente, primero se procesa su loop closure.
            const auto detect_start = std::chrono::steady_clock::now();
            did_work |= processSubmapSimilarities(loop_cycle);
            const auto detect_end = std::chrono::steady_clock::now();

            // Extracción: después se procesan los keyframes nuevos que llegaron desde addKeyframe().
            const auto extract_start = std::chrono::steady_clock::now();
            DescriptorExtractionTask extract_task;
            while (descriptor_extraction_queue_.try_dequeue(extract_task))
            {
                did_work |= extractDescriptor(extract_task);
            }
            const auto extract_end = std::chrono::steady_clock::now();

            // Verificación: se consumen candidatos y se intenta registrar la mejor coincidencia.
            const auto verify_start = std::chrono::steady_clock::now();
            did_work |= verifyLoop(batch_edges);
            const auto verify_end = std::chrono::steady_clock::now();

            // PGO: se consolidan las aristas válidas y se corrige la trayectoria global.
            const auto pgo_start = std::chrono::steady_clock::now();
            did_work |= poseGraphOptimization(batch_edges);
            const auto pgo_end = std::chrono::steady_clock::now();

            const float extract_ms = std::chrono::duration<float, std::milli>(extract_end - extract_start).count();
            const float detect_ms = std::chrono::duration<float, std::milli>(detect_end - detect_start).count();
            const float verify_ms = std::chrono::duration<float, std::milli>(verify_end - verify_start).count();
            const float pgo_ms = std::chrono::duration<float, std::milli>(pgo_end - pgo_start).count();
            const float cycle_ms = std::chrono::duration<float, std::milli>(pgo_end - cycle_start).count();

            loop_last_extract_ms_.store(extract_ms, std::memory_order_release);
            loop_last_detect_ms_.store(detect_ms, std::memory_order_release);
            loop_last_verify_ms_.store(verify_ms, std::memory_order_release);
            loop_last_total_ms_.store(cycle_ms, std::memory_order_release);
            loop_last_lock_hold_ms_.store(verify_ms, std::memory_order_release);

            constexpr float kLoopTimingPrintThresholdMs = 1.0f;
            const bool non_trivial_timing =
                extract_ms >= kLoopTimingPrintThresholdMs ||
                detect_ms >= kLoopTimingPrintThresholdMs ||
                verify_ms >= kLoopTimingPrintThresholdMs ||
                pgo_ms >= kLoopTimingPrintThresholdMs ||
                cycle_ms >= kLoopTimingPrintThresholdMs;

            if (did_work && non_trivial_timing)
            {
                std::cout << "[loopDetectionAndClosureThread][timing] cycle=" << loop_cycle
                          << " extract_ms=" << extract_ms
                          << " detect_ms=" << detect_ms
                          << " verify_ms=" << verify_ms
                          << " pgo_ms=" << pgo_ms
                          << " cycle_ms=" << cycle_ms
                          << std::endl;
            }

            if (did_work)
            {
                loop_last_extract_ms_.store(extract_ms, std::memory_order_release);
                loop_last_detect_ms_.store(detect_ms, std::memory_order_release);
                loop_last_verify_ms_.store(verify_ms, std::memory_order_release);
                loop_last_total_ms_.store(cycle_ms, std::memory_order_release);
                loop_last_lock_hold_ms_.store(verify_ms, std::memory_order_release);
            }

            if (!did_work)
            {
                std::unique_lock<std::mutex> wait_lock(loop_work_mutex_);
                loop_work_cv_.wait_for(wait_lock, std::chrono::milliseconds(10));
            }
        }

        std::cout << "[loopDetectionAndClosureThread] stopped" << std::endl;
    }

    bool GSSlam::extractDescriptor(const DescriptorExtractionTask &task)
    {
        if (!netvlad_)
        {
            std::cerr << "[LoopClosure::extractDescriptor][reject] NetVLAD not initialized" << std::endl;
            return false;
        }

        const int reported_dim = netvlad_->descriptorSize();
        if (reported_dim > 0 && reported_dim != descriptor_dim_)
        {
            std::cerr << "[LoopClosure::extractDescriptor][info] NetVLAD reports descriptor_size="
                      << reported_dim << " but backend expects " << descriptor_dim_ << std::endl;
            descriptor_dim_ = reported_dim;
        }

        if (task.submap_idx < 0 || task.submap_idx >= static_cast<int>(submaps_.size()))
        {
            std::cerr << "[LoopClosure::extractDescriptor][reject] invalid submap index "
                      << task.submap_idx << std::endl;
            return false;
        }

        Submap *submap = submaps_[static_cast<size_t>(task.submap_idx)].get();
        if (!submap)
        {
            std::cerr << "[LoopClosure::extractDescriptor][reject] submap is null at index "
                      << task.submap_idx << std::endl;
            return false;
        }

        if (task.keyframe_id < 0 || task.keyframe_id >= static_cast<int>(submap->keyframes.size()))
        {
            std::cerr << "[LoopClosure::extractDescriptor][reject] invalid keyframe index "
                      << task.keyframe_id << " in submap " << task.submap_idx << std::endl;
            return false;
        }

        KeyframeData &kf = submap->keyframes[static_cast<size_t>(task.keyframe_id)];
        if (kf.hasCpuDescriptor() || kf.hasGpuDescriptor())
        {
            return true;
        }

        const auto extract_start = std::chrono::steady_clock::now();
        std::vector<float> descriptor = netvlad_->extractDescriptor(task.color_image);
        const auto extract_end = std::chrono::steady_clock::now();
        const double extract_ms = std::chrono::duration<double, std::milli>(extract_end - extract_start).count();

        if (descriptor.empty())
        {
            std::cerr << "[LoopClosure::extractDescriptor][reject] failed to extract descriptor"
                      << " submap=" << task.submap_idx << " kf=" << task.keyframe_id << std::endl;
            return false;
        }

        if (static_cast<int>(descriptor.size()) != descriptor_dim_)
        {
            std::cerr << "[LoopClosure::extractDescriptor][info] descriptor dimension mismatch: got "
                      << descriptor.size() << ", expected " << descriptor_dim_
                      << " (submap=" << task.submap_idx << " kf=" << task.keyframe_id << ")" << std::endl;
            if (descriptor.size() > static_cast<size_t>(descriptor_dim_))
            {
                descriptor.resize(static_cast<size_t>(descriptor_dim_));
            }
            else
            {
                descriptor.resize(static_cast<size_t>(descriptor_dim_), 0.0f);
            }
        }

        //std::cout << "[LoopClosure::extractDescriptor] extracted descriptor in " << extract_ms
        //          << " ms, dim=" << descriptor.size()
        //         << " submap=" << task.submap_idx << " kf=" << task.keyframe_id << std::endl;

        double l2 = 0.0;
        for (float v : descriptor)
        {
            l2 += static_cast<double>(v) * static_cast<double>(v);
        }
        l2 = std::sqrt(std::max(1e-12, l2));
        if (std::abs(l2 - 1.0) > 1e-3)
        {
            std::cerr << "[LoopClosure::extractDescriptor][info] descriptor L2 norm = " << l2
                      << ", re-normalizing (submap=" << task.submap_idx << " kf=" << task.keyframe_id << ")" << std::endl;
            for (auto &v : descriptor)
            {
                v = static_cast<float>(v / l2);
            }
        }

        {
            //std::lock_guard<std::mutex> lock(optimization_mutex_);
            if (task.submap_idx < 0 || task.submap_idx >= static_cast<int>(submaps_.size()))
            {
                return false;
            }

            Submap *locked_submap = submaps_[static_cast<size_t>(task.submap_idx)].get();
            if (!locked_submap)
            {
                return false;
            }

            if (task.keyframe_id < 0 || task.keyframe_id >= static_cast<int>(locked_submap->keyframes.size()))
            {
                return false;
            }

            KeyframeData &locked_kf = locked_submap->keyframes[static_cast<size_t>(task.keyframe_id)];
            locked_kf.setDescriptor(std::move(descriptor));

            if (!locked_kf.hasGpuDescriptor())
            {
                locked_kf.copyDescriptorToGpu();
            }

            if (locked_kf.hasGpuDescriptor() && descriptor_database_count_ < MAX_DESCRIPTORS_COUNT)
            {
                const size_t append_size = locked_kf.netvlad_descriptor_gpu.size();
                thrust::copy(
                    locked_kf.netvlad_descriptor_gpu.begin(),
                    locked_kf.netvlad_descriptor_gpu.end(),
                    descriptor_database_gpu_.begin() + static_cast<std::ptrdiff_t>(descriptor_database_offset_));

                descriptor_to_keyframe_map.emplace_back(task.submap_idx, task.keyframe_id);
                descriptor_database_offset_ += append_size;
                descriptor_database_count_++;

                //std::cout << "[LoopClosure::extractDescriptor] appended descriptor to GPU DB"
                //          << " count=" << descriptor_database_count_ << "/" << MAX_DESCRIPTORS_COUNT
                //          << " offset=" << descriptor_database_offset_ << " floats" << std::endl;

                new_keyframe_added_to_submap_.store(true, std::memory_order_release);
                loop_kf_counter_.fetch_add(1, std::memory_order_relaxed);
                loop_work_cv_.notify_one();
            }
            else if (descriptor_database_count_ >= MAX_DESCRIPTORS_COUNT)
            {
                std::cerr << "[LoopClosure::extractDescriptor][reject] GPU descriptor database full! "
                          << "count=" << descriptor_database_count_ << "/" << MAX_DESCRIPTORS_COUNT << std::endl;
            }
        }

        return true;
    }

    bool GSSlam::processSubmapSimilarities(int &loop_cycle)
    {
        SubmapClosedEvent ev;
        if (!submap_closed_queue_.try_dequeue(ev))
        {
            return false;
        }

        if (ev.submap_id < 0 || ev.submap_id >= static_cast<int>(submaps_.size()))
        {
            std::cerr << "[LoopClosure::processSubmapSimilarities][reject] invalid submap index "
                      << ev.submap_id << std::endl;
            return true;
        }

        {
            //std::lock_guard<std::mutex> lock(optimization_mutex_);

            Submap *submap = submaps_[static_cast<size_t>(ev.submap_id)].get();
            if (!submap)
            {
                std::cerr << "[LoopClosure::processSubmapSimilarities][reject] submap null "
                          << ev.submap_id << std::endl;
                return true;
            }
        }

        loop_cycle++;
        std::cout << "[LoopClosure::processSubmapSimilarities] cycle=" << loop_cycle
                  << " processing_submap=" << ev.submap_id << std::endl;

        std::unordered_map<int, std::vector<uint32_t>> submap_to_indices;
        std::vector<uint32_t> query_indices;
        for (size_t idx = 0; idx < descriptor_to_keyframe_map.size(); ++idx)
        {
            const int submap_id = descriptor_to_keyframe_map[idx].first;
            const uint32_t db_index = static_cast<uint32_t>(idx);
            submap_to_indices[submap_id].push_back(db_index);
        }

        const int query_submap_id = static_cast<int>(ev.submap_id);
        const auto query_it = submap_to_indices.find(query_submap_id);
        if (query_it == submap_to_indices.end() || query_it->second.empty())
        {
            std::cout << "[LoopClosure::processSubmapSimilarities][reject] cycle=" << loop_cycle
                      << " reason=query submap has no descriptors" << std::endl;
            return true;
        }

        query_indices = query_it->second;
        const int descriptor_dim = descriptor_dim_;
        const int num_stored = static_cast<int>(descriptor_to_keyframe_map.size());
        if (descriptor_dim <= 0 || num_stored <= 0)
        {
            std::cout << "[LoopClosure::processSubmapSimilarities][reject] cycle=" << loop_cycle
                      << " reason=invalid descriptor setup" << std::endl;
            return true;
        }

        std::cout << "[LoopClosure::processSubmapSimilarities] cycle=" << loop_cycle
                  << " query_count=" << query_indices.size()
                  << " num_stored=" << num_stored
                  << " descriptor_dim=" << descriptor_dim << std::endl;

        thrust::device_vector<float> query_batch(query_indices.size() * static_cast<size_t>(descriptor_dim));
        for (size_t i = 0; i < query_indices.size(); ++i)
        {
            const size_t src_offset = static_cast<size_t>(query_indices[i]) * static_cast<size_t>(descriptor_dim);
            thrust::copy(
                descriptor_database_gpu_.begin() + static_cast<std::ptrdiff_t>(src_offset),
                descriptor_database_gpu_.begin() + static_cast<std::ptrdiff_t>(src_offset + static_cast<size_t>(descriptor_dim)),
                query_batch.begin() + static_cast<std::ptrdiff_t>(i * static_cast<size_t>(descriptor_dim)));
        }

        const float percentile_p = loop_self_similarity_percentile_;

        

        float query_self_threshold = computeSubmapSelfSimilarityGPU(
            query_indices,
            descriptor_database_gpu_,
            descriptor_dim,
            percentile_p,
            retrieval_stream_);

        if (ev.submap_id < submaps_.size() && submaps_[ev.submap_id])
        {
            submaps_[ev.submap_id]->self_similarity_percentile_score = query_self_threshold;
            submaps_[ev.submap_id]->has_descriptor_similarity_stats = true;
        }

        const uint32_t first_query_idx = query_indices.front();

        thrust::device_vector<float> cross_matrix = computeSimilarityMatrixGPU(
            query_batch,
            static_cast<int>(query_indices.size()),
            descriptor_database_gpu_,
            //num_stored, comparo solo con los anteriores
            static_cast<int>(first_query_idx),
            descriptor_dim,
            retrieval_stream_);

        if (cross_matrix.empty())
        {
            std::cout << "[LoopClosure::processSubmapSimilarities][reject] cycle=" << loop_cycle
                      << " reason=cross similarity matrix is empty" << std::endl;
            return true;
        }

        std::cout << "[LoopClosure::processSubmapSimilarities] cycle=" << loop_cycle
                  << " computing similarity and votes" << std::endl;

        std::unordered_map<int, CandidateStats> vote_stats;
        std::unordered_map<int, float> cached_hist_thresholds;
        const size_t query_limit = static_cast<size_t>(std::max(1, max_descriptor_batch_size_));
        size_t comparisons_total = 0;
        size_t rejected_low_query_threshold = 0;
        size_t rejected_same_submap = 0;
        size_t rejected_hist_gate = 0;
        size_t accepted_votes = 0;

        for (size_t qi = 0; qi < query_indices.size() && qi < query_limit; ++qi)
        {
            const int query_db_idx = static_cast<int>(query_indices[qi]);
            const size_t row_offset = qi * static_cast<size_t>(first_query_idx);

            std::vector<float> row_sims_cpu(static_cast<size_t>(first_query_idx));
            thrust::copy_n(
                cross_matrix.begin() + static_cast<std::ptrdiff_t>(row_offset),
                static_cast<size_t>(first_query_idx),
                row_sims_cpu.begin());

            for (int db_idx = 0; db_idx < first_query_idx; ++db_idx)
            {
                if (db_idx == query_db_idx)
                {
                    continue;
                }

                comparisons_total++;

                const float sim = row_sims_cpu[static_cast<size_t>(db_idx)];
                if (sim < query_self_threshold)
                {
                    rejected_low_query_threshold++;
                    continue;
                }

                const int hist_submap = descriptor_to_keyframe_map[static_cast<size_t>(db_idx)].first;
                if (hist_submap == query_submap_id)
                {
                    rejected_same_submap++;
                    continue;
                }

                float hist_threshold = query_self_threshold;
                const auto hist_cached = cached_hist_thresholds.find(hist_submap);
                if (hist_cached != cached_hist_thresholds.end())
                {
                    hist_threshold = hist_cached->second;
                }
                else
                {
                    float cached_value = 1.0f;
                    bool has_cache = false;
                    if (hist_submap >= 0 && hist_submap < static_cast<int>(submaps_.size()) && submaps_[static_cast<size_t>(hist_submap)])
                    {
                        const auto *hist_submap_ptr = submaps_[static_cast<size_t>(hist_submap)].get();
                        if (hist_submap_ptr->has_descriptor_similarity_stats)
                        {
                            cached_value = hist_submap_ptr->self_similarity_percentile_score;
                            has_cache = true;
                        }
                    }

                    if (!has_cache)
                    {
                        const auto indices_it = submap_to_indices.find(hist_submap);
                        if (indices_it != submap_to_indices.end() && indices_it->second.size() >= 1)
                        {
                            cached_value = computeSubmapSelfSimilarityGPU(
                                indices_it->second,
                                descriptor_database_gpu_,
                                descriptor_dim,
                                percentile_p,
                                retrieval_stream_);
                            has_cache = true;

                            if (hist_submap >= 0 && hist_submap < static_cast<int>(submaps_.size()) && submaps_[static_cast<size_t>(hist_submap)])
                            {
                                submaps_[static_cast<size_t>(hist_submap)]->self_similarity_percentile_score = cached_value;
                                submaps_[static_cast<size_t>(hist_submap)]->has_descriptor_similarity_stats = true;
                            }
                        }
                    }

                    hist_threshold = has_cache ? cached_value : query_self_threshold;
                    cached_hist_thresholds[hist_submap] = hist_threshold;
                }

                const float gate = std::min(query_self_threshold, hist_threshold);
                if (sim < gate)
                {
                    rejected_hist_gate++;
                    continue;
                }

                CandidateStats &stats = vote_stats[hist_submap];
                stats.votes += 1;
                stats.best_similarity = std::max(stats.best_similarity, sim);
                accepted_votes++;
            }
        }

        std::vector<std::pair<int, CandidateStats>> ranked_candidates;
        ranked_candidates.reserve(vote_stats.size());
        for (const auto &kv : vote_stats)
        {
            if (kv.second.votes >= min_votes_for_loop_closure_)
            {
                ranked_candidates.push_back(kv);
            }
        }

        std::sort(
            ranked_candidates.begin(), ranked_candidates.end(),
            [](const auto &a, const auto &b)
            {
                if (a.second.votes != b.second.votes)
                {
                    return a.second.votes > b.second.votes;
                }
                if (a.second.best_similarity != b.second.best_similarity)
                {
                    return a.second.best_similarity > b.second.best_similarity;
                }
                return a.first < b.first;
            });

        constexpr size_t kMaxLoopCandidates = 8;
        size_t enqueued_candidates = 0;
        for (size_t i = 0; i < ranked_candidates.size() && i < kMaxLoopCandidates; ++i)
        {
            const int hist_submap = ranked_candidates[i].first;
            const CandidateStats &stats = ranked_candidates[i].second;
            LoopCandidate cand(static_cast<int>(ev.submap_id), hist_submap, stats.best_similarity);
            cand.votes = stats.votes;
            loop_candidates_queue_.enqueue(cand);
            enqueued_candidates++;
            std::cout << "[LoopClosure::processSubmapSimilarities] cycle=" << loop_cycle
                      << " candidate_enqueued"
                      << " source_submap=" << ev.submap_id
                      << " target_submap=" << hist_submap
                      << " votes=" << stats.votes
                      << " best_similarity=" << stats.best_similarity
                      << std::endl;
        }

        std::cout << "[LoopClosure::processSubmapSimilarities] cycle=" << loop_cycle
                  << " summary"
                  << " comparisons=" << comparisons_total
                  << " rejected_query_threshold=" << rejected_low_query_threshold
                  << " rejected_same_submap=" << rejected_same_submap
                  << " rejected_hist_gate=" << rejected_hist_gate
                  << " accepted_votes=" << accepted_votes
                  << " unique_candidates=" << vote_stats.size()
                  << " ranked_candidates=" << ranked_candidates.size()
                  << " enqueued=" << enqueued_candidates
                  << std::endl;

        if (enqueued_candidates > 0)
        {
            loop_work_cv_.notify_one();
        }

        return true;
    }

        Pose inversePoseGlobal(const Pose &pose)
        {
            Eigen::Quaternionf q(
                pose.orientation.w,
                pose.orientation.x,
                pose.orientation.y,
                pose.orientation.z);
            q.normalize();

            const Eigen::Quaternionf q_inv = q.conjugate();
            const Eigen::Vector3f t(
                pose.position.x,
                pose.position.y,
                pose.position.z);
            const Eigen::Vector3f t_inv = -(q_inv * t);

            return Pose(
                make_float3(t_inv.x(), t_inv.y(), t_inv.z()),
                make_float4(q_inv.x(), q_inv.y(), q_inv.z(), q_inv.w()));
        }

        thrust::device_vector<float> computeSimilarityMatrixGPU(
            const thrust::device_vector<float> &left_rows,
            int left_count,
            const thrust::device_vector<float> &right_rows,
            int right_count,
            int dim,
            cudaStream_t stream)
        {
            if (left_rows.empty() || right_rows.empty() || left_count <= 0 || right_count <= 0 || dim <= 0)
            {
                return {};
            }

            thrust::device_vector<float> similarities(static_cast<size_t>(left_count) * static_cast<size_t>(right_count));

            cublasHandle_t handle = nullptr;
            if (cublasCreate(&handle) != CUBLAS_STATUS_SUCCESS)
            {
                return {};
            }

            if (stream != nullptr)
            {
                if (cublasSetStream(handle, stream) != CUBLAS_STATUS_SUCCESS)
                {
                    cublasDestroy(handle);
                    return {};
                }
            }

            const float alpha = 1.0f;
            const float beta = 0.0f;

            const cublasStatus_t status = cublasSgemm(
                handle,
                CUBLAS_OP_T,
                CUBLAS_OP_N,
                right_count,
                left_count,
                dim,
                &alpha,
                thrust::raw_pointer_cast(right_rows.data()),
                dim,
                thrust::raw_pointer_cast(left_rows.data()),
                dim,
                &beta,
                thrust::raw_pointer_cast(similarities.data()),
                right_count);

            cublasDestroy(handle);

            if (status != CUBLAS_STATUS_SUCCESS)
            {
                return {};
            }

            if (stream == nullptr)
            {
                cudaDeviceSynchronize();
            }

            return similarities;
        }

        float computeSubmapSelfSimilarityGPU(
            const std::vector<uint32_t> &descriptor_indices,
            const thrust::device_vector<float> &descriptor_database_gpu,
            int descriptor_dim,
            float percentile,
            cudaStream_t stream)
        {
            if (descriptor_indices.empty() || descriptor_dim <= 0)
            {
                return 0.0f;
            }

            thrust::device_vector<float> submap_descriptors(descriptor_indices.size() * static_cast<size_t>(descriptor_dim));
            for (size_t i = 0; i < descriptor_indices.size(); ++i)
            {
                const size_t src_offset = static_cast<size_t>(descriptor_indices[i]) * static_cast<size_t>(descriptor_dim);
                thrust::copy(
                    descriptor_database_gpu.begin() + static_cast<std::ptrdiff_t>(src_offset),
                    descriptor_database_gpu.begin() + static_cast<std::ptrdiff_t>(src_offset + static_cast<size_t>(descriptor_dim)),
                    submap_descriptors.begin() + static_cast<std::ptrdiff_t>(i * static_cast<size_t>(descriptor_dim)));
            }

            thrust::device_vector<float> similarity_matrix = computeSimilarityMatrixGPU(
                submap_descriptors,
                static_cast<int>(descriptor_indices.size()),
                submap_descriptors,
                static_cast<int>(descriptor_indices.size()),
                descriptor_dim,
                stream);

            if (similarity_matrix.empty())
            {
                return 0.0f;
            }

            const int n = static_cast<int>(descriptor_indices.size());
            const size_t total = static_cast<size_t>(n) * static_cast<size_t>(n);

            // Copy similarity matrix to host and extract off-diagonal values
            std::vector<float> host_sim(total);
            thrust::copy(similarity_matrix.begin(), similarity_matrix.end(), host_sim.begin());

            std::vector<float> offdiag;
            offdiag.reserve(total - static_cast<size_t>(n));
            for (int i = 0; i < n; ++i)
            {
                const size_t row_offset = static_cast<size_t>(i) * static_cast<size_t>(n);
                for (int j = 0; j < n; ++j)
                {
                    if (i == j) continue;
                    offdiag.push_back(host_sim[row_offset + static_cast<size_t>(j)]);
                }
            }

            if (offdiag.empty()) return 0.0f;

            // Compute percentile on host using nth_element for O(N)
            percentile = std::clamp(percentile, 0.0f, 100.0f);
            const size_t idx = static_cast<size_t>(std::round((percentile / 100.0f) * (offdiag.size() - 1)));
            std::nth_element(offdiag.begin(), offdiag.begin() + static_cast<std::ptrdiff_t>(idx), offdiag.end());
            return offdiag[idx];
        }

    bool GSSlam::processLoopDetectionEvent(int &loop_cycle)
    {
        SubmapClosedEvent ev;
        if (!submap_closed_queue_.try_dequeue(ev))
        {
            return false;
        }

        if (!netvlad_)
        {
            std::cerr << "[loopDetectionAndClosureThread][extraction][reject] NetVLAD not initialized" << std::endl;
            return true;
        }

        if (ev.submap_id < 0 || ev.submap_id >= static_cast<int>(submaps_.size()))
        {
            std::cerr << "[loopDetectionAndClosureThread][extraction][reject] invalid submap index "
                      << ev.submap_id << std::endl;
            return true;
        }

        {
            //std::lock_guard<std::mutex> lock(optimization_mutex_);

            Submap* submap = submaps_[static_cast<size_t>(ev.submap_id)].get();
            if (!submap)
            {
                std::cerr << "[loopDetectionAndClosureThread][extraction][reject] submap null "
                          << ev.submap_id << std::endl;
                return true;
            }

            const int reported_dim = netvlad_->descriptorSize();
            if (reported_dim > 0 && reported_dim != descriptor_dim_)
            {
                std::cerr << "[loopDetectionAndClosureThread][extraction][info] NetVLAD reports descriptor_size="
                          << reported_dim << " but backend expects " << descriptor_dim_ << std::endl;
                descriptor_dim_ = reported_dim;
            }

            for (size_t k = 0; k < submap->keyframes.size(); ++k)
            {
                KeyframeData &kf = submap->keyframes[k];
                if (kf.hasCpuDescriptor() || kf.hasGpuDescriptor())
                {
                    continue;
                }

                cv::Mat color_cpu;
                try
                {
                    kf.color_img.download(color_cpu);
                }
                catch (const cv::Exception &e)
                {
                    std::cerr << "[loopDetectionAndClosureThread][extraction][reject] failed to download image: "
                              << e.what() << std::endl;
                    continue;
                }

                const auto extract_start = std::chrono::steady_clock::now();
                std::vector<float> descriptor = netvlad_->extractDescriptor(color_cpu);
                const auto extract_end = std::chrono::steady_clock::now();
                const double extract_ms = std::chrono::duration<double, std::milli>(extract_end - extract_start).count();

                if (descriptor.empty())
                {
                    std::cerr << "[loopDetectionAndClosureThread][extraction][reject] Failed to extract descriptor "
                              << "submap=" << ev.submap_id << " kf=" << kf.keyframe_id << std::endl;
                    continue;
                }

                if (static_cast<int>(descriptor.size()) != descriptor_dim_)
                {
                    std::cerr << "[loopDetectionAndClosureThread][extraction][info] NetVLAD descriptor dimension mismatch: got "
                              << descriptor.size() << ", expected " << descriptor_dim_
                              << " (submap=" << ev.submap_id << " kf=" << kf.keyframe_id << ")" << std::endl;
                    if (descriptor.size() > static_cast<size_t>(descriptor_dim_))
                    {
                        descriptor.resize(static_cast<size_t>(descriptor_dim_));
                    }
                    else
                    {
                        descriptor.resize(static_cast<size_t>(descriptor_dim_), 0.0f);
                    }
                }

                //std::cout << "[loopDetectionAndClosureThread][extraction] Extracted descriptor in " << extract_ms
                //          << " ms, dim=" << descriptor.size()
                //          << " submap=" << ev.submap_id << " kf=" << kf.keyframe_id << std::endl;

                double l2 = 0.0;
                for (float v : descriptor)
                {
                    l2 += static_cast<double>(v) * static_cast<double>(v);
                }
                l2 = std::sqrt(std::max(1e-12, l2));
                if (std::abs(l2 - 1.0) > 1e-3)
                {
                    for (auto &v : descriptor)
                    {
                        v = static_cast<float>(v / l2);
                    }
                }

                kf.setDescriptor(std::move(descriptor));

                if (!kf.hasGpuDescriptor())
                {
                    kf.copyDescriptorToGpu();
                }

                if (kf.hasGpuDescriptor() && descriptor_database_count_ < MAX_DESCRIPTORS_COUNT)
                {
                    size_t append_size = kf.netvlad_descriptor_gpu.size();
                    thrust::copy(kf.netvlad_descriptor_gpu.begin(), kf.netvlad_descriptor_gpu.end(),
                                 descriptor_database_gpu_.begin() + static_cast<std::ptrdiff_t>(descriptor_database_offset_));

                    descriptor_to_keyframe_map.emplace_back(ev.submap_id, static_cast<int>(kf.keyframe_id));
                    descriptor_database_offset_ += append_size;
                    descriptor_database_count_++;

                    //std::cout << "[loopDetectionAndClosureThread][extraction] Appended descriptor to GPU DB "
                    //          << "count=" << descriptor_database_count_ << "/" << MAX_DESCRIPTORS_COUNT
                    //          << " offset=" << descriptor_database_offset_ << " floats" << std::endl;

                    new_keyframe_added_to_submap_.store(true, std::memory_order_release);
                    loop_kf_counter_.fetch_add(1, std::memory_order_relaxed);
                }
                else if (descriptor_database_count_ >= MAX_DESCRIPTORS_COUNT)
                {
                    std::cerr << "[loopDetectionAndClosureThread][extraction][reject] GPU descriptor database full! "
                              << "count=" << descriptor_database_count_ << "/" << MAX_DESCRIPTORS_COUNT << std::endl;
                }
            }
        }

        loop_cycle++;
        std::cout << "[loopDetectionAndClosureThread][detection] cycle=" << loop_cycle
                  << " processing_submap=" << ev.submap_id << std::endl;

        std::unordered_map<int, std::vector<uint32_t>> submap_to_indices;
        std::vector<uint32_t> query_indices;
        for (size_t idx = 0; idx < descriptor_to_keyframe_map.size(); ++idx)
        {
            const int submap_id = descriptor_to_keyframe_map[idx].first;
            const uint32_t db_index = static_cast<uint32_t>(idx);
            submap_to_indices[submap_id].push_back(db_index);
        }

        const int query_submap_id = static_cast<int>(ev.submap_id);
        const auto query_it = submap_to_indices.find(query_submap_id);
        if (query_it == submap_to_indices.end() || query_it->second.empty())
        {
            std::cout << "[loopDetectionAndClosureThread][detection][reject] cycle=" << loop_cycle
                      << " reason=query submap has no descriptors" << std::endl;
            return true;
        }

        query_indices = query_it->second;
        const int descriptor_dim = descriptor_dim_;
        const int num_stored = static_cast<int>(descriptor_to_keyframe_map.size());
        if (descriptor_dim <= 0 || num_stored <= 0)
        {
            std::cout << "[loopDetectionAndClosureThread][detection][reject] cycle=" << loop_cycle
                      << " reason=invalid descriptor setup" << std::endl;
            return true;
        }

        //std::cout << "[loopDetectionAndClosureThread][detection] cycle=" << loop_cycle
        //          << " query_count=" << query_indices.size()
        //          << " num_stored=" << num_stored
        //          << " descriptor_dim=" << descriptor_dim << std::endl;

        thrust::device_vector<float> query_batch(query_indices.size() * static_cast<size_t>(descriptor_dim));
        for (size_t i = 0; i < query_indices.size(); ++i)
        {
            const size_t src_offset = static_cast<size_t>(query_indices[i]) * static_cast<size_t>(descriptor_dim);
            thrust::copy(
                descriptor_database_gpu_.begin() + static_cast<std::ptrdiff_t>(src_offset),
                descriptor_database_gpu_.begin() + static_cast<std::ptrdiff_t>(src_offset + static_cast<size_t>(descriptor_dim)),
                query_batch.begin() + static_cast<std::ptrdiff_t>(i * static_cast<size_t>(descriptor_dim)));
        }

        const float percentile_p = loop_self_similarity_percentile_;
        float query_self_threshold = computeSubmapSelfSimilarityGPU(
            query_indices,
            descriptor_database_gpu_,
            descriptor_dim,
            percentile_p,
            retrieval_stream_);

        if (ev.submap_id < submaps_.size() && submaps_[ev.submap_id])
        {
            submaps_[ev.submap_id]->self_similarity_percentile_score = query_self_threshold;
            submaps_[ev.submap_id]->has_descriptor_similarity_stats = true;
        }

        thrust::device_vector<float> cross_matrix = computeSimilarityMatrixGPU(
            query_batch,
            static_cast<int>(query_indices.size()),
            descriptor_database_gpu_,
            num_stored,
            descriptor_dim,
            retrieval_stream_);

        if (cross_matrix.empty())
        {
            std::cout << "[loopDetectionAndClosureThread][detection][reject] cycle=" << loop_cycle
                      << " reason=cross similarity matrix is empty" << std::endl;
            return true;
        }

        //std::cout << "[loopDetectionAndClosureThread][detection] cycle=" << loop_cycle
        //          << " computing similarity and votes" << std::endl;

        std::unordered_map<int, CandidateStats> vote_stats;
        std::unordered_map<int, float> cached_hist_thresholds;
        const size_t query_limit = static_cast<size_t>(std::max(1, max_descriptor_batch_size_));
        size_t comparisons_total = 0;
        size_t rejected_low_query_threshold = 0;
        size_t rejected_same_submap = 0;
        size_t rejected_hist_gate = 0;
        size_t accepted_votes = 0;

        for (size_t qi = 0; qi < query_indices.size() && qi < query_limit; ++qi)
        {
            const int query_db_idx = static_cast<int>(query_indices[qi]);
            const size_t row_offset = qi * static_cast<size_t>(num_stored);

            std::vector<float> row_sims_cpu(static_cast<size_t>(num_stored));
            thrust::copy_n(
                cross_matrix.begin() + static_cast<std::ptrdiff_t>(row_offset),
                static_cast<size_t>(num_stored),
                row_sims_cpu.begin());

            for (int db_idx = 0; db_idx < num_stored; ++db_idx)
            {
                if (db_idx == query_db_idx)
                {
                    continue;
                }

                comparisons_total++;

                const float sim = row_sims_cpu[static_cast<size_t>(db_idx)];
                if (sim < query_self_threshold)
                {
                    rejected_low_query_threshold++;
                    continue;
                }

                const int hist_submap = descriptor_to_keyframe_map[static_cast<size_t>(db_idx)].first;
                if (hist_submap == query_submap_id)
                {
                    rejected_same_submap++;
                    continue;
                }

                float hist_threshold = query_self_threshold;
                const auto hist_cached = cached_hist_thresholds.find(hist_submap);
                if (hist_cached != cached_hist_thresholds.end())
                {
                    hist_threshold = hist_cached->second;
                }
                else
                {
                    float cached_value = 1.0f;
                    bool has_cache = false;
                    if (hist_submap >= 0 && hist_submap < static_cast<int>(submaps_.size()) && submaps_[static_cast<size_t>(hist_submap)])
                    {
                        const auto *hist_submap_ptr = submaps_[static_cast<size_t>(hist_submap)].get();
                        if (hist_submap_ptr->has_descriptor_similarity_stats)
                        {
                            cached_value = hist_submap_ptr->self_similarity_percentile_score;
                            has_cache = true;
                        }
                    }

                    if (!has_cache)
                    {
                        const auto indices_it = submap_to_indices.find(hist_submap);
                        if (indices_it != submap_to_indices.end() && indices_it->second.size() >= 1)
                        {
                            cached_value = computeSubmapSelfSimilarityGPU(
                                indices_it->second,
                                descriptor_database_gpu_,
                                descriptor_dim,
                                percentile_p,
                                retrieval_stream_);
                            has_cache = true;

                            if (hist_submap >= 0 && hist_submap < static_cast<int>(submaps_.size()) && submaps_[static_cast<size_t>(hist_submap)])
                            {
                                submaps_[static_cast<size_t>(hist_submap)]->self_similarity_percentile_score = cached_value;
                                submaps_[static_cast<size_t>(hist_submap)]->has_descriptor_similarity_stats = true;
                            }
                        }
                    }

                    hist_threshold = has_cache ? cached_value : query_self_threshold;
                    cached_hist_thresholds[hist_submap] = hist_threshold;
                }

                const float gate = std::min(query_self_threshold, hist_threshold);
                if (sim < gate)
                {
                    rejected_hist_gate++;
                    continue;
                }

                CandidateStats &stats = vote_stats[hist_submap];
                stats.votes += 1;
                stats.best_similarity = std::max(stats.best_similarity, sim);
                accepted_votes++;
            }
        }

        std::vector<std::pair<int, CandidateStats>> ranked_candidates;
        ranked_candidates.reserve(vote_stats.size());
        for (const auto &kv : vote_stats)
        {
            if (kv.second.votes >= min_votes_for_loop_closure_)
            {
                ranked_candidates.push_back(kv);
            }
        }

        std::sort(
            ranked_candidates.begin(), ranked_candidates.end(),
            [](const auto &a, const auto &b)
            {
                if (a.second.votes != b.second.votes)
                {
                    return a.second.votes > b.second.votes;
                }
                if (a.second.best_similarity != b.second.best_similarity)
                {
                    return a.second.best_similarity > b.second.best_similarity;
                }
                return a.first < b.first;
            });

        constexpr size_t kMaxLoopCandidates = 8;
        size_t enqueued_candidates = 0;
        for (size_t i = 0; i < ranked_candidates.size() && i < kMaxLoopCandidates; ++i)
        {
            const int hist_submap = ranked_candidates[i].first;
            const CandidateStats &stats = ranked_candidates[i].second;
                        // Store loop edges as current/new -> historical/old.
                        LoopCandidate cand(static_cast<int>(ev.submap_id), hist_submap, stats.best_similarity);
            cand.votes = stats.votes;
            loop_candidates_queue_.enqueue(cand);
            enqueued_candidates++;
            //std::cout << "[loopDetectionAndClosureThread][detection] cycle=" << loop_cycle
            //        << " candidate_enqueued"
             //                               << " source_submap=" << ev.submap_id
             //                               << " target_submap=" << hist_submap
              //        << " votes=" << stats.votes
               //       << " best_similarity=" << stats.best_similarity
                //      << std::endl;
        }

          //std::cout << "[loopDetectionAndClosureThread][detection] cycle=" << loop_cycle
          //      << " summary"
          //        << " comparisons=" << comparisons_total
          //        << " rejected_query_threshold=" << rejected_low_query_threshold
          //        << " rejected_same_submap=" << rejected_same_submap
          //        << " rejected_hist_gate=" << rejected_hist_gate
          //        << " accepted_votes=" << accepted_votes
          //        << " unique_candidates=" << vote_stats.size()
          //        << " ranked_candidates=" << ranked_candidates.size()
          //        << " enqueued=" << enqueued_candidates
          //        << std::endl;

        return true;
    }

    // ================= VERIFICACION DE CANDIDATOS DE LOOP =================
    // Entrada: candidatos (submapa fuente/objetivo + similitud + votos).
    // Salida: aristas LoopEdge validadas para entrar al bloque de PGO.
    bool GSSlam::verifyLoop(std::vector<LoopEdge> &batch_edges)
    {
        bool did_work = false;
        LoopCandidate candidate;
        while (loop_candidates_queue_.try_dequeue(candidate))
        {
            did_work = true;
            //std::cout << "[loopDetectionAndClosureThread][verification] draining queue" << std::endl;

            if (candidate.votes < min_votes_for_loop_closure_)
            {
                std::cout << "[loopDetectionAndClosureThread][verification][reject] reason=insufficient_votes required="
                          << min_votes_for_loop_closure_ << " got=" << candidate.votes << std::endl;
                continue;
            }
            if (candidate.source_submap_idx < 0 || candidate.target_submap_idx < 0)
            {
                std::cout << "[loopDetectionAndClosureThread][verification][reject] reason=negative_submap_index" << std::endl;
                continue;
            }
            if (candidate.source_submap_idx >= static_cast<int>(submaps_.size()) ||
                candidate.target_submap_idx >= static_cast<int>(submaps_.size()))
            {
                std::cout << "[loopDetectionAndClosureThread][verification][reject] reason=submap_index_out_of_range" << std::endl;
                continue;
            }

            std::shared_ptr<Submap> source_submap = submaps_[static_cast<size_t>(candidate.source_submap_idx)];
            std::shared_ptr<Submap> target_submap = submaps_[static_cast<size_t>(candidate.target_submap_idx)];
            if (!source_submap || !target_submap)
            {
                std::cout << "[loopDetectionAndClosureThread][verification][reject] reason=null_source_or_target" << std::endl;
                continue;
            }

            // Convencion usada en el pipeline:
            // edge.relative_pose representa T_source^-1 * T_target (source -> target).
            const Pose source_global = source_submap->getGlobalPose();
            const Pose target_global = target_submap->getGlobalPose();
            const Pose relative_init_guess = composePoses(inversePoseGlobal(source_global), target_global);

            const float initial_translation_error = poseTranslationError(source_global, target_global);
            const float initial_rotation_error = std::abs(normalizeAngleDeg(poseRotationDeg(source_global, target_global)));
            if (initial_translation_error > loop_verify_max_distance_m_ ||
                initial_rotation_error > loop_verify_max_rotation_deg_)
            {
                std::cout << "[loopDetectionAndClosureThread][verification][reject] reason=initial_pose_gate"
                          << " source=" << source_submap->submap_id
                          << " target=" << target_submap->submap_id
                          << " trans_m=" << initial_translation_error
                          << " rot_deg=" << initial_rotation_error
                          << " thresholds=(trans_m=" << loop_verify_max_distance_m_
                          << ", rot_deg=" << loop_verify_max_rotation_deg_ << ")"
                          << std::endl;
                continue;
            }

            const int temporal_gap = std::abs(candidate.target_submap_idx - candidate.source_submap_idx);
            const int min_submap_difference = std::max(0, loop_min_submap_difference_);
            if (temporal_gap <= min_submap_difference)
            {
                std::cout << "[loopDetectionAndClosureThread][verification][reject] reason=temporal_filter temporal_gap="
                          << temporal_gap << " min_submap_difference=" << min_submap_difference << std::endl;
                continue;
            }

            std::cout << "[loopDetectionAndClosureThread][verification][input]"
                      << " source_submap_id=" << source_submap->submap_id
                      << " target_submap_id=" << target_submap->submap_id
                      << " source_global=" << poseToString(source_global)
                      << " target_global=" << poseToString(target_global)
                      << " init_guess=" << poseToString(relative_init_guess)
                      << " similarity=" << candidate.similarity
                      << " votes=" << candidate.votes
                      << std::endl;

            LoopEdge edge;
            if (registerSubmaps(candidate, edge))
            {
                batch_edges.push_back(std::move(edge));
                std::cout << "[loopDetectionAndClosureThread][verification] edge added to batch batch_size="
                          << batch_edges.size() << std::endl;
            }
        }
        return did_work;
    }

    // ================= REGISTRO GEOMETRICO Y CONSTRUCCION DE EDGE =================
    // Este bloque invoca el registro Open3D y, si pasa los filtros,
    // produce una arista con pose relativa + confianza + matriz de informacion.
    bool GSSlam::registerSubmaps(const LoopCandidate &candidate, LoopEdge &edge_out)
    {
        std::cout << "[loopDetectionAndClosureThread][verification] candidate src=" << candidate.source_submap_idx
                  << " tgt=" << candidate.target_submap_idx
                  << " votes=" << candidate.votes << std::endl;

        const auto verification_start = std::chrono::steady_clock::now();

        std::shared_ptr<Submap> source_submap = submaps_[static_cast<size_t>(candidate.source_submap_idx)];
        std::shared_ptr<Submap> target_submap = submaps_[static_cast<size_t>(candidate.target_submap_idx)];

        const Pose source_global = source_submap->getGlobalPose();
        const Pose target_global = target_submap->getGlobalPose();
        const Pose relative_init_guess = composePoses(inversePoseGlobal(source_global), target_global);

        std::cout << "[loopDetectionAndClosureThread][verification][input]"
                  << " source_submap_id=" << source_submap->submap_id
                  << " target_submap_id=" << target_submap->submap_id
                  << " source_global=" << poseToString(source_global)
                  << " target_global=" << poseToString(target_global)
                  << " init_guess=" << poseToString(relative_init_guess)
                  << " similarity=" << candidate.similarity
                  << " votes=" << candidate.votes
                  << std::endl;

        float mean_residual = 0.0f;

        LoopEdge edge;
        if (!loop_closure_ || !loop_closure_->registerSubmapsOpen3D(
                source_submap.get(),
                target_submap.get(),
                candidate.similarity,
                relative_init_guess,
                edge,
                mean_residual))
        {
            std::cout << "[loopDetectionAndClosureThread][verification][reject] reason=registration_failed" << std::endl;
            return false;
        }

        const float confidence_threshold = loop_closure_->getConfiguration().loop_confidence_threshold;
        if (edge.confidence < confidence_threshold)
        {
            std::cout << "[loopDetectionAndClosureThread][verification][reject] reason=low_confidence confidence="
                      << edge.confidence << " threshold=" << confidence_threshold
                      << " inliers=" << edge.num_inliers << " mean_residual=" << mean_residual << std::endl;
            return false;
        }

        // Filtro adicional: consistencia de la pose registrada contra el estado actual.
        // Se proyecta target_global_pred = source_global * T_source_to_target y se compara
        // contra target_global vigente para rechazar saltos geometricamente inverosimiles.
        const Pose predicted_target_global = composePoses(source_global, edge.relative_pose);
        const float target_translation_error = poseTranslationError(predicted_target_global, target_global);
        const float target_rotation_error = poseRotationDeg(predicted_target_global, target_global);
        const float mean_res = mean_residual;
        const int inliers = edge.num_inliers;
        // thresholds (hay que hacer configurable)
        const float max_translation_thresh = 2.0f; // meters
        const float max_rotation_thresh_deg = 20.0f; // degrees
        const float max_mean_residual = 2.0f; // ICP RMS
        const int min_inliers = 30;
        if (target_translation_error > max_translation_thresh || target_rotation_error > max_rotation_thresh_deg ||
            mean_res > max_mean_residual || inliers < min_inliers)
        {
            std::cout << "[loopDetectionAndClosureThread][verification][reject] reason=post_registration_gate"
                      << " source=" << source_submap->submap_id
                      << " target=" << target_submap->submap_id
                      << " targ_trans_m=" << target_translation_error
                      << " targ_rot_deg=" << target_rotation_error
                      << " mean_res=" << mean_res
                      << " inliers=" << inliers
                      << " source_global=" << poseToString(source_global)
                      << " target_global=" << poseToString(target_global)
                      << " init_guess=" << poseToString(relative_init_guess)
                      << " returned_edge_pose=" << poseToString(edge.relative_pose)
                      << " predicted_target_global=" << poseToString(predicted_target_global)
                      << " thresholds=(trans_m=" << max_translation_thresh << ", rot_deg=" << max_rotation_thresh_deg
                      << ", mean_res=" << max_mean_residual << ", min_inliers=" << min_inliers << ")"
                      << std::endl;
            return false;
        }

        edge_out = edge;

        std::cout << "[loopDetectionAndClosureThread][verification] accepted edge source_submap_id="
                  << edge_out.source_submap_id << " target_submap_id=" << edge_out.target_submap_id
                  << " confidence=" << edge_out.confidence << " similarity=" << candidate.similarity
                  << " inliers=" << edge_out.num_inliers
                  << " relative_pose=" << poseToString(edge_out.relative_pose)
                  << " source_global_after=" << poseToString(source_global)
                  << " target_global_after=" << poseToString(target_global)
                  << std::endl;

        const auto verification_end = std::chrono::steady_clock::now();
        const float verification_ms = std::chrono::duration<float, std::milli>(verification_end - verification_start).count();
        std::cout << "[loopDetectionAndClosureThread][verification][timing]"
                  << " source_submap_id=" << edge_out.source_submap_id
                  << " target_submap_id=" << edge_out.target_submap_id
                  << " ms=" << verification_ms
                  << std::endl;

        return true;
    }

    // ================= OPTIMIZACION DEL GRAFO (PGO) =================
    // Toma las aristas validadas del batch y ejecuta:
    // - PGO completo (Ceres/Open3D) si esta habilitado apply_pgo_updates,
    // - o una actualizacion simplificada basada en la mejor arista.
    bool GSSlam::poseGraphOptimization(std::vector<LoopEdge> &batch_edges)
    {
        if (batch_edges.empty())
        {
            return false;
        }

        std::cout << "[loopDetectionAndClosureThread][pgo] batch_size=" << batch_edges.size()
                  << " starting update" << std::endl;

        for (size_t i = 0; i < batch_edges.size(); ++i)
        {
            const LoopEdge &edge = batch_edges[i];
            // Diagnostico de robustez numerica de la informacion de cada arista.
            const bool info_finite = edge.information_matrix.allFinite();
            const float info_diag_min = std::min({
                edge.information_matrix(0, 0),
                edge.information_matrix(1, 1),
                edge.information_matrix(2, 2),
                edge.information_matrix(3, 3),
                edge.information_matrix(4, 4),
                edge.information_matrix(5, 5)});
            const float info_diag_max = std::max({
                edge.information_matrix(0, 0),
                edge.information_matrix(1, 1),
                edge.information_matrix(2, 2),
                edge.information_matrix(3, 3),
                edge.information_matrix(4, 4),
                edge.information_matrix(5, 5)});

            std::cout << "[loopDetectionAndClosureThread][pgo][edge]"
                      << " idx=" << i
                      << " source_submap_id=" << edge.source_submap_id
                      << " target_submap_id=" << edge.target_submap_id
                      << " confidence=" << edge.confidence
                      << " similarity=" << edge.visual_similarity
                      << " num_inliers=" << edge.num_inliers
                      << " source_uncertainty=" << edge.source_uncertainty
                      << " target_uncertainty=" << edge.target_uncertainty
                      << " info_diag_min=" << info_diag_min
                      << " info_diag_max=" << info_diag_max
                      << " info_finite=" << (info_finite ? 1 : 0)
                      << " relative_pose=" << poseToString(edge.relative_pose)
                      << std::endl;

            if (!info_finite)
            {
                std::cout << "[loopDetectionAndClosureThread][pgo][warn] edge_information_non_finite idx=" << i
                          << " source_submap_id=" << edge.source_submap_id
                          << " target_submap_id=" << edge.target_submap_id << std::endl;
            }
        }

        int min_source = std::numeric_limits<int>::max();
        int max_source = std::numeric_limits<int>::min();
        int min_target = std::numeric_limits<int>::max();
        int max_target = std::numeric_limits<int>::min();
        const LoopEdge *best_edge = nullptr;
        for (const auto &edge : batch_edges)
        {
            min_source = std::min(min_source, edge.source_submap_id);
            max_source = std::max(max_source, edge.source_submap_id);
            min_target = std::min(min_target, edge.target_submap_id);
            max_target = std::max(max_target, edge.target_submap_id);
            if (!best_edge || edge.confidence > best_edge->confidence)
            {
                best_edge = &edge;
            }
        }
        std::cout << "[loopDetectionAndClosureThread][pgo][span]"
                  << " source_range=[" << min_source << ", " << max_source << "]"
                  << " target_range=[" << min_target << ", " << max_target << "]"
                  << std::endl;

        if (!loop_closure_)
        {
            std::cout << "[loopDetectionAndClosureThread][pgo][reject] loop_closure is null" << std::endl;
            batch_edges.clear();
            return true;
        }

        const auto pgo_start = std::chrono::steady_clock::now();
        const bool apply_pgo_updates = loop_closure_->getConfiguration().apply_pgo_updates;

        if (apply_pgo_updates)
        {
            const Pose last_pose_before = submaps_.empty() ? Pose::Identity() : submaps_.back()->getGlobalPose();
            const auto pgo_result = loop_closure_->optimizePoseGraph(submaps_, batch_edges, 20);
            const auto pgo_end = std::chrono::steady_clock::now();
            const float pgo_ms = std::chrono::duration<float, std::milli>(pgo_end - pgo_start).count();

            loop_last_pgo_ms_.store(pgo_ms, std::memory_order_release);
            loop_last_edges_.store(static_cast<int>(batch_edges.size()), std::memory_order_release);

            if (pgo_result.converged && pgo_result.corrected_submap_poses.size() == submaps_.size())
            {
                const bool all_poses_finite = std::all_of(
                    pgo_result.corrected_submap_poses.begin(),
                    pgo_result.corrected_submap_poses.end(),
                    [](const Pose &pose)
                    {
                        return std::isfinite(pose.position.x) && std::isfinite(pose.position.y) && std::isfinite(pose.position.z) &&
                               std::isfinite(pose.orientation.x) && std::isfinite(pose.orientation.y) && std::isfinite(pose.orientation.z) &&
                               std::isfinite(pose.orientation.w);
                    });

                std::cout << "[loopDetectionAndClosureThread][pgo][result]"
                          << " converged=1"
                          << " corrected_poses=" << pgo_result.corrected_submap_poses.size()
                          << " residual_error=" << pgo_result.residual_error
                          << " all_poses_finite=" << (all_poses_finite ? 1 : 0)
                          << std::endl;

                // Safety check: validate PGO deltas against configurable thresholds
                const auto &pgo_conf = loop_closure_->getConfiguration();
                const float max_allowed_trans = pgo_conf.pgo_max_translation_apply_m;
                const float max_allowed_rot = pgo_conf.pgo_max_rotation_apply_deg;
                float max_dpos = 0.0f;
                float max_drot = 0.0f;
                for (size_t ii = 0; ii < pgo_result.corrected_submap_poses.size(); ++ii)
                {
                    if (!submaps_[ii]) continue;
                    const Pose before = submaps_[ii]->getGlobalPose();
                    const Pose after = pgo_result.corrected_submap_poses[ii];
                    const float dpos = poseTranslationError(before, after);
                    const float drot = poseRotationDeg(before, after);
                    max_dpos = std::max(max_dpos, dpos);
                    max_drot = std::max(max_drot, drot);
                }

                if (max_dpos > max_allowed_trans || max_drot > max_allowed_rot)
                {
                    std::cout << "[loopDetectionAndClosureThread][pgo][reject_apply] PGO update exceeds safety thresholds"
                              << " max_dpos=" << max_dpos << " allowed_trans=" << max_allowed_trans
                              << " max_drot=" << max_drot << " allowed_rot=" << max_allowed_rot
                              << " batch_size=" << batch_edges.size()
                              << std::endl;
                    // Do not apply the PGO correction to avoid catastrophic jumps
                }
                else
                {
                    //std::lock_guard<std::mutex> lock(submap_pose_mutex_);
                    std::lock_guard<std::mutex> lock(optimization_mutex_);
                    updateSubmapChainRelativePoses(pgo_result.corrected_submap_poses);
                }

                if (!pgo_result.corrected_submap_poses.empty())
                {
                    const Pose &last_pose_after = pgo_result.corrected_submap_poses.back();
                    const float dx = last_pose_after.position.x - last_pose_before.position.x;
                    const float dy = last_pose_after.position.y - last_pose_before.position.y;
                    const float dz = last_pose_after.position.z - last_pose_before.position.z;
                    const float dpos = std::sqrt(dx * dx + dy * dy + dz * dz);

                    const float qdot_abs = std::abs(
                        last_pose_after.orientation.x * last_pose_before.orientation.x +
                        last_pose_after.orientation.y * last_pose_before.orientation.y +
                        last_pose_after.orientation.z * last_pose_before.orientation.z +
                        last_pose_after.orientation.w * last_pose_before.orientation.w);
                    const float qdot = std::clamp(qdot_abs, 0.0f, 1.0f);
                    constexpr float kRadToDeg = 57.29577951308232f;
                    const float drot_deg = 2.0f * std::acos(qdot) * kRadToDeg;

                    std::cout << "[loopDetectionAndClosureThread][pgo] applied dpos_m=" << dpos
                              << " drot_deg=" << drot_deg << std::endl;
                }

                std::cout << "[loopDetectionAndClosureThread][pgo][timing]"
                          << " batch_size=" << batch_edges.size()
                          << " solve_ms=" << pgo_ms
                          << " residual_error=" << pgo_result.residual_error
                          << " converged=1"
                          << " applied_updates=1"
                          << std::endl;
            }
            else
            {
                std::cout << "[loopDetectionAndClosureThread][pgo][reject] converged=" << (pgo_result.converged ? 1 : 0)
                          << " corrected_poses=" << pgo_result.corrected_submap_poses.size()
                          << " expected_submaps=" << submaps_.size() << std::endl;
                std::cout << "[loopDetectionAndClosureThread][pgo][timing]"
                          << " batch_size=" << batch_edges.size()
                          << " solve_ms=" << pgo_ms
                          << " residual_error=" << pgo_result.residual_error
                          << " converged=0"
                          << " applied_updates=0"
                          << std::endl;
            }
        }
        else
        {
            // Ruta simplificada (sin solver global): aplica solo la mejor arista.
            // OJO: aqui la convencion de inversion debe mantenerse consistente
            // con edge.relative_pose (source -> target) para evitar drift por signo.
            if (!best_edge)
            {
                std::cout << "[loopDetectionAndClosureThread][pgo][reject] reason=no_best_edge" << std::endl;
                batch_edges.clear();
                return true;
            }

            const int source_idx = best_edge->source_submap_id;
            const int target_idx = best_edge->target_submap_id;

            if (source_idx < 0 || target_idx < 0 ||
                source_idx >= static_cast<int>(submaps_.size()) ||
                target_idx >= static_cast<int>(submaps_.size()) ||
                !submaps_[static_cast<size_t>(source_idx)] ||
                !submaps_[static_cast<size_t>(target_idx)])
            {
                std::cout << "[loopDetectionAndClosureThread][pgo][reject] reason=edge_indices_out_of_range"
                          << " source_idx=" << source_idx
                          << " target_idx=" << target_idx
                          << std::endl;
                batch_edges.clear();
                return true;
            }

            std::vector<Pose> corrected_global_poses;
            corrected_global_poses.reserve(submaps_.size());
            for (const auto &submap : submaps_)
            {
                corrected_global_poses.push_back(submap ? submap->getGlobalPose() : Pose::Identity());
            }

            const Pose source_global_before = corrected_global_poses[static_cast<size_t>(source_idx)];
            const Pose target_global_before = corrected_global_poses[static_cast<size_t>(target_idx)];
            const Pose target_global_after = composePoses(
                source_global_before,
                invertPose(best_edge->relative_pose));

            submaps_[static_cast<size_t>(target_idx)]->T_global_cached = target_global_after;
            submaps_[static_cast<size_t>(target_idx)]->pose_cache_valid = true;
            if (target_idx == 0)
            {
                submaps_[static_cast<size_t>(target_idx)]->T_relative = Pose::Identity();
            }
            else
            {
                const Pose prev_global = submaps_[static_cast<size_t>(target_idx - 1)]->getGlobalPose();
                submaps_[static_cast<size_t>(target_idx)]->T_relative = composePoses(invertPose(prev_global), target_global_after);
            }

            corrected_global_poses[static_cast<size_t>(target_idx)] = target_global_after;
            for (size_t i = static_cast<size_t>(target_idx + 1); i < submaps_.size(); ++i)
            {
                if (!submaps_[i])
                {
                    continue;
                }
                corrected_global_poses[i] = composePoses(corrected_global_poses[i - 1], submaps_[i]->getRelativePose());
            }

            const Pose target_delta = composePoses(invertPose(target_global_before), target_global_after);

            {
                std::lock_guard<std::mutex> lock(optimization_mutex_);
                updateSubmapChainGlobalPosesFromIndex(static_cast<size_t>(target_idx));
            }

            const auto pgo_end = std::chrono::steady_clock::now();
            const float pgo_ms = std::chrono::duration<float, std::milli>(pgo_end - pgo_start).count();
            loop_last_pgo_ms_.store(pgo_ms, std::memory_order_release);
            loop_last_edges_.store(static_cast<int>(batch_edges.size()), std::memory_order_release);

            std::cout << "[loopDetectionAndClosureThread][pgo][apply_icp]"
                      << " source_submap_id=" << best_edge->source_submap_id
                      << " target_submap_id=" << best_edge->target_submap_id
                      << " confidence=" << best_edge->confidence
                      << " relative_pose=" << poseToString(best_edge->relative_pose)
                      << " source_global_before=" << poseToString(source_global_before)
                      << " target_global_before=" << poseToString(target_global_before)
                      << " target_global_after=" << poseToString(target_global_after)
                      << " target_delta=" << poseToString(target_delta)
                      << std::endl;

            std::cout << "[loopDetectionAndClosureThread][pgo][timing]"
                      << " batch_size=" << batch_edges.size()
                      << " update_ms=" << pgo_ms
                      << " residual_error=0"
                      << " converged=1"
                      << " applied_updates=1"
                      << std::endl;
        }

        batch_edges.clear();
        return true;
    }

} // namespace f_vigs_slam
