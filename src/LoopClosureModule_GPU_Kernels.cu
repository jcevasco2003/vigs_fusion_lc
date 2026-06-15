/**
 * @file LoopClosureModule_GPU_Kernels.cu
 * @brief GPU kernels for fast loop detection via NetVLAD descriptor matching
 * 
 * Implements GPU-accelerated cosineSimilarity computation and topk selection
 * for efficient loop candidate retrieval.
 */

#include <thrust/device_vector.h>
#include <thrust/copy.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>
#include <thrust/unique.h>
#include <thrust/reduce.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cub/cub.cuh>
#include <algorithm>
#include <cmath>

namespace f_vigs_slam {

    // ============================================================
    // GPU KERNEL: Compute cosine similarities
    // ============================================================
    /**
     * @brief Kernel to compute cosine similarity between query descriptor
     * and all stored descriptors in parallel.
     * 
     * Grid: (num_stored_descriptors,) - each block handles one stored descriptor
     * Block: (BLOCK_SIZE,) - each thread accumulates partial dot product
     * 
     * @param query_desc Query descriptor (1D, size = desc_dim)
     * @param all_descs All stored descriptors (2D, shape = [num_stored, desc_dim])
     * @param similarities Output similarities (1D, size = num_stored)
     * @param desc_dim Descriptor dimension (typically 256 for NetVLAD)
     * @param num_stored Number of stored descriptors
     */
    __global__ void computeCosineSimilarities_kernel(
        const float* query_desc,           // [desc_dim]
        const float* all_descs,            // [num_stored * desc_dim]
        float* similarities,               // [num_stored]
        int desc_dim,
        int num_stored)
    {
        int stored_idx = blockIdx.x;  // One block per stored descriptor
        
        if (stored_idx >= num_stored) return;

        // Compute dot product: query · stored_descriptor
        float dot_product = 0.0f;
        float query_norm_sq = 0.0f;
        float stored_norm_sq = 0.0f;
        
        // Each thread processes multiple elements (stride loop)
        for (int i = threadIdx.x; i < desc_dim; i += blockDim.x) {
            float q = query_desc[i];
            float s = all_descs[stored_idx * desc_dim + i];
            
            dot_product += q * s;
            query_norm_sq += q * q;
            stored_norm_sq += s * s;
        }

        // Reduce within block
        __shared__ float shared_dot[256];
        __shared__ float shared_q_norm[256];
        __shared__ float shared_s_norm[256];

        if (threadIdx.x < 256) {
            shared_dot[threadIdx.x] = dot_product;
            shared_q_norm[threadIdx.x] = query_norm_sq;
            shared_s_norm[threadIdx.x] = stored_norm_sq;
        }
        __syncthreads();

        // Block-level reduction
        for (int stride = 128; stride > 0; stride >>= 1) {
            if (threadIdx.x < stride) {
                shared_dot[threadIdx.x] += shared_dot[threadIdx.x + stride];
                shared_q_norm[threadIdx.x] += shared_q_norm[threadIdx.x + stride];
                shared_s_norm[threadIdx.x] += shared_s_norm[threadIdx.x + stride];
            }
            __syncthreads();
        }

        // Write result from thread 0
        if (threadIdx.x == 0) {
            float q_norm = sqrtf(shared_q_norm[0]);
            float s_norm = sqrtf(shared_s_norm[0]);
            
            if (q_norm > 1e-6f && s_norm > 1e-6f) {
                similarities[stored_idx] = shared_dot[0] / (q_norm * s_norm);
            } else {
                similarities[stored_idx] = 0.0f;
            }
        }
    }

    // ============================================================
    // Host wrapper function for GPU cosineSimilarity
    // ============================================================
    /**
     * @brief Host function to compute cosine similarities on GPU.
     * 
     * @param query_desc_gpu Query descriptor on GPU
     * @param all_descs_gpu All stored descriptors on GPU
     * @param num_stored Number of stored descriptors
     * @return thrust::device_vector<float> Similarities [num_stored]
     */
    thrust::device_vector<float> computeCosineSimilarities_GPU(
        const thrust::device_vector<float>& query_desc_gpu,
        const thrust::device_vector<float>& all_descs_gpu,
        int num_stored,
        cudaStream_t stream)
    {
        if (query_desc_gpu.empty() || all_descs_gpu.empty() || num_stored <= 0) {
            return thrust::device_vector<float>();
        }

        int desc_dim = static_cast<int>(query_desc_gpu.size());
        thrust::device_vector<float> similarities(num_stored);

        // NetVLAD ya llega normalizado, así que la similitud coseno se reduce a D * q.
        // Usamos cuBLAS porque esta multiplicación matriz-vector ya está altamente optimizada en CUDA.
        cublasHandle_t handle = nullptr;
        if (cublasCreate(&handle) != CUBLAS_STATUS_SUCCESS) {
            return thrust::device_vector<float>();
        }

        // ============================================================
        // PHASE 6: Use CUDA stream for async execution
        // Allows parallel GPU operations without blocking
        // ============================================================
        if (stream != nullptr) {
            if (cublasSetStream(handle, stream) != CUBLAS_STATUS_SUCCESS) {
                cublasDestroy(handle);
                return thrust::device_vector<float>();
            }
        }

        const float alpha = 1.0f;
        const float beta = 0.0f;

        // `all_descs_gpu` está en layout row-major como [num_stored x desc_dim].
        // cuBLAS usa column-major, así que pedimos la traspuesta para reinterpretar
        // correctamente la memoria sin copiar ni reordenar nada.
        const cublasStatus_t status = cublasSgemv(
            handle,
            CUBLAS_OP_T,
            desc_dim,
            num_stored,
            &alpha,
            thrust::raw_pointer_cast(all_descs_gpu.data()),
            desc_dim,
            thrust::raw_pointer_cast(query_desc_gpu.data()),
            1,
            &beta,
            thrust::raw_pointer_cast(similarities.data()),
            1);

        cublasDestroy(handle);

        if (status != CUBLAS_STATUS_SUCCESS) {
            return thrust::device_vector<float>();
        }

        // Only synchronize if stream is synchronous (NULL)
        if (stream == nullptr) {
            cudaDeviceSynchronize();
        }
        return similarities;
    }

    // ============================================================
    // GPU-accelerated topk selection
    // ============================================================
    /**
     * @brief Extract top-k indices and similarities using GPU sort.
     * 
     * @param similarities Device vector of similarities [num_stored]
     * @param k Number of top results
     * @param topk_indices Output indices of top-k
     * @param topk_sims Output top-k similarities
     */
    void selectTopk_GPU(
        const thrust::device_vector<float>& similarities,
        int k,
        thrust::device_vector<int>& topk_indices,
        thrust::device_vector<float>& topk_sims,
        cudaStream_t stream)
    {
        int num_stored = similarities.size();
        k = std::min(k, num_stored);

        // ============================================================
        // PHASE 6: Use CUB for efficient top-k selection (O(k log N))
        // Instead of full sort O(N log N), only extract top-k
        // ============================================================
        
        // Create (index, similarity) pairs
        thrust::device_vector<int> indices(num_stored);
        thrust::sequence(indices.begin(), indices.end(), 0);

        // Create parallel array for sorting
        thrust::device_vector<float> sims_copy = similarities;

        // Use CUB for descending radix sort (only top-k)
        void *d_temp_storage = nullptr;
        size_t temp_storage_bytes = 0;

        // Query temp storage size
        cub::DoubleBuffer<float> d_keys(thrust::raw_pointer_cast(sims_copy.data()),
                                         nullptr);
        cub::DoubleBuffer<int> d_values(thrust::raw_pointer_cast(indices.data()),
                                        nullptr);

        cub::DeviceRadixSort::SortPairsDescending(
            d_temp_storage, temp_storage_bytes,
            d_keys, d_values,
            num_stored, 0, 32, stream);

        // Allocate temp storage
        thrust::device_vector<char> temp_storage(temp_storage_bytes);
        d_temp_storage = thrust::raw_pointer_cast(temp_storage.data());

        // Run actual sort
        cub::DeviceRadixSort::SortPairsDescending(
            d_temp_storage, temp_storage_bytes,
            d_keys, d_values,
            num_stored, 0, 32, stream);

        // If no stream, synchronize
        if (stream == nullptr) {
            cudaDeviceSynchronize();
        }

        // Copy results to output (now in d_keys.Current() and d_values.Current())
        topk_sims.resize(k);
        topk_indices.resize(k);
        thrust::copy_n(d_keys.Current(), k, topk_sims.begin());
        thrust::copy_n(d_values.Current(), k, topk_indices.begin());

    }

} // namespace f_vigs_slam
