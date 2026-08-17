#include <ATen/ATen.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>
#include <math_constants.h>

namespace {
constexpr int threads = 256;
constexpr int tile_keys = 32;
constexpr int tile_queries = 8;
constexpr int max_head_dim = 128;

__global__ void scores_kernel(const float* q, const float* k, float* scores,
                              int sequence, int head_dim, int64_t count) {
    const int64_t index = static_cast<int64_t>(blockIdx.x) * threads + threadIdx.x;
    if (index >= count) return;
    const int key = index % sequence;
    const int query = (index / sequence) % sequence;
    if (key > query) {
        scores[index] = -CUDART_INF_F;
        return;
    }
    const int64_t pair = index / (static_cast<int64_t>(sequence) * sequence);
    const int64_t q_offset = (pair * sequence + query) * head_dim;
    const int64_t k_offset = (pair * sequence + key) * head_dim;
    float dot = 0.0f;
    for (int dim = 0; dim < head_dim; ++dim)
        dot += q[q_offset + dim] * k[k_offset + dim];
    scores[index] = dot * rsqrtf(static_cast<float>(head_dim));
}

__global__ void softmax_kernel(float* scores, int sequence) {
    const int lane = threadIdx.x;
    const int64_t offset = static_cast<int64_t>(blockIdx.x) * sequence;
    const float value = lane < sequence ? scores[offset + lane] : -CUDART_INF_F;
    __shared__ float reduction[threads];
    reduction[lane] = value;
    __syncthreads();
    for (int stride = threads / 2; stride > 0; stride /= 2) {
        if (lane < stride) reduction[lane] = fmaxf(reduction[lane], reduction[lane + stride]);
        __syncthreads();
    }
    __shared__ float maximum;
    if (lane == 0) maximum = reduction[0];
    __syncthreads();
    const float exponential = lane < sequence ? expf(value - maximum) : 0.0f;
    reduction[lane] = exponential;
    __syncthreads();
    for (int stride = threads / 2; stride > 0; stride /= 2) {
        if (lane < stride) reduction[lane] += reduction[lane + stride];
        __syncthreads();
    }
    if (lane < sequence) scores[offset + lane] = exponential / reduction[0];
}

__global__ void values_kernel(const float* scores, const float* v, float* out,
                              int sequence, int head_dim, int64_t count) {
    const int64_t index = static_cast<int64_t>(blockIdx.x) * threads + threadIdx.x;
    if (index >= count) return;
    const int dim = index % head_dim;
    const int query = (index / head_dim) % sequence;
    const int64_t pair = index / (static_cast<int64_t>(sequence) * head_dim);
    const int64_t scores_offset = (pair * sequence + query) * sequence;
    const int64_t v_offset = pair * sequence * head_dim + dim;
    float result = 0.0f;
    for (int key = 0; key <= query; ++key)
        result += scores[scores_offset + key] * v[v_offset + static_cast<int64_t>(key) * head_dim];
    out[index] = result;
}

// One query per block, streaming over 32-key tiles. The running numerator and
// denominator are rescaled whenever the maximum increases; no score matrix exists.
__global__ void tiled_kernel(const float* q, const float* k, const float* v, float* out,
                             int sequence, int head_dim) {
    const int dim = threadIdx.x;
    const int query = blockIdx.x % sequence;
    const int64_t pair = blockIdx.x / sequence;
    const int64_t q_offset = (pair * sequence + query) * head_dim;
    __shared__ float scores[tile_keys];
    __shared__ float weights[tile_keys];
    __shared__ float maximum, denominator, rescale;
    if (dim == 0) {
        maximum = -CUDART_INF_F;
        denominator = 0.0f;
    }
    __syncthreads();

    float numerator = 0.0f;
    for (int start = 0; start <= query; start += tile_keys) {
        const int valid = min(tile_keys, query - start + 1);
        if (dim < valid) {
            const int64_t k_offset = (pair * sequence + start + dim) * head_dim;
            float dot = 0.0f;
            for (int d = 0; d < head_dim; ++d)
                dot += q[q_offset + d] * k[k_offset + d];
            scores[dim] = dot * rsqrtf(static_cast<float>(head_dim));
        }
        __syncthreads();

        if (dim == 0) {
            float tile_maximum = -CUDART_INF_F;
            for (int j = 0; j < valid; ++j)
                tile_maximum = fmaxf(tile_maximum, scores[j]);
            const float next_maximum = fmaxf(maximum, tile_maximum);
            rescale = maximum == -CUDART_INF_F ? 0.0f : expf(maximum - next_maximum);
            float tile_denominator = 0.0f;
            for (int j = 0; j < valid; ++j) {
                weights[j] = expf(scores[j] - next_maximum);
                tile_denominator += weights[j];
            }
            denominator = denominator * rescale + tile_denominator;
            maximum = next_maximum;
        }
        __syncthreads();

        if (dim < head_dim) {
            numerator *= rescale;
            for (int j = 0; j < valid; ++j)
                numerator += weights[j] * v[(pair * sequence + start + j) * head_dim + dim];
        }
        __syncthreads();
    }
    if (dim < head_dim)
        out[q_offset + dim] = numerator / denominator;
}

// Eight warps handle eight queries while sharing each coalesced K/V tile.
template<int tile_head_dim>
__global__ void query_tiled_kernel(const float* q, const float* k, const float* v, float* out,
                                   int sequence, int input_head_dim) {
    const int head_dim = tile_head_dim == 64 ? 64 : input_head_dim;
    const int lane = threadIdx.x;
    const int row = threadIdx.y;
    const int thread = row * tile_keys + lane;
    const int query_blocks = (sequence + tile_queries - 1) / tile_queries;
    const int query_base = (blockIdx.x % query_blocks) * tile_queries;
    const int query = query_base + row;
    const int max_query = min(query_base + tile_queries - 1, sequence - 1);
    const int64_t pair = blockIdx.x / query_blocks;
    const int64_t pair_offset = pair * sequence * head_dim;

    __shared__ float q_tile[tile_queries * tile_head_dim];
    __shared__ float k_tile[tile_keys * tile_head_dim];
    __shared__ float v_tile[tile_keys * tile_head_dim];
    __shared__ float weights[tile_queries * tile_keys];
    for (int index = thread; index < tile_queries * head_dim; index += tile_queries * tile_keys) {
        const int q_row = index / head_dim;
        if (query_base + q_row < sequence)
            q_tile[index] = q[pair_offset + (query_base + q_row) * head_dim + index % head_dim];
    }
    __syncthreads();

    float maximum = -CUDART_INF_F;
    float denominator = 0.0f;
    float numerator[tile_head_dim / tile_keys] = {0.0f};
    for (int start = 0; start <= max_query; start += tile_keys) {
        const int key_count = min(tile_keys, sequence - start);
        for (int index = thread; index < key_count * head_dim; index += tile_queries * tile_keys) {
            const int key = index / head_dim;
            const int dim = index % head_dim;
            k_tile[dim * tile_keys + key] = k[pair_offset + (start + key) * head_dim + dim];
            v_tile[index] = v[pair_offset + (start + key) * head_dim + dim];
        }
        __syncthreads();

        const int valid = query < sequence && query >= start
            ? min(key_count, query - start + 1) : 0;
        // Every lane owns one key score. All 32 lanes participate in reductions,
        // including masked keys; valid is uniform within this query's warp.
        if (valid > 0) {
            float score = -CUDART_INF_F;
            if (lane < valid) {
                float dot = 0.0f;
                for (int dim = 0; dim < head_dim; ++dim)
                    dot += q_tile[row * head_dim + dim] * k_tile[dim * tile_keys + lane];
                score = dot * rsqrtf(static_cast<float>(head_dim));
            }
            float tile_maximum = score;
            for (int offset = tile_keys / 2; offset > 0; offset /= 2)
                tile_maximum = fmaxf(tile_maximum, __shfl_xor_sync(0xffffffff, tile_maximum, offset));
            const float next_maximum = fmaxf(maximum, tile_maximum);
            const float rescale = maximum == -CUDART_INF_F ? 0.0f : expf(maximum - next_maximum);
            const float weight = lane < valid ? expf(score - next_maximum) : 0.0f;
            weights[row * tile_keys + lane] = weight;
            float tile_denominator = weight;
            for (int offset = tile_keys / 2; offset > 0; offset /= 2)
                tile_denominator += __shfl_xor_sync(0xffffffff, tile_denominator, offset);
            denominator = denominator * rescale + tile_denominator;
            maximum = next_maximum;
            // Other lanes read this warp's weights below. No warp reads another
            // query's weights, so only warp synchronization is needed here.
            __syncwarp();
            for (int part = 0; part < tile_head_dim / tile_keys; ++part) {
                const int dim = lane + part * tile_keys;
                if (dim < head_dim) {
                    numerator[part] *= rescale;
                    for (int key = 0; key < valid; ++key)
                        numerator[part] += weights[row * tile_keys + key] * v_tile[key * head_dim + dim];
                }
            }
        }
        __syncthreads();
    }
    if (query < sequence) {
        for (int part = 0; part < tile_head_dim / tile_keys; ++part) {
            const int dim = lane + part * tile_keys;
            if (dim < head_dim)
                out[pair_offset + query * head_dim + dim] = numerator[part] / denominator;
        }
    }
}

// One new query attends to every cached key, including the current token.
__global__ void decode_kernel(const float* q, const float* k, const float* v, float* out,
                              int keys, int head_dim) {
    const int dim = threadIdx.x;
    const int64_t pair = blockIdx.x;
    const int64_t kv_offset = pair * keys * head_dim;
    const int64_t q_offset = pair * head_dim;
    __shared__ float scores[tile_keys], weights[tile_keys];
    __shared__ float maximum, denominator, rescale;
    if (dim == 0) {
        maximum = -CUDART_INF_F;
        denominator = 0.0f;
    }
    __syncthreads();

    float numerator = 0.0f;
    for (int start = 0; start < keys; start += tile_keys) {
        const int valid = min(tile_keys, keys - start);
        if (dim < valid) {
            float dot = 0.0f;
            const int64_t k_offset = kv_offset + (start + dim) * head_dim;
            for (int d = 0; d < head_dim; ++d)
                dot += q[q_offset + d] * k[k_offset + d];
            scores[dim] = dot * rsqrtf(static_cast<float>(head_dim));
        }
        __syncthreads();
        if (dim == 0) {
            float tile_maximum = -CUDART_INF_F;
            for (int j = 0; j < valid; ++j)
                tile_maximum = fmaxf(tile_maximum, scores[j]);
            const float next_maximum = fmaxf(maximum, tile_maximum);
            rescale = maximum == -CUDART_INF_F ? 0.0f : expf(maximum - next_maximum);
            float tile_denominator = 0.0f;
            for (int j = 0; j < valid; ++j) {
                weights[j] = expf(scores[j] - next_maximum);
                tile_denominator += weights[j];
            }
            denominator = denominator * rescale + tile_denominator;
            maximum = next_maximum;
        }
        __syncthreads();
        if (dim < head_dim) {
            numerator *= rescale;
            for (int j = 0; j < valid; ++j)
                numerator += weights[j] * v[kv_offset + (start + j) * head_dim + dim];
        }
        __syncthreads();
    }
    if (dim < head_dim)
        out[q_offset + dim] = numerator / denominator;
}
} // namespace

at::Tensor attention_cuda(const at::Tensor& q, const at::Tensor& k, const at::Tensor& v) {
    const c10::cuda::CUDAGuard guard(q.device());
    const int sequence = static_cast<int>(q.size(2));
    const int head_dim = static_cast<int>(q.size(3));
    auto scores = at::empty({q.size(0), q.size(1), sequence, sequence}, q.options());
    auto out = at::empty_like(q);
    const auto stream = c10::cuda::getCurrentCUDAStream(q.get_device());
    const int64_t score_count = scores.numel();
    const int64_t output_count = out.numel();
    scores_kernel<<<(score_count + threads - 1) / threads, threads, 0, stream.stream()>>>(
        q.data_ptr<float>(), k.data_ptr<float>(), scores.data_ptr<float>(), sequence, head_dim, score_count);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    softmax_kernel<<<q.numel() / head_dim, threads, 0, stream.stream()>>>(scores.data_ptr<float>(), sequence);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    values_kernel<<<(output_count + threads - 1) / threads, threads, 0, stream.stream()>>>(
        scores.data_ptr<float>(), v.data_ptr<float>(), out.data_ptr<float>(), sequence, head_dim, output_count);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}

at::Tensor attention_tiled_cuda(const at::Tensor& q, const at::Tensor& k, const at::Tensor& v) {
    const c10::cuda::CUDAGuard guard(q.device());
    const int sequence = static_cast<int>(q.size(2));
    const int head_dim = static_cast<int>(q.size(3));
    auto out = at::empty_like(q);
    const auto stream = c10::cuda::getCurrentCUDAStream(q.get_device());
    tiled_kernel<<<q.numel() / head_dim, 128, 0, stream.stream()>>>(
        q.data_ptr<float>(), k.data_ptr<float>(), v.data_ptr<float>(), out.data_ptr<float>(), sequence, head_dim);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}

at::Tensor attention_query_tiled_cuda(const at::Tensor& q, const at::Tensor& k, const at::Tensor& v) {
    const c10::cuda::CUDAGuard guard(q.device());
    const int sequence = static_cast<int>(q.size(2));
    const int head_dim = static_cast<int>(q.size(3));
    auto out = at::empty_like(q);
    const auto stream = c10::cuda::getCurrentCUDAStream(q.get_device());
    const int query_blocks = (sequence + tile_queries - 1) / tile_queries;
    if (head_dim == 64)
        query_tiled_kernel<64><<<q.size(0) * q.size(1) * query_blocks, dim3(tile_keys, tile_queries), 0, stream.stream()>>>(
            q.data_ptr<float>(), k.data_ptr<float>(), v.data_ptr<float>(), out.data_ptr<float>(), sequence, head_dim);
    else
        query_tiled_kernel<128><<<q.size(0) * q.size(1) * query_blocks, dim3(tile_keys, tile_queries), 0, stream.stream()>>>(
            q.data_ptr<float>(), k.data_ptr<float>(), v.data_ptr<float>(), out.data_ptr<float>(), sequence, head_dim);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}

at::Tensor attention_decode_cuda(const at::Tensor& q, const at::Tensor& k, const at::Tensor& v) {
    const c10::cuda::CUDAGuard guard(q.device());
    auto out = at::empty_like(q);
    const auto stream = c10::cuda::getCurrentCUDAStream(q.get_device());
    decode_kernel<<<q.size(0) * q.size(1), 128, 0, stream.stream()>>>(
        q.data_ptr<float>(), k.data_ptr<float>(), v.data_ptr<float>(), out.data_ptr<float>(),
        static_cast<int>(k.size(2)), static_cast<int>(q.size(3)));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}
