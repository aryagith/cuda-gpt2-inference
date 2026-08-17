#include <ATen/ATen.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <math_constants.h>

namespace {
// One warp owns 16 queries. WMMA performs QK and PV; online softmax stays FP32.
// ponytail: synchronous 16-key tiles; pipeline larger tiles if profiling warrants it.
__global__ void tensor_core_kernel(const half* q, const half* k, const half* v,
                                   half* out, int sequence) {
    using namespace nvcuda;
    const int lane = threadIdx.x;
    const int blocks = (sequence + 15) / 16;
    const int base = (blockIdx.x % blocks) * 16;
    const int64_t offset = static_cast<int64_t>(blockIdx.x / blocks) * sequence * 64;
    __shared__ __align__(32) half qs[16 * 64], ks[16 * 64], vs[16 * 64], p[16 * 16];
    __shared__ __align__(32) float scores[16 * 16], numerator[16 * 64];
    float maximum = -CUDART_INF_F, denominator = 0.0f;
    for (int i = lane; i < 16 * 64; i += 32) {
        qs[i] = base + i / 64 < sequence ? q[offset + base * 64 + i] : __float2half(0);
        numerator[i] = 0.0f;
    }
    __syncthreads();
    for (int start = 0; start <= min(base + 15, sequence - 1); start += 16) {
        for (int i = lane; i < 16 * 64; i += 32) {
            const bool valid = start + i / 64 < sequence;
            ks[i] = valid ? k[offset + start * 64 + i] : __float2half(0);
            vs[i] = valid ? v[offset + start * 64 + i] : __float2half(0);
        }
        __syncthreads();
        wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a;
        wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> b;
        wmma::fragment<wmma::accumulator, 16, 16, 16, float> c;
        wmma::fill_fragment(c, 0.0f);
        for (int d = 0; d < 64; d += 16) {
            wmma::load_matrix_sync(a, qs + d, 64);
            wmma::load_matrix_sync(b, ks + d, 64); // K rows are columns of K transpose.
            wmma::mma_sync(c, a, b, c);
        }
        wmma::store_matrix_sync(scores, c, 16, wmma::mem_row_major);
        __syncthreads();
        if (lane < 16) {
            const int query = base + lane;
            const int valid = query < sequence ? max(0, min(16, query - start + 1)) : 0;
            float next = maximum;
            for (int j = 0; j < valid; ++j) next = fmaxf(next, scores[lane * 16 + j] * 0.125f);
            const float rescale = maximum == -CUDART_INF_F ? 0.0f : expf(maximum - next);
            float sum = 0.0f;
            for (int j = 0; j < 16; ++j) {
                const float weight = j < valid ? expf(scores[lane * 16 + j] * 0.125f - next) : 0.0f;
                p[lane * 16 + j] = __float2half(weight);
                sum += weight;
            }
            denominator = denominator * rescale + sum;
            maximum = next;
            for (int d = 0; d < 64; ++d) numerator[lane * 64 + d] *= rescale;
        }
        __syncthreads();
        wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> values;
        wmma::load_matrix_sync(a, p, 16);
        for (int d = 0; d < 64; d += 16) {
            wmma::load_matrix_sync(values, vs + d, 64);
            wmma::load_matrix_sync(c, numerator + d, 64, wmma::mem_row_major);
            wmma::mma_sync(c, a, values, c);
            wmma::store_matrix_sync(numerator + d, c, 64, wmma::mem_row_major);
        }
        __syncthreads();
    }
    if (lane < 16 && base + lane < sequence)
        for (int d = 0; d < 64; ++d)
            out[offset + (base + lane) * 64 + d] = __float2half(numerator[lane * 64 + d] / denominator);
}
}

at::Tensor attention_tensor_core_cuda(const at::Tensor& q, const at::Tensor& k, const at::Tensor& v) {
    const c10::cuda::CUDAGuard guard(q.device());
    int major = 0;
    C10_CUDA_CHECK(cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, q.get_device()));
    TORCH_CHECK(major >= 7, "Tensor Core attention requires compute capability 7.0 or newer");
    auto out = at::empty_like(q);
    const int sequence = static_cast<int>(q.size(2));
    const auto stream = c10::cuda::getCurrentCUDAStream(q.get_device());
    tensor_core_kernel<<<q.size(0) * q.size(1) * ((sequence + 15) / 16), 32, 0, stream.stream()>>>(
        reinterpret_cast<const half*>(q.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(k.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(v.data_ptr<at::Half>()),
        reinterpret_cast<half*>(out.data_ptr<at::Half>()), sequence);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}
