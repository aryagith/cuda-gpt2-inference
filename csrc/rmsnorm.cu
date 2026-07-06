#include <ATen/ATen.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>

namespace {
constexpr int threads = 256;

__global__ void rmsnorm_kernel(const float* x, const float* weight, float* out,
                               int64_t width, float eps) {
    const int tid = threadIdx.x;
    const int64_t offset = static_cast<int64_t>(blockIdx.x) * width;
    __shared__ float warp_sums[threads / 32];
    float sum = 0.0f;
    for (int64_t col = tid; col < width; col += threads) {
        const float value = x[offset + col];
        sum += value * value;
    }
    for (int delta = 16; delta > 0; delta /= 2)
        sum += __shfl_down_sync(0xffffffff, sum, delta);
    if (tid % 32 == 0) warp_sums[tid / 32] = sum;
    __syncthreads();
    if (tid < 32) {
        float block_sum = tid < threads / 32 ? warp_sums[tid] : 0.0f;
        for (int delta = 16; delta > 0; delta /= 2)
            block_sum += __shfl_down_sync(0xffffffff, block_sum, delta);
        if (tid == 0) warp_sums[0] = block_sum;
    }
    __syncthreads();
    // ponytail: one block per row underfills the GPU for small row counts; split rows only if profiling justifies it.
    const float inverse_rms = rsqrtf(warp_sums[0] / static_cast<float>(width) + eps);
    for (int64_t col = tid; col < width; col += threads)
        out[offset + col] = x[offset + col] * inverse_rms * weight[col];
}
} // namespace

at::Tensor rmsnorm_cuda(const at::Tensor& x, const at::Tensor& weight, float eps) {
    const c10::cuda::CUDAGuard guard(x.device());
    auto out = at::empty_like(x);
    const auto stream = c10::cuda::getCurrentCUDAStream(x.get_device());
    rmsnorm_kernel<<<static_cast<unsigned>(x.size(0)), threads, 0, stream.stream()>>>(
        x.data_ptr<float>(), weight.data_ptr<float>(), out.data_ptr<float>(), x.size(1), eps);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}
