#include <ATen/ATen.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>

#include <tuple>

namespace {
constexpr int width = 768;
constexpr int threads = 256;

__device__ float warp_sum(float value) {
    for (int offset = 16; offset > 0; offset /= 2) {
        value += __shfl_down_sync(0xffffffff, value, offset);
    }
    return value;
}

__device__ float block_sum(float value, float* scratch) {
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    value = warp_sum(value);
    if (lane == 0) scratch[warp] = value;
    __syncthreads();
    if (warp == 0) {
        value = warp_sum(lane < 8 ? scratch[lane] : 0.0f);
        if (lane == 0) scratch[8] = value;
    }
    __syncthreads();
    return scratch[8];
}

__global__ void fused_residual_ln_kernel(
    const float* attention, const float* residual,
    const float* weight, const float* bias, float* summed, float* normalized,
    float eps) {
    __shared__ float mean_scratch[9];
    __shared__ float var_scratch[9];
    const int row = blockIdx.x;
    const int lane = threadIdx.x;
    const int base = row * width + lane;
    const float x0 = attention[base] + residual[base];
    const float x1 = attention[base + threads] + residual[base + threads];
    const float x2 = attention[base + 2 * threads] + residual[base + 2 * threads];
    summed[base] = x0;
    summed[base + threads] = x1;
    summed[base + 2 * threads] = x2;

    const float mean = block_sum(x0 + x1 + x2, mean_scratch) / width;
    const float d0 = x0 - mean;
    const float d1 = x1 - mean;
    const float d2 = x2 - mean;
    const float variance = block_sum(d0 * d0 + d1 * d1 + d2 * d2, var_scratch) / width;
    const float scale = rsqrtf(variance + eps);
    normalized[base] = (d0 * scale) * weight[lane] + bias[lane];
    normalized[base + threads] =
        (d1 * scale) * weight[lane + threads] + bias[lane + threads];
    normalized[base + 2 * threads] =
        (d2 * scale) * weight[lane + 2 * threads] + bias[lane + 2 * threads];
}
}  // namespace

std::tuple<at::Tensor, at::Tensor> fused_residual_ln_cuda(
    const at::Tensor& attention, const at::Tensor& residual,
    const at::Tensor& weight, const at::Tensor& bias, double eps) {
    const c10::cuda::CUDAGuard guard(attention.device());
    auto summed = at::empty_like(attention);
    auto normalized = at::empty_like(attention);
    const auto stream = c10::cuda::getCurrentCUDAStream(attention.get_device());
    const auto rows = static_cast<unsigned>(attention.numel() / width);
    fused_residual_ln_kernel<<<rows, threads, 0, stream.stream()>>>(
        attention.data_ptr<float>(), residual.data_ptr<float>(),
        weight.data_ptr<float>(), bias.data_ptr<float>(),
        summed.data_ptr<float>(), normalized.data_ptr<float>(),
        static_cast<float>(eps));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {summed, normalized};
}
