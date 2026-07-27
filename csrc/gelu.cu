#include <ATen/ATen.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>

namespace {
constexpr int threads = 256;

__global__ void gelu_new_kernel(const float* x, float* out, int64_t count) {
    const int64_t index = static_cast<int64_t>(blockIdx.x) * threads + threadIdx.x;
    if (index >= count) return;
    const float value = x[index];
    const float cubic = value * value * value;
    const float inner = 0.7978845608028654f * (value + 0.044715f * cubic);
    out[index] = 0.5f * value * (1.0f + tanhf(inner));
}
} // namespace

at::Tensor gelu_new_cuda(const at::Tensor& x) {
    const c10::cuda::CUDAGuard guard(x.device());
    auto out = at::empty_like(x);
    const auto stream = c10::cuda::getCurrentCUDAStream(x.get_device());
    const auto blocks = static_cast<unsigned>((x.numel() + threads - 1) / threads);
    gelu_new_kernel<<<blocks, threads, 0, stream.stream()>>>(
        x.data_ptr<float>(), out.data_ptr<float>(), x.numel());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}
