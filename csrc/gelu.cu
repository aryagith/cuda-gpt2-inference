#include <ATen/ATen.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>
#include <type_traits>

namespace {
constexpr int threads = 256;

__device__ float round_half(float value) {
    return static_cast<float>(static_cast<at::Half>(value));
}

template <typename scalar_t>
__global__ void gelu_new_kernel(const scalar_t* x, scalar_t* out, int64_t count) {
    const int64_t index = static_cast<int64_t>(blockIdx.x) * threads + threadIdx.x;
    if (index >= count) return;
    const float value = static_cast<float>(x[index]);
    if constexpr (std::is_same_v<scalar_t, at::Half>) {
        // Match NewGELUActivation's half rounding between eager operations,
        // while fusing their launches and computing each operation in float.
        const float cubic = round_half(round_half(value * value) * value);
        const float scaled = round_half(0.044715f * cubic);
        const float inner = round_half(0.7978845608028654f * round_half(value + scaled));
        const float gate = round_half(1.0f + round_half(tanhf(inner)));
        out[index] = static_cast<scalar_t>(round_half(0.5f * value) * gate);
        return;
    }
    const float cubic = value * value * value;
    const float inner = 0.7978845608028654f * (value + 0.044715f * cubic);
    out[index] = static_cast<scalar_t>(0.5f * value * (1.0f + tanhf(inner)));
}
} // namespace

at::Tensor gelu_new_cuda(const at::Tensor& x) {
    const c10::cuda::CUDAGuard guard(x.device());
    auto out = at::empty_like(x);
    const auto stream = c10::cuda::getCurrentCUDAStream(x.get_device());
    const auto blocks = static_cast<unsigned>((x.numel() + threads - 1) / threads);
    if (x.scalar_type() == at::kHalf) {
        gelu_new_kernel<at::Half><<<blocks, threads, 0, stream.stream()>>>(
            x.data_ptr<at::Half>(), out.data_ptr<at::Half>(), x.numel());
    } else {
        gelu_new_kernel<float><<<blocks, threads, 0, stream.stream()>>>(
            x.data_ptr<float>(), out.data_ptr<float>(), x.numel());
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}
