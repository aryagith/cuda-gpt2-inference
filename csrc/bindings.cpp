#include <torch/extension.h>
#include <cmath>
#include <limits>

at::Tensor rmsnorm_cuda(const at::Tensor& x, const at::Tensor& weight, float eps);

at::Tensor rmsnorm(const at::Tensor& x, const at::Tensor& weight, double eps) {
    TORCH_CHECK(x.is_cuda() && weight.is_cuda(), "expected CUDA tensors");
    TORCH_CHECK(x.device() == weight.device(), "inputs must share a device");
    TORCH_CHECK(x.scalar_type() == at::kFloat && weight.scalar_type() == at::kFloat,
                "only float32 is supported");
    TORCH_CHECK(x.dim() == 2 && weight.dim() == 1, "expected x [rows, width], weight [width]");
    TORCH_CHECK(x.size(0) > 0 && x.size(1) > 0 && weight.numel() == x.size(1),
                "invalid dimensions");
    TORCH_CHECK(x.size(0) <= std::numeric_limits<int>::max(), "too many rows");
    TORCH_CHECK(x.is_contiguous() && weight.is_contiguous(), "inputs must be contiguous");
    TORCH_CHECK(!x.requires_grad() && !weight.requires_grad(), "forward-only operator");
    TORCH_CHECK(std::isfinite(eps) && eps >= std::numeric_limits<float>::min()
                && eps <= std::numeric_limits<float>::max(), "invalid epsilon");
    return rmsnorm_cuda(x, weight, static_cast<float>(eps));
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
    module.def("rmsnorm", &rmsnorm, "Forward float32 RMSNorm");
}
