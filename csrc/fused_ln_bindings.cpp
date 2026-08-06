#include <torch/extension.h>
#include <c10/core/GradMode.h>

#include <cmath>
#include <limits>
#include <tuple>

std::tuple<at::Tensor, at::Tensor> fused_residual_ln_cuda(
    const at::Tensor& attention, const at::Tensor& residual,
    const at::Tensor& weight, const at::Tensor& bias, double eps);

std::tuple<at::Tensor, at::Tensor> fused_residual_ln(
    const at::Tensor& attention, const at::Tensor& residual,
    const at::Tensor& weight, const at::Tensor& bias, double eps) {
    TORCH_CHECK(attention.is_cuda() && attention.scalar_type() == at::kFloat &&
                attention.is_contiguous() && !attention.requires_grad() &&
                attention.dim() == 3 && attention.size(2) == 768 &&
                attention.numel() > 0,
                "attention must be nonempty contiguous [B,S,768] float32 CUDA without grad");
    TORCH_CHECK(residual.is_cuda() && residual.device() == attention.device() &&
                residual.scalar_type() == at::kFloat && residual.is_contiguous() &&
                !residual.requires_grad() && residual.sizes() == attention.sizes(),
                "residual must match attention");
    for (const auto& param : {weight, bias}) {
        TORCH_CHECK(param.is_cuda() && param.device() == attention.device() &&
                    param.scalar_type() == at::kFloat && param.is_contiguous() &&
                    (!param.requires_grad() || !c10::GradMode::is_enabled()) &&
                    param.dim() == 1 && param.numel() == 768,
                    "weight and bias must be contiguous [768] float32 CUDA without active autograd");
    }
    TORCH_CHECK(std::isfinite(eps) &&
                eps >= std::numeric_limits<float>::min() &&
                eps <= std::numeric_limits<float>::max(),
                "eps must be finite and representable as a positive normal float32");
    return fused_residual_ln_cuda(attention, residual, weight, bias, eps);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
    module.def("fused_residual_ln", &fused_residual_ln,
               "Forward float32 GPT-2 attention residual and LayerNorm");
}
