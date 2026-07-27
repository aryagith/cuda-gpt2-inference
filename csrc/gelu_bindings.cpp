#include <torch/extension.h>

at::Tensor gelu_new_cuda(const at::Tensor& x);

at::Tensor gelu_new(const at::Tensor& x) {
    TORCH_CHECK(x.is_cuda(), "expected a CUDA tensor");
    TORCH_CHECK(x.scalar_type() == at::kFloat, "only float32 is supported");
    TORCH_CHECK(x.is_contiguous(), "input must be contiguous");
    TORCH_CHECK(!x.requires_grad(), "forward-only operator");
    TORCH_CHECK(x.numel() > 0, "input must be nonempty");
    return gelu_new_cuda(x);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
    module.def("gelu_new", &gelu_new, "Forward float32 GPT-2 GELU activation");
}
