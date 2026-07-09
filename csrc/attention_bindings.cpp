#include <torch/extension.h>

at::Tensor attention_cuda(const at::Tensor& q, const at::Tensor& k, const at::Tensor& v);
at::Tensor attention_tiled_cuda(const at::Tensor& q, const at::Tensor& k, const at::Tensor& v);
at::Tensor attention_query_tiled_cuda(const at::Tensor& q, const at::Tensor& k, const at::Tensor& v);
at::Tensor attention_decode_cuda(const at::Tensor& q, const at::Tensor& k, const at::Tensor& v);

void validate(const at::Tensor& q, const at::Tensor& k, const at::Tensor& v) {
    TORCH_CHECK(q.is_cuda() && k.is_cuda() && v.is_cuda(), "expected CUDA tensors");
    TORCH_CHECK(q.device() == k.device() && q.device() == v.device(), "Q/K/V must share a device");
    TORCH_CHECK(q.scalar_type() == at::kFloat && k.scalar_type() == at::kFloat && v.scalar_type() == at::kFloat,
                "only float32 is supported");
    TORCH_CHECK(q.dim() == 4 && q.sizes().vec() == k.sizes().vec() && q.sizes().vec() == v.sizes().vec(),
                "expected matching Q/K/V [batch, heads, sequence, head_dim]");
    TORCH_CHECK(q.size(0) > 0 && q.size(1) > 0 && q.size(2) > 0 && q.size(3) > 0,
                "dimensions must be positive");
    TORCH_CHECK(q.is_contiguous() && k.is_contiguous() && v.is_contiguous(), "Q/K/V must be contiguous");
    TORCH_CHECK(!q.requires_grad() && !k.requires_grad() && !v.requires_grad(), "forward-only operator");
    TORCH_CHECK(q.size(2) <= 256 && q.size(3) <= 128 && q.numel() <= 16 * 1024 * 1024
                && q.numel() / q.size(3) * q.size(2) <= 16 * 1024 * 1024,
                "baseline workload exceeds sequence, head dimension, or buffer limit");
}

at::Tensor attention(const at::Tensor& q, const at::Tensor& k, const at::Tensor& v) {
    validate(q, k, v);
    return attention_cuda(q, k, v);
}

at::Tensor attention_tiled(const at::Tensor& q, const at::Tensor& k, const at::Tensor& v) {
    validate(q, k, v);
    return attention_tiled_cuda(q, k, v);
}

at::Tensor attention_query_tiled(const at::Tensor& q, const at::Tensor& k, const at::Tensor& v) {
    validate(q, k, v);
    return attention_query_tiled_cuda(q, k, v);
}

at::Tensor attention_decode(const at::Tensor& q, const at::Tensor& k, const at::Tensor& v) {
    TORCH_CHECK(q.is_cuda() && k.is_cuda() && v.is_cuda(), "expected CUDA tensors");
    TORCH_CHECK(q.device() == k.device() && q.device() == v.device(), "Q/K/V must share a device");
    TORCH_CHECK(q.scalar_type() == at::kFloat && k.scalar_type() == at::kFloat && v.scalar_type() == at::kFloat,
                "only float32 is supported");
    TORCH_CHECK(q.dim() == 4 && k.dim() == 4 && v.dim() == 4 && k.sizes() == v.sizes()
                && q.size(0) == k.size(0) && q.size(1) == k.size(1) && q.size(2) == 1
                && q.size(3) == k.size(3), "expected Q [B,H,1,D] and matching K/V [B,H,T,D]");
    TORCH_CHECK(q.size(0) > 0 && q.size(1) > 0 && q.size(3) > 0 && k.size(2) > 0,
                "dimensions must be positive");
    TORCH_CHECK(q.is_contiguous() && k.is_contiguous() && v.is_contiguous(), "Q/K/V must be contiguous");
    TORCH_CHECK(!q.requires_grad() && !k.requires_grad() && !v.requires_grad(), "forward-only operator");
    TORCH_CHECK(k.size(2) <= 1024 && q.size(3) <= 128 && k.numel() <= 16 * 1024 * 1024,
                "decode workload exceeds cache, head dimension, or buffer limit");
    return attention_decode_cuda(q, k, v);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
    module.def("attention", &attention, "Forward float32 causal attention baseline");
    module.def("attention_tiled", &attention_tiled, "Forward float32 causal attention with online softmax");
    module.def("attention_query_tiled", &attention_query_tiled, "Forward float32 attention with four-query tiles");
    module.def("attention_decode", &attention_decode, "Forward float32 attention over a K/V cache");
}
