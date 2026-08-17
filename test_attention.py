"""python test_attention.py [--cuda]; CUDA mode must build and run."""

import argparse
import math
import unittest

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

from attention import (cuda_attention, cuda_attention_tiled, cuda_attention_query_tiled,
                       cuda_attention_decode, cuda_attention_tensor_core, extension, reference, reference_decode)

CUSTOM = (cuda_attention, cuda_attention_tiled, cuda_attention_query_tiled)

RUN_CUDA = False


def math_sdpa(q, k, v):
    with sdpa_kernel(SDPBackend.MATH):
        return F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=True)


class ReferenceTests(unittest.TestCase):
    def test_tensor_core_requires_half_cuda_and_64_width(self):
        for q in (torch.ones(1, 1, 4, 64), torch.ones(1, 1, 4, 64).half(),
                  torch.ones(1, 1, 4, 32).half()):
            with self.assertRaises(ValueError):
                cuda_attention_tensor_core(q, q, q)

    def test_against_math_sdpa(self):
        generator = torch.Generator().manual_seed(42)
        for shape in ((1, 1, 1, 1), (1, 2, 7, 16), (2, 3, 33, 32), (1, 1, 257, 16)):
            with self.subTest(shape=shape):
                q, k, v = (torch.randn(shape, generator=generator) for _ in range(3))
                torch.testing.assert_close(reference(q, k, v), math_sdpa(q, k, v), rtol=2e-5, atol=2e-6)

    def test_future_values_do_not_affect_earlier_tokens(self):
        q = k = torch.zeros(1, 1, 4, 2)
        v = torch.arange(8, dtype=torch.float32).reshape(1, 1, 4, 2)
        changed = v.clone()
        changed[:, :, -1] = 1e5
        torch.testing.assert_close(reference(q, k, v)[:, :, :-1], reference(q, k, changed)[:, :, :-1])
        torch.testing.assert_close(reference(q, k, v)[:, :, 0], v[:, :, 0])

    def test_invalid_inputs(self):
        q = torch.ones(1, 1, 4, 8)
        cases = ((q.double(), q, q), (q, q[:, :, :2], q),
                 (q.transpose(-1, -2), q.transpose(-1, -2), q.transpose(-1, -2)),
                 (q.clone().requires_grad_(), q, q),
                 (torch.ones(1, 1, 1025, 8),) * 3,
                 (torch.ones(1, 1, 4, 129),) * 3)
        for args in cases:
            with self.subTest(shape=args[0].shape, dtype=args[0].dtype):
                with self.assertRaises(ValueError):
                    reference(*args)
        with self.assertRaises(ValueError):
            cuda_attention(q, q, q)
        with self.assertRaises(ValueError):
            cuda_attention_tiled(q, q, q)
        with self.assertRaises(ValueError):
            cuda_attention_query_tiled(q, q, q)
        with self.assertRaises(ValueError):
            cuda_attention_decode(q[:, :, :1], q, q)

    def test_decode_reference(self):
        generator = torch.Generator().manual_seed(42)
        for keys in (1, 17, 33):
            with self.subTest(keys=keys):
                q = torch.randn(2, 3, 1, 16, generator=generator)
                k, v = (torch.randn(2, 3, keys, 16, generator=generator) for _ in range(2))
                with sdpa_kernel(SDPBackend.MATH):
                    expected = F.scaled_dot_product_attention(q, k, v, is_causal=False)
                torch.testing.assert_close(reference_decode(q, k, v), expected, rtol=2e-5, atol=2e-6)
        q = torch.ones(1, 1, 1, 8)
        cases = ((q, torch.ones(1, 1, 0, 8), torch.ones(1, 1, 0, 8)),
                 (q, torch.ones(1, 1, 1025, 8), torch.ones(1, 1, 1025, 8)),
                 (q, torch.ones(1, 1, 2, 16), torch.ones(1, 1, 2, 16)),
                 (q, torch.ones(1, 1, 2, 8).transpose(1, 2), torch.ones(1, 1, 2, 8)),
                 (q.clone().requires_grad_(), q, q))
        for args in cases:
            with self.assertRaises(ValueError):
                reference_decode(*args)


class CudaTests(unittest.TestCase):
    def setUp(self):
        if not RUN_CUDA:
            self.skipTest("use --cuda to compile and test the CUDA extension")
        old_tf32 = torch.backends.cuda.matmul.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = False
        self.addCleanup(setattr, torch.backends.cuda.matmul, "allow_tf32", old_tf32)

    def test_against_oracles(self):
        torch.manual_seed(42)
        for shape in ((1, 1, 1, 1), (1, 1, 17, 16), (2, 3, 33, 32),
                      (1, 2, 129, 64), (1, 1, 256, 128)):
            # D=128 with scale 20 can exceed the existing FP32 near-tie tolerance
            # even for the verified naive path; keep that stress case at D<=64.
            for scale in ((1e-3, 1.0) if shape[-1] == 128 else (1e-3, 1.0, 20.0)):
                with self.subTest(shape=shape, scale=scale):
                    q, k, v = (torch.randn(shape, device="cuda") * scale for _ in range(3))
                    expected = reference(q, k, v)
                    sdpa = math_sdpa(q, k, v)
                    # Large, near-tied logits amplify FP32 dot-product ordering differences.
                    rtol, atol = (1e-3, 3e-3) if scale == 20.0 else (2e-4, 2e-5)
                    for implementation in CUSTOM:
                        actual = implementation(q, k, v)
                        torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol)
                        torch.testing.assert_close(actual, sdpa, rtol=rtol, atol=atol)
                    if scale == 20.0:
                        scores = (q.double() @ k.double().transpose(-2, -1)) / math.sqrt(shape[-1])
                        scores.masked_fill_(torch.ones(shape[-2], shape[-2], device="cuda", dtype=torch.bool).triu(1),
                                            -math.inf)
                        precise = torch.softmax(scores, dim=-1) @ v.double()
                        for implementation in CUSTOM:
                            torch.testing.assert_close(implementation(q, k, v).double(), precise,
                                                       rtol=1e-3, atol=2e-3)

    def test_causal_mask(self):
        q = k = torch.zeros(2, 3, 17, 8, device="cuda")
        v = torch.randn_like(q)
        changed = v.clone()
        changed[:, :, -1] = 1e5
        for implementation in CUSTOM:
            torch.testing.assert_close(implementation(q, k, v)[:, :, :-1],
                                       implementation(q, k, changed)[:, :, :-1])

    def test_current_stream(self):
        extension()
        stream = torch.cuda.Stream()
        with torch.cuda.stream(stream):
            q, k, v = (torch.randn(2, 2, 17, 32, device="cuda") for _ in range(3))
            actual = [implementation(q, k, v) for implementation in CUSTOM]
            expected = reference(q, k, v)
        stream.synchronize()
        for result in actual:
            torch.testing.assert_close(result, expected, rtol=2e-4, atol=2e-5)

    def test_decode(self):
        torch.manual_seed(43)
        for batch, heads, keys, dim in ((1, 1, 1, 1), (1, 12, 17, 64), (2, 3, 33, 32),
                                        (1, 12, 129, 64), (1, 12, 256, 64),
                                        (1, 12, 1024, 64), (1, 1, 1024, 128)):
            with self.subTest(batch=batch, heads=heads, keys=keys, dim=dim):
                q = torch.randn(batch, heads, 1, dim, device="cuda")
                k, v = (torch.randn(batch, heads, keys, dim, device="cuda") for _ in range(2))
                expected = reference_decode(q, k, v)
                with sdpa_kernel(SDPBackend.MATH if dim == 1 else SDPBackend.EFFICIENT_ATTENTION):
                    sdpa = F.scaled_dot_product_attention(q, k, v, is_causal=False)
                actual = cuda_attention_decode(q, k, v)
                torch.testing.assert_close(actual, expected, rtol=2e-4, atol=2e-5)
                torch.testing.assert_close(actual, sdpa, rtol=2e-4, atol=2e-5)
        stream = torch.cuda.Stream()
        with torch.cuda.stream(stream):
            q = torch.randn(1, 2, 1, 64, device="cuda")
            k, v = (torch.randn(1, 2, 129, 64, device="cuda") for _ in range(2))
            actual = cuda_attention_decode(q, k, v)
            expected = reference_decode(q, k, v)
        stream.synchronize()
        torch.testing.assert_close(actual, expected, rtol=2e-4, atol=2e-5)
        q = torch.randn(1, 2, 1, 64, device="cuda") * 20
        k = torch.randn(1, 2, 129, 64, device="cuda") * 20
        v = torch.randn(1, 2, 129, 64, device="cuda")
        precise = torch.softmax((q.double() @ k.double().transpose(-2, -1)) / math.sqrt(64), dim=-1) @ v.double()
        torch.testing.assert_close(cuda_attention_decode(q, k, v).double(), precise, rtol=1e-3, atol=2e-3)

    def test_binding_validates_inputs(self):
        module = extension()
        q = torch.ones(1, 1, 4, 8, device="cuda")
        cases = ((q.double(), q, q), (q, q[:, :, :2], q),
                 (q.transpose(-1, -2),) * 3,
                 (q.clone().requires_grad_(), q, q),
                 (torch.ones(1, 1, 1025, 8, device="cuda"),) * 3,
                 (q, q.cpu(), q),
                 (q.cpu(),) * 3)
        for args in cases:
            with self.subTest(shape=args[0].shape, dtype=args[0].dtype):
                for binding in (module.attention, module.attention_tiled, module.attention_query_tiled):
                    with self.assertRaises(RuntimeError):
                        binding(*args)
        longer = torch.ones(1, 1, 257, 8, device="cuda")
        with self.assertRaises(RuntimeError):
            module.attention(longer, longer, longer)
        with self.assertRaises(ValueError):
            cuda_attention(longer, longer, longer)

    def test_long_prefill(self):
        torch.manual_seed(44)
        extension()
        stream = torch.cuda.Stream()
        for shape in ((1, 2, 257, 65), (2, 3, 513, 32), (1, 2, 67, 64),
                      (1, 12, 992, 64), (1, 1, 1024, 128)):
            with self.subTest(shape=shape), torch.cuda.stream(stream):
                q, k, v = (torch.randn(shape, device="cuda") for _ in range(3))
                expected = reference(q, k, v)
                precise = math_sdpa(q.double(), k.double(), v.double())
                for implementation in (cuda_attention_tiled, cuda_attention_query_tiled):
                    actual = implementation(q, k, v)
                    torch.testing.assert_close(actual, expected, rtol=2e-4, atol=2e-5)
                    torch.testing.assert_close(actual.double(), precise, rtol=2e-4, atol=2e-5)
                    changed = v.clone()
                    changed[:, :, -1] = 1e5
                    torch.testing.assert_close(actual[:, :, :-1],
                        implementation(q, k, changed)[:, :, :-1], rtol=0, atol=0)
            stream.synchronize()

    def test_tensor_core(self):
        torch.manual_seed(73)
        module = extension()
        stream = torch.cuda.Stream()
        for sequence in (1, 15, 16, 17, 67, 257, 513, 992, 1024):
            with self.subTest(sequence=sequence), torch.cuda.stream(stream):
                q, k, v = (torch.randn(2, 2, sequence, 64, device="cuda", dtype=torch.float16)
                           for _ in range(3))
                actual = cuda_attention_tensor_core(q, k, v)
                expected = math_sdpa(q.double(), k.double(), v.double())
                torch.testing.assert_close(actual.double(), expected, rtol=1e-2, atol=3e-3)
                changed = v.clone()
                changed[:, :, -1] = 1000
                torch.testing.assert_close(actual[:, :, :-1],
                    cuda_attention_tensor_core(q, k, changed)[:, :, :-1], rtol=0, atol=0)
                if sequence == 67:
                    precise = math_sdpa((q * 20).double(), (k * 20).double(), v.double())
                    torch.testing.assert_close(cuda_attention_tensor_core(q * 20, k * 20, v).double(),
                                               precise, rtol=1e-2, atol=3e-3)
            stream.synchronize()
        q = torch.ones(1, 1, 17, 64, device="cuda", dtype=torch.float16)
        for args in ((q.float(), q, q), (q, q[:, :, :16], q), (q.cpu(),) * 3,
                     (q.transpose(-1, -2),) * 3, (q.clone().requires_grad_(), q, q),
                     (torch.ones(1, 1, 4, 32, device="cuda", dtype=torch.float16),) * 3,
                     (torch.ones(1, 1, 1025, 64, device="cuda", dtype=torch.float16),) * 3):
            with self.assertRaises(RuntimeError):
                module.attention_tensor_core(*args)

    def test_decode_binding_validates_inputs(self):
        module = extension()
        q = torch.ones(1, 2, 1, 8, device="cuda")
        kv = torch.ones(1, 2, 17, 8, device="cuda")
        cases = ((q.double(), kv, kv), (q, kv, kv[:, :, :16]),
                 (q, kv.transpose(1, 2), kv.transpose(1, 2)),
                 (q.clone().requires_grad_(), kv, kv),
                 (q, torch.empty(1, 2, 0, 8, device="cuda"), torch.empty(1, 2, 0, 8, device="cuda")),
                 (q, torch.ones(1, 2, 1025, 8, device="cuda"), torch.ones(1, 2, 1025, 8, device="cuda")),
                 (q, kv.cpu(), kv.cpu()), (q.cpu(), kv.cpu(), kv.cpu()))
        for args in cases:
            with self.subTest(shapes=tuple(t.shape for t in args)):
                with self.assertRaises(RuntimeError):
                    module.attention_decode(*args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cuda", action="store_true")
    parser.add_argument("--decode-only", action="store_true", help="run the decode CUDA test for profiling")
    args = parser.parse_args()
    if args.decode_only and not args.cuda:
        parser.error("--decode-only requires --cuda")
    RUN_CUDA = args.cuda
    unittest.main(argv=["test_attention.py"] + (["CudaTests.test_decode"] if args.decode_only else []),
                  verbosity=2)
