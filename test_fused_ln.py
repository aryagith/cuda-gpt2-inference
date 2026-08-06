"""python test_fused_ln.py [--cuda]"""

import argparse
import unittest

import torch

from fused_ln import cuda_fused_residual_ln, extension, reference


RUN_CUDA = False


class ReferenceTests(unittest.TestCase):
    def test_cpu_formula(self):
        attention = torch.arange(768, dtype=torch.float32).reshape(1, 1, 768) / 100
        residual = torch.ones_like(attention)
        weight, bias = torch.ones(768), torch.zeros(768)
        summed, normalized = reference(attention, residual, weight, bias, 1e-5)
        torch.testing.assert_close(summed, attention + residual)
        expected = (summed - summed.mean(dim=-1, keepdim=True)) / torch.sqrt(
            summed.var(dim=-1, keepdim=True, unbiased=False) + 1e-5)
        torch.testing.assert_close(normalized, expected, rtol=2e-5, atol=2e-6)
        with self.assertRaises(ValueError):
            cuda_fused_residual_ln(attention, residual, weight, bias, 1e-5)


class CudaTests(unittest.TestCase):
    def setUp(self):
        if not RUN_CUDA:
            self.skipTest("run with --cuda")

    def test_shapes_and_stream(self):
        generator = torch.Generator(device="cuda").manual_seed(42)
        weight = torch.randn(768, device="cuda", generator=generator)
        bias = torch.randn(768, device="cuda", generator=generator)
        stream = torch.cuda.Stream()
        for shape in ((1, 1, 768), (1, 17, 768), (2, 129, 768)):
            with self.subTest(shape=shape):
                attention = torch.randn(shape, device="cuda", generator=generator)
                residual = torch.randn(shape, device="cuda", generator=generator)
                stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(stream):
                    actual_sum, actual_norm = cuda_fused_residual_ln(
                        attention, residual, weight, bias, 1e-5)
                torch.cuda.current_stream().wait_stream(stream)
                expected_sum, expected_norm = reference(
                    attention, residual, weight, bias, 1e-5)
                torch.testing.assert_close(actual_sum, expected_sum, rtol=0, atol=0)
                torch.testing.assert_close(actual_norm, expected_norm,
                                           rtol=2e-5, atol=2e-5)

    def test_native_rejection(self):
        module = extension()
        x = torch.ones(1, 2, 768, device="cuda")
        weight = torch.ones(768, device="cuda")
        bias = torch.zeros(768, device="cuda")
        bad_calls = ((x.cpu(), x, weight, bias, 1e-5),
                     (x.double(), x, weight, bias, 1e-5),
                     (x.transpose(0, 1), x, weight, bias, 1e-5),
                     (x.clone().requires_grad_(), x, weight, bias, 1e-5),
                     (x[:, :0], x[:, :0], weight, bias, 1e-5),
                     (x, x[:, :1], weight, bias, 1e-5),
                     (x, x, weight.cpu(), bias, 1e-5),
                     (x, x, weight.clone().requires_grad_(), bias, 1e-5),
                     (x, x, weight, bias, 1e-50),
                     (x, x, weight, bias, float("nan")))
        for args in bad_calls:
            with self.assertRaises(RuntimeError):
                module.fused_residual_ln(*args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cuda", action="store_true")
    RUN_CUDA = parser.parse_args().cuda
    unittest.main(argv=["test_fused_ln.py"], verbosity=2)
