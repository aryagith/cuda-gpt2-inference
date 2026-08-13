"""python test_gelu.py [--cuda]; CUDA mode builds and checks the native kernel."""

import argparse
import math
import unittest

import torch
from transformers.activations import NewGELUActivation

from gelu_new import cuda_gelu_new, extension, reference


RUN_CUDA = False


class ReferenceTests(unittest.TestCase):
    def test_known_values_and_transformers(self):
        values = [-5.0, -1.0, 0.0, 1.0, 5.0]
        x = torch.tensor(values)
        expected = [0.5 * value * (1 + math.tanh(math.sqrt(2 / math.pi)
                    * (value + 0.044715 * value**3))) for value in values]
        torch.testing.assert_close(reference(x), torch.tensor(expected), rtol=2e-6, atol=2e-7)
        torch.testing.assert_close(reference(x), NewGELUActivation()(x))
        with self.assertRaises(ValueError):
            cuda_gelu_new(x)


class CudaTests(unittest.TestCase):
    def setUp(self):
        if not RUN_CUDA:
            self.skipTest("run with --cuda")

    def test_gpt2_shapes_and_stream(self):
        generator = torch.Generator(device="cuda").manual_seed(42)
        stream = torch.cuda.Stream()
        for shape in ((1,), (1, 1, 3072), (1, 129, 3072)):
            with self.subTest(shape=shape):
                x = torch.randn(shape, device="cuda", generator=generator) * 5
                stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(stream):
                    actual = cuda_gelu_new(x)
                torch.cuda.current_stream().wait_stream(stream)
                expected = NewGELUActivation()(x)
                torch.testing.assert_close(actual, expected, rtol=2e-5, atol=1e-5)
        x = torch.tensor([-20.0, -10.0, -1e-3, 0.0, 1e-3, 10.0, 20.0], device="cuda")
        torch.testing.assert_close(cuda_gelu_new(x), NewGELUActivation()(x), rtol=2e-5, atol=1e-5)

    def test_native_rejection(self):
        module = extension()
        x = torch.ones(2, 4, device="cuda")
        cases = (x.cpu(), x.double(), x.bfloat16(), x.int(), x.t(), x.clone().requires_grad_(), x[:0])
        for value in cases:
            with self.subTest(shape=value.shape, dtype=value.dtype):
                with self.assertRaises(RuntimeError):
                    module.gelu_new(value)

    def test_float16_oracle_and_stream(self):
        generator = torch.Generator(device="cuda").manual_seed(17)
        stream = torch.cuda.Stream()
        for shape in ((1,), (257,), (1, 1, 3072), (2, 129, 3072)):
            x = (torch.randn(shape, device="cuda", generator=generator) * 5).half()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                actual = cuda_gelu_new(x)
            torch.cuda.current_stream().wait_stream(stream)
            self.assertEqual(actual.dtype, torch.float16)
            # Native half rounds between eager operations; fusion preserves
            # those rounding points so this comparison is bit-exact.
            torch.testing.assert_close(actual, NewGELUActivation()(x), rtol=0, atol=0)
            # Half intermediates can lose low bits or cancel near negative tails.
            # About two half machine epsilons relative, plus 0.002 absolute,
            # cover this inherited rounding error.
            expected = NewGELUActivation()(x.double()).half()
            torch.testing.assert_close(actual, expected, rtol=2e-3, atol=2e-3)
        x = torch.tensor([-65504, -100, -20, -1e-3, 0, 1e-3, 20, 100, 65504],
                         dtype=torch.float16, device="cuda")
        torch.testing.assert_close(cuda_gelu_new(x), NewGELUActivation()(x.double()).half(),
                                   rtol=2e-3, atol=2e-3)
        # Every finite half bit pattern: verify rounding semantics beyond a
        # random sample, including signed zero and subnormal values.
        patterns = torch.arange(65536, device="cuda", dtype=torch.int32).to(torch.int16).view(torch.float16)
        finite = patterns[torch.isfinite(patterns)]
        torch.testing.assert_close(cuda_gelu_new(finite).view(torch.int16),
                                   NewGELUActivation()(finite).view(torch.int16), rtol=0, atol=0)
        for invalid in (x.cpu(), x[:0], x.clone().requires_grad_()):
            with self.assertRaises(RuntimeError):
                extension().gelu_new(invalid)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cuda", action="store_true")
    args = parser.parse_args()
    RUN_CUDA = args.cuda
    unittest.main(argv=["test_gelu.py"], verbosity=2)
