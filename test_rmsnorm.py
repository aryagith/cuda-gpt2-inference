"""python test_rmsnorm.py [--cuda]; CUDA mode must build and run, never silently skips."""

import argparse
import unittest

import torch
import torch.nn.functional as F

from rmsnorm import cuda_rmsnorm, extension, reference

RUN_CUDA = False


class ReferenceTests(unittest.TestCase):
    def test_known_values(self):
        x = torch.tensor([[3.0, 4.0], [0.0, 0.0]])
        expected = x / (12.5 + 1e-5) ** 0.5
        torch.testing.assert_close(reference(x, torch.ones(2)), expected)

    def test_against_pytorch(self):
        generator = torch.Generator().manual_seed(42)
        for width in (1, 31, 33, 255, 256, 257, 1024, 4096):
            with self.subTest(width=width):
                x = torch.randn(7, width, generator=generator)
                w = torch.randn(width, generator=generator)
                torch.testing.assert_close(reference(x, w), F.rms_norm(x, (width,), w, eps=1e-5))

    def test_invalid_inputs(self):
        x, w = torch.ones(2, 4), torch.ones(4)
        cases = [(x.double(), w, 1e-5), (x, w[:2], 1e-5),
                 (x[:, ::2], w[:2], 1e-5), (x[:0], w, 1e-5),
                 (x, w, 0.0), (x, w, float("nan")), (x, w, float("inf")),
                 (x, w, 1e-100), (x.clone().requires_grad_(), w, 1e-5)]
        for args in cases:
            with self.subTest(shape=args[0].shape, eps=args[2]):
                with self.assertRaises(ValueError):
                    reference(*args)
        with self.assertRaises(ValueError):
            cuda_rmsnorm(x, w)


class CudaTests(unittest.TestCase):
    def setUp(self):
        if not RUN_CUDA:
            self.skipTest("use --cuda to compile and test the CUDA extension")

    def test_against_pytorch(self):
        torch.manual_seed(42)
        for rows, width in ((1, 1), (3, 31), (7, 257), (32, 1024), (7, 4096)):
            for scale in (0.0, 1e-4, 1.0, 100.0):
                with self.subTest(rows=rows, width=width, scale=scale):
                    x = torch.randn(rows, width, device="cuda") * scale
                    w = torch.randn(width, device="cuda")
                    torch.testing.assert_close(cuda_rmsnorm(x, w), F.rms_norm(x, (width,), w, eps=1e-5),
                                               rtol=2e-5, atol=3e-6)

    def test_current_stream(self):
        extension()  # Compile before checking asynchronous stream behavior.
        stream = torch.cuda.Stream()
        with torch.cuda.stream(stream):
            x = torch.randn(65, 513, device="cuda")
            w = torch.randn(513, device="cuda")
            actual = cuda_rmsnorm(x, w)
            expected = F.rms_norm(x, (513,), w, eps=1e-5)
        stream.synchronize()
        torch.testing.assert_close(actual, expected, rtol=2e-5, atol=3e-6)

    def test_small_values_and_epsilon(self):
        x = torch.tensor([[0.0, 1e-6, -2e-6], [3e-6, -4e-6, 5e-6]], device="cuda")
        w = torch.tensor([0.5, -1.0, 2.0], device="cuda")
        for eps in (1e-10, 1e-5, 1e-2):
            with self.subTest(eps=eps):
                torch.testing.assert_close(cuda_rmsnorm(x, w, eps),
                                           F.rms_norm(x, (3,), w, eps=eps), rtol=2e-5, atol=3e-6)

    def test_binding_validates_inputs(self):
        module = extension()
        x, w = torch.ones(3, 8, device="cuda"), torch.ones(8, device="cuda")
        cases = ((x.double(), w, 1e-5), (x, w.double(), 1e-5),
                 (torch.ones(3, 16, device="cuda")[:, ::2], w, 1e-5),
                 (x, torch.ones(16, device="cuda")[::2], 1e-5),
                 (x.clone().requires_grad_(), w, 1e-5), (x, w, 0.0),
                 (x, w, float("nan")), (x, w, 1e-100),
                 (x.cpu(), w.cpu(), 1e-5))
        for args in cases:
            with self.subTest(dtype=args[0].dtype, stride=args[0].stride(), eps=args[2]):
                with self.assertRaises(RuntimeError):
                    module.rmsnorm(*args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cuda", action="store_true")
    RUN_CUDA = parser.parse_args().cuda
    unittest.main(argv=["test_rmsnorm.py"], verbosity=2)
