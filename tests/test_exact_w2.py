import math
import unittest

import torch

from codes.metrics.distances import histograms_to_quantiles, pairwise_distances, pairwise_w2_exact


class ExactW2Tests(unittest.TestCase):
    def test_translated_uniforms_return_unsquared_distance(self) -> None:
        edges = torch.arange(5, dtype=torch.float64)
        left = torch.tensor([[0.5, 0.5, 0.0, 0.0]], dtype=torch.float64)
        right = torch.tensor([[0.0, 0.0, 0.5, 0.5]], dtype=torch.float64)
        distance = pairwise_w2_exact(left, right, edges)
        self.assertEqual(distance.item(), 2.0)
        shifted = pairwise_w2_exact(left, right, edges + 17.0)
        torch.testing.assert_close(shifted, distance, rtol=0.0, atol=1e-14)

    def test_distinct_uniforms_have_analytic_mean_and_width_distance(self) -> None:
        # U(0,2) against U(1,7): W2^2 = (1-4)^2 + (2-6)^2 / 12.
        edges = torch.tensor([0.0, 1.0, 2.0, 4.0, 7.0], dtype=torch.float64)
        left = torch.tensor([[0.5, 0.5, 0.0, 0.0]], dtype=torch.float64)
        right = torch.tensor([[0.0, 1 / 6, 2 / 6, 3 / 6]], dtype=torch.float64)
        actual = pairwise_w2_exact(left, right, edges)
        self.assertAlmostEqual(actual.item(), math.sqrt(9 + 16 / 12), places=13)

    def test_zero_mass_bins_and_quantile_jumps(self) -> None:
        edges = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0, 3.0], dtype=torch.float64)
        # Leading, internal, and trailing empty bins; the gap contributes a
        # quantile jump at probability 1/2, which has zero integration measure.
        left = torch.tensor([[0.0, 0.5, 0.0, 0.5, 0.0]], dtype=torch.float64)
        right = torch.tensor([[0.0, 0.0, 1.0, 0.0, 0.0]], dtype=torch.float64)
        actual = pairwise_w2_exact(left, right, edges)
        self.assertAlmostEqual(actual.item(), math.sqrt(7 / 12), places=14)

    def test_sparse_identity_and_symmetry_are_finite(self) -> None:
        generator = torch.Generator().manual_seed(311)
        mass = torch.rand((9, 13), generator=generator, dtype=torch.float64)
        mass[mass < 0.65] = 0.0
        mass[:, 6] += 0.1
        mass /= mass.sum(-1, keepdim=True)
        edges = torch.linspace(-5.0, 6.0, 14, dtype=torch.float64)
        actual = pairwise_w2_exact(mass, mass, edges, chunk_size=3)
        self.assertTrue(bool(torch.isfinite(actual).all()))
        torch.testing.assert_close(actual.diagonal(), torch.zeros(9, dtype=torch.float64), rtol=0.0, atol=0.0)
        torch.testing.assert_close(actual, actual.T, rtol=0.0, atol=0.0)

    def test_chunking_grouping_and_float64_normalization(self) -> None:
        generator = torch.Generator().manual_seed(23)
        left = torch.rand((2, 7, 8), generator=generator, dtype=torch.float32)
        right = torch.rand((2, 5, 8), generator=generator, dtype=torch.float32)
        left /= left.sum(-1, keepdim=True)
        right /= right.sum(-1, keepdim=True)
        edges = torch.linspace(-2.0, 4.0, 9, dtype=torch.float32)
        full = pairwise_w2_exact(left, right, edges, chunk_size=None)
        chunked = pairwise_w2_exact(left, right, edges, chunk_size=2)
        self.assertEqual(full.dtype, torch.float64)
        self.assertEqual(full.shape, (2, 7, 5))
        self.assertEqual(full.device, left.device)
        torch.testing.assert_close(chunked, full, rtol=0.0, atol=0.0)
        for group in range(2):
            separate = pairwise_w2_exact(left[group], right[group], edges)
            torch.testing.assert_close(separate, full[group], rtol=0.0, atol=0.0)
        normalized_left = left.double() / left.double().sum(-1, keepdim=True)
        normalized_right = right.double() / right.double().sum(-1, keepdim=True)
        normalized = pairwise_w2_exact(normalized_left, normalized_right, edges)
        torch.testing.assert_close(normalized, full, rtol=1e-13, atol=1e-13)

    def test_matches_high_resolution_inverse_cdf_quadrature(self) -> None:
        generator = torch.Generator().manual_seed(101)
        left = torch.rand((3, 7), generator=generator, dtype=torch.float64) + 0.2
        right = torch.rand((2, 7), generator=generator, dtype=torch.float64) + 0.2
        left /= left.sum(-1, keepdim=True)
        right /= right.sum(-1, keepdim=True)
        edges = torch.tensor([-4.0, -3.9, -1.7, 0.2, 1.1, 3.0, 3.4, 6.2], dtype=torch.float64)
        left_q = histograms_to_quantiles(left, edges, num_quantiles=65536)
        right_q = histograms_to_quantiles(right, edges, num_quantiles=65536)
        quadrature = (left_q[:, None] - right_q[None]).square().mean(-1).sqrt()
        actual = pairwise_w2_exact(left, right, edges, chunk_size=2)
        torch.testing.assert_close(actual, quadrature, rtol=0.0, atol=1e-8)

    def test_tiny_probability_intervals_and_rounded_terminal_cdfs(self) -> None:
        edges = torch.arange(9, dtype=torch.float64)
        epsilon = torch.finfo(torch.float64).eps
        masses = torch.tensor(
            [
                [0.0, 0.7, 0.0, 0.3 - epsilon, epsilon, 0.0, 0.0, 0.0],
                [epsilon / 4, 0.0, 0.2, epsilon / 4, 0.8, 0.0, 0.0, 0.0],
                [0.0, 0.1, 0.1, 0.1, 0.1, 0.1, 0.5, 0.0],
                [0.0, 1.0, 1e-300, 0.0, 0.0, 0.0, 0.0, 0.0],
            ],
            dtype=torch.float64,
        )
        actual = pairwise_w2_exact(masses, masses, edges, chunk_size=2)
        self.assertTrue(bool(torch.isfinite(actual).all()))
        torch.testing.assert_close(actual.diagonal(), torch.zeros(4, dtype=torch.float64), rtol=0.0, atol=0.0)
        torch.testing.assert_close(actual, actual.T, rtol=0.0, atol=0.0)

    def test_existing_w2_retains_midpoint_approximation(self) -> None:
        edges = torch.tensor([0.0, 1.0, 3.0], dtype=torch.float64)
        left = torch.tensor([[1.0, 0.0]], dtype=torch.float64)
        right = torch.tensor([[0.0, 1.0]], dtype=torch.float64)
        old = pairwise_distances(left, right, metric="w2", log_rate_edges=edges, num_quantiles=1)
        self.assertEqual(old.item(), 1.5)
        exact = pairwise_w2_exact(left, right, edges)
        self.assertAlmostEqual(exact.item(), math.sqrt(7 / 3), places=14)
        self.assertNotEqual(old.item(), exact.item())

    def test_input_validation(self) -> None:
        edges = torch.tensor([0.0, 1.0, 2.0], dtype=torch.float64)
        good = torch.tensor([[0.5, 0.5]], dtype=torch.float64)
        for bad, message in (
            (torch.tensor([[0.0, 0.0]]), "positive mass"),
            (torch.tensor([[-0.5, 1.5]]), "negative"),
            (torch.tensor([[0.2, 0.2]]), "unit mass"),
            (torch.tensor([[float("nan"), 0.5]]), "non-finite"),
        ):
            with self.subTest(message=message), self.assertRaisesRegex(ValueError, message):
                pairwise_w2_exact(bad, good, edges)
        with self.assertRaisesRegex(ValueError, "same number of dimensions"):
            pairwise_w2_exact(good[None], good, edges)
        with self.assertRaisesRegex(ValueError, "same group count"):
            pairwise_w2_exact(good[None].expand(2, -1, -1), good[None], edges)
        with self.assertRaisesRegex(ValueError, "exactly N"):
            pairwise_w2_exact(good, good, torch.arange(4, dtype=torch.float64))
        with self.assertRaisesRegex(ValueError, "strictly increasing"):
            pairwise_w2_exact(good, good, torch.tensor([0.0, 1.0, 1.0]))
        for invalid_chunk in (0, -1, True, 1.5):
            with self.subTest(chunk=invalid_chunk), self.assertRaisesRegex(ValueError, "chunk_size"):
                pairwise_w2_exact(good, good, edges, chunk_size=invalid_chunk)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is unavailable")
    def test_cuda_and_device_validation(self) -> None:
        masses = torch.tensor([[0.0, 1.0], [1.0, 0.0]], dtype=torch.float64)
        edges = torch.tensor([0.0, 1.0, 3.0], dtype=torch.float64)
        cpu = pairwise_w2_exact(masses, masses, edges)
        gpu = pairwise_w2_exact(masses.cuda(), masses.cuda(), edges.cuda())
        torch.testing.assert_close(cpu, gpu.cpu(), rtol=1e-13, atol=1e-13)
        with self.assertRaisesRegex(ValueError, "same device"):
            pairwise_w2_exact(masses, masses.cuda(), edges)
        with self.assertRaisesRegex(ValueError, "same device"):
            pairwise_w2_exact(masses.cuda(), masses.cuda(), edges)


if __name__ == "__main__":
    unittest.main()
