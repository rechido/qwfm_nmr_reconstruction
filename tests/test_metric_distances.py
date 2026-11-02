import math
import unittest

import torch

from codes.metrics import histograms_to_quantiles, pairwise_distances


class GroundDistanceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.edges = torch.tensor([0.0, 1.0, 2.0], dtype=torch.float64)
        self.left_bin = torch.tensor([[1.0, 0.0]], dtype=torch.float64)
        self.right_bin = torch.tensor([[0.0, 1.0]], dtype=torch.float64)

    def test_exact_two_bin_hand_case(self) -> None:
        l2 = pairwise_distances(self.left_bin, self.right_bin, metric="l2")
        w1 = pairwise_distances(
            self.left_bin,
            self.right_bin,
            metric="w1",
            log_rate_edges=self.edges,
        )
        w2 = pairwise_distances(
            self.left_bin,
            self.right_bin,
            metric="w2",
            log_rate_edges=self.edges,
            num_quantiles=17,
        )
        self.assertAlmostEqual(l2.item(), math.sqrt(2.0), places=12)
        self.assertAlmostEqual(w1.item(), 1.0, places=12)
        self.assertAlmostEqual(w2.item(), 1.0, places=12)

    def test_inverse_cdf_midpoints(self) -> None:
        quantiles = histograms_to_quantiles(
            self.left_bin,
            self.edges,
            num_quantiles=4,
            row_chunk_size=1,
        )
        expected = torch.tensor([[0.125, 0.375, 0.625, 0.875]], dtype=torch.float64)
        torch.testing.assert_close(quantiles, expected, rtol=0.0, atol=1e-14)

    def test_w1_integrates_a_cdf_zero_crossing_exactly(self) -> None:
        edges = torch.tensor([0.0, 1.0, 2.0, 3.0], dtype=torch.float64)
        first = torch.tensor([[0.6, 0.0, 0.4]], dtype=torch.float64)
        second = torch.tensor([[0.2, 0.7, 0.1]], dtype=torch.float64)
        distance = pairwise_distances(first, second, metric="w1", log_rate_edges=edges)
        expected = 0.2 + 0.5 * (0.4**2 + 0.3**2) / (0.4 + 0.3) + 0.15
        self.assertAlmostEqual(distance.item(), expected, places=13)

    def test_chunking_does_not_change_distances(self) -> None:
        generator = torch.Generator().manual_seed(21)
        left = torch.rand((7, 6), generator=generator, dtype=torch.float64)
        right = torch.rand((5, 6), generator=generator, dtype=torch.float64)
        left /= left.sum(dim=-1, keepdim=True)
        right /= right.sum(dim=-1, keepdim=True)
        edges = torch.linspace(-2.0, 3.0, 7, dtype=torch.float64)
        for metric in ("l2", "w1", "w2"):
            kwargs = {
                "metric": metric,
                "log_rate_edges": edges,
                "num_quantiles": 31,
            }
            full = pairwise_distances(left, right, chunk_size=None, **kwargs)
            chunked = pairwise_distances(left, right, chunk_size=2, **kwargs)
            torch.testing.assert_close(chunked, full, rtol=0.0, atol=1e-13)

    def test_grouped_distance_is_condition_isolated(self) -> None:
        group_left = torch.stack((self.left_bin, self.right_bin), dim=0)
        group_right = torch.stack((self.right_bin, self.right_bin), dim=0)
        distances = pairwise_distances(
            group_left,
            group_right,
            metric="w1",
            log_rate_edges=self.edges,
            chunk_size=1,
        )
        expected = torch.tensor([[[1.0]], [[0.0]]], dtype=torch.float64)
        torch.testing.assert_close(distances, expected, rtol=0.0, atol=1e-14)

    def test_aliases_resolve_to_same_metric(self) -> None:
        short = pairwise_distances(
            self.left_bin, self.right_bin, metric="w1", log_rate_edges=self.edges
        )
        explicit = pairwise_distances(
            self.left_bin,
            self.right_bin,
            metric="log_w1",
            log_rate_edges=self.edges,
        )
        torch.testing.assert_close(short, explicit)

    def test_invalid_histogram_is_rejected(self) -> None:
        invalid = torch.tensor([[0.2, 0.2]], dtype=torch.float64)
        with self.assertRaisesRegex(ValueError, "unit mass"):
            pairwise_distances(invalid, self.right_bin, metric="l2")


if __name__ == "__main__":
    unittest.main()
