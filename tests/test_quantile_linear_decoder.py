"""Decoder ablation contracts: valid mass, atoms, tails, and unchanged sampling."""

from contextlib import nullcontext
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

import torch

CODES_DIR = Path(__file__).resolve().parents[1] / "codes"
if str(CODES_DIR) not in sys.path:
    sys.path.insert(0, str(CODES_DIR))

from samplers.quantile_wasserstein_sampler import QuantileWassersteinSampler
from utils.nmr_utils import (
    decode_quantiles,
    histogram_to_quantiles,
    quantile_wasserstein_representation_signature,
    quantiles_to_histograms,
    quantiles_to_histograms_linear_cdf,
    qwfm_decoder_signature,
)
from utils.predictive_model_utils import PredictiveModel


class _ZeroVelocity:
    def __init__(self):
        self.calls = 0

    def __call__(self, inputs, time):
        self.calls += 1
        return SimpleNamespace(sample=torch.zeros_like(inputs[:, :1]))


class LinearCDFDecoderTests(unittest.TestCase):
    def test_uniform_distribution_is_exact_on_nonuniform_bins(self):
        edges = torch.tensor([0., .03, .2, .45, .78, 1.], dtype=torch.float64)
        q = ((torch.arange(17, dtype=torch.float64) + .5) / 17)[None]
        actual = decode_quantiles(q, edges, "linear_cdf")
        torch.testing.assert_close(actual[0], edges.diff(), atol=2e-15, rtol=0)
        self.assertFalse(torch.allclose(actual * 17, (actual * 17).round()))

    def test_endpoint_knots_use_bounded_half_step_extrapolation(self):
        q = torch.tensor([[.2, .4]], dtype=torch.float64)
        edges = torch.tensor([0., .15, .3, .5, .8, 1.], dtype=torch.float64)
        # The extended knots .1 and .5 produce Uniform(.1, .5), not tails
        # stretching to the full support [0,1].
        torch.testing.assert_close(decode_quantiles(q, edges, "linear_cdf"),
                                   torch.tensor([[.125, .375, .5, 0., 0.]], dtype=torch.float64))
        # Clipping an extrapolated endpoint knot preserves its probability 0.
        # It does not create an atom by pointwise clipping an extended line.
        q = torch.tensor([[.05, .5]], dtype=torch.float64)
        edges = torch.tensor([0., .025, .05, 1.], dtype=torch.float64)
        torch.testing.assert_close(decode_quantiles(q, edges, "linear_cdf"),
                                   torch.tensor([[.125, .125, .75]], dtype=torch.float64))

    def test_plateau_retains_atom_and_assigns_internal_boundary_to_right(self):
        edges = torch.tensor([0., .4, .6, 1.], dtype=torch.float64)
        q = torch.tensor([[.1, .4, .4, .8]], dtype=torch.float64)
        torch.testing.assert_close(decode_quantiles(q, edges, "linear_cdf"),
                                   torch.tensor([[.375, .375, .25]], dtype=torch.float64))
        q = torch.tensor([[0., 0., 0.], [.4, .4, .4], [1., 1., 1.]], dtype=torch.float64)
        torch.testing.assert_close(decode_quantiles(q, edges, "linear_cdf"),
                                   torch.tensor([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]], dtype=torch.float64))

    def test_single_quantile_is_point_mass_with_same_boundary_convention(self):
        edges = torch.tensor([0., .5, 1.], dtype=torch.float64)
        q = torch.tensor([[.5], [1.], [0.]], dtype=torch.float64)
        torch.testing.assert_close(decode_quantiles(q, edges, "linear_cdf"),
                                   quantiles_to_histograms(q, edges))

    def test_finite_normalized_dtype_preserved_and_inputs_unchanged(self):
        rng = torch.Generator().manual_seed(13)
        edges = torch.linspace(-2., 3., 129, dtype=torch.float64)
        q = (torch.rand((37, 256), generator=rng) * 5 - 2).sort(dim=-1).values
        before = q.clone()
        output = decode_quantiles(q, edges, "linear_cdf")
        self.assertEqual(output.dtype, q.dtype)
        self.assertEqual(output.device, q.device)
        self.assertTrue(bool(torch.isfinite(output).all()))
        self.assertTrue(bool((output >= 0).all()))
        torch.testing.assert_close(output.sum(-1), torch.ones(37), atol=2e-7, rtol=0)
        self.assertTrue(torch.equal(q, before))

    def test_empty_batch_and_very_narrow_bins(self):
        edges = torch.tensor([0., .5 - 1e-10, .5 + 1e-10, 1.], dtype=torch.float64)
        self.assertEqual(decode_quantiles(torch.empty(0, 8), edges, "linear_cdf").shape, (0, 3))
        q = ((torch.arange(8, dtype=torch.float64) + .5) / 8)[None]
        torch.testing.assert_close(decode_quantiles(q, edges, "linear_cdf")[0],
                                   edges.diff(), atol=2e-15, rtol=0)

    def test_invalid_inputs_fail_without_silently_sorting_or_clipping(self):
        edges = torch.tensor([0., .5, 1.])
        for q in (torch.tensor([[.6, .4]]), torch.tensor([[float("nan"), .4]]),
                  torch.tensor([[-.1, .4]]), torch.tensor([[.4, 1.1]]), torch.empty(2, 0)):
            with self.assertRaises(ValueError):
                decode_quantiles(q, edges, "linear_cdf")
        for bad_edges in (torch.tensor([0., .5, .5]), torch.tensor([0., float("inf"), 1.])):
            with self.assertRaises(ValueError):
                decode_quantiles(torch.tensor([[.2, .8]]), bad_edges, "linear_cdf")

    def test_smooth_oracle_roundtrip_improves_without_claiming_exact_inverse(self):
        edges = torch.linspace(0, 1, 129, dtype=torch.float64)
        centers = (edges[:-1] + edges[1:]) / 2
        histograms = torch.stack((
            torch.exp(-.5 * ((centers - .43) / .12).square()),
            torch.exp(-.5 * ((centers - .25) / .08).square())
            + .7 * torch.exp(-.5 * ((centers - .73) / .10).square()),
        ))
        histograms /= histograms.sum(-1, keepdim=True)
        q = histogram_to_quantiles(histograms, edges, num_quantiles=128)
        hard = decode_quantiles(q, edges)
        interpolated = decode_quantiles(q, edges, "linear_cdf")
        self.assertTrue(bool(((interpolated - histograms).square().sum(-1)
                             < (hard - histograms).square().sum(-1)).all()))
        self.assertFalse(torch.equal(interpolated, histograms))

    def test_default_and_checkpoint_contract_are_unchanged(self):
        q = torch.tensor([[.1, .3, .3, 1.]])
        edges = torch.linspace(0, 1, 9)
        self.assertTrue(torch.equal(decode_quantiles(q, edges), quantiles_to_histograms(q, edges)))
        self.assertEqual(quantile_wasserstein_representation_signature()["protocol_version"],
                         "normalized_log_quantile_hard_bin_v2")
        self.assertNotEqual(qwfm_decoder_signature("hard_bin"), qwfm_decoder_signature("linear_cdf"))
        with self.assertRaises(ValueError):
            decode_quantiles(q, edges, "unknown")

    def test_sampler_raw_quantile_option_does_not_change_trajectory_or_rng(self):
        condition = torch.ones(2, 5, 16)
        edges = torch.linspace(0, 1, 9, dtype=torch.float64)
        regular_model, raw_model = _ZeroVelocity(), _ZeroVelocity()
        regular = QuantileWassersteinSampler(regular_model, "cpu", 71, edges, 1024)
        raw = QuantileWassersteinSampler(raw_model, "cpu", 71, edges, 1024)
        expected, diagnostics = regular.sample(condition, 3, 4)
        actual, extras = raw.sample(condition, 3, 4, return_quantiles=True)
        self.assertTrue(torch.equal(expected, actual))
        self.assertEqual(extras["quantiles"].shape, (2, 3, 16))
        for key in diagnostics:
            self.assertTrue(torch.equal(diagnostics[key], extras[key]))
        self.assertTrue(torch.equal(regular.generator.get_state(), raw.generator.get_state()))
        self.assertEqual(regular_model.calls, 4)
        self.assertEqual(raw_model.calls, 4)
        q = extras["quantiles"].reshape(-1, 16)
        self.assertTrue(torch.equal(decode_quantiles(q, edges).reshape_as(actual), actual))
        linear = decode_quantiles(q, edges, "linear_cdf")
        self.assertEqual(raw_model.calls, 4)  # Both decoders share one trajectory.
        self.assertFalse(torch.equal(actual.reshape_as(linear), linear))

    def test_predictive_model_quantile_opt_in_shape_and_non_qwfm_rejection(self):
        wrapper = PredictiveModel.__new__(PredictiveModel)
        wrapper.method = "quantile_wasserstein"
        wrapper.deterministic = False
        wrapper.args = SimpleNamespace(num_measurements=2, num_spectrum_bins=8,
                                       noise_standard_deviation=.005)
        wrapper.info = {"state_size": 16}
        wrapper.accelerator = SimpleNamespace(device=torch.device("cpu"), autocast=nullcontext)
        wrapper.grid = SimpleNamespace(echo_times=torch.tensor([.008, .512]))
        wrapper.sampler = QuantileWassersteinSampler(
            _ZeroVelocity(), "cpu", 71, torch.linspace(0, 1, 9), 1024)
        measurements = torch.tensor([[.8, .03], [.7, .04]])
        hist, q = wrapper.sample(measurements, 3, nfe=2, return_quantiles=True)
        self.assertEqual(hist.shape, (2, 3, 8))
        self.assertEqual(q.shape, (2, 3, 16))
        self.assertEqual(q.dtype, torch.float32)
        self.assertIsInstance(wrapper.sample(measurements, 3, nfe=2), torch.Tensor)
        wrapper.method = "conditional_vae"
        with self.assertRaisesRegex(ValueError, "only supported"):
            wrapper.sample(measurements, 3, return_quantiles=True)


if __name__ == "__main__":
    unittest.main()
