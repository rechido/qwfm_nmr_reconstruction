from __future__ import annotations

from typing import Dict, Tuple

import torch

from utils.deterministic_regression_utils import (
    deterministic_regression_forward,
    make_normalized_log_rate_positions,
)


class DeterministicRegressionSampler:
    """Expose one deterministic point prediction in the common sample layout.

    The baseline has no random source and no ODE.  It therefore supports exactly
    one unique output per condition; repeating it to emulate a generative sample
    budget would corrupt diversity and per-generated-sample timing metrics.
    """

    def __init__(self, model, log_rate_edges: torch.Tensor):
        self.model = model
        self.normalized_log_rate_positions = make_normalized_log_rate_positions(
            log_rate_edges
        )

    @torch.no_grad()
    def predict(self, condition: torch.Tensor) -> torch.Tensor:
        """Return point predictions with shape ``[batch, bins]``."""

        return deterministic_regression_forward(
            self.model,
            condition,
            self.normalized_log_rate_positions,
        )

    @torch.no_grad()
    def sample(
        self,
        condition: torch.Tensor,
        num_samples_per_condition: int = 1,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Return ``[batch, 1, bins]`` for shared histogram evaluators."""

        if (
            isinstance(num_samples_per_condition, bool)
            or not isinstance(num_samples_per_condition, int)
            or num_samples_per_condition != 1
        ):
            raise ValueError(
                "Deterministic regression supports exactly one unique sample "
                "per condition."
            )
        predictions = self.predict(condition)
        return predictions[:, None, :], {}
