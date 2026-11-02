from __future__ import annotations

from typing import Dict, Tuple

import torch


class ConditionalVAESampler:
    """Draw independent K-shot spectra from the conditional VAE prior."""

    def __init__(self, model, *, max_decode_batch_size: int | None = None):
        if max_decode_batch_size is not None and (
            isinstance(max_decode_batch_size, bool)
            or not isinstance(max_decode_batch_size, int)
            or max_decode_batch_size <= 0
        ):
            raise ValueError("max_decode_batch_size must be positive or None.")
        self.model = model
        self.max_decode_batch_size = max_decode_batch_size

    @torch.no_grad()
    def sample(
        self,
        condition: torch.Tensor,
        num_samples_per_condition: int = 1,
        *,
        generator: torch.Generator | None = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Return independent samples with shape ``[batch, K, bins]``.

        All K samples are decoded from independently drawn standard-normal
        latents.  Chunking only limits decoder memory and does not change the
        random draw order or returned sample ordering.
        """

        if (
            isinstance(num_samples_per_condition, bool)
            or not isinstance(num_samples_per_condition, int)
            or num_samples_per_condition <= 0
        ):
            raise ValueError("num_samples_per_condition must be positive.")
        if not isinstance(condition, torch.Tensor) or condition.ndim != 3:
            raise ValueError(
                "condition must be a tensor with shape [batch, channels, bins]."
            )
        batch_size = condition.shape[0]
        latent_dim = int(self.model.config.latent_dim)
        latent = torch.randn(
            (batch_size, num_samples_per_condition, latent_dim),
            device=condition.device,
            dtype=condition.dtype,
            generator=generator,
        )
        expanded_condition = condition[:, None, :, :].expand(
            -1, num_samples_per_condition, -1, -1
        )
        flat_latent = latent.reshape(batch_size * num_samples_per_condition, latent_dim)
        flat_condition = expanded_condition.reshape(
            batch_size * num_samples_per_condition,
            condition.shape[1],
            condition.shape[2],
        )
        chunk_size = self.max_decode_batch_size or flat_latent.shape[0]
        decoded_chunks = []
        for start in range(0, flat_latent.shape[0], chunk_size):
            stop = min(start + chunk_size, flat_latent.shape[0])
            decoded_chunks.append(
                self.model.decode(
                    flat_latent[start:stop],
                    flat_condition[start:stop],
                )
            )
        samples = torch.cat(decoded_chunks, dim=0).reshape(
            batch_size,
            num_samples_per_condition,
            condition.shape[-1],
        )
        return samples, {}


__all__ = ["ConditionalVAESampler"]
