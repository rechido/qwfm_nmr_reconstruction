from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Mapping

import torch
import torch.nn as nn
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_utils import ModelMixin


CONDITIONAL_FLOW_ARCHITECTURE_NAME = "conditional_realnvp_alr_mlp_v1"
CONDITIONAL_FLOW_INPUT_REPRESENTATION = "broadcast_measurement_echo_sigma_v1"
CONDITIONAL_FLOW_OUTPUT_PARAMETERIZATION = "inverse_alr_simplex_v1"
CONDITIONAL_FLOW_TARGET_REPRESENTATION = "interiorized_alr_v1"
DEFAULT_NUM_COUPLING_LAYERS = 8
DEFAULT_HIDDEN_DIM = 256
DEFAULT_CONTEXT_DIM = 256
DEFAULT_NUM_CONDITION_BLOCKS = 3
DEFAULT_SCALE_LIMIT = 1.5


def _positive_integer(value: Any, *, name: str, minimum: int = 1) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer of at least {minimum}.")
    return int(value)


def _finite_positive(value: Any, *, name: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) <= 0.0
    ):
        raise ValueError(f"{name} must be finite and positive.")
    return float(value)


def interiorize_simplex(
    histograms: torch.Tensor,
    interior_mass: float,
) -> torch.Tensor:
    """Move possibly sparse histograms into the open simplex.

    ``(1-a) * normalize(x) + a / N`` is injective on the closed simplex for a
    fixed ``a`` and gives every bin a strictly positive mass.  This is needed
    because an additive-log-ratio (ALR) chart is undefined on the boundary.
    The fixed transformation is part of the checkpointed CNF data contract.
    """

    if not isinstance(histograms, torch.Tensor):
        raise TypeError("histograms must be a torch.Tensor.")
    if histograms.ndim < 2 or histograms.shape[-1] < 2:
        raise ValueError("histograms must end in a simplex dimension of at least two.")
    if not histograms.is_floating_point():
        raise TypeError("histograms must have a floating-point dtype.")
    alpha = _finite_positive(interior_mass, name="interior_mass")
    if alpha >= 1.0:
        raise ValueError("interior_mass must be smaller than one.")
    if not bool(torch.isfinite(histograms).all()) or bool((histograms < 0.0).any()):
        raise ValueError("histograms must be finite and nonnegative.")
    totals = histograms.sum(dim=-1, keepdim=True)
    if bool((totals <= 0.0).any()) or not bool(torch.isfinite(totals).all()):
        raise ValueError("every histogram must have positive finite mass.")
    normalized = histograms / totals
    result = (1.0 - alpha) * normalized + alpha / float(histograms.shape[-1])
    if bool((result <= 0.0).any()) or not bool(torch.isfinite(result).all()):
        raise RuntimeError("simplex interiorization produced an invalid value.")
    return result


def simplex_to_alr(
    histograms: torch.Tensor,
    *,
    interior_mass: float,
) -> torch.Tensor:
    """Map histograms to an ``N-1`` dimensional ALR chart.

    The final spectrum bin is the fixed reference component.  Calculations are
    performed in at least float32 to keep log/ratio operations stable under
    mixed precision.
    """

    work_dtype = (
        torch.float32
        if histograms.dtype in (torch.float16, torch.bfloat16)
        else histograms.dtype
    )
    interior = interiorize_simplex(
        histograms.to(dtype=work_dtype),
        interior_mass,
    )
    return (interior[..., :-1].log() - interior[..., -1:].log()).contiguous()


def alr_to_simplex(alr_coordinates: torch.Tensor) -> torch.Tensor:
    """Invert the final-bin ALR chart with a numerically stable softmax."""

    if not isinstance(alr_coordinates, torch.Tensor):
        raise TypeError("alr_coordinates must be a torch.Tensor.")
    if alr_coordinates.ndim < 2 or alr_coordinates.shape[-1] < 1:
        raise ValueError("alr_coordinates must end in a nonempty ALR dimension.")
    if not alr_coordinates.is_floating_point():
        raise TypeError("alr_coordinates must have a floating-point dtype.")
    if not bool(torch.isfinite(alr_coordinates).all()):
        raise ValueError("alr_coordinates must be finite.")
    work_dtype = (
        torch.float32
        if alr_coordinates.dtype in (torch.float16, torch.bfloat16)
        else alr_coordinates.dtype
    )
    coordinates = alr_coordinates.to(dtype=work_dtype)
    reference = torch.zeros_like(coordinates[..., :1])
    return torch.softmax(torch.cat((coordinates, reference), dim=-1), dim=-1)


class _ResidualMLPBlock(nn.Module):
    def __init__(self, width: int):
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.linear1 = nn.Linear(width, width)
        self.linear2 = nn.Linear(width, width)
        self.activation = nn.SiLU()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        hidden = self.linear1(self.activation(self.norm(inputs)))
        hidden = self.linear2(self.activation(hidden))
        return inputs + hidden


class _ConditionEncoder(nn.Module):
    def __init__(
        self,
        condition_channels: int,
        context_dim: int,
        num_blocks: int,
    ):
        super().__init__()
        self.input_norm = nn.LayerNorm(condition_channels)
        self.input_projection = nn.Linear(condition_channels, context_dim)
        self.blocks = nn.ModuleList(
            _ResidualMLPBlock(context_dim) for _ in range(num_blocks)
        )
        self.output_norm = nn.LayerNorm(context_dim)

    def forward(self, condition_vector: torch.Tensor) -> torch.Tensor:
        hidden = self.input_projection(self.input_norm(condition_vector))
        for block in self.blocks:
            hidden = block(hidden)
        return self.output_norm(hidden)


class _ConditionalAffineCoupling(nn.Module):
    """One conditional RealNVP affine coupling transform."""

    def __init__(
        self,
        dimension: int,
        context_dim: int,
        hidden_dim: int,
        mask: torch.Tensor,
        scale_limit: float,
    ):
        super().__init__()
        if mask.shape != (dimension,):
            raise ValueError("coupling mask must have shape [dimension].")
        self.register_buffer("mask", mask.to(dtype=torch.float32).contiguous())
        self.scale_limit = float(scale_limit)
        self.network = nn.Sequential(
            nn.Linear(dimension + context_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 2 * dimension),
        )
        # Starting at the identity prevents extreme initial likelihood values.
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)

    def _scale_and_shift(
        self,
        coordinates: torch.Tensor,
        context: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mask = self.mask.to(device=coordinates.device, dtype=coordinates.dtype)
        parameters = self.network(torch.cat((coordinates * mask, context), dim=-1))
        raw_scale, shift = parameters.chunk(2, dim=-1)
        inverse_mask = 1.0 - mask
        scale = self.scale_limit * torch.tanh(raw_scale) * inverse_mask
        shift = shift * inverse_mask
        return scale, shift, mask

    def encode(
        self,
        coordinates: torch.Tensor,
        context: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        scale, shift, mask = self._scale_and_shift(coordinates, context)
        encoded = coordinates * mask + (1.0 - mask) * (
            coordinates * torch.exp(scale) + shift
        )
        return encoded, scale.sum(dim=-1)

    def decode(
        self,
        encoded: torch.Tensor,
        context: torch.Tensor,
    ) -> torch.Tensor:
        # Masked coordinates are unchanged, so the conditioner sees exactly
        # the values used by encode and the inverse is analytic.
        scale, shift, mask = self._scale_and_shift(encoded, context)
        return encoded * mask + (1.0 - mask) * ((encoded - shift) * torch.exp(-scale))


class ConditionalSimplexFlow(ModelMixin, ConfigMixin):
    """Conditional RealNVP density in an ALR chart of the spectrum simplex.

    AttentionUNet1D is intentionally not reused here: a generic U-Net update is
    not bijective and has no tractable Jacobian determinant.  RealNVP coupling
    layers provide exact inversion and exact conditional likelihood, while a
    shared residual MLP encodes the same measurement/echo/sigma condition used
    by the other experiments.  The condition is broadcast along the spectrum
    axis by ``make_condition``; reading one position loses no information.
    """

    _skip_layerwise_casting_patterns = ["norm"]

    @register_to_config
    def __init__(
        self,
        num_spectrum_bins: int,
        num_measurements: int,
        num_coupling_layers: int = DEFAULT_NUM_COUPLING_LAYERS,
        hidden_dim: int = DEFAULT_HIDDEN_DIM,
        context_dim: int = DEFAULT_CONTEXT_DIM,
        num_condition_blocks: int = DEFAULT_NUM_CONDITION_BLOCKS,
        scale_limit: float = DEFAULT_SCALE_LIMIT,
        architecture_name: str = CONDITIONAL_FLOW_ARCHITECTURE_NAME,
    ):
        super().__init__()
        num_spectrum_bins = _positive_integer(
            num_spectrum_bins, name="num_spectrum_bins", minimum=2
        )
        num_measurements = _positive_integer(
            num_measurements, name="num_measurements", minimum=2
        )
        num_coupling_layers = _positive_integer(
            num_coupling_layers, name="num_coupling_layers", minimum=2
        )
        hidden_dim = _positive_integer(hidden_dim, name="hidden_dim")
        context_dim = _positive_integer(context_dim, name="context_dim")
        num_condition_blocks = _positive_integer(
            num_condition_blocks, name="num_condition_blocks"
        )
        scale_limit = _finite_positive(scale_limit, name="scale_limit")
        if architecture_name != CONDITIONAL_FLOW_ARCHITECTURE_NAME:
            raise ValueError(
                f"architecture_name must be {CONDITIONAL_FLOW_ARCHITECTURE_NAME!r}."
            )

        self.alr_dimension = num_spectrum_bins - 1
        self.condition_channels = 2 * num_measurements + 1
        self.condition_encoder = _ConditionEncoder(
            self.condition_channels,
            context_dim,
            num_condition_blocks,
        )
        layers = []
        for layer_index in range(num_coupling_layers):
            # Alternating parity masks work for both odd and even ALR sizes and
            # guarantee every coordinate is transformed by adjacent layers.
            mask = ((torch.arange(self.alr_dimension) + layer_index) % 2 == 0).to(
                torch.float32
            )
            layers.append(
                _ConditionalAffineCoupling(
                    self.alr_dimension,
                    context_dim,
                    hidden_dim,
                    mask,
                    scale_limit,
                )
            )
        self.coupling_layers = nn.ModuleList(layers)

    def _encode_condition(self, condition: torch.Tensor) -> torch.Tensor:
        if not isinstance(condition, torch.Tensor):
            raise TypeError("condition must be a torch.Tensor.")
        if condition.ndim != 3:
            raise ValueError(
                "condition must have shape [batch, channels, bins], "
                f"got {tuple(condition.shape)}."
            )
        if condition.shape[1] != self.condition_channels:
            raise ValueError(
                f"condition has {condition.shape[1]} channels; expected "
                f"{self.condition_channels}."
            )
        if condition.shape[2] != self.config.num_spectrum_bins:
            raise ValueError(
                f"condition length is {condition.shape[2]}; expected "
                f"{self.config.num_spectrum_bins}."
            )
        if not condition.is_floating_point() or not bool(
            torch.isfinite(condition).all()
        ):
            raise ValueError("condition must be a finite floating-point tensor.")
        return self.condition_encoder(condition[:, :, 0])

    def encode(
        self,
        alr_coordinates: torch.Tensor,
        condition: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if alr_coordinates.ndim != 2 or alr_coordinates.shape[1] != self.alr_dimension:
            raise ValueError(
                "alr_coordinates must have shape [batch, bins-1], "
                f"got {tuple(alr_coordinates.shape)}."
            )
        if alr_coordinates.shape[0] != condition.shape[0]:
            raise ValueError("ALR coordinates and condition batch sizes must match.")
        context = self._encode_condition(condition)
        encoded = alr_coordinates
        log_absolute_determinant = torch.zeros(
            encoded.shape[0], device=encoded.device, dtype=encoded.dtype
        )
        for layer in self.coupling_layers:
            encoded, layer_logdet = layer.encode(encoded, context)
            log_absolute_determinant = log_absolute_determinant + layer_logdet
        return encoded, log_absolute_determinant

    def decode(
        self,
        base_coordinates: torch.Tensor,
        condition: torch.Tensor,
    ) -> torch.Tensor:
        if (
            base_coordinates.ndim != 2
            or base_coordinates.shape[1] != self.alr_dimension
        ):
            raise ValueError(
                "base_coordinates must have shape [batch, bins-1], "
                f"got {tuple(base_coordinates.shape)}."
            )
        if base_coordinates.shape[0] != condition.shape[0]:
            raise ValueError("base coordinates and condition batch sizes must match.")
        context = self._encode_condition(condition)
        decoded = base_coordinates
        for layer in reversed(self.coupling_layers):
            decoded = layer.decode(decoded, context)
        return decoded

    def log_prob(
        self,
        alr_coordinates: torch.Tensor,
        condition: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        latent, logdet = self.encode(alr_coordinates, condition)
        log_base = -0.5 * (latent.square() + math.log(2.0 * math.pi)).sum(dim=-1)
        return log_base + logdet, latent, logdet

    def forward(
        self,
        alr_coordinates: torch.Tensor,
        condition: torch.Tensor,
        return_diagnostics: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Evaluate log density through the standard ``nn.Module`` call path.

        DistributedDataParallel only intercepts ``forward``; it does not proxy
        arbitrary methods such as :meth:`log_prob`.  Training therefore asks
        for the auxiliary tensors through this flag so gradient reduction and
        Accelerator's accumulation/no-sync hooks remain active.
        """

        outputs = self.log_prob(alr_coordinates, condition)
        if return_diagnostics:
            return outputs
        return outputs[0]


CONDITIONAL_FLOW_MODEL_CLASS = ConditionalSimplexFlow


def get_conditional_flow_architecture_signature(
    *,
    num_spectrum_bins: int,
    num_measurements: int,
    num_coupling_layers: int = DEFAULT_NUM_COUPLING_LAYERS,
    hidden_dim: int = DEFAULT_HIDDEN_DIM,
    context_dim: int = DEFAULT_CONTEXT_DIM,
    num_condition_blocks: int = DEFAULT_NUM_CONDITION_BLOCKS,
    scale_limit: float = DEFAULT_SCALE_LIMIT,
) -> dict[str, Any]:
    return {
        "architecture_name": CONDITIONAL_FLOW_ARCHITECTURE_NAME,
        "model_class": CONDITIONAL_FLOW_MODEL_CLASS.__name__,
        "diffusers_base_class": "ModelMixin",
        "flow_family": "conditional_realnvp_affine_coupling",
        "simplex_chart": "final_component_additive_log_ratio",
        "num_spectrum_bins": _positive_integer(
            num_spectrum_bins, name="num_spectrum_bins", minimum=2
        ),
        "alr_dimension": int(num_spectrum_bins) - 1,
        "num_measurements": _positive_integer(
            num_measurements, name="num_measurements", minimum=2
        ),
        "condition_channels": 2 * int(num_measurements) + 1,
        "num_coupling_layers": _positive_integer(
            num_coupling_layers, name="num_coupling_layers", minimum=2
        ),
        "hidden_dim": _positive_integer(hidden_dim, name="hidden_dim"),
        "context_dim": _positive_integer(context_dim, name="context_dim"),
        "num_condition_blocks": _positive_integer(
            num_condition_blocks, name="num_condition_blocks"
        ),
        "scale_limit": _finite_positive(scale_limit, name="scale_limit"),
        "condition_encoder": "residual_mlp",
        "condition_input_representation": CONDITIONAL_FLOW_INPUT_REPRESENTATION,
        "output_parameterization": CONDITIONAL_FLOW_OUTPUT_PARAMETERIZATION,
    }


def create_conditional_flow_model(
    **architecture: Any,
) -> ConditionalSimplexFlow:
    signature = get_conditional_flow_architecture_signature(**architecture)
    return CONDITIONAL_FLOW_MODEL_CLASS(
        num_spectrum_bins=signature["num_spectrum_bins"],
        num_measurements=signature["num_measurements"],
        num_coupling_layers=signature["num_coupling_layers"],
        hidden_dim=signature["hidden_dim"],
        context_dim=signature["context_dim"],
        num_condition_blocks=signature["num_condition_blocks"],
        scale_limit=signature["scale_limit"],
    )


def _model_config_mapping(model: ConditionalSimplexFlow) -> Mapping[str, Any]:
    config = getattr(model, "config", None)
    if config is None or not hasattr(config, "get"):
        raise TypeError("The CNF model does not expose a Diffusers config mapping.")
    return config


def validate_conditional_flow_model_config(
    model: ConditionalSimplexFlow,
    **architecture: Any,
) -> None:
    if not isinstance(model, CONDITIONAL_FLOW_MODEL_CLASS):
        raise TypeError(
            f"Expected {CONDITIONAL_FLOW_MODEL_CLASS.__name__}, "
            f"got {type(model).__name__}."
        )
    expected = get_conditional_flow_architecture_signature(**architecture)
    config = _model_config_mapping(model)
    names = (
        "architecture_name",
        "num_spectrum_bins",
        "num_measurements",
        "num_coupling_layers",
        "hidden_dim",
        "context_dim",
        "num_condition_blocks",
        "scale_limit",
    )
    mismatches = [
        f"{name}: checkpoint={config.get(name)!r}, required={expected[name]!r}"
        for name in names
        if config.get(name) != expected[name]
    ]
    if mismatches:
        raise ValueError(
            "CNF checkpoint architecture mismatch: " + "; ".join(mismatches)
        )


def load_conditional_flow_model_pretrained(
    path: str | Path,
) -> ConditionalSimplexFlow:
    model_path = Path(path)
    loaded_config = CONDITIONAL_FLOW_MODEL_CLASS.load_config(model_path)
    if loaded_config.get("_class_name") != CONDITIONAL_FLOW_MODEL_CLASS.__name__:
        raise ValueError(
            f"Checkpoint model class mismatch: {loaded_config.get('_class_name')!r}."
        )
    if loaded_config.get("architecture_name") != CONDITIONAL_FLOW_ARCHITECTURE_NAME:
        raise ValueError("Checkpoint CNF architecture signature mismatch.")
    return CONDITIONAL_FLOW_MODEL_CLASS.from_pretrained(model_path)


__all__ = [
    "CONDITIONAL_FLOW_ARCHITECTURE_NAME",
    "CONDITIONAL_FLOW_INPUT_REPRESENTATION",
    "CONDITIONAL_FLOW_MODEL_CLASS",
    "CONDITIONAL_FLOW_OUTPUT_PARAMETERIZATION",
    "CONDITIONAL_FLOW_TARGET_REPRESENTATION",
    "ConditionalSimplexFlow",
    "alr_to_simplex",
    "create_conditional_flow_model",
    "get_conditional_flow_architecture_signature",
    "interiorize_simplex",
    "load_conditional_flow_model_pretrained",
    "simplex_to_alr",
    "validate_conditional_flow_model_config",
]
