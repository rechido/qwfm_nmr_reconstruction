"""Small training-time logging and fixed-evaluation helpers."""

from __future__ import annotations

import json
import math
import os
import tempfile
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset

from utils.nmr_utils import make_condition, sample_standard_dirichlet


def nested_config_values_match(
    current: Any,
    saved: Any,
    *,
    absolute_tolerance: float = 1.0e-12,
) -> bool:
    """Compare JSON-like config trees with strict shape and tolerant floats."""

    if isinstance(current, bool) or isinstance(saved, bool):
        return type(current) is type(saved) and current == saved
    if isinstance(current, (int, float)) and isinstance(saved, (int, float)):
        if not math.isfinite(float(current)) or not math.isfinite(float(saved)):
            return current == saved
        return math.isclose(
            float(current),
            float(saved),
            rel_tol=0.0,
            abs_tol=float(absolute_tolerance),
        )
    if isinstance(current, Mapping) and isinstance(saved, Mapping):
        return current.keys() == saved.keys() and all(
            nested_config_values_match(
                current[key],
                saved[key],
                absolute_tolerance=absolute_tolerance,
            )
            for key in current
        )
    if isinstance(current, (list, tuple)) and isinstance(saved, (list, tuple)):
        return len(current) == len(saved) and all(
            nested_config_values_match(
                left,
                right,
                absolute_tolerance=absolute_tolerance,
            )
            for left, right in zip(current, saved)
        )
    return current == saved


def require_fresh_training_history(output_directory: str | os.PathLike[str]) -> None:
    """Refuse to append to artifacts left by a run with no committed checkpoint."""

    output_directory = Path(output_directory)
    known_artifacts = (
        output_directory / "training_config.json",
        output_directory / "loss_history.jsonl",
        output_directory / "eval" / "fixed_validation" / "metrics.jsonl",
        output_directory / "eval" / "fm_generalization" / "metrics.jsonl",
    )
    existing = [path for path in known_artifacts if path.exists() or path.is_symlink()]
    if existing:
        rendered = "\n".join(str(path) for path in existing)
        raise FileExistsError(
            "Training artifacts exist but no checkpoint was selected. Use a new "
            "output_dir or deliberately remove/recover the stale run:\n" + rendered
        )


class FixedFMProbeDataset(Dataset):
    """Pair a fixed dataset with an index-aligned flow-matching probe bank.

    Wrapping the train probe and validation datasets with the same bank before
    passing their loaders to ``Accelerator.prepare`` preserves the pairing
    under distributed dataloader sharding. The bank lives on CPU and is
    transferred by the training entry point along with each data batch.
    """

    def __init__(
        self,
        dataset: Any,
        source_histograms: torch.Tensor,
        flow_times: torch.Tensor,
    ) -> None:
        if not hasattr(dataset, "__len__") or not hasattr(dataset, "__getitem__"):
            raise TypeError("dataset must implement __len__ and __getitem__.")
        if source_histograms.ndim != 2:
            raise ValueError(
                "source_histograms must have shape [samples, histogram bins]."
            )
        if flow_times.ndim != 1:
            raise ValueError("flow_times must have shape [samples].")
        sample_count = len(dataset)
        if sample_count <= 0:
            raise ValueError("dataset must contain at least one sample.")
        if source_histograms.shape[0] != sample_count:
            raise ValueError(
                "source_histograms and dataset must contain the same number of samples."
            )
        if flow_times.shape[0] != sample_count:
            raise ValueError(
                "flow_times and dataset must contain the same number of samples."
            )
        if source_histograms.device.type != "cpu" or flow_times.device.type != "cpu":
            raise ValueError("The fixed flow-matching probe bank must live on CPU.")
        if not bool(torch.isfinite(source_histograms).all()):
            raise ValueError("source_histograms must contain only finite values.")
        if bool((source_histograms <= 0.0).any()):
            raise ValueError("source_histograms must be strictly positive.")
        if not bool(torch.isfinite(flow_times).all()) or bool(
            ((flow_times < 0.0) | (flow_times >= 1.0)).any()
        ):
            raise ValueError("flow_times must be finite and lie in [0, 1).")

        self.dataset = dataset
        self.source_histograms = source_histograms.detach().clone().contiguous()
        self.flow_times = flow_times.detach().clone().contiguous()

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(
        self, index: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        sample = self.dataset[index]
        if not isinstance(sample, (tuple, list)) or len(sample) not in (2, 3):
            raise ValueError(
                "The wrapped dataset must return (target, measurement) or "
                "(target, measurement, loss_weight)."
            )
        target, measurement = sample[:2]
        return (
            target,
            measurement,
            self.source_histograms[index],
            self.flow_times[index],
        )


def build_fixed_fm_probe_bank(
    num_samples: int,
    num_bins: int,
    source_seed: int,
    time_seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build a reproducible CPU source/time bank without touching global RNGs."""

    _positive_integer(num_samples, "num_samples")
    _positive_integer(num_bins, "num_bins")
    for seed, name in ((source_seed, "source_seed"), (time_seed, "time_seed")):
        if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**63:
            raise ValueError(f"{name} must be an integer in [0, 2**63).")

    source_generator = torch.Generator(device="cpu").manual_seed(source_seed)
    time_generator = torch.Generator(device="cpu").manual_seed(time_seed)
    source_histograms = sample_standard_dirichlet(
        num_samples,
        num_bins,
        device=torch.device("cpu"),
        output_dtype=torch.float64,
        generator=source_generator,
    )
    flow_times = torch.rand(
        num_samples,
        dtype=torch.float64,
        device="cpu",
        generator=time_generator,
    )
    return source_histograms, flow_times


def _positive_integer(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer.")
    return value


def _synchronize_cuda(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def start_distributed_wall_timer(accelerator: Any) -> float:
    """Start a wall timer after every rank and its CUDA work are ready."""

    accelerator.wait_for_everyone()
    _synchronize_cuda(accelerator.device)
    return time.perf_counter()


def slowest_distributed_elapsed_seconds(
    accelerator: Any,
    started_at: float,
) -> float:
    """Return elapsed wall time for the slowest distributed rank."""

    accelerator.wait_for_everyone()
    _synchronize_cuda(accelerator.device)
    local_elapsed = torch.tensor(
        [time.perf_counter() - started_at],
        device=accelerator.device,
        dtype=torch.float64,
    )
    gathered_elapsed = accelerator.gather(local_elapsed)
    return float(gathered_elapsed.max().item())


def format_console_log_record(
    record: Mapping[str, Any],
    *,
    indent: int | None = None,
    sort_keys: bool = False,
) -> str:
    """Render floats with four significant digits for terminal-only logs."""

    items = list(record.items())
    if sort_keys:
        items.sort(key=lambda item: item[0])
    fields: list[str] = []
    for key, value in items:
        rendered = format(value, ".4g") if isinstance(value, float) else repr(value)
        fields.append(f"{key!r}: {rendered}")
    if indent is None:
        return "{" + ", ".join(fields) + "}"
    if isinstance(indent, bool) or not isinstance(indent, int) or indent < 0:
        raise ValueError("indent must be a nonnegative integer or None.")
    prefix = " " * indent
    return "{\n" + ",\n".join(prefix + field for field in fields) + "\n}"


@torch.no_grad()
def collect_euler_samples(
    accelerator: Any,
    loader: Any,
    sampler: Any,
    echo_times: torch.Tensor,
    state_size: int,
    noise_standard_deviation: float,
    num_samples_per_condition: int,
    num_ode_steps: int,
) -> tuple[
    torch.Tensor | None,
    torch.Tensor | None,
    torch.Tensor | None,
    dict[str, float],
    float,
]:
    """Run one fixed Euler evaluation and gather its result.

    The first three return values are concatenated CPU tensors on the main
    process and ``None`` elsewhere. Reduced sampler diagnostics and the
    slowest-rank wall time are returned on every process.
    """

    _positive_integer(state_size, "state_size")
    _positive_integer(
        num_samples_per_condition,
        "num_samples_per_condition",
    )
    _positive_integer(num_ode_steps, "num_ode_steps")
    if (
        isinstance(noise_standard_deviation, bool)
        or not isinstance(noise_standard_deviation, (int, float))
        or not math.isfinite(float(noise_standard_deviation))
        or float(noise_standard_deviation) <= 0.0
    ):
        raise ValueError("noise_standard_deviation must be finite and positive.")

    generated_batches: list[torch.Tensor] = []
    target_batches: list[torch.Tensor] = []
    measurement_batches: list[torch.Tensor] = []
    diagnostic_totals: dict[str, float] = {}
    local_batch_count = 0

    accelerator.wait_for_everyone()
    _synchronize_cuda(accelerator.device)
    start_time = time.perf_counter()

    for targets, measurements in loader:
        local_batch_count += 1
        condition = make_condition(
            measurements,
            echo_times,
            output_size=state_size,
            noise_standard_deviation=noise_standard_deviation,
        )
        samples, diagnostics = sampler.sample(
            condition,
            num_samples_per_condition=num_samples_per_condition,
            num_ode_steps=num_ode_steps,
        )
        gathered_samples, gathered_targets, gathered_measurements = (
            accelerator.gather_for_metrics((samples, targets, measurements))
        )
        if accelerator.is_main_process:
            generated_batches.append(gathered_samples.detach().cpu())
            target_batches.append(gathered_targets.detach().cpu())
            measurement_batches.append(gathered_measurements.detach().cpu())

        for name, value in diagnostics.items():
            if not isinstance(value, torch.Tensor) or value.numel() != 1:
                raise ValueError(
                    f"Sampler diagnostic {name!r} must be a scalar tensor."
                )
            reduced = accelerator.reduce(value, reduction="sum")
            diagnostic_totals[name] = diagnostic_totals.get(name, 0.0) + float(
                reduced.item()
            )

    _synchronize_cuda(accelerator.device)
    local_elapsed = torch.tensor(
        [time.perf_counter() - start_time],
        device=accelerator.device,
        dtype=torch.float64,
    )
    elapsed = float(accelerator.gather(local_elapsed).max().item())
    accelerator.wait_for_everyone()

    if local_batch_count == 0:
        raise RuntimeError("Evaluation loader produced no batches.")

    if accelerator.is_main_process:
        samples_cpu = torch.cat(generated_batches, dim=0)
        targets_cpu = torch.cat(target_batches, dim=0)
        measurements_cpu = torch.cat(measurement_batches, dim=0)
    else:
        samples_cpu = None
        targets_cpu = None
        measurements_cpu = None

    return (
        samples_cpu,
        targets_cpu,
        measurements_cpu,
        diagnostic_totals,
        elapsed,
    )


def reduce_fixed_fm_probe_statistics(
    accelerator: Any,
    error_squared_sum: torch.Tensor,
    velocity_sum: torch.Tensor,
    velocity_squared_sum: torch.Tensor,
    sample_count: torch.Tensor,
) -> dict[str, float | int]:
    """Reduce global sufficient statistics for fixed FM probe metrics.

    Each vector contains per-coordinate sums over local probe samples.  The
    normalized metric is computed only after distributed reduction, using the
    variance of the target velocity over the complete global probe.
    """

    vectors = (error_squared_sum, velocity_sum, velocity_squared_sum)
    if any(vector.ndim != 1 for vector in vectors):
        raise ValueError(
            "FM probe sufficient statistics must be one-dimensional vectors."
        )
    if not (
        error_squared_sum.shape == velocity_sum.shape == velocity_squared_sum.shape
    ):
        raise ValueError(
            "FM probe sufficient-statistic vectors must have identical shapes."
        )
    if error_squared_sum.numel() == 0:
        raise ValueError("FM probe sufficient-statistic vectors cannot be empty.")
    if sample_count.numel() != 1:
        raise ValueError("sample_count must be a scalar tensor.")
    devices = {tensor.device for tensor in (*vectors, sample_count)}
    if len(devices) != 1:
        raise ValueError("All FM probe statistics must be on the same device.")
    if not all(bool(torch.isfinite(vector).all()) for vector in vectors):
        raise ValueError("FM probe sufficient statistics must be finite.")
    if bool((error_squared_sum < 0.0).any()) or bool(
        (velocity_squared_sum < 0.0).any()
    ):
        raise ValueError("Squared FM probe sums must be nonnegative.")
    if not bool(torch.isfinite(sample_count).all()) or float(sample_count.item()) < 0:
        raise ValueError("sample_count must be finite and nonnegative.")

    reduced_error_squared_sum = accelerator.reduce(
        error_squared_sum.to(torch.float64), reduction="sum"
    )
    reduced_velocity_sum = accelerator.reduce(
        velocity_sum.to(torch.float64), reduction="sum"
    )
    reduced_velocity_squared_sum = accelerator.reduce(
        velocity_squared_sum.to(torch.float64), reduction="sum"
    )
    reduced_sample_count = accelerator.reduce(
        sample_count.to(torch.float64), reduction="sum"
    )
    count = float(reduced_sample_count.item())
    if not math.isfinite(count) or count <= 0.0 or not count.is_integer():
        raise RuntimeError(
            "The globally reduced FM probe sample count must be a positive integer."
        )

    coordinate_mse = reduced_error_squared_sum / count
    velocity_mean = reduced_velocity_sum / count
    coordinate_variance = (
        reduced_velocity_squared_sum / count - velocity_mean.square()
    ).clamp_min(0.0)
    loss = coordinate_mse.mean()
    normalized_velocity_mse = (
        coordinate_mse / coordinate_variance.clamp_min(1.0e-12)
    ).mean()
    if not bool(torch.isfinite(loss)) or not bool(
        torch.isfinite(normalized_velocity_mse)
    ):
        raise RuntimeError("The globally reduced FM probe metrics are not finite.")
    return {
        "loss": float(loss.item()),
        "normalized_velocity_mse": float(normalized_velocity_mse.item()),
        "sample_count": int(count),
    }


def make_fm_generalization_record(
    iteration: int,
    train_metrics: Mapping[str, Any],
    validation_metrics: Mapping[str, Any],
    *,
    model_weight: str = "raw",
) -> dict[str, float | int | str]:
    """Create one strict equal-size train/validation FM comparison record."""

    if (
        isinstance(iteration, bool)
        or not isinstance(iteration, int)
        or iteration < 0
    ):
        raise ValueError("iteration must be a nonnegative integer.")
    if not isinstance(model_weight, str) or not model_weight.strip():
        raise ValueError("model_weight must be a nonempty string.")

    metric_values: dict[str, float] = {}
    for split_name, metrics in (
        ("train_probe", train_metrics),
        ("validation", validation_metrics),
    ):
        if not isinstance(metrics, Mapping):
            raise TypeError(f"{split_name} metrics must be a mapping.")
        for metric_name in ("loss", "normalized_velocity_mse"):
            value = metrics.get(metric_name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"{split_name} {metric_name} must be a finite number.")
            numeric_value = float(value)
            if not math.isfinite(numeric_value) or numeric_value < 0.0:
                raise ValueError(
                    f"{split_name} {metric_name} must be finite and nonnegative."
                )
            metric_values[f"{split_name}_{metric_name}"] = numeric_value

    train_count = train_metrics.get("sample_count")
    validation_count = validation_metrics.get("sample_count")
    for split_name, count in (
        ("train_probe", train_count),
        ("validation", validation_count),
    ):
        if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
            raise ValueError(f"{split_name} sample_count must be a positive integer.")
    if train_count != validation_count:
        raise ValueError(
            "Train and validation FM probes must contain the same sample count."
        )

    train_loss = metric_values["train_probe_loss"]
    validation_loss = metric_values["validation_loss"]
    train_normalized = metric_values["train_probe_normalized_velocity_mse"]
    validation_normalized = metric_values["validation_normalized_velocity_mse"]
    return {
        "iteration": iteration,
        "model_weight": model_weight.strip(),
        "num_probe_samples": train_count,
        "train_probe_loss": train_loss,
        "validation_loss": validation_loss,
        "loss_gap": validation_loss - train_loss,
        "train_probe_normalized_velocity_mse": train_normalized,
        "validation_normalized_velocity_mse": validation_normalized,
        "normalized_velocity_mse_gap": (validation_normalized - train_normalized),
    }


def append_jsonl(path: str | os.PathLike[str], record: Mapping[str, Any]) -> None:
    """Append one strict JSON object to a JSON-lines file."""

    if not isinstance(record, Mapping):
        raise TypeError("record must be a mapping.")
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(dict(record), sort_keys=True, allow_nan=False)
    with output_path.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def read_jsonl(path: str | os.PathLike[str]) -> list[dict[str, Any]]:
    """Read valid object records while ignoring blank or damaged lines."""

    input_path = Path(path)
    if not input_path.exists():
        return []

    records: list[dict[str, Any]] = []
    with input_path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except (json.JSONDecodeError, UnicodeDecodeError):
                continue
            if isinstance(record, dict):
                records.append(record)
    return records


def _atomic_replace_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            descriptor = -1
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass


def _restore_latest_fixed_evaluation_metrics(
    fixed_directory: Path,
    max_iteration: int,
) -> int | None:
    metrics_path = fixed_directory / "metrics.jsonl"
    latest_metrics_path = fixed_directory / "latest_metrics.json"
    records: list[tuple[int, dict[str, Any]]] = []
    for record in read_jsonl(metrics_path):
        iteration = _record_iteration(record)
        if iteration is not None and iteration <= max_iteration:
            records.append((iteration, record))
    latest_iteration: int | None = None
    if records:
        latest_iteration, latest_record = max(records, key=lambda item: item[0])
        latest_record = dict(latest_record)
        latest_record["iteration"] = latest_iteration
        payload = (
            json.dumps(
                latest_record,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            + "\n"
        ).encode("utf-8")
        _atomic_replace_bytes(latest_metrics_path, payload)
    else:
        latest_metrics_path.unlink(missing_ok=True)
    return latest_iteration


def restore_latest_fixed_evaluation_metrics(
    directory: str | os.PathLike[str],
    max_iteration: int,
    *,
    discard_latest_images: bool,
) -> None:
    """Restore metrics at or before a resumed optimizer iteration.

    When ``discard_latest_images`` is true, the two latest-only images are
    discarded because their model iteration cannot be verified without an
    image archive. The next fixed evaluation recreates both images.
    """

    if (
        isinstance(max_iteration, bool)
        or not isinstance(max_iteration, int)
        or max_iteration < 0
    ):
        raise ValueError("max_iteration must be a nonnegative integer.")
    if not isinstance(discard_latest_images, bool):
        raise TypeError("discard_latest_images must be a boolean.")
    fixed_directory = Path(directory)
    _restore_latest_fixed_evaluation_metrics(
        fixed_directory,
        max_iteration,
    )
    if discard_latest_images:
        for latest_filename in (
            "latest_reconstruction.png",
            "latest_reconstruction_grid.png",
        ):
            latest_path = fixed_directory / latest_filename
            if latest_path.is_symlink():
                raise RuntimeError(
                    "Refusing to remove a symlinked latest fixed-evaluation "
                    f"image: {latest_path}"
                )
            latest_path.unlink(missing_ok=True)


def truncate_jsonl_after_iteration(
    path: str | os.PathLike[str],
    max_iteration: int,
) -> None:
    """Keep the last valid record per iteration through the rollback limit."""

    if (
        isinstance(max_iteration, bool)
        or not isinstance(max_iteration, int)
        or max_iteration < 0
    ):
        raise ValueError("max_iteration must be a nonnegative integer.")
    history_path = Path(path)
    if not history_path.exists():
        return

    records_by_iteration: dict[int, dict[str, Any]] = {}
    for record in read_jsonl(history_path):
        iteration = _record_iteration(record)
        if iteration is None:
            continue
        if iteration <= max_iteration:
            canonical_record = dict(record)
            canonical_record["iteration"] = iteration
            records_by_iteration[iteration] = canonical_record

    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{history_path.name}.",
        suffix=".tmp",
        dir=history_path.parent,
    )
    try:
        handle = os.fdopen(descriptor, "w", encoding="utf-8")
        descriptor = -1
        with handle:
            for iteration in sorted(records_by_iteration):
                handle.write(
                    json.dumps(
                        records_by_iteration[iteration],
                        sort_keys=True,
                        allow_nan=False,
                    )
                    + "\n"
                )
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, history_path)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass


def _finite_number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    numeric_value = float(value)
    if not math.isfinite(numeric_value):
        return None
    return numeric_value


def _record_iteration(record: Mapping[str, Any]) -> int | None:
    iteration = _finite_number(record.get("iteration"))
    if iteration is not None and iteration.is_integer() and iteration >= 0:
        return int(iteration)
    return None


def _series(
    records: list[dict[str, Any]],
    key: str,
) -> tuple[list[float], list[float]]:
    points: list[tuple[float, float]] = []
    for record in records:
        iteration = _record_iteration(record)
        value = _finite_number(record.get(key))
        if value is None and isinstance(record.get("metrics"), Mapping):
            value = _finite_number(record["metrics"].get(key))
        if iteration is not None and value is not None:
            points.append((float(iteration), value))
    points.sort(key=lambda point: point[0])
    return (
        [point[0] for point in points],
        [point[1] for point in points],
    )


def _plot_series(
    axis: Any,
    records: list[dict[str, Any]],
    series: tuple[tuple[str, str], ...],
    title: str,
) -> None:
    plotted_values: list[float] = []
    for key, label in series:
        iterations, values = _series(records, key)
        if values:
            axis.plot(iterations, values, label=label)
            plotted_values.extend(values)
    axis.set_title(title)
    axis.set_xlabel("Optimizer iteration")
    axis.grid(alpha=0.3)
    if not plotted_values:
        axis.text(
            0.5,
            0.5,
            "No data",
            ha="center",
            va="center",
            transform=axis.transAxes,
        )
    else:
        if all(value > 0.0 for value in plotted_values):
            axis.set_yscale("log")
        axis.legend()


def save_training_curves(
    loss_history_path: str | os.PathLike[str],
    evaluation_history_path: str | os.PathLike[str],
    output_path: str | os.PathLike[str],
) -> None:
    """Atomically replace a three-panel PNG of train and fixed-eval curves."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    loss_records = read_jsonl(loss_history_path)
    evaluation_records = read_jsonl(evaluation_history_path)
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)

    figure, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    _plot_series(
        axes[0],
        loss_records,
        (("loss", "loss"),),
        "Training loss",
    )
    has_histogram_metric = any(
        _finite_number(record.get("normalized_histogram_mse")) is not None
        for record in loss_records
    )
    if has_histogram_metric:
        normalized_series = (("normalized_histogram_mse", "normalized histogram MSE"),)
        normalized_title = "Normalized histogram MSE"
    else:
        normalized_series = (("normalized_velocity_mse", "normalized velocity MSE"),)
        normalized_title = "Normalized velocity MSE"
    _plot_series(
        axes[1],
        loss_records,
        normalized_series,
        normalized_title,
    )
    _plot_series(
        axes[2],
        evaluation_records,
        (
            ("sample_nrmse_mean", "NRMSE"),
            ("sample_log_w1_mean", "log-W1"),
            (
                "forward_chi_square_per_measurement_mean",
                "chi-square / measurement",
            ),
        ),
        "Fixed evaluation",
    )
    figure.tight_layout()

    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".png",
        dir=destination.parent,
    )
    os.close(descriptor)
    try:
        figure.savefig(temporary_name, dpi=150, format="png")
        plt.close(figure)
        os.replace(temporary_name, destination)
    finally:
        if plt.fignum_exists(figure.number):
            plt.close(figure)
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass


def save_fm_generalization_curves(
    history_path: str | os.PathLike[str],
    output_path: str | os.PathLike[str],
) -> None:
    """Atomically save fixed train/validation FM metrics and their gaps."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    records = read_jsonl(history_path)
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)

    figure, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    _plot_series(
        axes[0],
        records,
        (
            ("train_probe_loss", "fixed train probe"),
            ("validation_loss", "fixed validation"),
        ),
        "Flow-matching loss",
    )
    _plot_series(
        axes[1],
        records,
        (
            (
                "train_probe_normalized_velocity_mse",
                "fixed train probe",
            ),
            (
                "validation_normalized_velocity_mse",
                "fixed validation",
            ),
        ),
        "Normalized velocity MSE",
    )
    _plot_series(
        axes[2],
        records,
        (
            ("loss_gap", "loss gap"),
            (
                "normalized_velocity_mse_gap",
                "normalized velocity MSE gap",
            ),
        ),
        "Validation - train gap",
    )
    figure.tight_layout()

    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".png",
        dir=destination.parent,
    )
    os.close(descriptor)
    try:
        figure.savefig(temporary_name, dpi=150, format="png")
        plt.close(figure)
        os.replace(temporary_name, destination)
    finally:
        if plt.fignum_exists(figure.number):
            plt.close(figure)
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
