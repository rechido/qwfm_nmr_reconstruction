"""Illustrate histogram interpolation and quantile transport without model execution."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def transport_paths() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return matched Gaussian endpoints and the two paths on the same grid."""
    position = np.linspace(-2.1, 2.1, 600)
    flow_time = np.linspace(0.0, 1.0, 121)
    source_mean, target_mean, width = -1.05, 1.05, 0.18

    def gaussian(mean):
        return np.exp(-0.5 * ((position - mean) / width) ** 2) / (
            np.sqrt(2 * np.pi) * width)

    source, target = gaussian(source_mean), gaussian(target_mean)
    histogram = ((1 - flow_time[:, None]) * source[None, :]
                 + flow_time[:, None] * target[None, :])
    quantile = np.stack([gaussian((1 - t) * source_mean + t * target_mean)
                         for t in flow_time])
    integrate = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
    for path in (histogram, quantile):
        path /= integrate(path, position, axis=1)[:, None]
    return position, flow_time, histogram, quantile


def main(argv: list[str] | None = None) -> dict:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output_dir", type=Path,
                        default=Path("outputs/visualizations/concept"))
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args(argv)
    if args.dpi < 1:
        parser.error("--dpi must be positive.")
    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    position, flow_time, histogram, quantile = transport_paths()
    with plt.rc_context({"pdf.fonttype": 42, "ps.fonttype": 42}):
        figure, axes = plt.subplots(1, 2, figsize=(7.1, 3.0), sharey=True)
        figure.subplots_adjust(left=.10, right=.99, bottom=.22, top=.86, wspace=.09)
        maximum = max(histogram.max(), quantile.max())
        for axis, path, title in (
            (axes[0], histogram, "(a) Histogram interpolation"),
            (axes[1], quantile, "(b) Quantile transport"),
        ):
            axis.imshow(path, aspect="auto", origin="lower",
                        extent=(position[0], position[-1], flow_time[0], flow_time[-1]),
                        cmap="magma", vmin=0, vmax=maximum, rasterized=True)
            axis.set_title(title, fontsize=11)
            axis.set_xlabel("Log-rate coordinate", fontsize=10)
            axis.set_xticks([-2, 0, 2])
            axis.set_yticks([0, .5, 1])
        axes[0].set_ylabel("Flow time", fontsize=10)
        files = {}
        for extension in ("png", "pdf"):
            path = output / f"transport_paths.{extension}"
            figure.savefig(path, dpi=args.dpi, facecolor="white")
            files[extension] = str(path)
        plt.close(figure)
    record = {
        "protocol": "histogram_and_quantile_transport_illustration_v1",
        "files": files,
        "source_mean": -1.05, "target_mean": 1.05, "standard_deviation": .18,
        "position_count": len(position), "flow_time_count": len(flow_time),
        "description": "Fixed-bin mixtures change endpoint weights; quantile interpolation "
                       "translates a Gaussian profile between the same endpoints.",
        "model_execution": False, "experimental_results": False,
    }
    (output / "transport_paths.json").write_text(json.dumps(record, indent=2) + "\n")
    print(f"Saved transport-path illustration to {output}")
    return record


if __name__ == "__main__":
    main()
