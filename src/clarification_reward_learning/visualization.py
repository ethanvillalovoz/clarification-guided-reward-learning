from collections.abc import Sequence
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from .simulation import ExperimentResult

INK = "#1f2321"
MUTED = "#68716b"
GREEN = "#426f60"
AMBER = "#a66a34"
PAPER = "#f4f5f2"
LINE = "#d8ddd8"


def _style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "figure.facecolor": PAPER,
            "axes.facecolor": PAPER,
            "axes.edgecolor": LINE,
            "axes.labelcolor": MUTED,
            "text.color": INK,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "axes.titleweight": "bold",
        }
    )


def save_comparison_plot(comparison: dict[str, ExperimentResult], output_path: Path) -> None:
    _style()
    correction_only = comparison["correction_only"]
    clarification = comparison["with_clarification"]
    x = np.arange(len(clarification.steps) + 1)
    x_labels = ["prior", *[f"object {step.timestep + 1}" for step in clarification.steps]]

    def true_trajectory(result: ExperimentResult) -> list[float]:
        start = 1.0 / len(result.hypothesis_names)
        return [start, *[step.posterior[result.true_hypothesis_index] for step in result.steps]]

    def entropy_trajectory(result: ExperimentResult) -> list[float]:
        return [1.0, *[step.entropy_after_clarification for step in result.steps]]

    figure, axes = plt.subplots(1, 2, figsize=(11, 4.4), constrained_layout=True)
    for result, color, marker in [
        (correction_only, AMBER, "o"),
        (clarification, GREEN, "s"),
    ]:
        axes[0].plot(
            x,
            true_trajectory(result),
            color=color,
            marker=marker,
            linewidth=2,
            label=result.condition,
        )
        axes[1].plot(
            x,
            entropy_trajectory(result),
            color=color,
            marker=marker,
            linewidth=2,
            label=result.condition,
        )

    axes[0].set_title("Posterior on the designated true hypothesis", loc="left")
    axes[0].set_ylabel("posterior probability")
    axes[0].set_ylim(0, 1.02)
    axes[1].set_title("Normalized hypothesis entropy", loc="left")
    axes[1].set_ylabel("normalized entropy")
    axes[1].set_ylim(0, 1.02)
    for axis in axes:
        axis.set_xticks(x, x_labels)
        axis.grid(axis="y", color=LINE, linewidth=0.8)
        axis.spines[["top", "right"]].set_visible(False)
        axis.legend(frameon=False, fontsize=9)

    figure.suptitle("Illustrative simulation trace", fontsize=15, fontweight="bold")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def _belief_states(result: ExperimentResult) -> list[tuple[str, Sequence[float]]]:
    states: list[tuple[str, Sequence[float]]] = [
        ("Uniform prior", [1.0 / len(result.hypothesis_names)] * len(result.hypothesis_names))
    ]
    for step in result.steps:
        states.append(
            (f"Object {step.timestep + 1}: after state correction", step.after_correction)
        )
        if step.clarification_asked:
            states.append(
                (f"Object {step.timestep + 1}: after feature clarification", step.posterior)
            )
    return states


def save_belief_animation(result: ExperimentResult, output_path: Path) -> None:
    _style()
    frames = []
    true_index = result.true_hypothesis_index
    for title, beliefs in _belief_states(result):
        figure, axis = plt.subplots(figsize=(9.6, 5.4), constrained_layout=True)
        colors = [GREEN if index == true_index else "#b8c0ba" for index in range(len(beliefs))]
        axis.barh(result.hypothesis_names, beliefs, color=colors, edgecolor="none")
        axis.set_xlim(0, 1)
        axis.set_xlabel("belief probability")
        axis.set_title(title, loc="left", fontsize=15)
        axis.grid(axis="x", color=LINE, linewidth=0.8)
        axis.spines[["top", "right", "left"]].set_visible(False)
        axis.text(
            1.0,
            -0.16,
            "Green denotes the designated true hypothesis in this simulation.",
            transform=axis.transAxes,
            ha="right",
            color=MUTED,
            fontsize=8,
        )
        figure.canvas.draw()
        width, height = figure.canvas.get_width_height()
        rgba = np.asarray(figure.canvas.buffer_rgba()).copy()
        frame = Image.fromarray(rgba).convert("RGB")
        frames.append(frame.convert("P", palette=Image.Palette.ADAPTIVE, colors=128))
        plt.close(figure)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=[1400] + [1150] * (len(frames) - 1),
        loop=0,
        optimize=True,
    )
