from collections.abc import Sequence
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyArrowPatch, Rectangle
from PIL import Image

from .simulation import ExperimentResult

INK = "#1f2321"
MUTED = "#68716b"
GREEN = "#426f60"
AMBER = "#a66a34"
PAPER = "#f4f5f2"
LINE = "#d8ddd8"
COBALT = "#315bd6"
CORAL = "#dd654f"
YELLOW = "#e6c94d"
VIOLET = "#8b70c9"


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


def _draw_cup(axis, center: tuple[float, float], *, alpha: float = 1.0) -> None:
    x, y = center
    axis.add_patch(
        Rectangle(
            (x - 0.12, y - 0.12),
            0.24,
            0.24,
            facecolor=YELLOW,
            edgecolor=INK,
            linewidth=1.4,
            alpha=alpha,
        )
    )
    axis.add_patch(
        Circle(
            (x + 0.16, y),
            0.075,
            facecolor="none",
            edgecolor=INK,
            linewidth=1.4,
            alpha=alpha,
        )
    )


def save_reasoning_snapshot(result: ExperimentResult, output_path: Path) -> None:
    """Render one correction and clarification update as an inspectable research figure."""
    _style()
    step = next((candidate for candidate in result.steps if candidate.corrected), result.steps[0])
    true_index = result.true_hypothesis_index

    figure = plt.figure(figsize=(14.2, 7.2), constrained_layout=True)
    grid = figure.add_gridspec(1, 3, width_ratios=(1.0, 0.82, 1.35))
    scene_axis = figure.add_subplot(grid[0, 0])
    explanation_axis = figure.add_subplot(grid[0, 1])
    belief_axis = figure.add_subplot(grid[0, 2])

    figure.suptitle(
        "A correction narrows the reward hypothesis; clarification sharpens it",
        x=0.03,
        ha="left",
        fontsize=18,
        fontweight="bold",
    )

    scene_axis.set_title(step.object_label, loc="left", fontsize=13, pad=14)
    scene_axis.set_xlim(0, 2)
    scene_axis.set_ylim(0, 2)
    scene_axis.set_aspect("equal")
    scene_axis.axis("off")
    quadrant_centers = {
        "Q1": (0.5, 1.5),
        "Q2": (1.5, 1.5),
        "Q3": (0.5, 0.5),
        "Q4": (1.5, 0.5),
    }
    for quadrant, (x, y) in quadrant_centers.items():
        scene_axis.add_patch(
            Rectangle(
                (x - 0.46, y - 0.46),
                0.92,
                0.92,
                facecolor="#ebe9e1",
                edgecolor=LINE,
                linewidth=1.2,
            )
        )
        scene_axis.text(x - 0.38, y + 0.34, quadrant, color=MUTED, fontsize=9)

    robot_center = quadrant_centers[step.robot_action]
    human_center = quadrant_centers[step.human_action]
    _draw_cup(scene_axis, robot_center, alpha=0.35)
    _draw_cup(scene_axis, human_center)
    scene_axis.add_patch(
        FancyArrowPatch(
            robot_center,
            human_center,
            arrowstyle="-|>",
            mutation_scale=15,
            linewidth=2,
            color=CORAL,
            connectionstyle="arc3,rad=-0.18",
        )
    )
    scene_axis.text(
        0.04,
        -0.12,
        f"robot: {step.robot_action}   human correction: {step.human_action}",
        transform=scene_axis.transAxes,
        color=MUTED,
        fontsize=9,
    )

    explanation_axis.axis("off")
    explanation_axis.set_xlim(0, 1)
    explanation_axis.set_ylim(0, 1)
    explanation_axis.text(0, 0.91, "Observed signal", color=COBALT, fontsize=9, fontweight="bold")
    explanation_axis.text(0, 0.83, "State correction", fontsize=15, fontweight="bold")
    explanation_axis.text(
        0,
        0.74,
        "The corrected state is compatible with\nseveral reward explanations.",
        color=MUTED,
        fontsize=10,
        linespacing=1.45,
    )
    explanation_axis.plot([0, 1], [0.62, 0.62], color=LINE, linewidth=1)
    explanation_axis.text(0, 0.54, "Follow-up question", color=CORAL, fontsize=9, fontweight="bold")
    explanation_axis.text(0, 0.46, "Which features mattered?", fontsize=14, fontweight="bold")
    chip_x = 0.0
    chip_labels = [feature.replace("kind", "object type") for feature in step.clarification_answer]
    for index, label in enumerate(chip_labels):
        width = 0.27 if len(label) < 8 else 0.42
        explanation_axis.add_patch(
            Rectangle(
                (chip_x, 0.31),
                width,
                0.08,
                facecolor="#fff",
                edgecolor=VIOLET if index else COBALT,
                linewidth=1.4,
            )
        )
        explanation_axis.text(chip_x + width / 2, 0.35, label, ha="center", va="center", fontsize=9)
        chip_x += width + 0.05
    explanation_axis.text(
        0,
        0.18,
        f"entropy {step.entropy_after_correction:.2f}  ->  {step.entropy_after_clarification:.2f}",
        color=MUTED,
        fontsize=10,
    )

    y = np.arange(len(result.hypothesis_names))
    height = 0.34
    belief_axis.barh(
        y - height / 2,
        step.after_correction,
        height,
        color="#c8c9c4",
        label="after correction",
    )
    posterior_colors = [COBALT if index == true_index else VIOLET for index in y]
    belief_axis.barh(
        y + height / 2,
        step.posterior,
        height,
        color=posterior_colors,
        label="after clarification",
    )
    belief_axis.set_yticks(y, result.hypothesis_names)
    belief_axis.invert_yaxis()
    belief_axis.set_xlim(0, 1)
    belief_axis.set_xlabel("posterior probability")
    belief_axis.set_title("Belief update", loc="left", fontsize=13, pad=14)
    belief_axis.grid(axis="x", color=LINE, linewidth=0.8)
    belief_axis.spines[["top", "right", "left"]].set_visible(False)
    belief_axis.legend(frameon=False, loc="lower right", fontsize=9)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=170, bbox_inches="tight")
    plt.close(figure)


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
        frames.append(frame)
        plt.close(figure)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=[1400] + [1150] * (len(frames) - 1),
        loop=0,
        quality=80,
        method=6,
    )
