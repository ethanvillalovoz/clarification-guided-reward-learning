from collections.abc import Sequence
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle
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

_OVERVIEW_QUADRANT_CENTERS = {
    "Q1": (0.40, 0.56),  # upper-right
    "Q2": (0.16, 0.56),  # upper-left
    "Q3": (0.16, 0.30),  # lower-left
    "Q4": (0.40, 0.30),  # lower-right
}


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
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
            "svg.hashsalt": "clarification-guided-reward-learning",
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


def _draw_cup(
    axis,
    center: tuple[float, float],
    *,
    alpha: float = 1.0,
    scale: float = 1.0,
) -> None:
    x, y = center
    axis.add_patch(
        Rectangle(
            (x - 0.12 * scale, y - 0.12 * scale),
            0.24 * scale,
            0.24 * scale,
            facecolor=YELLOW,
            edgecolor=INK,
            linewidth=1.4 * scale,
            alpha=alpha,
        )
    )
    axis.add_patch(
        Circle(
            (x + 0.16 * scale, y),
            0.075 * scale,
            facecolor="none",
            edgecolor=INK,
            linewidth=1.4 * scale,
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


def _panel_background(axis, *, edgecolor: str = LINE) -> None:
    axis.add_patch(
        FancyBboxPatch(
            (0.01, 0.01),
            0.98,
            0.98,
            boxstyle="round,pad=0.012,rounding_size=0.035",
            facecolor="#fbfcfa",
            edgecolor=edgecolor,
            linewidth=0.75,
            transform=axis.transAxes,
            clip_on=False,
            zorder=-10,
        )
    )


def _panel_title(axis, label: str, title: str, subtitle: str) -> None:
    axis.text(
        0.035,
        0.94,
        label,
        transform=axis.transAxes,
        ha="left",
        va="top",
        color=COBALT,
        fontsize=9.2,
        fontweight="bold",
        clip_on=True,
    )
    axis.text(
        0.105,
        0.94,
        title,
        transform=axis.transAxes,
        ha="left",
        va="top",
        color=INK,
        fontsize=9.2,
        fontweight="bold",
        clip_on=True,
    )
    axis.text(
        0.035,
        0.79,
        subtitle,
        transform=axis.transAxes,
        ha="left",
        va="top",
        color=MUTED,
        fontsize=7.2,
        linespacing=1.2,
        clip_on=True,
    )


def _belief_panel(
    axis,
    *,
    values: Sequence[float],
    names: Sequence[str],
    true_index: int,
    entropy: float,
    label: str,
    title: str,
    subtitle: str,
    accent: str,
) -> None:
    _panel_background(axis)
    _panel_title(axis, label, title, subtitle)
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.axis("off")

    bar_left = 0.43
    bar_span = 0.41
    scale_max = 0.65
    row_centers = np.linspace(0.66, 0.20, len(values))
    bar_height = 0.055
    for tick in (0.0, 0.3, 0.6):
        x = bar_left + bar_span * tick / scale_max
        axis.plot(
            [x, x],
            [0.15, 0.70],
            color=LINE,
            linewidth=0.55,
            transform=axis.transAxes,
            zorder=0,
        )
    for index, (position, value, name) in enumerate(zip(row_centers, values, names, strict=True)):
        width = bar_span * value / scale_max
        is_true = index == true_index
        axis.text(
            0.035,
            position,
            name,
            transform=axis.transAxes,
            va="center",
            color=INK if is_true else MUTED,
            fontsize=7.0,
            fontweight="bold" if is_true else "normal",
        )
        axis.add_patch(
            Rectangle(
                (bar_left, position - bar_height / 2),
                width,
                bar_height,
                transform=axis.transAxes,
                facecolor=accent if is_true else "#c8cdd0",
                edgecolor=COBALT if is_true else "none",
                linewidth=0.8,
                zorder=2,
            )
        )
        axis.text(
            min(bar_left + width + 0.012, 0.94),
            position,
            f"{value:.3f}",
            transform=axis.transAxes,
            va="center",
            color=INK if is_true else MUTED,
            fontsize=7.0,
            fontweight="bold" if is_true else "normal",
        )
    axis.text(
        bar_left,
        0.085,
        "posterior probability",
        transform=axis.transAxes,
        ha="left",
        color=MUTED,
        fontsize=6.5,
    )
    axis.text(
        0.955,
        0.085,
        f"H = {entropy:.3f}",
        transform=axis.transAxes,
        ha="right",
        va="center",
        color=MUTED,
        fontsize=7.2,
        fontweight="bold",
    )


def _small_action_box(
    axis,
    x: float,
    y: float,
    label: str,
    quadrant: str,
    color: str,
) -> None:
    axis.add_patch(
        FancyBboxPatch(
            (x, y),
            0.135,
            0.31,
            boxstyle="round,pad=0.012,rounding_size=0.02",
            facecolor="white",
            edgecolor=color,
            linewidth=1.0,
            transform=axis.transAxes,
        )
    )
    axis.text(
        x + 0.0675,
        y + 0.225,
        label,
        transform=axis.transAxes,
        ha="center",
        va="center",
        color=MUTED,
        fontsize=7.0,
    )
    axis.text(
        x + 0.0675,
        y + 0.10,
        quadrant,
        transform=axis.transAxes,
        ha="center",
        va="center",
        color=color,
        fontsize=10.0,
        fontweight="bold",
    )


def _validate_text_insets(figure, axes: Sequence, *, minimum_points: float = 3.0) -> None:
    """Reject exports whose panel text crosses the requested axes inset."""
    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    minimum_pixels = minimum_points * figure.dpi / 72.0
    for axis in axes:
        boundary = axis.get_window_extent(renderer)
        for artist in axis.texts:
            if not artist.get_visible() or not artist.get_text().strip():
                continue
            bounds = artist.get_window_extent(renderer)
            if (
                bounds.x0 < boundary.x0 + minimum_pixels
                or bounds.x1 > boundary.x1 - minimum_pixels
                or bounds.y0 < boundary.y0 + minimum_pixels
                or bounds.y1 > boundary.y1 - minimum_pixels
            ):
                msg = f"Panel text violates the minimum inset: {artist.get_text()!r}"
                raise ValueError(msg)


def save_system_overview(
    comparison: dict[str, ExperimentResult],
    output_stem: Path,
) -> list[Path]:
    """Render a wide, evidence-backed overview of the default simulation mechanism."""
    _style()
    correction_only = comparison["correction_only"]
    clarification = comparison["with_clarification"]
    correction_step = clarification.steps[0]
    baseline_next = correction_only.steps[1]
    clarified_next = clarification.steps[1]
    true_index = clarification.true_hypothesis_index

    figure = plt.figure(figsize=(7.16, 5.60), facecolor="white")
    left = 0.018
    right = 0.517
    panel_width = 0.465
    correction_axis = figure.add_axes((left, 0.695, panel_width, 0.285))
    before_axis = figure.add_axes((right, 0.695, panel_width, 0.285))
    question_axis = figure.add_axes((left, 0.390, panel_width, 0.275))
    after_axis = figure.add_axes((right, 0.390, panel_width, 0.275))
    outcome_axis = figure.add_axes((left, 0.085, 0.964, 0.275))

    # A: the observed state correction.
    _panel_background(correction_axis)
    _panel_title(
        correction_axis,
        "A",
        "Observed state correction",
        correction_step.object_label,
    )
    correction_axis.set_xlim(0, 1)
    correction_axis.set_ylim(0, 1)
    correction_axis.axis("off")
    centers = _OVERVIEW_QUADRANT_CENTERS
    for quadrant, (x, y) in centers.items():
        correction_axis.add_patch(
            Rectangle(
                (x - 0.115, y - 0.115),
                0.23,
                0.23,
                facecolor="#ebece8",
                edgecolor=LINE,
                linewidth=0.7,
            )
        )
        correction_axis.text(
            x - 0.095,
            y + 0.075,
            quadrant,
            color=MUTED,
            fontsize=7.2,
            fontweight="bold",
        )
    robot_center = centers[correction_step.robot_action]
    human_center = centers[correction_step.human_action]
    _draw_cup(correction_axis, robot_center, alpha=0.28, scale=0.38)
    _draw_cup(correction_axis, human_center, scale=0.38)
    correction_axis.add_patch(
        FancyArrowPatch(
            robot_center,
            human_center,
            arrowstyle="-|>",
            mutation_scale=10,
            linewidth=1.6,
            color=CORAL,
            connectionstyle="arc3,rad=-0.18",
        )
    )
    correction_axis.text(
        0.60,
        0.62,
        "ROBOT PLACEMENT",
        transform=correction_axis.transAxes,
        ha="left",
        color=MUTED,
        fontsize=6.5,
        fontweight="bold",
    )
    _draw_cup(correction_axis, (0.62, 0.51), alpha=0.28, scale=0.34)
    correction_axis.text(
        0.70,
        0.51,
        correction_step.robot_action,
        ha="left",
        va="center",
        color=INK,
        fontsize=9.0,
        fontweight="bold",
    )
    correction_axis.text(
        0.60,
        0.35,
        "HUMAN CORRECTION",
        transform=correction_axis.transAxes,
        ha="left",
        color=MUTED,
        fontsize=6.5,
        fontweight="bold",
    )
    _draw_cup(correction_axis, (0.62, 0.24), scale=0.34)
    correction_axis.text(
        0.70,
        0.24,
        correction_step.human_action,
        ha="left",
        va="center",
        color=CORAL,
        fontsize=9.0,
        fontweight="bold",
    )

    # B and D: aligned posterior displays before and after clarification.
    _belief_panel(
        before_axis,
        values=correction_step.after_correction,
        names=clarification.hypothesis_names,
        true_index=true_index,
        entropy=correction_step.entropy_after_correction,
        label="B",
        title="Correction remains ambiguous",
        subtitle="Several reward hypotheses predict this correction.",
        accent="#8eadd8",
    )
    _belief_panel(
        after_axis,
        values=correction_step.posterior,
        names=clarification.hypothesis_names,
        true_index=true_index,
        entropy=correction_step.entropy_after_clarification,
        label="D",
        title="Belief after clarification",
        subtitle="The answer favors the designated true hypothesis.",
        accent=COBALT,
    )

    # C: the entropy-gated feature question.
    _panel_background(question_axis)
    _panel_title(
        question_axis,
        "C",
        "Entropy-gated clarification",
        (
            f"H = {correction_step.entropy_after_correction:.3f} ≥ "
            f"τ = {clarification.clarification_threshold:.2f}, so the robot asks."
        ),
    )
    question_axis.axis("off")
    question_axis.add_patch(
        FancyBboxPatch(
            (0.055, 0.43),
            0.48,
            0.25,
            boxstyle="round,pad=0.025,rounding_size=0.045",
            facecolor="white",
            edgecolor=INK,
            linewidth=1.0,
            transform=question_axis.transAxes,
        )
    )
    question_axis.text(
        0.295,
        0.555,
        "Which features\nmattered?",
        transform=question_axis.transAxes,
        ha="center",
        va="center",
        color=INK,
        fontsize=9.0,
        fontweight="bold",
    )
    display_features = [
        "object type" if feature == "kind" else feature
        for feature in correction_step.clarification_answer
    ]
    question_axis.text(
        0.775,
        0.66,
        "HUMAN ANSWER",
        transform=question_axis.transAxes,
        ha="center",
        color=MUTED,
        fontsize=6.5,
        fontweight="bold",
    )
    chip_specs = [
        (0.59, 0.47, 0.15, display_features[0]),
        (0.77, 0.47, 0.19, display_features[1]),
    ]
    for x, y, width, feature in chip_specs:
        question_axis.add_patch(
            FancyBboxPatch(
                (x, y),
                width,
                0.12,
                boxstyle="round,pad=0.01,rounding_size=0.025",
                facecolor="#edf2ff",
                edgecolor=COBALT,
                linewidth=0.8,
                transform=question_axis.transAxes,
            )
        )
        question_axis.text(
            x + width / 2,
            y + 0.06,
            feature,
            transform=question_axis.transAxes,
            ha="center",
            va="center",
            color=COBALT,
            fontsize=7.0,
            fontweight="bold",
        )
    question_axis.text(
        0.775,
        0.31,
        f"answer likelihood = {clarification.clarification_accuracy:.1f}",
        transform=question_axis.transAxes,
        ha="center",
        color=MUTED,
        fontsize=7.0,
    )

    # E: a directly observed consequence on the next object in the same trace.
    _panel_background(outcome_axis)
    outcome_axis.axis("off")
    outcome_axis.text(
        0.018,
        0.94,
        "E",
        transform=outcome_axis.transAxes,
        va="top",
        color=COBALT,
        fontsize=9.2,
        fontweight="bold",
    )
    outcome_axis.text(
        0.055,
        0.94,
        "Consequence on the next object",
        transform=outcome_axis.transAxes,
        va="top",
        color=INK,
        fontsize=9.2,
        fontweight="bold",
    )
    outcome_axis.text(
        0.055,
        0.76,
        f"Same trace, next object: {baseline_next.object_label}",
        transform=outcome_axis.transAxes,
        va="top",
        color=MUTED,
        fontsize=7.2,
    )
    outcome_axis.plot(
        [0.50, 0.50],
        [0.13, 0.69],
        transform=outcome_axis.transAxes,
        color=LINE,
        linewidth=0.8,
    )
    outcome_axis.text(
        0.25,
        0.61,
        "CORRECTION ONLY",
        transform=outcome_axis.transAxes,
        ha="center",
        color=AMBER,
        fontsize=7.2,
        fontweight="bold",
    )
    _small_action_box(
        outcome_axis,
        0.12,
        0.22,
        "robot",
        baseline_next.robot_action,
        AMBER,
    )
    outcome_axis.annotate(
        "",
        xy=(0.315, 0.375),
        xytext=(0.265, 0.375),
        xycoords="axes fraction",
        arrowprops={"arrowstyle": "-|>", "color": CORAL, "linewidth": 1.2},
    )
    _small_action_box(
        outcome_axis,
        0.325,
        0.22,
        "human",
        baseline_next.human_action,
        CORAL,
    )
    outcome_axis.text(
        0.29,
        0.10,
        "✕  another correction",
        transform=outcome_axis.transAxes,
        ha="center",
        color=CORAL,
        fontsize=7.2,
        fontweight="bold",
    )

    outcome_axis.text(
        0.75,
        0.61,
        "WITH CLARIFICATION",
        transform=outcome_axis.transAxes,
        ha="center",
        color=GREEN,
        fontsize=7.2,
        fontweight="bold",
    )
    _small_action_box(
        outcome_axis,
        0.65,
        0.22,
        "robot",
        clarified_next.robot_action,
        GREEN,
    )
    outcome_axis.text(
        0.81,
        0.375,
        f"= human {clarified_next.human_action}",
        transform=outcome_axis.transAxes,
        ha="left",
        va="center",
        color=GREEN,
        fontsize=8.0,
        fontweight="bold",
    )
    outcome_axis.text(
        0.80,
        0.10,
        "✓  accepted without correction",
        transform=outcome_axis.transAxes,
        ha="center",
        color=GREEN,
        fontsize=7.2,
        fontweight="bold",
    )

    figure.text(
        0.50,
        0.025,
        (
            "Illustrative deterministic trace  •  6 hand-authored hypotheses  •  "
            "3 objects  •  β=2.0  •  no statistical uncertainty"
        ),
        ha="center",
        color=MUTED,
        fontsize=7.0,
    )

    _validate_text_insets(
        figure,
        [correction_axis, before_axis, question_axis, after_axis, outcome_axis],
    )

    output_stem.parent.mkdir(parents=True, exist_ok=True)
    outputs = [output_stem.with_suffix(suffix) for suffix in (".svg", ".pdf", ".png")]
    for path in outputs:
        metadata = {"Title": "Clarification-guided reward learning system overview"}
        if path.suffix == ".pdf":
            metadata.update(
                {
                    "Author": "Ethan Villalovoz",
                    "CreationDate": None,
                    "ModDate": None,
                }
            )
        elif path.suffix == ".svg":
            metadata["Creator"] = "clarification-reward-demo"
            metadata["Date"] = None
        figure.savefig(
            path,
            dpi=300 if path.suffix == ".png" else None,
            facecolor="white",
            metadata=metadata,
        )
        if path.suffix == ".svg":
            lines = path.read_text(encoding="utf-8").splitlines()
            path.write_text("\n".join(line.rstrip() for line in lines) + "\n", encoding="utf-8")
    plt.close(figure)
    return outputs
