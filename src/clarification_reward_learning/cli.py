import argparse
import json
from pathlib import Path

from .simulation import run_comparison
from .visualization import (
    save_belief_animation,
    save_comparison_plot,
    save_reasoning_snapshot,
    save_system_overview,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the clarification-guided reward-learning reference simulation."
    )
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/latest"))
    parser.add_argument("--beta", type=float, default=2.0)
    parser.add_argument("--clarification-accuracy", type=float, default=0.8)
    parser.add_argument("--clarification-threshold", type=float, default=0.55)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    comparison = run_comparison(
        beta=args.beta,
        clarification_accuracy=args.clarification_accuracy,
        clarification_threshold=args.clarification_threshold,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    payload = {name: result.to_dict() for name, result in comparison.items()}
    (args.output_dir / "trace.json").write_text(json.dumps(payload, indent=2) + "\n")
    save_comparison_plot(comparison, args.output_dir / "comparison.png")
    save_reasoning_snapshot(
        comparison["with_clarification"],
        args.output_dir / "reasoning-snapshot.png",
    )
    save_belief_animation(
        comparison["with_clarification"],
        args.output_dir / "belief-update.webp",
    )
    save_system_overview(comparison, args.output_dir / "system-overview")

    correction_only = comparison["correction_only"].final_true_posterior
    clarification = comparison["with_clarification"].final_true_posterior
    print("Illustrative simulation complete")
    print(f"  correction only final posterior: {correction_only:.4f}")
    print(f"  with clarification final posterior: {clarification:.4f}")
    print(f"  artifacts: {args.output_dir.resolve()}")
