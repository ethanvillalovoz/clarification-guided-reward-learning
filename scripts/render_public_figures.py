"""Regenerate the committed public figure exports from the default simulation."""

from pathlib import Path

from clarification_reward_learning.simulation import run_comparison
from clarification_reward_learning.visualization import save_system_overview

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    outputs = save_system_overview(
        run_comparison(),
        ROOT / "docs" / "media" / "system-overview",
    )
    for output in outputs:
        print(output.relative_to(ROOT))


if __name__ == "__main__":
    main()
