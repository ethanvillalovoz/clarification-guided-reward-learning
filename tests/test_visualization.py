from pathlib import Path

from PIL import Image

from clarification_reward_learning.simulation import run_comparison
from clarification_reward_learning.visualization import (
    _OVERVIEW_QUADRANT_CENTERS,
    save_system_overview,
)


def test_overview_uses_standard_cartesian_quadrant_layout():
    q1 = _OVERVIEW_QUADRANT_CENTERS["Q1"]
    q2 = _OVERVIEW_QUADRANT_CENTERS["Q2"]
    q3 = _OVERVIEW_QUADRANT_CENTERS["Q3"]
    q4 = _OVERVIEW_QUADRANT_CENTERS["Q4"]

    assert q1[0] > q2[0] and q1[1] == q2[1]
    assert q3[0] < q4[0] and q3[1] == q4[1]
    assert q1[1] > q4[1] and q2[1] > q3[1]


def test_system_overview_exports_vector_and_raster_formats(tmp_path: Path):
    outputs = save_system_overview(run_comparison(), tmp_path / "system-overview")

    assert [path.suffix for path in outputs] == [".svg", ".pdf", ".png"]
    assert all(path.is_file() and path.stat().st_size > 0 for path in outputs)
    assert outputs[1].read_bytes().startswith(b"%PDF")

    svg = outputs[0].read_text(encoding="utf-8")
    assert "Entropy-gated clarification" in svg
    assert "no statistical uncertainty" in svg

    with Image.open(outputs[2]) as image:
        assert image.size == (2148, 1680)
