import json
from pathlib import Path

import pytest

from clarification_reward_learning.simulation import run_comparison, run_simulation


def test_default_simulation_is_deterministic_and_normalized():
    first = run_simulation()
    second = run_simulation()

    assert first.to_dict() == second.to_dict()
    assert sum(first.final_beliefs) == pytest.approx(1.0)
    assert len(first.steps) == 3


def test_clarification_condition_is_compared_against_same_correction_trace():
    comparison = run_comparison()
    correction_only = comparison["correction_only"]
    clarification = comparison["with_clarification"]

    assert [step.human_action for step in correction_only.steps] == [
        step.human_action for step in clarification.steps
    ]
    assert clarification.final_true_posterior > correction_only.final_true_posterior
    assert clarification.steps[-1].entropy_after_clarification < (
        correction_only.steps[-1].entropy_after_clarification
    )


def test_invalid_true_hypothesis_is_rejected():
    with pytest.raises(ValueError, match="out of range"):
        run_simulation(true_hypothesis_index=99)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"hypotheses": []}, "hypothesis"),
        ({"objects": []}, "object"),
        ({"clarification_threshold": 1.1}, "threshold"),
    ],
)
def test_invalid_simulation_configuration_is_rejected(kwargs, message):
    with pytest.raises(ValueError, match=message):
        run_simulation(**kwargs)


def test_default_configuration_matches_committed_reference_trace():
    reference_path = Path(__file__).resolve().parents[1] / "examples" / "reference-trace.json"
    expected = json.loads(reference_path.read_text())
    actual = {name: result.to_dict() for name, result in run_comparison().items()}

    assert actual == expected
