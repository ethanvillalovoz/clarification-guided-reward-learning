import numpy as np
import pytest

from clarification_reward_learning.inference import (
    bayesian_update,
    bradley_terry_probability,
    normalized_entropy,
    uniform_prior,
    update_from_correction,
    update_from_feature_clarification,
)
from clarification_reward_learning.models import default_problem


def test_uniform_prior_is_normalized():
    prior = uniform_prior(4)

    np.testing.assert_allclose(prior, [0.25, 0.25, 0.25, 0.25])
    assert float(prior.sum()) == pytest.approx(1.0)


def test_bradley_terry_prefers_higher_reward():
    preferred = bradley_terry_probability(1.0, 0.0, beta=2.0)
    reversed_preference = bradley_terry_probability(0.0, 1.0, beta=2.0)

    assert preferred > 0.5
    assert reversed_preference < 0.5
    assert preferred + reversed_preference == pytest.approx(1.0)


def test_bayesian_update_rejects_zero_evidence():
    with pytest.raises(ValueError, match="positive"):
        bayesian_update([0.5, 0.5], [0.0, 0.0])


def test_state_correction_increases_compatible_hypotheses():
    hypotheses, objects, true_index = default_problem()
    prior = uniform_prior(len(hypotheses))
    true_action = hypotheses[true_index].preferred_quadrant(objects[0])

    posterior = update_from_correction(
        prior,
        hypotheses,
        objects[0],
        robot_action="Q1",
        human_action=true_action,
    )

    assert posterior[true_index] > prior[true_index]
    assert float(posterior.sum()) == pytest.approx(1.0)


def test_feature_answer_reduces_entropy_and_boosts_exact_structure():
    hypotheses, _objects, true_index = default_problem()
    prior = uniform_prior(len(hypotheses))

    posterior = update_from_feature_clarification(
        prior,
        hypotheses,
        hypotheses[true_index].relevant_features,
    )

    assert posterior[true_index] > prior[true_index]
    assert normalized_entropy(posterior) < normalized_entropy(prior)


def test_feature_answer_is_order_independent():
    hypotheses, _objects, _true_index = default_problem()
    prior = uniform_prior(len(hypotheses))

    expected = update_from_feature_clarification(
        prior,
        hypotheses,
        ["color", "kind"],
    )
    reversed_answer = update_from_feature_clarification(
        prior,
        hypotheses,
        ["kind", "color"],
    )

    np.testing.assert_allclose(reversed_answer, expected)


@pytest.mark.parametrize("features", [[], ["temperature"]])
def test_feature_answer_rejects_invalid_features(features):
    hypotheses, _objects, _true_index = default_problem()

    with pytest.raises(ValueError, match="feature"):
        update_from_feature_clarification(
            uniform_prior(len(hypotheses)),
            hypotheses,
            features,
        )
