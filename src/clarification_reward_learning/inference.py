import math
from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

from .models import QUADRANTS, VALID_FEATURES, ObjectSpec, Quadrant, RewardHypothesis

Beliefs = NDArray[np.float64]


def normalize(weights: Sequence[float]) -> Beliefs:
    values = np.asarray(weights, dtype=float)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("Belief weights must be a non-empty vector")
    if not np.all(np.isfinite(values)) or np.any(values < 0):
        raise ValueError("Belief weights must be finite and non-negative")
    total = float(values.sum())
    if total <= 0:
        raise ValueError("At least one belief weight must be positive")
    return values / total


def uniform_prior(hypothesis_count: int) -> Beliefs:
    if hypothesis_count <= 0:
        raise ValueError("Hypothesis count must be positive")
    return np.full(hypothesis_count, 1.0 / hypothesis_count, dtype=float)


def bradley_terry_probability(
    preferred_reward: float,
    alternative_reward: float,
    beta: float,
) -> float:
    if beta <= 0:
        raise ValueError("Beta must be positive")
    difference = beta * (preferred_reward - alternative_reward)
    if difference >= 0:
        return 1.0 / (1.0 + math.exp(-difference))
    exponent = math.exp(difference)
    return exponent / (1.0 + exponent)


def bayesian_update(prior: Sequence[float], likelihoods: Sequence[float]) -> Beliefs:
    prior_values = normalize(prior)
    likelihood_values = np.asarray(likelihoods, dtype=float)
    if likelihood_values.shape != prior_values.shape:
        raise ValueError("Likelihood vector must match the prior")
    if not np.all(np.isfinite(likelihood_values)) or np.any(likelihood_values < 0):
        raise ValueError("Likelihoods must be finite and non-negative")
    return normalize(prior_values * likelihood_values)


def correction_likelihoods(
    hypotheses: Sequence[RewardHypothesis],
    object_spec: ObjectSpec,
    robot_action: Quadrant,
    human_action: Quadrant,
    beta: float = 2.0,
) -> Beliefs:
    likelihoods = []
    for hypothesis in hypotheses:
        human_reward = hypothesis.reward(object_spec, human_action)
        if robot_action != human_action:
            robot_reward = hypothesis.reward(object_spec, robot_action)
            likelihood = bradley_terry_probability(human_reward, robot_reward, beta)
        else:
            rewards = np.array(
                [hypothesis.reward(object_spec, action) for action in QUADRANTS],
                dtype=float,
            )
            logits = beta * rewards
            logits -= float(logits.max())
            probabilities = np.exp(logits) / np.exp(logits).sum()
            likelihood = float(probabilities[QUADRANTS.index(human_action)])
        likelihoods.append(likelihood)
    return np.asarray(likelihoods, dtype=float)


def update_from_correction(
    prior: Sequence[float],
    hypotheses: Sequence[RewardHypothesis],
    object_spec: ObjectSpec,
    robot_action: Quadrant,
    human_action: Quadrant,
    beta: float = 2.0,
) -> Beliefs:
    return bayesian_update(
        prior,
        correction_likelihoods(
            hypotheses,
            object_spec,
            robot_action,
            human_action,
            beta,
        ),
    )


def update_from_feature_clarification(
    prior: Sequence[float],
    hypotheses: Sequence[RewardHypothesis],
    observed_features: Sequence[str],
    answer_accuracy: float = 0.8,
) -> Beliefs:
    if not 0.5 < answer_accuracy <= 1.0:
        raise ValueError("Clarification accuracy must be in (0.5, 1.0]")
    observed = frozenset(observed_features)
    if not observed:
        raise ValueError("A clarification answer must include at least one feature")
    if not observed.issubset(VALID_FEATURES):
        raise ValueError("Clarification answer contains an unsupported feature")
    likelihoods = [
        answer_accuracy
        if frozenset(hypothesis.relevant_features) == observed
        else 1.0 - answer_accuracy
        for hypothesis in hypotheses
    ]
    return bayesian_update(prior, likelihoods)


def entropy(beliefs: Sequence[float]) -> float:
    probabilities = normalize(beliefs)
    positive = probabilities[probabilities > 0]
    return float(-np.sum(positive * np.log(positive)))


def normalized_entropy(beliefs: Sequence[float]) -> float:
    probabilities = normalize(beliefs)
    if probabilities.size == 1:
        return 0.0
    return entropy(probabilities) / math.log(probabilities.size)


def select_expected_reward_action(
    beliefs: Sequence[float],
    hypotheses: Sequence[RewardHypothesis],
    object_spec: ObjectSpec,
) -> Quadrant:
    probabilities = normalize(beliefs)
    if len(hypotheses) != probabilities.size:
        raise ValueError("Hypothesis count must match the belief vector")
    expected_rewards = {
        action: sum(
            probability * hypothesis.reward(object_spec, action)
            for probability, hypothesis in zip(probabilities, hypotheses, strict=True)
        )
        for action in QUADRANTS
    }
    return max(QUADRANTS, key=lambda action: expected_rewards[action])
