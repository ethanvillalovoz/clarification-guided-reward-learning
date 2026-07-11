from dataclasses import asdict, dataclass
from typing import Any

from numpy.typing import NDArray

from .inference import (
    normalized_entropy,
    select_expected_reward_action,
    uniform_prior,
    update_from_correction,
    update_from_feature_clarification,
)
from .models import ObjectSpec, Quadrant, RewardHypothesis, default_problem


def _vector(values: NDArray) -> list[float]:
    return [round(float(value), 8) for value in values]


@dataclass(frozen=True)
class StepTrace:
    timestep: int
    object_label: str
    robot_action: Quadrant
    human_action: Quadrant
    corrected: bool
    prior: list[float]
    after_correction: list[float]
    clarification_asked: bool
    clarification_answer: list[str]
    posterior: list[float]
    entropy_after_correction: float
    entropy_after_clarification: float


@dataclass(frozen=True)
class ExperimentResult:
    condition: str
    hypothesis_names: list[str]
    true_hypothesis_index: int
    beta: float
    clarification_accuracy: float
    clarification_threshold: float
    steps: list[StepTrace]

    @property
    def final_beliefs(self) -> list[float]:
        if not self.steps:
            return _vector(uniform_prior(len(self.hypothesis_names)))
        return self.steps[-1].posterior

    @property
    def final_true_posterior(self) -> float:
        return self.final_beliefs[self.true_hypothesis_index]

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["final_beliefs"] = self.final_beliefs
        payload["final_true_posterior"] = round(self.final_true_posterior, 8)
        return payload


def run_simulation(
    *,
    use_clarification: bool = True,
    hypotheses: list[RewardHypothesis] | None = None,
    objects: list[ObjectSpec] | None = None,
    true_hypothesis_index: int | None = None,
    beta: float = 2.0,
    clarification_accuracy: float = 0.8,
    clarification_threshold: float = 0.55,
) -> ExperimentResult:
    default_hypotheses, default_objects, default_true_index = default_problem()
    active_hypotheses = default_hypotheses if hypotheses is None else hypotheses
    active_objects = default_objects if objects is None else objects
    if not active_hypotheses:
        raise ValueError("At least one reward hypothesis is required")
    if not active_objects:
        raise ValueError("At least one object is required")
    if not 0.0 <= clarification_threshold <= 1.0:
        raise ValueError("Clarification threshold must be in [0.0, 1.0]")
    active_true_index = (
        default_true_index if true_hypothesis_index is None else true_hypothesis_index
    )
    if not 0 <= active_true_index < len(active_hypotheses):
        raise ValueError("True hypothesis index is out of range")

    beliefs = uniform_prior(len(active_hypotheses))
    true_hypothesis = active_hypotheses[active_true_index]
    traces = []

    for timestep, object_spec in enumerate(active_objects):
        prior = beliefs.copy()
        robot_action = select_expected_reward_action(beliefs, active_hypotheses, object_spec)
        human_action = true_hypothesis.preferred_quadrant(object_spec)
        after_correction = update_from_correction(
            beliefs,
            active_hypotheses,
            object_spec,
            robot_action,
            human_action,
            beta,
        )
        uncertainty = normalized_entropy(after_correction)
        ask_clarification = use_clarification and uncertainty >= clarification_threshold
        clarification_answer = list(true_hypothesis.relevant_features) if ask_clarification else []
        if ask_clarification:
            beliefs = update_from_feature_clarification(
                after_correction,
                active_hypotheses,
                clarification_answer,
                clarification_accuracy,
            )
        else:
            beliefs = after_correction

        traces.append(
            StepTrace(
                timestep=timestep,
                object_label=object_spec.label,
                robot_action=robot_action,
                human_action=human_action,
                corrected=robot_action != human_action,
                prior=_vector(prior),
                after_correction=_vector(after_correction),
                clarification_asked=ask_clarification,
                clarification_answer=clarification_answer,
                posterior=_vector(beliefs),
                entropy_after_correction=round(uncertainty, 8),
                entropy_after_clarification=round(normalized_entropy(beliefs), 8),
            )
        )

    return ExperimentResult(
        condition="correction + clarification" if use_clarification else "correction only",
        hypothesis_names=[hypothesis.name for hypothesis in active_hypotheses],
        true_hypothesis_index=active_true_index,
        beta=beta,
        clarification_accuracy=clarification_accuracy,
        clarification_threshold=clarification_threshold,
        steps=traces,
    )


def run_comparison(**kwargs) -> dict[str, ExperimentResult]:
    return {
        "correction_only": run_simulation(use_clarification=False, **kwargs),
        "with_clarification": run_simulation(use_clarification=True, **kwargs),
    }
