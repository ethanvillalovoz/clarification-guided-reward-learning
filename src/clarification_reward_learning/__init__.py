"""Clarification-guided reward-learning reference implementation."""

from .models import ObjectSpec, RewardHypothesis, default_problem
from .simulation import ExperimentResult, run_comparison, run_simulation

__all__ = [
    "ExperimentResult",
    "ObjectSpec",
    "RewardHypothesis",
    "default_problem",
    "run_comparison",
    "run_simulation",
]

__version__ = "1.1.0"
