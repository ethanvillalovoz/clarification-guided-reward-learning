from dataclasses import dataclass
from typing import Literal

Quadrant = Literal["Q1", "Q2", "Q3", "Q4"]
QUADRANTS: tuple[Quadrant, ...] = ("Q1", "Q2", "Q3", "Q4")
VALID_FEATURES = frozenset({"color", "material", "kind"})


@dataclass(frozen=True)
class ObjectSpec:
    label: str
    color: str
    material: str
    kind: str

    def feature_value(self, feature: str) -> str:
        if feature not in VALID_FEATURES:
            raise ValueError(f"Unsupported object feature: {feature}")
        return getattr(self, feature)


@dataclass(frozen=True)
class RewardHypothesis:
    name: str
    relevant_features: tuple[str, ...]
    preferred_quadrants: dict[tuple[str, ...], Quadrant]
    default_quadrant: Quadrant = "Q1"

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("Hypothesis name cannot be empty")
        if not self.relevant_features:
            raise ValueError("A hypothesis must depend on at least one feature")
        if len(set(self.relevant_features)) != len(self.relevant_features):
            raise ValueError("Hypothesis features must be unique")
        if not set(self.relevant_features).issubset(VALID_FEATURES):
            raise ValueError("Hypothesis contains an unsupported feature")
        if self.default_quadrant not in QUADRANTS:
            raise ValueError("Default quadrant is invalid")

    def feature_signature(self, object_spec: ObjectSpec) -> tuple[str, ...]:
        return tuple(object_spec.feature_value(feature) for feature in self.relevant_features)

    def preferred_quadrant(self, object_spec: ObjectSpec) -> Quadrant:
        return self.preferred_quadrants.get(
            self.feature_signature(object_spec),
            self.default_quadrant,
        )

    def reward(self, object_spec: ObjectSpec, quadrant: Quadrant) -> float:
        return 1.0 if quadrant == self.preferred_quadrant(object_spec) else 0.0


def default_problem() -> tuple[list[RewardHypothesis], list[ObjectSpec], int]:
    objects = [
        ObjectSpec("yellow glass cup", color="yellow", material="glass", kind="cup"),
        ObjectSpec("red glass cup", color="red", material="glass", kind="cup"),
        ObjectSpec("purple ceramic bowl", color="purple", material="ceramic", kind="bowl"),
    ]
    hypotheses = [
        RewardHypothesis(
            "color",
            ("color",),
            {("yellow",): "Q2", ("red",): "Q4", ("purple",): "Q3"},
        ),
        RewardHypothesis(
            "material",
            ("material",),
            {("glass",): "Q1", ("ceramic",): "Q3"},
        ),
        RewardHypothesis(
            "object type",
            ("kind",),
            {("cup",): "Q4", ("bowl",): "Q3"},
        ),
        RewardHypothesis(
            "color + object type",
            ("color", "kind"),
            {
                ("yellow", "cup"): "Q2",
                ("red", "cup"): "Q1",
                ("purple", "bowl"): "Q3",
            },
        ),
        RewardHypothesis(
            "material + object type",
            ("material", "kind"),
            {("glass", "cup"): "Q4", ("ceramic", "bowl"): "Q3"},
        ),
        RewardHypothesis(
            "color + material",
            ("color", "material"),
            {
                ("yellow", "glass"): "Q1",
                ("red", "glass"): "Q4",
                ("purple", "ceramic"): "Q2",
            },
        ),
    ]
    true_hypothesis_index = 3
    return hypotheses, objects, true_hypothesis_index
