import hashlib
import time
from dataclasses import dataclass, field

import numpy as np
from scipy import stats


@dataclass
class ABExperiment:
    experiment_id: str
    name: str
    control_model: str
    treatment_model: str
    traffic_split: float
    start_date: float = field(default_factory=time.time)
    end_date: float | None = None
    status: str = "running"


@dataclass
class ExperimentResults:
    experiment_id: str
    control_ctr: float
    treatment_ctr: float
    control_samples: int
    treatment_samples: int
    p_value: float
    confidence_interval: tuple[float, float]
    is_significant: bool


class ExperimentManager:
    """A/B testing with deterministic hash-based variant assignment."""

    def __init__(self) -> None:
        self._experiments: dict[str, ABExperiment] = {}
        self._outcomes: dict[str, list[dict]] = {}

    def create_experiment(self, name: str, control_model: str,
                          treatment_model: str,
                          traffic_split: float = 0.5) -> ABExperiment:
        exp_id = f"exp_{int(time.time())}_{name}"
        exp = ABExperiment(
            experiment_id=exp_id, name=name,
            control_model=control_model,
            treatment_model=treatment_model,
            traffic_split=traffic_split,
        )
        self._experiments[exp_id] = exp
        self._outcomes[exp_id] = []
        return exp

    def assign_variant(self, user_id: str, experiment_id: str) -> str:
        """Deterministic assignment -- same user always gets same variant."""
        exp = self._experiments[experiment_id]
        h = int(hashlib.sha256(
            f"{user_id}:{experiment_id}".encode()
        ).hexdigest(), 16)
        return "treatment" if (h % 1000) / 1000 < exp.traffic_split else "control"

    def record_outcome(self, experiment_id: str, user_id: str,
                       variant: str, clicked: bool) -> None:
        self._outcomes[experiment_id].append({
            "user_id": user_id, "variant": variant,
            "clicked": clicked, "timestamp": time.time(),
        })

    def get_results(self, experiment_id: str) -> ExperimentResults:
        outcomes = self._outcomes.get(experiment_id, [])
        control = [o["clicked"] for o in outcomes if o["variant"] == "control"]
        treatment = [o["clicked"] for o in outcomes if o["variant"] == "treatment"]

        analyzer = StatisticalAnalyzer()
        return ExperimentResults(
            experiment_id=experiment_id,
            control_ctr=np.mean(control) if control else 0.0,
            treatment_ctr=np.mean(treatment) if treatment else 0.0,
            control_samples=len(control),
            treatment_samples=len(treatment),
            p_value=analyzer.compute_p_value(control, treatment),
            confidence_interval=analyzer.compute_confidence_interval(
                treatment, control
            ),
            is_significant=analyzer.compute_p_value(control, treatment) < 0.05,
        )


class StatisticalAnalyzer:
    """Statistical testing utilities for A/B experiments."""

    @staticmethod
    def compute_p_value(control: list[bool], treatment: list[bool]) -> float:
        if len(control) < 2 or len(treatment) < 2:
            return 1.0
        table = [
            [sum(control), len(control) - sum(control)],
            [sum(treatment), len(treatment) - sum(treatment)],
        ]
        _, p, _, _ = stats.chi2_contingency(table)
        return p

    @staticmethod
    def compute_confidence_interval(
        treatment: list[bool], control: list[bool], confidence: float = 0.95
    ) -> tuple[float, float]:
        if not treatment or not control:
            return (0.0, 0.0)
        p_t, p_c = np.mean(treatment), np.mean(control)
        diff = p_t - p_c
        se = np.sqrt(
            p_t * (1 - p_t) / len(treatment)
            + p_c * (1 - p_c) / len(control)
        )
        z = stats.norm.ppf(1 - (1 - confidence) / 2)
        return (diff - z * se, diff + z * se)

    @staticmethod
    def compute_minimum_sample_size(
        baseline_ctr: float, mde: float,
        alpha: float = 0.05, power: float = 0.8
    ) -> int:
        """Minimum samples per variant to detect the given effect size."""
        p1, p2 = baseline_ctr, baseline_ctr + mde
        z_a = stats.norm.ppf(1 - alpha / 2)
        z_b = stats.norm.ppf(power)
        pooled_var = p1 * (1 - p1) + p2 * (1 - p2)
        return int(np.ceil((z_a + z_b) ** 2 * pooled_var / mde ** 2))
