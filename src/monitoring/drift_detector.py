import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import stats


@dataclass
class DriftReport:
    timestamp: float
    features_drifted: list[str]
    drift_scores: dict[str, float]
    is_alert: bool
    details: str


@dataclass
class Alert:
    alert_id: str
    alert_type: str
    message: str
    severity: str  # INFO, WARNING, CRITICAL
    timestamp: float = field(default_factory=time.time)
    acknowledged: bool = False


class FeatureDriftDetector:
    """Detects feature drift using PSI (numeric) and chi-squared (categorical)."""

    def __init__(self, reference_stats: dict | None = None) -> None:
        self._ref = reference_stats or {}

    def compute_reference_stats(self, df: pd.DataFrame) -> dict:
        """Compute baseline distributions from a reference DataFrame."""
        result = {}
        for col in df.select_dtypes(include=[np.number]).columns:
            result[col] = {
                "mean": float(df[col].mean()),
                "std": float(df[col].std()),
                "quantiles": df[col].quantile([0.1, 0.25, 0.5, 0.75, 0.9]).to_dict(),
                "type": "numeric",
            }
        for col in df.select_dtypes(include=["object", "category"]).columns:
            result[col] = {
                "frequencies": df[col].value_counts(normalize=True).to_dict(),
                "type": "categorical",
            }
        self._ref = result
        return result

    def detect_drift(self, current_df: pd.DataFrame,
                     threshold: float = 0.2) -> DriftReport:
        drifted, scores = [], {}
        for col, ref in self._ref.items():
            if col not in current_df.columns:
                continue
            if ref["type"] == "numeric":
                psi = self._psi(ref, current_df[col].dropna().values)
                scores[col] = psi
                if psi > threshold:
                    drifted.append(col)
            elif ref["type"] == "categorical":
                p_val = self._cat_drift(ref["frequencies"], current_df[col])
                scores[col] = p_val
                if p_val < 0.05:
                    drifted.append(col)

        return DriftReport(
            timestamp=time.time(),
            features_drifted=drifted,
            drift_scores=scores,
            is_alert=len(drifted) > 0,
            details=(f"{len(drifted)} features drifted: {drifted}"
                     if drifted else "No drift detected"),
        )

    @staticmethod
    def _psi(ref: dict, values: np.ndarray, n_bins: int = 10) -> float:
        """Population Stability Index."""
        quantiles = list(ref["quantiles"].values())
        bins = [-np.inf] + sorted(set(quantiles)) + [np.inf]
        expected = np.diff(
            stats.norm.cdf(bins, loc=ref["mean"], scale=max(ref["std"], 1e-6))
        )
        actual = np.histogram(values, bins=bins)[0] / max(len(values), 1)
        expected = np.clip(expected, 1e-6, None)
        actual = np.clip(actual, 1e-6, None)
        return float(np.sum((actual - expected) * np.log(actual / expected)))

    @staticmethod
    def _cat_drift(ref_freq: dict, series: pd.Series) -> float:
        cur = series.value_counts(normalize=True).to_dict()
        cats = set(ref_freq) | set(cur)
        n = len(series)
        observed = [cur.get(c, 1e-6) * n for c in cats]
        expected = [ref_freq.get(c, 1e-6) * n for c in cats]
        _, p = stats.chisquare(observed, expected)
        return p


class PredictionDriftDetector:
    """Detects concept drift via sliding window comparison."""

    def __init__(self, window_size: int = 1000) -> None:
        self._window = window_size
        self._actuals: list[bool] = []

    def add_observation(self, predicted: float, actual: bool) -> None:
        self._actuals.append(actual)

    def detect_concept_drift(self) -> bool:
        if len(self._actuals) < self._window * 2:
            return False
        old = self._actuals[-self._window * 2: -self._window]
        new = self._actuals[-self._window:]
        old_rate, new_rate = np.mean(old), np.mean(new)
        pooled = (sum(old) + sum(new)) / (len(old) + len(new))
        se = np.sqrt(pooled * (1 - pooled) * (1 / len(old) + 1 / len(new)))
        if se < 1e-10:
            return False
        return abs(old_rate - new_rate) / se > 2.576


class AlertManager:
    """Manages operational alerts with severity levels."""

    def __init__(self) -> None:
        self._alerts: list[Alert] = []
        self._counter = 0

    def register_alert(self, alert_type: str, message: str,
                       severity: str = "WARNING") -> Alert:
        self._counter += 1
        alert = Alert(alert_id=f"alert_{self._counter}",
                      alert_type=alert_type, message=message,
                      severity=severity)
        self._alerts.append(alert)
        return alert

    def get_active_alerts(self) -> list[Alert]:
        return [a for a in self._alerts if not a.acknowledged]

    def acknowledge_alert(self, alert_id: str) -> None:
        for a in self._alerts:
            if a.alert_id == alert_id:
                a.acknowledged = True
                break
