import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score, log_loss, precision_score, recall_score, f1_score,
)

from src.training.models import BaseCTRModel


class ModelEvaluator:
    """Comprehensive model evaluation and reporting."""

    @staticmethod
    def compute_metrics(y_true: np.ndarray, y_pred_proba: np.ndarray) -> dict:
        y_pred = (y_pred_proba >= 0.5).astype(int)
        return {
            "auc_roc": roc_auc_score(y_true, y_pred_proba),
            "log_loss": log_loss(y_true, y_pred_proba),
            "precision": precision_score(y_true, y_pred, zero_division=0),
            "recall": recall_score(y_true, y_pred, zero_division=0),
            "f1": f1_score(y_true, y_pred, zero_division=0),
            "calibration_error": ModelEvaluator._ece(y_true, y_pred_proba),
        }

    @staticmethod
    def _ece(y_true: np.ndarray, y_pred: np.ndarray, n_bins: int = 10) -> float:
        """Expected Calibration Error."""
        edges = np.linspace(0, 1, n_bins + 1)
        ece = 0.0
        for i in range(n_bins):
            mask = (y_pred >= edges[i]) & (y_pred < edges[i + 1])
            if mask.sum() == 0:
                continue
            ece += mask.sum() * abs(y_true[mask].mean() - y_pred[mask].mean())
        return ece / len(y_true)

    @staticmethod
    def compute_lift_chart(y_true: np.ndarray, y_pred_proba: np.ndarray,
                           n_bins: int = 10) -> pd.DataFrame:
        df = pd.DataFrame({"actual": y_true, "predicted": y_pred_proba})
        df["decile"] = pd.qcut(df["predicted"], n_bins, labels=False,
                               duplicates="drop")
        lift = df.groupby("decile").agg(
            avg_predicted=("predicted", "mean"),
            avg_actual=("actual", "mean"),
            count=("actual", "count"),
        ).reset_index()
        lift["lift"] = lift["avg_actual"] / max(y_true.mean(), 1e-9)
        return lift

    @staticmethod
    def generate_report(model: BaseCTRModel, X_test: pd.DataFrame,
                        y_test: pd.Series, output_dir: str) -> dict:
        """Generate evaluation report and save artifacts to output_dir."""
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        y_proba = model.predict_proba(X_test)
        metrics = ModelEvaluator.compute_metrics(y_test.values, y_proba)
        lift = ModelEvaluator.compute_lift_chart(y_test.values, y_proba)

        lift.to_csv(out / "lift_chart.csv", index=False)
        (out / "metrics.json").write_text(json.dumps(metrics, indent=2))

        importance = model.get_feature_importance()
        if importance:
            sorted_imp = dict(
                sorted(importance.items(), key=lambda x: x[1], reverse=True)[:20]
            )
            (out / "feature_importance.json").write_text(
                json.dumps(sorted_imp, indent=2)
            )

        return metrics


class BiasAnalyzer:
    """Checks model fairness across segments."""

    @staticmethod
    def analyze_by_segment(y_true: np.ndarray, y_pred_proba: np.ndarray,
                           segments: pd.Series) -> pd.DataFrame:
        results = []
        for seg in segments.unique():
            mask = segments == seg
            if mask.sum() < 10:
                continue
            m = ModelEvaluator.compute_metrics(y_true[mask], y_pred_proba[mask])
            m["segment"] = seg
            m["count"] = int(mask.sum())
            results.append(m)
        return pd.DataFrame(results)
