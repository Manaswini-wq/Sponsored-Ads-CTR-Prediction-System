import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score, log_loss, precision_score, recall_score, f1_score,
)

from src.training.models import BaseCTRModel
from src.utils.logger import get_logger

logger = get_logger(__name__)


class Trainer:
    """Manages model training with checkpointing and metric tracking."""

    def __init__(self, checkpoint_dir: str = "artifacts/checkpoints") -> None:
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.history: list[dict] = []

    def train(self, model: BaseCTRModel, X_train: pd.DataFrame,
              y_train: pd.Series, X_val: pd.DataFrame,
              y_val: pd.Series) -> dict:
        """Train model and return metrics dict."""
        logger.info(
            f"Training {model.__class__.__name__} on {len(X_train)} samples"
        )
        start = time.time()
        model.fit(X_train, y_train, X_val=X_val, y_val=y_val)

        train_proba = model.predict_proba(X_train)
        val_proba = model.predict_proba(X_val)
        val_pred = (val_proba >= 0.5).astype(int)

        metrics = {
            "train_auc": roc_auc_score(y_train, train_proba),
            "val_auc": roc_auc_score(y_val, val_proba),
            "train_logloss": log_loss(y_train, train_proba),
            "val_logloss": log_loss(y_val, val_proba),
            "val_precision": precision_score(y_val, val_pred, zero_division=0),
            "val_recall": recall_score(y_val, val_pred, zero_division=0),
            "val_f1": f1_score(y_val, val_pred, zero_division=0),
            "training_time_seconds": round(time.time() - start, 2),
        }
        self.history.append(metrics)
        logger.info(f"Val AUC: {metrics['val_auc']:.4f} | "
                     f"Val LogLoss: {metrics['val_logloss']:.4f}")

        model.save(str(self.checkpoint_dir / "best_model.joblib"))
        return metrics


class ExperimentTracker:
    """Tracks experiments as JSON files for reproducibility."""

    def __init__(self, experiments_dir: str = "experiments") -> None:
        self.dir = Path(experiments_dir)
        self.dir.mkdir(parents=True, exist_ok=True)

    def log_experiment(self, name: str, params: dict, metrics: dict,
                       artifacts: dict | None = None) -> str:
        exp_id = f"{name}_{int(time.time())}"
        experiment = {
            "id": exp_id, "name": name, "timestamp": time.time(),
            "params": params, "metrics": metrics,
            "artifacts": artifacts or {},
        }
        (self.dir / f"{exp_id}.json").write_text(
            json.dumps(experiment, indent=2, default=str)
        )
        logger.info(f"Logged experiment: {exp_id}")
        return exp_id

    def list_experiments(self) -> list[dict]:
        return [
            json.loads(p.read_text())
            for p in sorted(self.dir.glob("*.json"))
        ]

    def get_best_experiment(self, metric: str = "val_auc",
                            higher_is_better: bool = True) -> dict | None:
        exps = self.list_experiments()
        if not exps:
            return None
        return sorted(
            exps, key=lambda e: e["metrics"].get(metric, 0),
            reverse=higher_is_better,
        )[0]


class HyperparameterTuner:
    """Grid and random search over hyperparameters."""

    def __init__(self, trainer: Trainer) -> None:
        self.trainer = trainer

    def grid_search(
        self, model_class: type[BaseCTRModel], param_grid: dict[str, list],
        X_train: pd.DataFrame, y_train: pd.Series,
        X_val: pd.DataFrame, y_val: pd.Series,
    ) -> tuple[dict, list[dict]]:
        import itertools
        keys = list(param_grid.keys())
        results: list[dict] = []
        best_auc, best_params = -1.0, {}

        for values in itertools.product(*param_grid.values()):
            params = dict(zip(keys, values))
            model = model_class(**params)
            metrics = self.trainer.train(model, X_train, y_train, X_val, y_val)
            results.append({"params": params, **metrics})
            if metrics["val_auc"] > best_auc:
                best_auc = metrics["val_auc"]
                best_params = params

        return best_params, results

    def random_search(
        self, model_class: type[BaseCTRModel],
        param_distributions: dict[str, Any],
        X_train: pd.DataFrame, y_train: pd.Series,
        X_val: pd.DataFrame, y_val: pd.Series,
        n_trials: int = 20, seed: int = 42,
    ) -> tuple[dict, list[dict]]:
        rng = np.random.RandomState(seed)
        results: list[dict] = []
        best_auc, best_params = -1.0, {}

        for _ in range(n_trials):
            params = {}
            for k, v in param_distributions.items():
                if isinstance(v, list):
                    params[k] = rng.choice(v)
                elif isinstance(v, tuple) and len(v) == 2:
                    params[k] = rng.uniform(v[0], v[1])
                else:
                    params[k] = v

            model = model_class(**params)
            metrics = self.trainer.train(model, X_train, y_train, X_val, y_val)
            results.append({"params": params, **metrics})
            if metrics["val_auc"] > best_auc:
                best_auc = metrics["val_auc"]
                best_params = params

        return best_params, results
