from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd


class BaseCTRModel(ABC):
    """Common interface for all CTR prediction models."""

    @abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.Series, **kwargs: Any) -> None:
        ...

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        ...

    @abstractmethod
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        ...

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: str) -> "BaseCTRModel":
        return joblib.load(path)

    def get_feature_importance(self) -> dict[str, float] | None:
        return None


class GradientBoostedCTRModel(BaseCTRModel):
    """LightGBM-based CTR model. Primary production model."""

    def __init__(self, learning_rate: float = 0.05, num_trees: int = 500,
                 max_depth: int = 7, **kwargs: Any) -> None:
        self.params = {
            "objective": "binary",
            "metric": "binary_logloss",
            "learning_rate": learning_rate,
            "num_leaves": 2 ** max_depth - 1,
            "max_depth": max_depth,
            "verbose": -1,
            **kwargs,
        }
        self.num_trees = num_trees
        self.model = None
        self._feature_names: list[str] = []

    def fit(self, X: pd.DataFrame, y: pd.Series,
            X_val: pd.DataFrame | None = None,
            y_val: pd.Series | None = None, **kwargs: Any) -> None:
        import lightgbm as lgb

        self._feature_names = list(X.columns)
        train_set = lgb.Dataset(X, label=y)
        valid_sets = [train_set]
        callbacks = [lgb.log_evaluation(period=50)]

        if X_val is not None and y_val is not None:
            valid_sets.append(lgb.Dataset(X_val, label=y_val))
            callbacks.append(lgb.early_stopping(stopping_rounds=30))

        self.model = lgb.train(
            self.params, train_set,
            num_boost_round=self.num_trees,
            valid_sets=valid_sets,
            callbacks=callbacks,
        )

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return (self.predict_proba(X) >= 0.5).astype(int)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict(X, num_iteration=self.model.best_iteration)

    def get_feature_importance(self) -> dict[str, float]:
        importance = self.model.feature_importance(importance_type="gain")
        return dict(zip(self._feature_names, importance))


class LogisticBaselineModel(BaseCTRModel):
    """Logistic regression baseline for comparison."""

    def __init__(self, C: float = 1.0) -> None:
        from sklearn.linear_model import LogisticRegression
        self.model = LogisticRegression(C=C, max_iter=1000, solver="lbfgs")
        self._feature_names: list[str] = []

    def fit(self, X: pd.DataFrame, y: pd.Series, **kwargs: Any) -> None:
        self._feature_names = list(X.columns)
        self.model.fit(X, y)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict(X)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict_proba(X)[:, 1]

    def get_feature_importance(self) -> dict[str, float]:
        coefs = np.abs(self.model.coef_[0])
        return dict(zip(self._feature_names, coefs))


class DeepCTRModel(BaseCTRModel):
    """DeepFM-style neural CTR model using PyTorch."""

    def __init__(self, embed_dim: int = 8,
                 hidden_dims: tuple = (256, 128, 64),
                 dropout: float = 0.2, lr: float = 0.001,
                 epochs: int = 10) -> None:
        self.embed_dim = embed_dim
        self.hidden_dims = hidden_dims
        self.dropout = dropout
        self.lr = lr
        self.epochs = epochs
        self.model = None

    def _build_model(self, input_dim: int) -> None:
        import torch
        import torch.nn as nn

        class DeepFM(nn.Module):
            def __init__(self, in_dim: int, embed_dim: int,
                         hidden: tuple, drop: float) -> None:
                super().__init__()
                self.linear = nn.Linear(in_dim, 1)
                self.fm_first = nn.Linear(in_dim, embed_dim)
                self.fm_second = nn.Linear(in_dim, embed_dim)
                layers: list[nn.Module] = []
                prev = in_dim
                for h in hidden:
                    layers += [nn.Linear(prev, h), nn.BatchNorm1d(h),
                               nn.ReLU(), nn.Dropout(drop)]
                    prev = h
                layers.append(nn.Linear(prev, 1))
                self.deep = nn.Sequential(*layers)
                self.sigmoid = nn.Sigmoid()

            def forward(self, x: "torch.Tensor") -> "torch.Tensor":
                linear_out = self.linear(x)
                fm1 = self.fm_first(x)
                fm2 = self.fm_second(x)
                fm_out = (fm1 * fm2).sum(dim=1, keepdim=True)
                deep_out = self.deep(x)
                return self.sigmoid(linear_out + fm_out + deep_out).squeeze()

        self.model = DeepFM(input_dim, self.embed_dim,
                            self.hidden_dims, self.dropout)

    def fit(self, X: pd.DataFrame, y: pd.Series, **kwargs: Any) -> None:
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset

        self._build_model(X.shape[1])
        dataset = TensorDataset(
            torch.FloatTensor(X.values), torch.FloatTensor(y.values)
        )
        loader = DataLoader(dataset, batch_size=1024, shuffle=True)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        criterion = nn.BCELoss()

        self.model.train()
        for _ in range(self.epochs):
            for xb, yb in loader:
                pred = self.model(xb)
                loss = criterion(pred, yb)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return (self.predict_proba(X) >= 0.5).astype(int)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        import torch
        self.model.eval()
        with torch.no_grad():
            return self.model(torch.FloatTensor(X.values)).numpy()


MODEL_REGISTRY: dict[str, type[BaseCTRModel]] = {
    "lightgbm": GradientBoostedCTRModel,
    "logistic": LogisticBaselineModel,
    "deepfm": DeepCTRModel,
}


def create_model(name: str, **kwargs: Any) -> BaseCTRModel:
    """Factory function to create a model by name."""
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {name}. Available: {list(MODEL_REGISTRY)}")
    return MODEL_REGISTRY[name](**kwargs)
