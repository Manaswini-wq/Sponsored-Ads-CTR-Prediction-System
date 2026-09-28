import time
from typing import Any

from pydantic import BaseModel

from src.training.models import BaseCTRModel
from src.feature_store.feature_store import OnlineFeatureStore


class PredictionRequest(BaseModel):
    user_id: str
    ad_id: str
    query: str
    position: int = 1
    device_type: str = "desktop"
    context: dict[str, Any] = {}


class PredictionResponse(BaseModel):
    ctr_probability: float
    model_version: str
    latency_ms: float
    features_used: list[str]


class CTRPredictor:
    """Core prediction engine for real-time CTR inference."""

    def __init__(self, model: BaseCTRModel,
                 feature_store: OnlineFeatureStore | None = None,
                 model_version: str = "v1") -> None:
        self.model = model
        self.feature_store = feature_store
        self.model_version = model_version

    def predict(self, request: PredictionRequest) -> PredictionResponse:
        start = time.perf_counter()
        features = self._build_features(request)

        import pandas as pd
        X = pd.DataFrame([features])
        proba = float(self.model.predict_proba(X)[0])
        latency = (time.perf_counter() - start) * 1000

        return PredictionResponse(
            ctr_probability=round(proba, 6),
            model_version=self.model_version,
            latency_ms=round(latency, 2),
            features_used=list(features.keys()),
        )

    def batch_predict(self, requests: list[PredictionRequest]) -> list[PredictionResponse]:
        return [self.predict(r) for r in requests]

    def _build_features(self, req: PredictionRequest) -> dict[str, float]:
        """Build feature vector from request + feature store lookups."""
        device_map = {"mobile": 0, "desktop": 1, "tablet": 2}
        features: dict[str, float] = {
            "position": float(req.position),
            "query_length": float(len(req.query)),
            "query_word_count": float(len(req.query.split())),
            "device_type": float(device_map.get(req.device_type, 1)),
        }

        if self.feature_store:
            for prefix, key in [("user_", f"user:{req.user_id}"),
                                ("ad_", f"ad:{req.ad_id}")]:
                stored = self.feature_store.get_features(key)
                if stored:
                    features.update({
                        f"{prefix}{k}": float(v)
                        for k, v in stored.items()
                        if isinstance(v, (int, float))
                    })

        defaults = {
            "hour": 12.0, "day_of_week": 3.0, "is_weekend": 0.0,
            "is_business_hours": 1.0, "user_historical_ctr": 0.03,
            "user_total_impressions": 100.0, "user_total_clicks": 3.0,
            "ad_historical_ctr": 0.03, "ad_total_impressions": 500.0,
            "ad_total_clicks": 15.0, "ad_quality_score": 0.5,
            "query_historical_ctr": 0.03, "user_ad_affinity": 0.0,
            "position_avg_ctr": 0.03, "ad_category": 0.0,
            "user_segment": 0.0,
        }
        for k, v in defaults.items():
            features.setdefault(k, v)

        return features
