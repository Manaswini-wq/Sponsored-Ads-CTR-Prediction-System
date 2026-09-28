import numpy as np
import pandas as pd
from src.training.models import LogisticBaselineModel
from src.serving.predictor import CTRPredictor, PredictionRequest


def test_predictor():
    rng = np.random.RandomState(0)
    X = pd.DataFrame(rng.randn(100, 5), columns=[f"f{i}" for i in range(5)])
    y = pd.Series((X["f0"] > 0).astype(int))

    model = LogisticBaselineModel()
    model.fit(X, y)
    model._feature_names = list(X.columns)

    predictor = CTRPredictor(model=model)
    req = PredictionRequest(
        user_id="u1", ad_id="a1", query="test query",
        position=1, device_type="mobile",
    )
    resp = predictor.predict(req)

    assert 0.0 <= resp.ctr_probability <= 1.0
    assert resp.latency_ms >= 0
    assert resp.model_version == "v1"


def test_batch_predict():
    rng = np.random.RandomState(0)
    X = pd.DataFrame(rng.randn(100, 5), columns=[f"f{i}" for i in range(5)])
    y = pd.Series((X["f0"] > 0).astype(int))

    model = LogisticBaselineModel()
    model.fit(X, y)

    predictor = CTRPredictor(model=model)
    requests = [
        PredictionRequest(user_id=f"u{i}", ad_id="a1", query="test")
        for i in range(5)
    ]
    responses = predictor.batch_predict(requests)
    assert len(responses) == 5
