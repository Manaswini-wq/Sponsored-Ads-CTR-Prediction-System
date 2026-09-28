import numpy as np
import pandas as pd
from src.training.models import LogisticBaselineModel, create_model


def test_logistic_baseline():
    rng = np.random.RandomState(0)
    X = pd.DataFrame(rng.randn(200, 5), columns=[f"f{i}" for i in range(5)])
    y = pd.Series((X["f0"] > 0).astype(int))

    model = LogisticBaselineModel()
    model.fit(X, y)

    proba = model.predict_proba(X)
    assert proba.shape == (200,)
    assert 0 <= proba.min() and proba.max() <= 1

    preds = model.predict(X)
    assert set(preds).issubset({0, 1})

    importance = model.get_feature_importance()
    assert len(importance) == 5


def test_create_model_factory():
    model = create_model("logistic", C=0.5)
    assert isinstance(model, LogisticBaselineModel)

    try:
        create_model("nonexistent")
        assert False, "Should have raised"
    except ValueError:
        pass
