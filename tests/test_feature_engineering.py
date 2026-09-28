import pandas as pd
from src.feature_store.feature_engineering import (
    UserFeatureBuilder, AdFeatureBuilder, FeaturePipeline,
)


def _sample_df() -> pd.DataFrame:
    return pd.DataFrame({
        "user_id": ["u1", "u1", "u2", "u2", "u2"],
        "ad_id": ["a1", "a2", "a1", "a2", "a3"],
        "query": ["shoes", "shoes", "hat", "hat", "hat"],
        "position": [1, 2, 1, 3, 5],
        "clicked": [1, 0, 0, 1, 0],
    })


def test_user_feature_builder():
    df = _sample_df()
    feats = UserFeatureBuilder().build(df)
    assert "user_historical_ctr" in feats.columns
    u1 = feats[feats["user_id"] == "u1"].iloc[0]
    assert u1["user_total_impressions"] == 2
    assert u1["user_total_clicks"] == 1


def test_ad_feature_builder():
    df = _sample_df()
    feats = AdFeatureBuilder().build(df)
    assert "ad_quality_score" in feats.columns
    assert len(feats) == 3


def test_feature_pipeline():
    df = _sample_df()
    result = FeaturePipeline().build_features(df)
    assert "user_historical_ctr" in result.columns
    assert "ad_historical_ctr" in result.columns
    assert "position_avg_ctr" in result.columns
    assert len(result) == len(df)
