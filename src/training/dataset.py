import numpy as np
import pandas as pd


FEATURE_COLUMNS = [
    "position", "hour", "day_of_week", "is_weekend", "is_business_hours",
    "query_length", "query_word_count", "user_historical_ctr",
    "user_total_impressions", "user_total_clicks",
    "ad_historical_ctr", "ad_total_impressions", "ad_total_clicks",
    "ad_quality_score", "query_historical_ctr",
    "user_ad_affinity", "position_avg_ctr",
]
LABEL_COLUMN = "clicked"
CATEGORICAL_COLUMNS = ["device_type", "ad_category", "user_segment"]


def train_test_split_by_time(
    df: pd.DataFrame, time_col: str = "timestamp", train_ratio: float = 0.8
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Temporal split to avoid data leakage from future information.

    Returns (train_df, val_df, test_df).
    """
    df_sorted = df.sort_values(time_col).reset_index(drop=True)
    n = len(df_sorted)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + (1 - train_ratio) / 2))
    return df_sorted[:train_end], df_sorted[train_end:val_end], df_sorted[val_end:]


def prepare_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Extract feature matrix X and label y from a DataFrame.

    Encodes categorical columns as integer codes.
    """
    df = df.copy()
    for col in CATEGORICAL_COLUMNS:
        if col in df.columns:
            df[col] = df[col].astype("category").cat.codes

    feature_cols = [
        c for c in FEATURE_COLUMNS + CATEGORICAL_COLUMNS if c in df.columns
    ]
    X = df[feature_cols].fillna(0.0)
    y = df[LABEL_COLUMN]
    return X, y


class DataValidator:
    """Validates training data quality."""

    @staticmethod
    def check_class_balance(y: pd.Series, min_ratio: float = 0.01) -> dict:
        pos_rate = y.mean()
        return {
            "positive_rate": pos_rate,
            "negative_rate": 1 - pos_rate,
            "is_balanced": pos_rate >= min_ratio,
        }

    @staticmethod
    def check_missing_features(X: pd.DataFrame) -> dict[str, float]:
        return (X.isnull().sum() / len(X)).to_dict()

    @staticmethod
    def check_feature_variance(X: pd.DataFrame, min_var: float = 1e-6) -> list[str]:
        """Returns feature names with near-zero variance."""
        return X.columns[X.var() < min_var].tolist()
