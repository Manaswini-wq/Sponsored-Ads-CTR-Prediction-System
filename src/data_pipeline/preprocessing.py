import re

import numpy as np
import pandas as pd

STOPWORDS = {"the", "a", "an", "is", "in", "on", "at", "to", "for", "of", "and", "or", "with"}


class ClickStreamPreprocessor:
    """Cleans and validates click-stream DataFrames."""

    REQUIRED_COLUMNS = [
        "user_id", "ad_id", "query", "position",
        "device_type", "timestamp", "clicked",
    ]

    def validate_schema(self, df: pd.DataFrame) -> list[str]:
        """Returns list of missing required columns."""
        return [c for c in self.REQUIRED_COLUMNS if c not in df.columns]

    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """Full cleaning pipeline: validate, dedupe, fill missing."""
        missing = self.validate_schema(df)
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
        df = self.remove_duplicates(df)
        df = self.handle_missing_values(df)
        return df

    def remove_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        return df.drop_duplicates(subset=["user_id", "ad_id", "timestamp"])

    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["query"] = df["query"].fillna("")
        df["device_type"] = df["device_type"].fillna("unknown")
        df["position"] = df["position"].fillna(df["position"].median())
        df = df.dropna(subset=["user_id", "ad_id", "clicked"])
        return df


class TimeFeatureExtractor:
    """Extracts temporal features from timestamp column."""

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        if "hour" not in df.columns:
            dt = pd.to_datetime(df["timestamp"], unit="s")
            df["hour"] = dt.dt.hour
            df["day_of_week"] = dt.dt.dayofweek
        df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)
        df["is_business_hours"] = ((df["hour"] >= 9) & (df["hour"] <= 17)).astype(int)
        return df


class TextPreprocessor:
    """Processes query text into numeric features."""

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["query_clean"] = df["query"].str.lower().str.strip()
        df["query_length"] = df["query_clean"].str.len()
        df["query_word_count"] = (
            df["query_clean"].str.split().str.len().fillna(0).astype(int)
        )
        return df

    @staticmethod
    def tokenize(text: str) -> list[str]:
        if not isinstance(text, str) or not text:
            return []
        words = re.findall(r"\w+", text.lower())
        return [w for w in words if w not in STOPWORDS]
