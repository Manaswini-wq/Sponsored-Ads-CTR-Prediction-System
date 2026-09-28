import numpy as np
import pandas as pd


class UserFeatureBuilder:
    """Computes user-level aggregate features from historical data."""

    def build(self, df: pd.DataFrame) -> pd.DataFrame:
        agg = df.groupby("user_id").agg(
            user_total_impressions=("clicked", "count"),
            user_total_clicks=("clicked", "sum"),
        ).reset_index()
        agg["user_historical_ctr"] = (
            agg["user_total_clicks"] / agg["user_total_impressions"]
        ).fillna(0.0)
        return agg


class AdFeatureBuilder:
    """Computes ad-level features."""

    def build(self, df: pd.DataFrame) -> pd.DataFrame:
        agg = df.groupby("ad_id").agg(
            ad_total_impressions=("clicked", "count"),
            ad_total_clicks=("clicked", "sum"),
        ).reset_index()
        agg["ad_historical_ctr"] = (
            agg["ad_total_clicks"] / agg["ad_total_impressions"]
        ).fillna(0.0)
        agg["ad_quality_score"] = np.clip(
            agg["ad_historical_ctr"] * 10 + 0.5, 0, 1
        )
        return agg


class QueryFeatureBuilder:
    """Computes query-level features."""

    def build(self, df: pd.DataFrame) -> pd.DataFrame:
        agg = df.groupby("query").agg(
            query_impressions=("clicked", "count"),
            query_clicks=("clicked", "sum"),
        ).reset_index()
        agg["query_historical_ctr"] = (
            agg["query_clicks"] / agg["query_impressions"]
        ).fillna(0.0)
        agg["query_length"] = agg["query"].str.len()
        agg["query_word_count"] = agg["query"].str.split().str.len()
        return agg


class CrossFeatureBuilder:
    """Computes interaction features between users, ads, and positions."""

    def build(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        user_ad = df.groupby(["user_id", "ad_id"]).agg(
            user_ad_impressions=("clicked", "count"),
            user_ad_clicks=("clicked", "sum"),
        ).reset_index()
        user_ad["user_ad_affinity"] = (
            user_ad["user_ad_clicks"] / user_ad["user_ad_impressions"]
        ).fillna(0.0)

        pos = df.groupby("position").agg(
            position_avg_ctr=("clicked", "mean")
        ).reset_index()

        return user_ad, pos


class FeaturePipeline:
    """Orchestrates all feature builders into a single feature DataFrame."""

    def __init__(self) -> None:
        self.user_builder = UserFeatureBuilder()
        self.ad_builder = AdFeatureBuilder()
        self.query_builder = QueryFeatureBuilder()
        self.cross_builder = CrossFeatureBuilder()

    def build_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Build and merge all features onto each impression row."""
        user_feats = self.user_builder.build(df)
        ad_feats = self.ad_builder.build(df)
        query_feats = self.query_builder.build(df)
        user_ad_feats, pos_feats = self.cross_builder.build(df)

        result = df.copy()
        result = result.merge(user_feats, on="user_id", how="left")
        result = result.merge(ad_feats, on="ad_id", how="left")
        result = result.merge(query_feats, on="query", how="left")
        result = result.merge(
            user_ad_feats[["user_id", "ad_id", "user_ad_affinity"]],
            on=["user_id", "ad_id"], how="left",
        )
        result = result.merge(pos_feats, on="position", how="left")
        result = result.fillna(0.0)
        return result
