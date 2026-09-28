import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass
class AdImpression:
    user_id: str
    ad_id: str
    query: str
    position: int
    device_type: str
    timestamp: float
    ad_category: str
    user_segment: str


@dataclass
class ClickEvent:
    impression: AdImpression
    clicked: bool
    click_timestamp: float | None = None


CATEGORIES = ["electronics", "clothing", "sports", "books", "home", "beauty", "toys", "automotive"]
DEVICES = ["mobile", "desktop", "tablet"]
QUERIES = {
    "electronics": ["wireless headphones", "laptop stand", "usb cable", "phone case", "bluetooth speaker"],
    "clothing": ["running shoes", "winter jacket", "cotton shirt", "denim jeans", "dress socks"],
    "sports": ["yoga mat", "resistance bands", "water bottle", "gym bag", "tennis racket"],
    "books": ["python programming", "sci fi novel", "cookbook", "history book", "self help"],
    "home": ["desk lamp", "throw pillow", "storage bins", "wall shelf", "door mat"],
    "beauty": ["face cream", "shampoo", "sunscreen", "lip balm", "hair dryer"],
    "toys": ["building blocks", "board game", "puzzle set", "action figure", "stuffed animal"],
    "automotive": ["car charger", "dash cam", "seat cover", "air freshener", "floor mat"],
}
POSITION_CTR_MULTIPLIER = {1: 1.0, 2: 0.7, 3: 0.5, 4: 0.35, 5: 0.2}


class ClickStreamGenerator:
    """Generates realistic synthetic ad click-stream data.

    Models position bias, user-ad category affinity, time-of-day effects,
    device effects, and user segment value.
    """

    def __init__(self, num_users: int = 10000, num_ads: int = 5000,
                 base_ctr: float = 0.03, seed: int = 42) -> None:
        self.rng = np.random.RandomState(seed)
        self.num_users = num_users
        self.num_ads = num_ads
        self.base_ctr = base_ctr
        self._build_user_profiles()
        self._build_ad_catalog()

    def _build_user_profiles(self) -> None:
        self.user_profiles: dict[str, dict] = {}
        for i in range(self.num_users):
            uid = f"u_{i:06d}"
            preferred = self.rng.choice(CATEGORIES, size=2, replace=False).tolist()
            segment = self.rng.choice(
                ["high_value", "medium_value", "low_value"], p=[0.1, 0.3, 0.6]
            )
            self.user_profiles[uid] = {
                "preferred_categories": preferred, "segment": segment
            }

    def _build_ad_catalog(self) -> None:
        self.ad_catalog: dict[str, dict] = {}
        for i in range(self.num_ads):
            aid = f"a_{i:06d}"
            cat = self.rng.choice(CATEGORIES)
            quality = self.rng.uniform(0.5, 1.5)
            self.ad_catalog[aid] = {"category": cat, "quality_score": quality}

    def _compute_ctr(self, user_id: str, ad_id: str, position: int,
                     hour: int, device: str) -> float:
        profile = self.user_profiles[user_id]
        ad = self.ad_catalog[ad_id]
        ctr = self.base_ctr

        if ad["category"] in profile["preferred_categories"]:
            ctr *= 2.5
        ctr *= POSITION_CTR_MULTIPLIER.get(position, 0.15)
        ctr *= ad["quality_score"]
        if 18 <= hour <= 22:
            ctr *= 1.3
        elif 2 <= hour <= 6:
            ctr *= 0.6
        if device == "mobile":
            ctr *= 1.1
        elif device == "tablet":
            ctr *= 0.9
        if profile["segment"] == "high_value":
            ctr *= 1.4

        return min(ctr, 0.95)

    def generate(self, num_samples: int) -> pd.DataFrame:
        """Generate synthetic click-stream data.

        Args:
            num_samples: Number of ad impressions to generate.

        Returns:
            DataFrame with impression data and click labels.
        """
        records = []
        user_ids = list(self.user_profiles.keys())
        ad_ids = list(self.ad_catalog.keys())

        for _ in range(num_samples):
            uid = self.rng.choice(user_ids)
            aid = self.rng.choice(ad_ids)
            position = int(self.rng.choice([1, 2, 3, 4, 5]))
            device = self.rng.choice(DEVICES)
            hour = int(self.rng.randint(0, 24))
            day_of_week = int(self.rng.randint(0, 7))
            ts = time.time() - self.rng.randint(0, 86400 * 30)
            ad_cat = self.ad_catalog[aid]["category"]
            query = self.rng.choice(QUERIES[ad_cat])
            segment = self.user_profiles[uid]["segment"]
            ctr = self._compute_ctr(uid, aid, position, hour, device)
            clicked = int(self.rng.random() < ctr)

            records.append({
                "user_id": uid, "ad_id": aid, "query": query,
                "position": position, "device_type": device,
                "timestamp": ts, "ad_category": ad_cat,
                "user_segment": segment, "hour": hour,
                "day_of_week": day_of_week, "clicked": clicked,
            })

        return pd.DataFrame(records)


class BatchDataLoader:
    """Loads and saves data from CSV or Parquet files."""

    @staticmethod
    def load(path: str) -> pd.DataFrame:
        p = Path(path)
        if p.suffix == ".csv":
            return pd.read_csv(p)
        elif p.suffix == ".parquet":
            return pd.read_parquet(p)
        raise ValueError(f"Unsupported format: {p.suffix}")

    @staticmethod
    def save(df: pd.DataFrame, path: str) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        if p.suffix == ".csv":
            df.to_csv(p, index=False)
        elif p.suffix == ".parquet":
            df.to_parquet(p, index=False)
        else:
            raise ValueError(f"Unsupported format: {p.suffix}")
