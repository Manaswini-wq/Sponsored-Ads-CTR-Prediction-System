import json
import time
from typing import Any

from src.utils.logger import get_logger

logger = get_logger(__name__)


class FeatureStore:
    """Base feature store interface."""

    def put_features(self, entity_id: str, features: dict[str, Any], ttl: int = 3600) -> None:
        raise NotImplementedError

    def get_features(self, entity_id: str) -> dict[str, Any] | None:
        raise NotImplementedError

    def batch_get(self, entity_ids: list[str]) -> dict[str, dict[str, Any] | None]:
        return {eid: self.get_features(eid) for eid in entity_ids}

    def batch_put(self, features_map: dict[str, dict[str, Any]], ttl: int = 3600) -> None:
        for eid, feats in features_map.items():
            self.put_features(eid, feats, ttl)


class OnlineFeatureStore(FeatureStore):
    """Redis-backed store for low-latency online serving.

    Falls back to in-memory dict if Redis is unavailable.
    """

    def __init__(self, redis_host: str = "localhost", redis_port: int = 6379,
                 fallback_to_memory: bool = True) -> None:
        self._memory: dict[str, dict] = {}
        self._use_redis = False
        try:
            import redis
            self._redis = redis.Redis(
                host=redis_host, port=redis_port, decode_responses=True
            )
            self._redis.ping()
            self._use_redis = True
            logger.info("Connected to Redis feature store")
        except Exception:
            if fallback_to_memory:
                logger.info("Redis unavailable, using in-memory feature store")
            else:
                raise

    def put_features(self, entity_id: str, features: dict[str, Any], ttl: int = 3600) -> None:
        payload = json.dumps({"features": features, "version": int(time.time())})
        if self._use_redis:
            self._redis.setex(f"feat:{entity_id}", ttl, payload)
        else:
            self._memory[entity_id] = {
                "payload": payload, "expires": time.time() + ttl
            }

    def get_features(self, entity_id: str) -> dict[str, Any] | None:
        if self._use_redis:
            raw = self._redis.get(f"feat:{entity_id}")
            return json.loads(raw)["features"] if raw else None
        entry = self._memory.get(entity_id)
        if entry is None or time.time() > entry["expires"]:
            return None
        return json.loads(entry["payload"])["features"]


class OfflineFeatureStore(FeatureStore):
    """Parquet-backed store for batch training workloads."""

    def __init__(self, storage_dir: str = "data/features") -> None:
        import os
        self._dir = storage_dir
        os.makedirs(storage_dir, exist_ok=True)
        self._cache: dict[str, dict[str, Any]] = {}

    def put_features(self, entity_id: str, features: dict[str, Any], ttl: int = 0) -> None:
        self._cache[entity_id] = features

    def get_features(self, entity_id: str) -> dict[str, Any] | None:
        return self._cache.get(entity_id)

    def save_to_parquet(self, path: str) -> None:
        import pandas as pd
        rows = [{"entity_id": k, **v} for k, v in self._cache.items()]
        pd.DataFrame(rows).to_parquet(path, index=False)

    def load_from_parquet(self, path: str) -> None:
        import pandas as pd
        df = pd.read_parquet(path)
        for _, row in df.iterrows():
            eid = row["entity_id"]
            self._cache[eid] = {
                k: v for k, v in row.items() if k != "entity_id"
            }
