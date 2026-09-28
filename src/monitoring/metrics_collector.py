import threading
import time
from collections import deque

import numpy as np


class MetricsCollector:
    """Thread-safe sliding window metrics collection.

    Tracks prediction latency percentiles, request counts, and error rates.
    """

    _instance = None
    _lock = threading.Lock()

    def __new__(cls) -> "MetricsCollector":
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialized = False
            return cls._instance

    def __init__(self, window_size: int = 10000) -> None:
        if self._initialized:
            return
        self._latencies: deque = deque(maxlen=window_size)
        self._fs_latencies: deque = deque(maxlen=window_size)
        self._counts: dict[str, int] = {"success": 0, "error": 0}
        self._versions: dict[str, int] = {}
        self._data_lock = threading.Lock()
        self._initialized = True

    def record_prediction_latency(self, latency_ms: float) -> None:
        with self._data_lock:
            self._latencies.append((time.time(), latency_ms))

    def record_prediction_request(self, model_version: str, status: str) -> None:
        with self._data_lock:
            self._counts[status] = self._counts.get(status, 0) + 1
            self._versions[model_version] = self._versions.get(model_version, 0) + 1

    def record_feature_store_latency(self, latency_ms: float) -> None:
        with self._data_lock:
            self._fs_latencies.append((time.time(), latency_ms))

    def get_summary(self) -> dict:
        with self._data_lock:
            lat = [v for _, v in self._latencies]
            total = self._counts.get("success", 0) + self._counts.get("error", 0)
            return {
                "prediction_latency": self._pct(lat),
                "total_requests": total,
                "error_rate": self._counts.get("error", 0) / max(total, 1),
                "request_counts": dict(self._counts),
                "model_versions": dict(self._versions),
            }

    @staticmethod
    def _pct(values: list[float]) -> dict:
        if not values:
            return {"p50": 0, "p95": 0, "p99": 0}
        a = np.array(values)
        return {
            "p50": round(float(np.percentile(a, 50)), 2),
            "p95": round(float(np.percentile(a, 95)), 2),
            "p99": round(float(np.percentile(a, 99)), 2),
        }
