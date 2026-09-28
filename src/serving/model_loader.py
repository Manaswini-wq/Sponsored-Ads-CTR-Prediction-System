import threading
import time
from dataclasses import dataclass, field

from src.training.models import BaseCTRModel
from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class ModelInfo:
    name: str
    version: str
    path: str
    stage: str = "staging"
    registered_at: float = field(default_factory=time.time)
    metadata: dict = field(default_factory=dict)


class ModelRegistry:
    """Tracks registered model versions and promotion stages."""

    def __init__(self) -> None:
        self._models: dict[str, list[ModelInfo]] = {}

    def register_model(self, name: str, version: str, path: str,
                       metadata: dict | None = None) -> ModelInfo:
        info = ModelInfo(name=name, version=version, path=path,
                         metadata=metadata or {})
        self._models.setdefault(name, []).append(info)
        logger.info(f"Registered model {name} v{version}")
        return info

    def get_model(self, name: str, version: str | None = None) -> ModelInfo | None:
        versions = self._models.get(name, [])
        if not versions:
            return None
        if version:
            return next((m for m in versions if m.version == version), None)
        return versions[-1]

    def list_models(self) -> list[ModelInfo]:
        return [m for vs in self._models.values() for m in vs]

    def promote_model(self, name: str, version: str, stage: str) -> None:
        model = self.get_model(name, version)
        if model:
            model.stage = stage
            logger.info(f"Promoted {name} v{version} -> {stage}")


class ModelLoader:
    """Thread-safe model loading with atomic hot-reload."""

    def __init__(self) -> None:
        self._model: BaseCTRModel | None = None
        self._lock = threading.Lock()
        self._loaded_at: float | None = None
        self._model_path: str | None = None

    def load_model(self, path: str) -> BaseCTRModel:
        logger.info(f"Loading model from {path}")
        model = BaseCTRModel.load(path)
        with self._lock:
            self._model = model
            self._model_path = path
            self._loaded_at = time.time()
        return model

    def get_model(self) -> BaseCTRModel | None:
        with self._lock:
            return self._model

    def hot_reload(self, new_path: str) -> None:
        """Atomically swap model in memory -- zero downtime."""
        logger.info(f"Hot-reloading model from {new_path}")
        new_model = BaseCTRModel.load(new_path)
        with self._lock:
            self._model = new_model
            self._model_path = new_path
            self._loaded_at = time.time()
        logger.info("Hot-reload complete")

    def health_check(self) -> dict:
        with self._lock:
            return {
                "model_loaded": self._model is not None,
                "model_path": self._model_path,
                "loaded_at": self._loaded_at,
            }
