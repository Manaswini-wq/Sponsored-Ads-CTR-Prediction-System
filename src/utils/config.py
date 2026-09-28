from pydantic_settings import BaseSettings
from pydantic import Field


class ModelConfig(BaseSettings):
    model_type: str = Field(default="lightgbm")
    learning_rate: float = Field(default=0.05)
    num_trees: int = Field(default=500)
    max_depth: int = Field(default=7)
    feature_fraction: float = Field(default=0.8)
    min_data_in_leaf: int = Field(default=50)
    early_stopping_rounds: int = Field(default=30)

    model_config = {"env_prefix": "MODEL_"}


class ServingConfig(BaseSettings):
    host: str = Field(default="0.0.0.0")
    port: int = Field(default=8000)
    workers: int = Field(default=4)
    timeout_ms: int = Field(default=50)
    max_batch_size: int = Field(default=100)
    model_path: str = Field(default="artifacts/model.joblib")

    model_config = {"env_prefix": "SERVING_"}


class FeatureStoreConfig(BaseSettings):
    redis_host: str = Field(default="localhost")
    redis_port: int = Field(default=6379)
    ttl_seconds: int = Field(default=3600)
    fallback_to_memory: bool = Field(default=True)

    model_config = {"env_prefix": "FEATURE_STORE_"}


class PipelineConfig(BaseSettings):
    batch_size: int = Field(default=4096)
    checkpoint_dir: str = Field(default="artifacts/checkpoints")
    experiment_name: str = Field(default="ctr_experiment")

    model_config = {"env_prefix": "PIPELINE_"}


class AppConfig:
    """Bundles all configuration sections."""

    def __init__(self) -> None:
        self.model = ModelConfig()
        self.serving = ServingConfig()
        self.feature_store = FeatureStoreConfig()
        self.pipeline = PipelineConfig()
