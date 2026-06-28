from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        # Avoid warnings for fields like `model_path`
        protected_namespaces=("settings_",),
    )

    app_name: str = "ml-course-ensemble-api"
    api_v1_prefix: str = "/api/v1"

    # Model path inside the container/image
    model_path: str = "models/model.joblib"

    # Optional: comma-separated feature names. If provided, we will order inputs.
    feature_names: str | None = None

    @property
    def model_path_obj(self) -> Path:
        return Path(self.model_path)

    @property
    def feature_list(self) -> list[str] | None:
        if not self.feature_names:
            return None
        feats = [f.strip() for f in self.feature_names.split(",") if f.strip()]
        return feats or None


@lru_cache
def get_settings() -> Settings:
    return Settings()


# Convenience singleton for importers
settings = get_settings()
