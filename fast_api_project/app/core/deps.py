from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from app.core.config import settings
from app.core.model import ModelService


@lru_cache(maxsize=1)
def get_model_service() -> ModelService:
    return ModelService.load(model_path=settings.model_path_obj, feature_names=settings.feature_list)
