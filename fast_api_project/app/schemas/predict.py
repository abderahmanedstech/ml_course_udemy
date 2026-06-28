from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    # Accept a free-form feature dict (works for most sklearn models)
    features: dict[str, float | int | bool] = Field(
        ..., description="Mapping of feature name to numeric value"
    )


class PredictResponse(BaseModel):
    prediction: Any
    proba: list[float] | None = None
    n_features: int
