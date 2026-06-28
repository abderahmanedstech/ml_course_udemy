from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from app.core.deps import get_model_service
from app.core.model import ModelService
from app.schemas.predict import PredictRequest, PredictResponse

router = APIRouter(tags=["predict"])


@router.post("/predict", response_model=PredictResponse)
def predict(
    payload: PredictRequest, model_service: ModelService = Depends(get_model_service)
) -> PredictResponse:
    try:
        result = model_service.predict(payload.features)
        return PredictResponse(**result)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
