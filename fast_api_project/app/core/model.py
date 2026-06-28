from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class ModelService:
    model: Any
    feature_names: list[str] | None = None

    @classmethod
    def load(cls, model_path: Path, feature_names: list[str] | None = None) -> "ModelService":
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")

        model = joblib.load(model_path)
        logger.info("Loaded model from %s", model_path)

        # Try to auto-extract feature names if a pandas DataFrame was used during training
        auto_features = None
        try:
            if hasattr(model, "feature_names_in_"):
                auto_features = list(getattr(model, "feature_names_in_"))
        except Exception:
            auto_features = None

        final_features = feature_names or auto_features
        if final_features:
            logger.info("Using %d feature names", len(final_features))
        else:
            logger.info("No feature_names provided; will accept any columns/order")

        return cls(model=model, feature_names=final_features)

    def _to_frame(self, features: dict[str, float | int | bool]) -> pd.DataFrame:
        row = pd.DataFrame([features])

        # Ensure booleans become 0/1 for sklearn estimators
        for col in row.columns:
            if row[col].dtype == bool:
                row[col] = row[col].astype(int)

        if self.feature_names:
            # Ensure all required columns exist
            for col in self.feature_names:
                if col not in row.columns:
                    row[col] = 0
            row = row[self.feature_names]

        row = row.fillna(0)
        return row

    def predict(self, features: dict[str, float | int | bool]) -> dict[str, Any]:
        X = self._to_frame(features)

        pred = self.model.predict(X)
        pred_value = pred[0] if isinstance(pred, (list, np.ndarray)) else pred

        proba = None
        if hasattr(self.model, "predict_proba"):
            try:
                probs = self.model.predict_proba(X)
                proba = probs[0].tolist() if hasattr(probs, "__len__") else None
            except Exception:
                proba = None

        return {
            "prediction": int(pred_value) if isinstance(pred_value, (np.integer, int, bool)) else pred_value,
            "proba": proba,
            "n_features": int(X.shape[1]),
        }
