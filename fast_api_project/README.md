# FastAPI Model Deployment

This folder contains a small, best-practice-ish FastAPI service for serving a saved scikit-learn model (Joblib).

## Expected model file

Place your model at:

- `fast_api_project/models/model.joblib`

Or change `MODEL_PATH` via environment variable.

## Endpoints

- `GET /api/v1/health` → service health
- `POST /api/v1/predict` → returns prediction (and optional probabilities)

### Example request

```json
{
  "features": {
    "Age": 22,
    "Fare": 7.25,
    "Pclass": 3,
    "Sex_male": 1
  }
}
```

If you set `FEATURE_NAMES` (comma-separated), the API will:
- enforce column order
- fill missing features with `0`

## Run without typing commands

Use VS Code:

- **Run and Debug** → select **FastAPI (uvicorn)**
- or **Terminal → Run Task…** → **FastAPI: Run (uvicorn)**

(Those tasks/configs are included in `.vscode/`.)
