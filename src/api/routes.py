"""
api/routes.py
──────────────
FastAPI APIRouter containing all route handlers:

  GET  /health   — liveness + readiness probe
  POST /train    — trigger a full training run
  POST /predict  — predict house price from feature payload
  POST /explain  — predict + SHAP feature attribution

Dependency injection
────────────────────
The loaded model is stored on ``app.state`` (set in main.py lifespan),
then injected into each handler via the ``get_model`` dependency.
This avoids module-level global state and makes handlers unit-testable.
"""

from __future__ import annotations

import numpy as np
from fastapi import APIRouter, Depends, HTTPException, Request, status

from src.api.schemas import (
    ErrorResponse,
    ExplainResponse,
    HealthResponse,
    HouseFeatures,
    PredictionResponse,
    TrainRequest,
    TrainResponse,
)
from src.features.feature_engineering import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    TARGET,
    prepare_inference_df,
)
from src.models.trainer import (
    load_best_model,
    load_metadata,
    predict_price,
    train_and_evaluate,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)
router = APIRouter()



def get_model(request: Request):
    """
    FastAPI dependency that retrieves the loaded pipeline from app state.

    Raises 503 if no model is loaded (i.e., /train has never been called).
    """
    model = getattr(request.app.state, "model", None)
    if model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded. Call POST /train first.",
        )
    return model



@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Health & readiness check",
    tags=["Operations"],
)
def health_check(request: Request) -> HealthResponse:
    """
    Liveness + readiness probe.

    Returns 200 whether or not a model is loaded.
    Kubernetes/load-balancer liveness checks should use this endpoint.
    CI/CD readiness checks should additionally verify ``model_loaded == true``.
    """
    model_loaded = getattr(request.app.state, "model", None) is not None
    meta = load_metadata()

    return HealthResponse(
        status="ok",
        model_loaded=model_loaded,
        best_model_name=meta.get("best_model_name"),
        best_rmse=meta.get("best_rmse"),
    )



@router.post(
    "/train",
    response_model=TrainResponse,
    summary="Train all candidate models and persist the best",
    tags=["Training"],
)
def train(
    body: TrainRequest,
    request: Request,
) -> TrainResponse:
    """
    Trigger a full training run:

    1. Load + clean data from *data_path* (defaults to bundled CSV).
    2. Train Linear Regression, Random Forest, XGBoost.
    3. Evaluate all on a held-out test set.
    4. Persist the best model to ``saved_models/best_model.joblib``.
    5. Hot-swap the in-memory model on ``app.state``.

    This endpoint is *synchronous* — for very large datasets you would
    move training to a background task or a dedicated worker queue.
    """
    logger.info("Training run requested", extra={"data_path": body.data_path})

    try:
        result = train_and_evaluate(data_path=body.data_path)
    except Exception as exc:
        logger.error("Training failed", extra={"error": str(exc)})
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Training failed: {exc}",
        )

    #Hot-swap the loaded model without restarting the server
    try:
        request.app.state.model = load_best_model()
        logger.info("In-memory model hot-swapped after training")
    except Exception as exc:
        logger.warning("Could not hot-swap model", extra={"error": str(exc)})

    return TrainResponse(
        status="success",
        best_model_name=result["best_model_name"],
        best_rmse=result["best_rmse"],
        best_r2=result["best_r2"],
        all_results=result["all_results"],
        training_time_s=result["training_time_s"],
        model_path=result["model_path"],
    )



@router.post(
    "/predict",
    response_model=PredictionResponse,
    summary="Predict house price from feature payload",
    tags=["Inference"],
)
def predict(
    features: HouseFeatures,
    model=Depends(get_model),
) -> PredictionResponse:
    """
    Predict the market price of a house from its physical and locational attributes.

    The model applies the *same* preprocessing pipeline that was fitted during
    training, so you never need to scale or encode features yourself.

    **Example curl**:
    ```bash
    curl -X POST http://localhost:8000/predict \\
      -H 'Content-Type: application/json' \\
      -d '{"bedrooms":3,"bathrooms":2,"sqft_living":1800,...}'
    ```
    """
    input_dict = features.model_dump()

    #Apply default date fields when caller omitted them
    if input_dict.get("sale_year") is None:
        input_dict["sale_year"] = 2024
    if input_dict.get("sale_month") is None:
        input_dict["sale_month"] = 6
    if input_dict.get("sale_dayofweek") is None:
        input_dict["sale_dayofweek"] = 1

    input_df = prepare_inference_df(input_dict)

    try:
        price = predict_price(model, input_df)
    except Exception as exc:
        logger.error("Prediction error", extra={"error": str(exc)})
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Prediction failed: {exc}",
        )

    meta = load_metadata()
    model_name = meta.get("best_model_name", "unknown")

    logger.info(
        "Prediction made",
        extra={
            "predicted_price": round(price, 2),
            "city": features.city,
            "sqft_living": features.sqft_living,
        },
    )

    return PredictionResponse(
        predicted_price=round(price, 2),
        model_used=model_name,
        confidence_note=(
            "This is a statistical estimate. Actual market prices may vary "
            "based on market conditions not captured in the model."
        ),
    )


#SHAP explain endpoint omitted for brevity — see full code in src/api/routes.py
@router.post(
    "/explain",
    response_model=ExplainResponse,
    summary="Predict + SHAP feature attribution",
    tags=["Inference"],
)
def explain(
    features: HouseFeatures,
    model=Depends(get_model),
) -> ExplainResponse:
    """
    Return a prediction **and** SHAP values for the top contributing features.

    SHAP (SHapley Additive exPlanations) decomposes the prediction into
    additive contributions from each feature — answering "which features
    pushed the price up / down and by how much?"

    Implementation note:
    SHAP's ``TreeExplainer`` is used for Random Forest / XGBoost.
    A ``LinearExplainer`` is used for Linear Regression.
    Both run in under a second for a single row.
    """
    import shap

    input_dict = features.model_dump()
    if input_dict.get("sale_year") is None:
        input_dict["sale_year"] = 2024
    if input_dict.get("sale_month") is None:
        input_dict["sale_month"] = 6
    if input_dict.get("sale_dayofweek") is None:
        input_dict["sale_dayofweek"] = 1

    input_df = prepare_inference_df(input_dict)

    #Run prediction
    try:
        price = predict_price(model, input_df)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Prediction failed: {exc}")

    #Extract preprocessor + raw estimator from Pipeline
    preprocessor = model.named_steps["preprocessor"]
    estimator    = model.named_steps["model"]

    #Transform the input through the preprocessing step
    X_transformed = preprocessor.transform(input_df)

    # Recover feature names from the ColumnTransformer
    try:
        feature_names = preprocessor.get_feature_names_out().tolist()
    except Exception:
        feature_names = [f"feature_{i}" for i in range(X_transformed.shape[1])]

    #Choose SHAP explainer
    try:
        estimator_type = type(estimator).__name__

        if estimator_type in ("RandomForestRegressor", "XGBRegressor"):
            explainer  = shap.TreeExplainer(estimator)
            shap_vals  = explainer.shap_values(X_transformed)
        else:
            # Linear models
            explainer  = shap.LinearExplainer(estimator, X_transformed)
            shap_vals  = explainer.shap_values(X_transformed)

        shap_row = shap_vals[0] if shap_vals.ndim > 1 else shap_vals

        # Map feature name → SHAP value
        shap_map = {
            name: round(float(val), 4)
            for name, val in zip(feature_names, shap_row)
        }

        # Return top-10 by absolute magnitude
        top_features = dict(
            sorted(shap_map.items(), key=lambda kv: abs(kv[1]), reverse=True)[:10]
        )

    except Exception as exc:
        logger.warning("SHAP computation failed", extra={"error": str(exc)})
        # Graceful fallback: return empty attribution rather than 500
        shap_map    = {}
        top_features = {}

    return ExplainResponse(
        predicted_price=round(price, 2),
        shap_values=shap_map,
        top_features=top_features,
    )
