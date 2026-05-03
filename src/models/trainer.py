"""
models/trainer.py
──────────────────
Orchestrates the full ML training workflow:

  1. Load & clean data            (data.data_loader)
  2. Add domain features          (features.feature_engineering)
  3. Train/test split
  4. Train candidate models inside sklearn Pipelines
  5. Evaluate with RMSE (and R²)
  6. Select the best model
  7. Persist with joblib
  8. (Optional) Log to MLflow

Candidate models
────────────────
  • Linear Regression  — interpretable baseline
  • Random Forest      — robust non-linear ensemble
  • XGBoost            — gradient-boosted trees (often best on tabular data)

Reproducibility
───────────────
  All random states are seeded via ``RANDOM_STATE``.
  The Pipeline saves the *fitted* preprocessor alongside the model,
  guaranteeing identical transformations at inference time.
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any, Dict, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor

from src.data.data_loader import load_and_clean
from src.features.feature_engineering import (
    TARGET,
    NUMERIC_FEATURES,
    CATEGORICAL_FEATURES,
    add_domain_features,
    build_full_pipeline,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)

#Paths 
PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = Path(os.getenv("MODEL_DIR", str(PROJECT_ROOT / "saved_models")))
MODEL_DIR.mkdir(parents=True, exist_ok=True)

BEST_MODEL_PATH = MODEL_DIR / "best_model.joblib"
METADATA_PATH   = MODEL_DIR / "training_metadata.joblib"

#Hyper-parameters (sensible defaults; swap for Optuna/GridSearch)
RANDOM_STATE  = 42
TEST_SIZE     = 0.20
N_JOBS        = -1   # use all available cores


#Model definitions
def _get_candidate_models() -> Dict[str, Any]:
    """
    Return a dict of {name: unfitted_estimator} for all candidate models.

    Each estimator will be wrapped in a full Pipeline (preprocessor + model)
    inside :func:`train_and_evaluate`.
    """
    return {
        "linear_regression": LinearRegression(n_jobs=N_JOBS),
        "random_forest": RandomForestRegressor(
            n_estimators=200,
            max_depth=15,
            min_samples_leaf=3,
            random_state=RANDOM_STATE,
            n_jobs=N_JOBS,
        ),
        "xgboost": XGBRegressor(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=RANDOM_STATE,
            n_jobs=N_JOBS,
            verbosity=0,
        ),
    }


#Model training and evaluation
def train_and_evaluate(data_path: str | None = None) -> Dict[str, Any]:
    """
    Full training run — load data, train all candidates, pick the winner.

    Parameters
    ----------
    data_path : str | None
        Optional override for the CSV path.  Defaults to the env-var /
        project-relative default in :mod:`data_loader`.

    Returns
    -------
    dict
        Summary with keys:
          ``best_model_name``, ``best_rmse``, ``best_r2``,
          ``all_results``, ``model_path``, ``training_time_s``
    """
    run_start = time.time()

    #Load & clean
    logger.info("Starting training run")
    df = load_and_clean(data_path)

    # Feature engineering
    df = add_domain_features(df)

    # Drop rows that still have NaN in any feature column after engineering
    feature_cols = NUMERIC_FEATURES + CATEGORICAL_FEATURES + [TARGET]
    df = df[feature_cols].dropna()

    X = df.drop(columns=[TARGET])
    y = df[TARGET]

    logger.info(
        "Feature matrix ready",
        extra={"samples": len(X), "features": X.shape[1]},
    )

    #Train / test split 
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    logger.info(
        "Data split",
        extra={"train": len(X_train), "test": len(X_test)},
    )

    #Train all candidates
    candidates = _get_candidate_models()
    results: Dict[str, Dict] = {}

    for name, estimator in candidates.items():
        logger.info(f"Training {name}…")
        t0 = time.time()

        pipeline = build_full_pipeline(estimator)
        pipeline.fit(X_train, y_train)

        y_pred = pipeline.predict(X_test)
        rmse   = float(np.sqrt(mean_squared_error(y_test, y_pred)))
        r2     = float(r2_score(y_test, y_pred))
        elapsed = round(time.time() - t0, 2)

        results[name] = {
            "pipeline":   pipeline,
            "rmse":       rmse,
            "r2":         r2,
            "train_time": elapsed,
        }
        logger.info(
            f"{name} done",
            extra={"rmse": round(rmse, 2), "r2": round(r2, 4), "s": elapsed},
        )

    #Select best model (lowest RMSE)
    best_name = min(results, key=lambda n: results[n]["rmse"])
    best      = results[best_name]

    logger.info(
        "Best model selected",
        extra={"model": best_name, "rmse": round(best["rmse"], 2)},
    )

    #Persist
    joblib.dump(best["pipeline"], BEST_MODEL_PATH)

    metadata = {
        "best_model_name": best_name,
        "best_rmse":       best["rmse"],
        "best_r2":         best["r2"],
        "all_results": {
            n: {"rmse": v["rmse"], "r2": v["r2"], "train_time": v["train_time"]}
            for n, v in results.items()
        },
        "feature_cols":    list(X.columns),
        "train_rows":      len(X_train),
        "test_rows":       len(X_test),
    }
    joblib.dump(metadata, METADATA_PATH)
    logger.info("Model and metadata saved", extra={"path": str(BEST_MODEL_PATH)})

    total_time = round(time.time() - run_start, 2)
    return {**metadata, "model_path": str(BEST_MODEL_PATH), "training_time_s": total_time}


#Inference utilities
def load_best_model():
    """
    Load the persisted best model from disk.

    Raises
    ------
    FileNotFoundError
        If no model has been trained yet (``saved_models/best_model.joblib``
        does not exist).
    """
    if not BEST_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"No trained model found at {BEST_MODEL_PATH}. "
            "Call POST /train first."
        )
    return joblib.load(BEST_MODEL_PATH)


def load_metadata() -> Dict:
    """Return the training metadata dict, or an empty dict if unavailable."""
    if METADATA_PATH.exists():
        return joblib.load(METADATA_PATH)
    return {}


def predict_price(pipeline, input_df: pd.DataFrame) -> float:
    """
    Run inference and return the predicted price as a Python float.

    Parameters
    ----------
    pipeline :
        Fitted sklearn Pipeline (preprocessor + estimator).
    input_df :
        Single-row DataFrame (output of
        :func:`features.feature_engineering.prepare_inference_df`).

    Returns
    -------
    float
        Predicted house price in USD.
    """
    prediction = pipeline.predict(input_df)[0]
    return max(float(prediction), 0.0)   # prices cannot be negative
