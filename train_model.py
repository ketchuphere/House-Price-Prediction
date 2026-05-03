#!/usr/bin/env python3
"""
train_model.py
───────────────
Standalone CLI script to train the house-price prediction models.

Run this once before starting the API server so that
``saved_models/best_model.joblib`` exists and the server loads immediately.

Usage
─────
    # From the project root:
    python train_model.py

    # With a custom dataset path:
    DATA_PATH=/path/to/custom.csv python train_model.py

    # With MLflow tracking (set MLFLOW_TRACKING_URI first):
    MLFLOW_TRACKING_URI=http://localhost:5000 python train_model.py
"""

from __future__ import annotations

import json
import os
import sys
import time

#Ensure the project root is on sys.path so `src.*` imports resolve
sys.path.insert(0, os.path.dirname(__file__))

from src.models.trainer import train_and_evaluate
from src.utils.logger import get_logger

logger = get_logger("train_model")


def main() -> None:
    logger.info("═" * 60)
    logger.info("  House Price Prediction — Model Training Script")
    logger.info("═" * 60)

    start = time.time()
    result = train_and_evaluate()
    elapsed = round(time.time() - start, 2)

    #Pretty-print results
    print("\n" + "═" * 60)
    print("  TRAINING RESULTS")
    print("═" * 60)
    print(f"\n  Best model  : {result['best_model_name'].upper()}")
    print(f"  Best RMSE   : ${result['best_rmse']:,.2f}")
    print(f"  Best R²     : {result['best_r2']:.4f}")
    print(f"  Total time  : {elapsed}s")
    print(f"  Saved to    : {result['model_path']}")

    print("\n  All model results:")
    for name, metrics in result["all_results"].items():
        marker = "✓" if name == result["best_model_name"] else " "
        print(
            f"  {marker} {name:<25}"
            f"  RMSE=${metrics['rmse']:>12,.2f}"
            f"  R²={metrics['r2']:.4f}"
            f"  ({metrics['train_time']}s)"
        )

    print("\n" + "═" * 60)

    #Optional MLflow logging
    mlflow_uri = os.getenv("MLFLOW_TRACKING_URI")
    if mlflow_uri:
        _log_to_mlflow(result, mlflow_uri)


def _log_to_mlflow(result: dict, tracking_uri: str) -> None:
    """Log the best model run to MLflow if the URI is configured."""
    try:
        import mlflow

        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment("house_price_prediction")

        with mlflow.start_run(run_name=result["best_model_name"]):
            mlflow.log_param("model_type",    result["best_model_name"])
            mlflow.log_param("train_rows",    result.get("train_rows"))
            mlflow.log_param("test_rows",     result.get("test_rows"))
            mlflow.log_metric("rmse",         result["best_rmse"])
            mlflow.log_metric("r2",           result["best_r2"])
            mlflow.log_metric("training_time_s", result["training_time_s"])

            # Log all competing models as child metrics
            for name, metrics in result["all_results"].items():
                mlflow.log_metric(f"{name}_rmse", metrics["rmse"])
                mlflow.log_metric(f"{name}_r2",   metrics["r2"])

            mlflow.log_artifact(result["model_path"])

        logger.info("Run logged to MLflow", extra={"uri": tracking_uri})
    except Exception as exc:
        logger.warning("MLflow logging failed", extra={"error": str(exc)})


if __name__ == "__main__":
    main()
