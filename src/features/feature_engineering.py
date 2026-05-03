"""
features/feature_engineering.py
─────────────────────────────────
Defines the feature-engineering pipeline that sits between raw cleaned data
and the ML model.

Architecture
────────────
We use scikit-learn's ``ColumnTransformer`` + ``Pipeline`` pattern so that:

  • All transformations are reproducible (fit on train, applied to test/prod).
  • The whole transformer can be serialised with ``joblib`` alongside the model.
  • New inference requests pass through identical preprocessing automatically.

Columns handled
───────────────
  Numeric : StandardScaler  (zero-mean, unit-variance)
  Categorical : OneHotEncoder (handle_unknown="ignore" for unseen cities)

Custom features added before the pipeline
──────────────────────────────────────────
  ``house_age``       — current year minus yr_built
  ``since_renovated`` — years since last renovation (0 if never renovated)
  ``total_sqft``      — sqft_living + sqft_basement
  ``bed_bath_ratio``  — bedrooms / (bathrooms + 1e-6)
  ``is_renovated``    — binary flag: was the house ever renovated?
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.utils.logger import get_logger

logger = get_logger(__name__)

#Reference year used to compute age-related features
REFERENCE_YEAR = 2024

#Column lists (kept centralised so API schemas stay in sync)
NUMERIC_FEATURES = [
    "bedrooms",
    "bathrooms",
    "sqft_living",
    "sqft_lot",
    "floors",
    "waterfront",
    "view",
    "condition",
    "sqft_above",
    "sqft_basement",
    "yr_built",
    "yr_renovated",
    "sale_year",
    "sale_month",
    "sale_dayofweek",
    # Engineered
    "house_age",
    "since_renovated",
    "total_sqft",
    "bed_bath_ratio",
    "is_renovated",
]

CATEGORICAL_FEATURES = [
    "city",
    "statezip",
]

TARGET = "price"


#feature engineering functions
def add_domain_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Enrich the DataFrame with domain-knowledge-derived features.

    These are computed *before* the sklearn pipeline so that they are visible
    to both numeric scalers and tree-based models (which don't need scaling
    but benefit from explicit feature interactions).

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned dataframe (output of :func:`data_loader.clean_data`).

    Returns
    -------
    pd.DataFrame
        Copy of *df* with additional columns.
    """
    df = df.copy()

    #How old is the house?
    df["house_age"] = REFERENCE_YEAR - df["yr_built"]

    #How long since the last renovation? (0 means never renovated)
    df["since_renovated"] = np.where(
        df["yr_renovated"] > 0,
        REFERENCE_YEAR - df["yr_renovated"],
        df["house_age"],   # treat "never renovated" as age of the house
    )

    #Combined square footage
    df["total_sqft"] = df["sqft_living"] + df["sqft_basement"]

    #Bedroom-to-bathroom ratio (useful signal for over/under-bathromed homes)
    df["bed_bath_ratio"] = df["bedrooms"] / (df["bathrooms"] + 1e-6)

    #Simple binary renovation flag
    df["is_renovated"] = (df["yr_renovated"] > 0).astype(int)

    logger.debug("Domain features added", extra={"new_cols": 5})
    return df


#pipeline construction functions
def build_preprocessor() -> ColumnTransformer:
    """
    Build and return the sklearn ``ColumnTransformer`` that handles:

    * Numeric columns  → :class:`~sklearn.preprocessing.StandardScaler`
    * Categorical cols → :class:`~sklearn.preprocessing.OneHotEncoder`

    The transformer is *unfitted*; call ``.fit(X_train)`` or ``.fit_transform``
    downstream inside the full :func:`build_full_pipeline`.

    Returns
    -------
    ColumnTransformer
    """
    numeric_transformer = Pipeline(
        steps=[("scaler", StandardScaler())]
    )

    categorical_transformer = Pipeline(
        steps=[
            (
                "ohe",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            )
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, NUMERIC_FEATURES),
            ("cat", categorical_transformer, CATEGORICAL_FEATURES),
        ],
        remainder="drop",   #silently ignore any extra columns
        verbose_feature_names_out=False,
    )
    return preprocessor


def build_full_pipeline(estimator) -> Pipeline:
    """
    Wrap a scikit-learn–compatible *estimator* inside a full end-to-end
    ``Pipeline``:

        preprocessor  →  estimator

    Parameters
    ----------
    estimator :
        Any sklearn-compatible regressor (LinearRegression, RandomForest, XGB…).

    Returns
    -------
    sklearn.pipeline.Pipeline
        Unfitted pipeline.
    """
    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            ("model", estimator),
        ]
    )


#inference preparation functions
def prepare_inference_df(input_dict: dict) -> pd.DataFrame:
    """
    Convert a raw prediction request payload (as a Python dict) into a
    feature-engineered DataFrame ready to be passed to a fitted pipeline.

    The function fills in any missing date-derived columns with sensible
    defaults so that callers only need to supply the core house attributes.

    Parameters
    ----------
    input_dict : dict
        Keys matching the house feature fields defined in the Pydantic schema.

    Returns
    -------
    pd.DataFrame
        Single-row DataFrame with all expected columns present.
    """
    df = pd.DataFrame([input_dict])

    #Provide default date-features if caller omitted them or passed None
    if "sale_year" not in df.columns or df["sale_year"].isna().all():
        df["sale_year"] = REFERENCE_YEAR
    if "sale_month" not in df.columns or df["sale_month"].isna().all():
        df["sale_month"] = 6
    if "sale_dayofweek" not in df.columns or df["sale_dayofweek"].isna().all():
        df["sale_dayofweek"] = 1
    if "statezip" not in df.columns or df["statezip"].isna().all():
        df["statezip"] = "WA 98000"

    df = add_domain_features(df)
    return df
