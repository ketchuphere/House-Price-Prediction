"""
api/schemas.py
───────────────
Pydantic v2 schemas for all API request and response bodies.

Pydantic handles:
  • Type coercion   (e.g., "3" → 3)
  • Field validation (e.g., bedrooms must be >= 0)
  • Automatic OpenAPI / Swagger documentation

Design choices:
  • All house-feature fields have sensible defaults so callers only need to
    supply what they know; the feature-engineering layer fills the gaps.
  • Response models are strict (no extra fields) to prevent info leakage.
"""

from __future__ import annotations

from typing import Dict, Optional

from pydantic import BaseModel, Field, field_validator


#request schemas
class HouseFeatures(BaseModel):
    """
    Input payload for POST /predict and POST /explain.

    All fields correspond to columns in the Seattle house-price dataset.
    Date-derived fields (sale_year, sale_month, sale_dayofweek) are optional;
    the server fills in current-year defaults when omitted.
    """

    bedrooms:      float = Field(...,  ge=0, le=30,  description="Number of bedrooms")
    bathrooms:     float = Field(...,  ge=0, le=20,  description="Number of bathrooms (can be fractional)")
    sqft_living:   int   = Field(...,  ge=1,         description="Interior living area in sq ft")
    sqft_lot:      int   = Field(...,  ge=1,         description="Lot size in sq ft")
    floors:        float = Field(1.0,  ge=1, le=5,   description="Number of floors")
    waterfront:    int   = Field(0,    ge=0, le=1,   description="1 if waterfront property, else 0")
    view:          int   = Field(0,    ge=0, le=4,   description="View quality index (0–4)")
    condition:     int   = Field(3,    ge=1, le=5,   description="Overall condition (1–5)")
    sqft_above:    int   = Field(0,    ge=0,         description="Square footage above basement")
    sqft_basement: int   = Field(0,    ge=0,         description="Basement square footage")
    yr_built:      int   = Field(...,  ge=1800, le=2024, description="Year the house was built")
    yr_renovated:  int   = Field(0,    ge=0,         description="Year last renovated (0 = never)")
    city:          str   = Field("Seattle",          description="City name")
    statezip:      str   = Field("WA 98000",         description="State + ZIP code")

    #Optional date-derived fields (server defaults to 2024-06 if omitted)
    sale_year:      Optional[int] = Field(None, ge=2000, le=2030)
    sale_month:     Optional[int] = Field(None, ge=1,    le=12)
    sale_dayofweek: Optional[int] = Field(None, ge=0,    le=6)

    @field_validator("sqft_above")
    @classmethod
    def sqft_above_default(cls, v: int, info) -> int:
        """Default sqft_above to sqft_living when caller omits it."""
        if v == 0 and "sqft_living" in (info.data or {}):
            return info.data["sqft_living"]
        return v

    model_config = {"json_schema_extra": {
        "example": {
            "bedrooms": 3,
            "bathrooms": 2.0,
            "sqft_living": 1800,
            "sqft_lot": 5000,
            "floors": 1.0,
            "waterfront": 0,
            "view": 0,
            "condition": 3,
            "sqft_above": 1800,
            "sqft_basement": 0,
            "yr_built": 1995,
            "yr_renovated": 0,
            "city": "Seattle",
            "statezip": "WA 98103",
        }
    }}


class TrainRequest(BaseModel):
    """Optional body for POST /train — lets callers override the data path."""
    data_path: Optional[str] = Field(
        None,
        description="Absolute path to CSV on the server (defaults to bundled dataset).",
    )


#response schemas
class PredictionResponse(BaseModel):
    """Response envelope for POST /predict."""
    predicted_price:  float  = Field(..., description="Predicted house price in USD")
    model_used:       str    = Field(..., description="Name of the model that produced this prediction")
    confidence_note:  str    = Field(..., description="Human-readable caveat about the prediction")


class ModelMetrics(BaseModel):
    """Per-model evaluation results stored after training."""
    rmse:       float
    r2:         float
    train_time: float


class TrainResponse(BaseModel):
    """Response envelope for POST /train."""
    status:            str
    best_model_name:   str
    best_rmse:         float
    best_r2:           float
    all_results:       Dict[str, ModelMetrics]
    training_time_s:   float
    model_path:        str


class HealthResponse(BaseModel):
    """Response envelope for GET /health."""
    status:          str
    model_loaded:    bool
    best_model_name: Optional[str]
    best_rmse:       Optional[float]


class ExplainResponse(BaseModel):
    """Response envelope for POST /explain."""
    predicted_price: float
    shap_values:     Dict[str, float]
    top_features:    Dict[str, float]


class ErrorResponse(BaseModel):
    """Standard error envelope."""
    detail: str
