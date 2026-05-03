"""
api/bridge.py
──────────────
Bridge endpoint that translates the simplified frontend payload into the full
ML feature schema, runs inference, and returns the frontend-friendly response.

Frontend payload (from predict.ts):
    { location, square_feet, bedrooms, bathrooms, year_built }

ML model expects:
    { bedrooms, bathrooms, sqft_living, sqft_lot, floors, waterfront,
      view, condition, sqft_above, sqft_basement, yr_built, yr_renovated,
      city, statezip, sale_year, sale_month, sale_dayofweek }

Response the frontend reads:
    { price, low, high, confidence }

Location parsing
────────────────
We extract the city name from free-text like:
  "Seattle, WA" → city="Seattle", statezip="WA 98000"
  "Austin, TX"  → city="Austin",  statezip="TX 78700"
  "Seattle"     → city="Seattle", statezip="WA 98000"  (Seattle default)

Unknown cities fall back to Seattle defaults so inference always runs.

Confidence heuristic
────────────────────
We can't get a true confidence interval from a point-estimate model without
calibration sets, so we derive a synthetic but meaningful value from:
  • R² of the best model (stored in metadata)
  • Whether the city was recognised
  • Whether the house features are in the training distribution
"""

from __future__ import annotations

import re
from datetime import datetime
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Request, status
from pydantic import BaseModel, Field

from src.features.feature_engineering import prepare_inference_df
from src.models.trainer import load_metadata, predict_price
from src.utils.logger import get_logger

logger = get_logger(__name__)
bridge_router = APIRouter()

# City → (state_abbr, zip_prefix) look-up table
# Covers the most common US markets; unknown cities fall through to the default.
CITY_ZIP_MAP: dict[str, tuple[str, str]] = {
    "seattle":       ("WA", "98101"),
    "bellevue":      ("WA", "98004"),
    "tacoma":        ("WA", "98401"),
    "redmond":       ("WA", "98052"),
    "kirkland":      ("WA", "98033"),
    "san francisco": ("CA", "94102"),
    "sf":            ("CA", "94102"),
    "los angeles":   ("CA", "90001"),
    "la":            ("CA", "90001"),
    "san jose":      ("CA", "95101"),
    "san diego":     ("CA", "92101"),
    "portland":      ("OR", "97201"),
    "new york":      ("NY", "10001"),
    "nyc":           ("NY", "10001"),
    "boston":        ("MA", "02101"),
    "chicago":       ("IL", "60601"),
    "austin":        ("TX", "78701"),
    "dallas":        ("TX", "75201"),
    "houston":       ("TX", "77001"),
    "denver":        ("CO", "80201"),
    "miami":         ("FL", "33101"),
    "atlanta":       ("GA", "30301"),
    "phoenix":       ("AZ", "85001"),
    "minneapolis":   ("MN", "55401"),
    "detroit":       ("MI", "48201"),
    "nashville":     ("TN", "37201"),
    "charlotte":     ("NC", "28201"),
    "raleigh":       ("NC", "27601"),
    "columbus":      ("OH", "43201"),
    "indianapolis":  ("IN", "46201"),
    "memphis":       ("TN", "38101"),
    "louisville":    ("KY", "40201"),
}

DEFAULT_CITY     = "Seattle"
DEFAULT_STATEZIP = "WA 98101"


# Schemas
class FrontendPredictRequest(BaseModel):
    """
    Simplified payload sent by the React frontend (src/lib/predict.ts).
    Field names use snake_case matching what predict.ts serialises.
    """
    location:    str   = Field(..., description="Free-text city/state, e.g. 'Seattle, WA'")
    square_feet: int   = Field(..., ge=100, le=50_000)
    bedrooms:    int   = Field(..., ge=0, le=20)
    bathrooms:   float = Field(..., ge=0, le=20)
    year_built:  int   = Field(..., ge=1800, le=datetime.now().year + 1)

    model_config = {"json_schema_extra": {"example": {
        "location": "Seattle, WA",
        "square_feet": 1800,
        "bedrooms": 3,
        "bathrooms": 2.0,
        "year_built": 1998,
    }}}


class FrontendPredictResponse(BaseModel):
    """Response shape the React frontend reads (src/lib/predict.ts)."""
    price:      float = Field(..., description="Point-estimate price in USD")
    low:        float = Field(..., description="Lower bound of predicted range")
    high:       float = Field(..., description="Upper bound of predicted range")
    confidence: float = Field(..., ge=0, le=1, description="Confidence score 0–1")

    # Extra fields for power users / debugging (ignored by frontend)
    model_used: Optional[str] = None
    city_matched: Optional[bool] = None


#Helper functions for the bridge endpoint
def _parse_location(raw: str) -> tuple[str, str, bool]:
    """
    Extract a city name and statezip string from a free-text location.

    Returns
    -------
    (city, statezip, matched)
        city      — display city name (Title Case)
        statezip  — e.g. "WA 98101"
        matched   — True if city was found in CITY_ZIP_MAP
    """
    # Normalise: strip punctuation, lowercase
    normalised = raw.strip().lower()

    # Remove state abbreviation suffix like ", WA" or " WA"
    city_part = re.split(r",\s*|\s+[A-Z]{2}$", normalised)[0].strip()

    if city_part in CITY_ZIP_MAP:
        state, zipcode = CITY_ZIP_MAP[city_part]
        return city_part.title(), f"{state} {zipcode}", True

    # Try partial match (e.g. "downtown seattle" → "seattle")
    for key, (state, zipcode) in CITY_ZIP_MAP.items():
        if key in city_part or city_part in key:
            return key.title(), f"{state} {zipcode}", True

    # Unknown city — use Seattle as proxy (in-distribution for the model)
    return city_part.title() or DEFAULT_CITY, DEFAULT_STATEZIP, False


def _confidence_score(r2: float, city_matched: bool, sqft: int) -> float:
    """
    Derive a synthetic confidence value from model quality + input signals.

    Base = R² of the best trained model (objective model quality).
    Penalise:
      -0.08 if the city was not in our training distribution
      -0.05 if the square footage is outside the typical training range (500–5000)
    """
    base = float(r2) * 0.98   # slight haircut; R² is optimistic on seen data
    if not city_matched:
        base -= 0.08
    if not (500 <= sqft <= 5_000):
        base -= 0.05
    return round(max(0.35, min(0.97, base)), 4)


def _price_range(price: float, confidence: float) -> tuple[float, float]:
    """
    Derive a low/high range from the point estimate.

    Spread is inversely proportional to confidence:
    high confidence → ±10%, low confidence → ±20%
    """
    spread = 0.10 + (1 - confidence) * 0.10   # 10%–20%
    low  = round(price * (1 - spread) / 1_000) * 1_000
    high = round(price * (1 + spread) / 1_000) * 1_000
    return float(low), float(high)


def get_model(request: Request):
    """Dependency: retrieve loaded model or raise 503."""
    model = getattr(request.app.state, "model", None)
    if model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded. Call POST /train first.",
        )
    return model


#Bridge endpoint
@bridge_router.post(
    "/predict",
    response_model=FrontendPredictResponse,
    summary="[Frontend] Simplified prediction endpoint",
    tags=["Frontend Bridge"],
)
def frontend_predict(
    body: FrontendPredictRequest,
    model=Depends(get_model),
) -> FrontendPredictResponse:
    """
    Accepts the simplified frontend payload, maps it to the full ML feature
    schema, runs the trained pipeline, and returns a frontend-friendly response.

    This endpoint intentionally shadows the old ``POST /predict`` so the React
    app can call ``/predict`` without any path prefix changes.
    The full ML endpoint is still available at ``POST /api/predict``.
    """
   
    city, statezip, city_matched = _parse_location(body.location)
    logger.info(
        "Bridge predict",
        extra={"city": city, "matched": city_matched, "sqft": body.square_feet},
    )

    #Build full feature dict using sensible defaults for missing fields
    now = datetime.now()
    feature_dict = {
        "bedrooms":      body.bedrooms,
        "bathrooms":     body.bathrooms,
        "sqft_living":   body.square_feet,
        "sqft_lot":      body.square_feet * 3,   # median lot ≈ 3× living area
        "floors":        1.0 if body.square_feet < 2_000 else 2.0,
        "waterfront":    0,
        "view":          0,
        "condition":     3,                       # average condition
        "sqft_above":    body.square_feet,        # assume no basement
        "sqft_basement": 0,
        "yr_built":      body.year_built,
        "yr_renovated":  0,
        "city":          city,
        "statezip":      statezip,
        "sale_year":     now.year,
        "sale_month":    now.month,
        "sale_dayofweek": now.weekday(),
    }

    #Inference
    try:
        input_df = prepare_inference_df(feature_dict)
        price    = predict_price(model, input_df)
    except Exception as exc:
        logger.error("Bridge inference error", extra={"error": str(exc)})
        raise HTTPException(status_code=422, detail=f"Prediction failed: {exc}")

    #Build response
    meta       = load_metadata()
    r2         = meta.get("best_r2", 0.78)
    model_name = meta.get("best_model_name", "unknown")
    confidence = _confidence_score(r2, city_matched, body.square_feet)
    low, high  = _price_range(price, confidence)

    logger.info(
        "Bridge predict done",
        extra={"price": round(price, 2), "confidence": confidence, "city": city},
    )

    return FrontendPredictResponse(
        price=round(price, 2),
        low=low,
        high=high,
        confidence=confidence,
        model_used=model_name,
        city_matched=city_matched,
    )
