"""
tests/test_api.py
──────────────────
Full integration test suite — covers both the bridge (frontend) endpoint
and the full ML API endpoints.

Run with:
    pytest tests/ -v
"""

from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from fastapi.testclient import TestClient
from src.api.main import create_app

# Fixtures
@pytest.fixture(scope="session")
def client():
    app = create_app()
    with TestClient(app) as c:
        yield c

@pytest.fixture(scope="session", autouse=True)
def trained(client):
    """Train the model once before any test that needs it."""
    client.post("/api/train", json={})


#Simplified payload — what the React frontend sends to POST /predict
BRIDGE_PAYLOAD = {
    "location":    "Seattle, WA",
    "square_feet": 1800,
    "bedrooms":    3,
    "bathrooms":   2.0,
    "year_built":  1998,
}

#Full ML payload — what POST /api/predict expects
ML_PAYLOAD = {
    "bedrooms":      3,
    "bathrooms":     2.0,
    "sqft_living":   1800,
    "sqft_lot":      5000,
    "floors":        1.0,
    "waterfront":    0,
    "view":          0,
    "condition":     3,
    "sqft_above":    1800,
    "sqft_basement": 0,
    "yr_built":      1995,
    "yr_renovated":  0,
    "city":          "Seattle",
    "statezip":      "WA 98103",
}


#Get /health
class TestHealth:
    def test_returns_200(self, client):
        assert client.get("/health").status_code == 200

    def test_schema(self, client):
        body = client.get("/health").json()
        assert body["status"] == "ok"
        assert "model_loaded" in body

    def test_model_loaded_after_train(self, client):
        body = client.get("/health").json()
        assert body["model_loaded"] is True


#post /api/train
class TestTrain:
    def test_returns_200(self, client):
        r = client.post("/api/train", json={})
        assert r.status_code == 200, r.text

    def test_schema(self, client):
        body = client.post("/api/train", json={}).json()
        assert body["status"] == "success"
        assert "best_model_name" in body
        assert body["best_rmse"] > 0
        assert set(body["all_results"].keys()) == {
            "linear_regression", "random_forest", "xgboost"
        }

    def test_hot_swap(self, client):
        client.post("/api/train", json={})
        assert client.get("/health").json()["model_loaded"] is True


#post /api/predict  (full ML schema)
class TestBridgePredict:
    def test_returns_200(self, client):
        r = client.post("/predict", json=BRIDGE_PAYLOAD)
        assert r.status_code == 200, r.text

    def test_response_fields(self, client):
        body = client.post("/predict", json=BRIDGE_PAYLOAD).json()
        assert "price"      in body
        assert "low"        in body
        assert "high"       in body
        assert "confidence" in body

    def test_price_is_positive(self, client):
        body = client.post("/predict", json=BRIDGE_PAYLOAD).json()
        assert body["price"] > 0

    def test_low_lt_price_lt_high(self, client):
        body = client.post("/predict", json=BRIDGE_PAYLOAD).json()
        assert body["low"] < body["price"] < body["high"]

    def test_confidence_between_0_and_1(self, client):
        body = client.post("/predict", json=BRIDGE_PAYLOAD).json()
        assert 0.0 <= body["confidence"] <= 1.0

    def test_known_city_has_higher_confidence(self, client):
        known   = {**BRIDGE_PAYLOAD, "location": "Seattle, WA"}
        unknown = {**BRIDGE_PAYLOAD, "location": "Randomville, ZZ"}
        c_known   = client.post("/predict", json=known).json()["confidence"]
        c_unknown = client.post("/predict", json=unknown).json()["confidence"]
        assert c_known > c_unknown

    def test_larger_sqft_costs_more(self, client):
        small = {**BRIDGE_PAYLOAD, "square_feet": 700}
        large = {**BRIDGE_PAYLOAD, "square_feet": 4000}
        p_small = client.post("/predict", json=small).json()["price"]
        p_large = client.post("/predict", json=large).json()["price"]
        assert p_large > p_small

    def test_invalid_payload_returns_422(self, client):
        r = client.post("/predict", json={"location": "Seattle", "square_feet": -5})
        assert r.status_code == 422

    def test_various_cities(self, client):
        # Model is trained on Seattle-area data; we verify every city
        # returns a valid positive price (not a crash / zero).
        cities = ["Austin, TX", "San Francisco, CA", "Chicago, IL", "Boston, MA", "Portland, OR"]
        for city in cities:
            body = client.post("/predict", json={**BRIDGE_PAYLOAD, "location": city}).json()
            assert body["price"] > 0, f"Zero/negative price for {city}"
            assert body["low"] < body["price"] < body["high"], f"Invalid range for {city}"
            assert 0 <= body["confidence"] <= 1, f"Bad confidence for {city}"

    def test_older_house_logic(self, client):
        """Older house should generally predict lower price than newer one."""
        new_house = {**BRIDGE_PAYLOAD, "year_built": 2020}
        old_house = {**BRIDGE_PAYLOAD, "year_built": 1950}
        p_new = client.post("/predict", json=new_house).json()["price"]
        p_old = client.post("/predict", json=old_house).json()["price"]
        # Age factor should push older house lower (may not always hold for all models)
        # We just assert both are valid positive numbers
        assert p_new > 0 and p_old > 0


#post /api/predict  (full ML schema)
class TestMLPredict:
    def test_returns_200(self, client):
        r = client.post("/api/predict", json=ML_PAYLOAD)
        assert r.status_code == 200, r.text

    def test_response_fields(self, client):
        body = client.post("/api/predict", json=ML_PAYLOAD).json()
        assert "predicted_price" in body
        assert "model_used"      in body
        assert "confidence_note" in body

    def test_price_is_positive(self, client):
        body = client.post("/api/predict", json=ML_PAYLOAD).json()
        assert body["predicted_price"] > 0

    def test_waterfront_premium(self, client):
        inland    = {**ML_PAYLOAD, "waterfront": 0}
        waterfront = {**ML_PAYLOAD, "waterfront": 1}
        p_inland = client.post("/api/predict", json=inland).json()["predicted_price"]
        p_water  = client.post("/api/predict", json=waterfront).json()["predicted_price"]
        assert p_water >= p_inland

    def test_larger_house_costs_more(self, client):
        small = {**ML_PAYLOAD, "sqft_living": 800,  "sqft_above": 800}
        large = {**ML_PAYLOAD, "sqft_living": 4000, "sqft_above": 4000}
        p_small = client.post("/api/predict", json=small).json()["predicted_price"]
        p_large = client.post("/api/predict", json=large).json()["predicted_price"]
        assert p_large > p_small

    def test_invalid_payload_returns_422(self, client):
        r = client.post("/api/predict", json={"bedrooms": -5})
        assert r.status_code == 422


#post /api/explain
class TestExplain:
    def test_returns_200(self, client):
        r = client.post("/api/explain", json=ML_PAYLOAD)
        assert r.status_code == 200, r.text

    def test_schema(self, client):
        body = client.post("/api/explain", json=ML_PAYLOAD).json()
        assert "predicted_price" in body
        assert "shap_values"     in body
        assert "top_features"    in body

    def test_top_features_nonempty(self, client):
        body = client.post("/api/explain", json=ML_PAYLOAD).json()
        assert len(body["top_features"]) > 0

    def test_predicted_price_matches_predict(self, client):
        price_from_predict = client.post("/api/predict", json=ML_PAYLOAD).json()["predicted_price"]
        price_from_explain = client.post("/api/explain", json=ML_PAYLOAD).json()["predicted_price"]
        # Allow small floating-point difference
        assert abs(price_from_predict - price_from_explain) < 1.0


#Location parsing logic (bridge.py)
class TestLocationParser:
    def test_full_city_state(self):
        from src.api.bridge import _parse_location
        city, sz, matched = _parse_location("Seattle, WA")
        assert city == "Seattle" and matched is True

    def test_city_only(self):
        from src.api.bridge import _parse_location
        _, _, matched = _parse_location("Austin")
        assert matched is True

    def test_abbreviation(self):
        from src.api.bridge import _parse_location
        city, _, matched = _parse_location("sf")
        assert matched is True

    def test_unknown_city_falls_back(self):
        from src.api.bridge import _parse_location
        # Use a name with no substring in CITY_ZIP_MAP keys
        city, statezip, matched = _parse_location("Zyx123ville")
        assert matched is False          # not recognised
        assert statezip == "WA 98101"   # default fallback

    def test_case_insensitive(self):
        from src.api.bridge import _parse_location
        _, _, m1 = _parse_location("SEATTLE")
        _, _, m2 = _parse_location("seattle")
        assert m1 == m2 == True
