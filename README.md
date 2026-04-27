# 🏠 House Price Prediction System
 
> Full-stack ML app — React frontend + FastAPI backend + scikit-learn pipeline
 
---
 
## What This Project Does
 
You enter a location, square footage, bedrooms, bathrooms, and year built into the React UI. The FastAPI backend translates that into 17 ML features, runs it through a trained sklearn pipeline, and returns a predicted price, a low/high range, and a confidence score.
 
---
 
## Tech Stack
 
| Layer | Technology |
|---|---|
| Frontend | React 18, Vite, TypeScript, shadcn/ui, Tailwind |
| Backend | Python, FastAPI, Pydantic v2, Uvicorn |
| ML | scikit-learn, XGBoost, pandas, numpy, joblib |
| Explainability | SHAP |
| Infra | Docker, nginx, docker-compose |
 
---
 
## System Architecture
 
```
┌─────────────────────────────────────────────────────────┐
│                      DEVELOPMENT                        │
│                                                         │
│   Browser :8080  ──►  Vite Dev Server  ──►  FastAPI     │
│                       (HMR + proxy)         :8000       │
└─────────────────────────────────────────────────────────┘
 
┌─────────────────────────────────────────────────────────┐
│                      PRODUCTION                         │
│                                                         │
│   Browser :80  ──►  nginx  ──►  FastAPI (internal)      │
│                    /static      /predict /api/*         │
└─────────────────────────────────────────────────────────┘
 
┌─────────────────────────────────────────────────────────┐
│                    SHARED ML LAYER                      │
│                                                         │
│  Bridge  ──►  Feature Engineering  ──►  ML Pipeline     │
│ /predict      (17 features built)     sklearn + joblib  │
│                                              │          │
│                                           SHAP          │
│                                        /api/explain     │
└─────────────────────────────────────────────────────────┘
```
 
---
 
## Request Flow
 
```
Browser                 Bridge                Feature Eng.          Model
  │                       │                       │                    │
  │  POST /predict        │                       │                    │
  │  {                    │                       │                    │
  │    location,          │  _parse_location()    │                    │
  │    square_feet,  ───► │  fill ML defaults ──► │  add_domain_   ──► │  predict()
  │    bedrooms,          │                       │  features()        │
  │    bathrooms,         │                       │  17 features       │
  │    year_built         │                       │  ready             │
  │  }                    │                       │                    │
  │                       │                       │                    │
  │ ◄────────────────────────────────────────────────────────────────  │
  │  { price, low, high, confidence }                                  │
```
 
---
 
## ML Pipeline
 
```
Raw CSV  ──►  Clean  ──►  Engineer  ──►  Preprocess  ──►  Train ×3  ──►  Save Best
4,600 rows    outlier      +5 domain     StandardScaler    LR · RF ·      min RMSE
              fence        features      + OneHotEncoder   XGBoost        (joblib)
```
 
### Training Results (your dataset)
 
| Model | RMSE | R² | Time |
|---|---|---|---|
| ✅ **Linear Regression** | **$123,074** | **0.785** | 0.09s |
| XGBoost | $125,183 | 0.778 | 1.6s |
| Random Forest | $139,981 | 0.722 | 6.2s |
 
### Engineered Features Added Automatically
 
| Feature | Formula |
|---|---|
| `house_age` | 2024 − `yr_built` |
| `since_renovated` | 2024 − `yr_renovated` (or `house_age` if never renovated) |
| `total_sqft` | `sqft_living` + `sqft_basement` |
| `bed_bath_ratio` | `bedrooms` / (`bathrooms` + 1e-6) |
| `is_renovated` | 1 if `yr_renovated` > 0, else 0 |
 
---
 
## Project Structure
 
```
house_price_prediction/
│
├── src/
│   ├── data/
│   │   └── data_loader.py              # Load CSV → clean → IQR outlier fence
│   │
│   ├── features/
│   │   └── feature_engineering.py     # Domain features + ColumnTransformer pipeline
│   │
│   ├── models/
│   │   └── trainer.py                 # Train LR/RF/XGB, pick best by RMSE, joblib
│   │
│   ├── api/
│   │   ├── main.py                    # FastAPI app factory, lifespan, static serving
│   │   ├── routes.py                  # /health  /api/train  /api/predict  /api/explain
│   │   ├── bridge.py           ✨     # POST /predict — translates frontend → ML schema
│   │   └── schemas.py                 # Pydantic v2 request/response models
│   │
│   └── utils/
│       └── logger.py                  # Structured JSON logging
│
├── frontend/                          # React + Vite + shadcn/ui (your original code)
│   ├── src/lib/predict.ts             # Calls POST /predict (no changes needed)
│   ├── vite.config.ts          ✨     # Updated: proxy /predict → :8000 in dev
│   └── Dockerfile              ✨     # New: node builder → nginx runtime
│
├── data/
│   └── data.csv                       # Seattle house dataset (4,600 rows)
│
├── saved_models/                      # joblib artifacts written here at runtime
├── tests/
│   └── test_api.py                    # 31 integration tests — all passing ✅
│
├── train_model.py                     # Standalone training CLI
├── Dockerfile                         # Backend: Python multi-stage image
├── docker-compose.yml          ✨     # Production: api + nginx frontend
├── docker-compose.dev.yml      ✨     # Dev: hot-reload both services
├── nginx.conf                  ✨     # Reverse proxy config
├── Makefile                           # Convenience commands
└── requirements.txt
 
✨ = new or updated in the full-stack integration
```
 
---
 
## How to Run
 
### Option A — Local Development (Recommended)
 
Two servers run side-by-side. Edits to Python or React reload instantly.
 
**Step 1 — Install dependencies**
 
```bash
pip install -r requirements.txt
cd frontend && npm install && cd ..
```
 
**Step 2 — Train the ML model**
 
```bash
python train_model.py
```
 
Expected output:
```
  Best model  : LINEAR_REGRESSION
  Best RMSE   : $123,074
  Best R²     : 0.785
  Saved to    : saved_models/best_model.joblib
```
 
**Step 3 — Start both servers**
 
```bash
make dev
```
 
Or in two separate terminals:
 
```bash
# Terminal 1
uvicorn src.api.main:app --port 8000 --reload
 
# Terminal 2
cd frontend && npm run dev
```
 
**Step 4 — Open in browser**
 
| What | URL |
|---|---|
| React UI | http://localhost:8080 |
| Swagger API docs | http://localhost:8000/docs |
| Health check | http://localhost:8000/health |
 
> Vite automatically proxies `POST /predict` from `:8080` to `:8000`.
> No CORS issues, no configuration needed.
 
---
 
### Option B — Docker (Production-like)
 
Everything containerised. nginx serves the React build and proxies API calls.
 
**Step 1 — Build and start**
 
```bash
docker compose up --build -d
```
 
This builds the React app, packages Python into a container, and starts both services.
 
**Step 2 — Train the model (first boot only)**
 
```bash
curl -X POST http://localhost/api/train \
  -H "Content-Type: application/json" \
  -d '{}'
```
 
The model is saved to a Docker volume and persists across container restarts.
 
**Step 3 — Open in browser**
 
| What | URL |
|---|---|
| React UI | http://localhost |
| Swagger API docs | http://localhost/docs |
| Health check | http://localhost/health |
 
**To stop:**
 
```bash
docker compose down
```
 
---
 
### Option C — Docker Dev Mode (hot-reload in containers)
 
```bash
docker compose -f docker-compose.dev.yml up --build
```
 
Backend at `:8000` with `--reload`. Frontend at `:8080` with Vite HMR. Both inside Docker.
 
---
 
## All Make Commands
 
```bash
make install     # pip install + npm install
make train       # train models, save best to saved_models/
make dev         # start backend + frontend dev servers (hot reload)
make build       # build React for production → frontend/dist/
make up          # docker compose up --build -d
make down        # stop Docker containers
make logs        # tail Docker container logs
make test        # run pytest (31 tests)
make clean       # remove build artefacts and model files
```
 
---
 
## API Reference
 
### `POST /predict` — Frontend Bridge
 
Accepts the simplified 5-field payload from the React app. Internally maps location → city/statezip, fills ML defaults, runs inference.
 
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "location":    "Seattle, WA",
    "square_feet": 1800,
    "bedrooms":    3,
    "bathrooms":   2.0,
    "year_built":  1998
  }'
```
 
Response:
```json
{
  "price": 487350.00,
  "low":   426000.00,
  "high":  549000.00,
  "confidence": 0.769,
  "model_used": "linear_regression",
  "city_matched": true
}
```
 
---
 
### `GET /health` — Health Check
 
```bash
curl http://localhost:8000/health
```
 
```json
{
  "status": "ok",
  "model_loaded": true,
  "best_model_name": "linear_regression",
  "best_rmse": 123074.3
}
```
 
---
 
### `POST /api/train` — Trigger Training
 
```bash
curl -X POST http://localhost:8000/api/train \
  -H "Content-Type: application/json" \
  -d '{}'
```
 
```json
{
  "status": "success",
  "best_model_name": "linear_regression",
  "best_rmse": 123074.30,
  "best_r2": 0.7853,
  "all_results": {
    "linear_regression": { "rmse": 123074.30, "r2": 0.7853, "train_time": 0.09 },
    "random_forest":     { "rmse": 139980.51, "r2": 0.7222, "train_time": 6.19 },
    "xgboost":           { "rmse": 125182.83, "r2": 0.7778, "train_time": 1.61 }
  },
  "training_time_s": 7.97
}
```
 
---
 
### `POST /api/predict` — Full ML Schema
 
For direct ML access with all 14 feature fields.
 
```bash
curl -X POST http://localhost:8000/api/predict \
  -H "Content-Type: application/json" \
  -d '{
    "bedrooms": 3, "bathrooms": 2.0, "sqft_living": 1800,
    "sqft_lot": 5000, "floors": 1.0, "waterfront": 0,
    "view": 0, "condition": 3, "sqft_above": 1800,
    "sqft_basement": 0, "yr_built": 1995, "yr_renovated": 0,
    "city": "Seattle", "statezip": "WA 98103"
  }'
```
 
---
 
### `POST /api/explain` — SHAP Feature Attribution
 
Same payload as `/api/predict`. Returns prediction + top-10 features ranked by impact.
 
```json
{
  "predicted_price": 487350.0,
  "shap_values": {
    "sqft_living": 42310.5,
    "city_Seattle": 18750.2,
    "house_age": -9200.1,
    "...": "..."
  },
  "top_features": {
    "sqft_living": 42310.5,
    "city_Seattle": 18750.2
  }
}
```
 
> Positive SHAP values pushed the price **up**. Negative values pushed it **down**.
 
---
 
## How the Bridge Works
 
The frontend sends 5 fields. The ML model needs 17. The bridge (`src/api/bridge.py`) fills the gap:
 
| Frontend field | ML field(s) derived |
|---|---|
| `location: "Seattle, WA"` | `city = "Seattle"`, `statezip = "WA 98101"` |
| `square_feet: 1800` | `sqft_living`, `sqft_above`, `sqft_lot = sqft × 3` |
| `year_built: 1998` | `yr_built`, then `house_age = 26`, `since_renovated = 26` |
| `bedrooms`, `bathrooms` | passed through + `bed_bath_ratio` computed |
| *(date of request)* | `sale_year`, `sale_month`, `sale_dayofweek` auto-filled |
 
**Confidence score** = model R² × 0.98, minus 0.08 if city not recognised, minus 0.05 if sqft is outside training range.
 
**Price range** = ±10–20% spread around the point estimate, wider when confidence is lower.
 
---
 
## Running Tests
 
```bash
pytest tests/ -v
```
 
31 tests, 0 failures. Covers:
 
- Health check schema and model readiness
- Train: all 3 models present in results, hot-swap works
- Bridge `/predict`: fields, positive price, `low < price < high`, confidence in 0–1, known vs unknown city confidence delta, sqft scaling, 5 different cities, invalid payload rejection
- ML `/api/predict`: full schema, waterfront premium, size scaling, validation
- SHAP `/api/explain`: schema, non-empty top features, price matches `/api/predict`
- Location parser unit tests: `"Seattle, WA"`, city-only, abbreviation `"sf"`, unknown city fallback, case-insensitive
---
 
## Environment Variables
 
| Variable | Default | Description |
|---|---|---|
| `DATA_PATH` | `data/data.csv` | Path to training CSV |
| `MODEL_DIR` | `saved_models/` | Directory for joblib artifacts |
| `MLFLOW_TRACKING_URI` | *(unset)* | MLflow server URL for experiment logging |
 
---
 
## MLflow (Optional)
 
```bash
# Start a local MLflow server
mlflow server --host 0.0.0.0 --port 5000
 
# Train with experiment tracking
MLFLOW_TRACKING_URI=http://localhost:5000 python train_model.py
 
# Open the UI
open http://localhost:5000
```
 
---
 
## Dataset
 
Seattle-area house sales — 4,600 rows, 18 columns. Target column: `price`.
 
Key columns: `bedrooms`, `bathrooms`, `sqft_living`, `sqft_lot`, `floors`, `waterfront`, `view`, `condition`, `sqft_above`, `sqft_basement`, `yr_built`, `yr_renovated`, `city`, `statezip`.