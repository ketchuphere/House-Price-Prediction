"""
api/main.py
────────────
FastAPI application factory.

Route layout
────────────
  POST /predict          ← bridge endpoint (simplified frontend payload)
  GET  /health           ← liveness probe
  POST /api/train        ← full training trigger
  POST /api/predict      ← full ML schema (power users / curl)
  POST /api/explain      ← SHAP explainability
  GET  /                 ← serves built React frontend (production)
  GET  /assets/*         ← static assets

Development workflow
────────────────────
  Backend  → uvicorn src.api.main:app --port 8000 --reload
  Frontend → cd frontend && npm run dev  (Vite proxies /predict → :8000)

Production workflow
───────────────────
  1. cd frontend && npm run build        (outputs to frontend/dist/)
  2. uvicorn src.api.main:app --port 8000
     FastAPI serves dist/ at "/"  +  API at "/predict" and "/api/*"
"""

from __future__ import annotations

import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from src.api.bridge import bridge_router
from src.api.routes import router as api_router
from src.models.trainer import load_best_model
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Path to the Vite production build output
FRONTEND_DIST = Path(__file__).resolve().parents[2] / "frontend" / "dist"



@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("House Price Prediction API starting…")
    try:
        app.state.model = load_best_model()
        logger.info("Pre-trained model loaded on startup")
    except FileNotFoundError:
        app.state.model = None
        logger.warning("No model found — call POST /api/train first")
    yield
    logger.info("API shutting down")


def create_app() -> FastAPI:
    app = FastAPI(
        title="House Price Prediction API",
        description=(
            "Full-stack ML API for predicting house prices.\n\n"
            "**Frontend bridge:** `POST /predict` (simplified payload)\n\n"
            "**ML API:**\n"
            "- `POST /api/train` — train all models\n"
            "- `POST /api/predict` — full feature prediction\n"
            "- `POST /api/explain` — SHAP attribution\n"
            "- `GET  /health` — health check"
        ),
        version="2.0.0",
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=[
            "http://localhost:8080",
            "http://localhost:3000",
            "http://localhost:8000",
        ],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    #Request latency logging
    @app.middleware("http")
    async def log_requests(request: Request, call_next):
        start = time.perf_counter()
        response = await call_next(request)
        latency_ms = round((time.perf_counter() - start) * 1000, 1)
        logger.info(
            "HTTP",
            extra={
                "method": request.method,
                "path": request.url.path,
                "status": response.status_code,
                "ms": latency_ms,
            },
        )
        return response

    #Global error handler
    @app.exception_handler(Exception)
    async def global_exc(request: Request, exc: Exception):
        logger.error("Unhandled exception", extra={"error": str(exc)})
        return JSONResponse(status_code=500, content={"detail": "Internal server error"})

    #Bridge endpoint (simplified payload for React frontend)
    app.include_router(bridge_router)

    #Core ML routes (health, full predict, explain, train)
    app.include_router(api_router)

    #Same ML routes also under /api prefix
    app.include_router(api_router, prefix="/api", tags=["ML API (prefixed)"])

    #Serve React frontend static files (production only)
    if FRONTEND_DIST.exists():
        logger.info("Serving React frontend", extra={"path": str(FRONTEND_DIST)})

        assets_dir = FRONTEND_DIST / "assets"
        if assets_dir.exists():
            app.mount(
                "/assets",
                StaticFiles(directory=str(assets_dir)),
                name="assets",
            )

        @app.get("/favicon.ico", include_in_schema=False)
        async def favicon():
            f = FRONTEND_DIST / "favicon.ico"
            return FileResponse(str(f)) if f.exists() else JSONResponse({}, status_code=404)

        @app.get("/{full_path:path}", include_in_schema=False)
        async def spa_fallback(full_path: str):
            index = FRONTEND_DIST / "index.html"
            if index.exists():
                return FileResponse(str(index))
            return JSONResponse(
                {"detail": "Frontend not built. Run: cd frontend && npm run build"},
                status_code=404,
            )
    else:
        logger.warning("Frontend dist not found — API-only mode.")

    return app


app = create_app()
