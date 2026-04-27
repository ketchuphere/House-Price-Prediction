# Makefile for House Price Prediction project
.PHONY: help install train serve dev build up down logs test clean

help:
	@echo ""
	@echo "  House Price Prediction — Available Commands"
	@echo "  ─────────────────────────────────────────────────────────"
	@echo "  make install     Install Python + Node dependencies"
	@echo "  make train       Train ML models (saves best to saved_models/)"
	@echo "  make serve       Run FastAPI backend only (port 8000)"
	@echo "  make dev         Run backend + frontend dev servers together"
	@echo "  make build       Build React frontend for production"
	@echo "  make up          Build + start full-stack via Docker Compose"
	@echo "  make down        Stop Docker Compose services"
	@echo "  make logs        Tail all service logs"
	@echo "  make test        Run Python test suite"
	@echo "  make clean       Remove saved models, build artefacts, caches"
	@echo ""


install:
	pip install -r requirements.txt
	cd frontend && npm install


train:
	python train_model.py


serve:
	uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload


dev:
	@echo "Starting backend on :8000 and frontend on :8080 …"
	@echo "Press Ctrl+C to stop."
	@trap 'kill %1 %2' INT; \
	  uvicorn src.api.main:app --port 8000 --reload & \
	  (cd frontend && npm run dev) & \
	  wait


build:
	cd frontend && npm run build
	@echo "React build complete → frontend/dist/"


prod: build
	uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --workers 4


up:
	docker compose up --build -d
	@echo ""
	@echo "  Services started:"
	@echo "    UI  → http://localhost"
	@echo "    API → http://localhost/docs"
	@echo ""

up-dev:
	docker compose -f docker-compose.dev.yml up --build

down:
	docker compose down

logs:
	docker compose logs -f


test:
	pytest tests/ -v

test-watch:
	pytest tests/ -v --tb=short -f


clean:
	rm -rf saved_models/*.joblib frontend/dist __pycache__ \
	       src/**/__pycache__ .pytest_cache logs/*.log
	@echo "Cleaned artefacts."
