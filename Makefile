# Makefile for Web Scraper project

.PHONY: help install setup dev test clean docker-build docker-up docker-down migrate

help:
	@echo "Web Scraper - Available commands:"
	@echo "  make install       - Install Python dependencies"
	@echo "  make setup         - Complete setup (install + download models)"
	@echo "  make dev           - Start development environment"
	@echo "  make test          - Run tests"
	@echo "  make clean         - Clean up temporary files"
	@echo "  make docker-build  - Build Docker images"
	@echo "  make docker-up     - Start Docker services"
	@echo "  make docker-down   - Stop Docker services"
	@echo "  make migrate       - Run database migrations"
	@echo "  make format        - Format code with black"
	@echo "  make lint          - Lint code with ruff"

install:
	pip install -r requirements.txt

setup: install
	@echo "Installing Playwright browsers..."
	playwright install chromium
	@echo "Downloading spaCy model..."
	python -m spacy download en_core_web_sm
	@echo "Setup complete!"

dev:
	@echo "Starting development servers..."
	@echo "Starting API server..."
	uvicorn api.app:app --reload --host 0.0.0.0 --port 8000

test:
	pytest tests/ -v --cov=. --cov-report=html

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "htmlcov" -exec rm -rf {} + 2>/dev/null || true
	rm -f .coverage

docker-build:
	docker-compose build

docker-up:
	docker-compose up -d
	@echo "Services starting..."
	@echo "API: http://localhost:8000"
	@echo "Grafana: http://localhost:3000"
	@echo "Flower: http://localhost:5555"
	@echo "MinIO Console: http://localhost:9001"

docker-down:
	docker-compose down

docker-logs:
	docker-compose logs -f

migrate:
	alembic upgrade head

format:
	black .
	ruff format .

lint:
	ruff check .

typecheck:
	mypy .

quality: format lint typecheck
	@echo "Code quality checks passed!"
