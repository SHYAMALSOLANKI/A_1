.PHONY: lint test run up down migrate install-hooks verify

# Variables
COMPOSE_FILE = infra/docker/docker-compose.yml

lint:
	poetry run ruff check .
	poetry run black --check .
	poetry run mypy .

test:
	poetry run pytest

run:
	cd services/orchestrator && poetry run uvicorn main:app --reload

up:
	docker-compose -f $(COMPOSE_FILE) up --build -d

down:
	docker-compose -f $(COMPOSE_FILE) down

migrate:
	@echo "Running migrations (placeholder)..."

verify:
	@echo "Running Verification..."
	python scripts/verify.py
