.PHONY: help up down logs lint format type-check test install fetch-docs index reindex smoke

help:
	@echo "Available commands:"
	@echo "  make install       - Install Python dependencies"
	@echo "  make up            - Start Docker services and check Ollama"
	@echo "  make down          - Stop Docker services"
	@echo "  make logs          - Show Docker service logs"
	@echo "  make ollama-status - Check Ollama status (native install)"
	@echo "  make lint          - Run ruff linter"
	@echo "  make format        - Format code with black + ruff"
	@echo "  make type-check    - Run mypy"
	@echo "  make test          - Run pytest"

install:
	uv venv --python 3.12
	uv pip install -e ".[dev]"

up:
	docker compose up -d
	@echo "Waiting for services..."
	@sleep 3
	@curl -sf http://localhost:6333/collections > /dev/null && echo "✓ Qdrant is up" || echo "✗ Qdrant not ready"
	@curl -sf http://localhost:11434/api/tags > /dev/null && echo "✓ Ollama is up (native)" || echo "✗ Ollama not ready — run 'brew services start ollama'"

down:
	docker compose down

logs:
	docker compose logs -f

ollama-status:
	@brew services list | grep ollama || echo "Ollama not installed via brew"
	@curl -sf http://localhost:11434/api/tags > /dev/null && echo "✓ Ollama API responding" || echo "✗ Ollama API not responding"

lint:
	ruff check .

format:
	ruff check --fix .
	black .

type-check:
	mypy api indexing evaluation

test:
	pytest -v

fetch-docs:
	./indexing/fetch_docs.sh

index:
	python -m indexing.run_indexing

reindex:
	python -m indexing.run_indexing --recreate

smoke:
	python -m indexing.smoke_test "how to define a path parameter in FastAPI"