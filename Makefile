.PHONY: help install up down logs ollama-status \
        lint format type-check test \
        fetch-docs index reindex smoke \
        build rebuild api-logs api-shell health ask warmup \
        eval mlflow-ui clean

# Default question for `make ask` if Q is not provided
Q ?= How do I define a path parameter in FastAPI?

help:
	@echo "Available commands:"
	@echo ""
	@echo "  Setup:"
	@echo "    make install       - Install Python dependencies (uv venv + editable install)"
	@echo ""
	@echo "  Lifecycle:"
	@echo "    make up            - Start all Docker services (Qdrant + API) and check Ollama"
	@echo "    make down          - Stop Docker services"
	@echo "    make build         - Build the API Docker image"
	@echo "    make rebuild       - Rebuild the API image without cache and restart"
	@echo "    make logs          - Tail logs of all services"
	@echo "    make api-logs      - Tail logs of the API service only"
	@echo "    make api-shell     - Open a shell inside the running API container"
	@echo "    make ollama-status - Check Ollama status (native install)"
	@echo ""
	@echo "  RAG API:"
	@echo "    make health        - GET /health"
	@echo "    make ask Q=\"...\"   - POST /ask with a question (default: path-params example)"
	@echo "    make warmup        - Send a warmup question to load the LLM into Ollama RAM"
	@echo ""
	@echo "  Indexing:"
	@echo "    make fetch-docs    - Download FastAPI documentation"
	@echo "    make index         - Run the indexing pipeline (incremental)"
	@echo "    make reindex       - Recreate the collection (CHUNK_SIZE=1024 CHUNK_OVERLAP=100 by default)"
	@echo "    make smoke         - Run a smoke retrieval test"
	@echo ""
	@echo "  Quality:"
	@echo "    make lint          - Run ruff linter"
	@echo "    make format        - Format code with ruff + black"
	@echo "    make type-check    - Run mypy"
	@echo "    make test          - Run pytest"
	@echo ""
	@echo "  Misc:"
	@echo "    make clean         - Remove caches and build artifacts"

# Setup

install:
	uv venv --python 3.12
	uv pip install -e ".[dev]"

# Lifecycle

up:
	docker compose up -d
	@echo "Waiting for services..."
	@sleep 3
	@curl -sf http://localhost:6333/collections > /dev/null && echo "✓ Qdrant is up" || echo "✗ Qdrant not ready"
	@curl -sf http://localhost:11434/api/tags > /dev/null && echo "✓ Ollama is up (native)" || echo "✗ Ollama not ready — start the Ollama app or run 'ollama serve'"
	@curl -sf http://localhost:8000/health > /dev/null && echo "✓ API is up" || echo "… API still starting (model loading takes ~10-20s on first run); check 'make api-logs'"

down:
	docker compose down

build:
	docker compose build api

rebuild:
	docker compose build --no-cache api
	docker compose up -d api

logs:
	docker compose logs -f

api-logs:
	docker compose logs -f api

api-shell:
	docker compose exec api /bin/bash

ollama-status:
	@curl -sf http://localhost:11434/api/tags > /dev/null && echo "✓ Ollama API responding" || echo "✗ Ollama API not responding — start the Ollama app or run 'ollama serve'"

# RAG API

health:
	@curl -sf http://localhost:8000/health | python -m json.tool || echo "✗ API not reachable on http://localhost:8000"

ask:
	@curl -s -X POST http://localhost:8000/ask \
		-H 'Content-Type: application/json' \
		-d '{"question": $(call quote,$(Q)), "top_k": 5, "include_contexts": false}' \
		| python -m json.tool

warmup:
	@echo "Sending warmup request — this loads the LLM into Ollama's RAM..."
	@curl -s -X POST http://localhost:8000/ask \
		-H 'Content-Type: application/json' \
		-d '{"question": "What is FastAPI?", "top_k": 3, "include_contexts": false}' \
		> /dev/null && echo "✓ Warmup complete" || echo "✗ Warmup failed"

# Indexing

fetch-docs:
	./indexing/fetch_docs.sh

index:
	python -m indexing.run_indexing

CHUNK_SIZE ?= 1024
CHUNK_OVERLAP ?= 100

reindex:
	python -m indexing.run_indexing --recreate --chunk-size $(CHUNK_SIZE) --overlap $(CHUNK_OVERLAP)
	rm -f data/bm25_index.pkl

smoke:
	python -m indexing.smoke_test "how to define a path parameter in FastAPI"

# Quality 

lint:
	ruff check .

format:
	ruff check --fix .
	black .

type-check:
	mypy api indexing evaluation

test:
	pytest -v

# Evaluation

CONFIG ?= configs/baseline.yaml

eval:
	python evaluation/run_eval.py --config $(CONFIG)

mlflow-ui:
	@open http://localhost:5000 || xdg-open http://localhost:5000

# Misc

clean:
	rm -rf .ruff_cache .mypy_cache .pytest_cache __pycache__ */__pycache__ */*/__pycache__
	@echo "✓ Cleaned local caches"

# Helpers 

# Safely JSON-quote a make variable: escape backslashes and double-quotes,
# then wrap in double-quotes. Lets `make ask Q='...'` survive apostrophes,
# quotes, and shell metacharacters in the question.
quote = "$(subst ",\",$(subst \,\\,$(1)))"
