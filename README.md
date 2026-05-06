# DocsRAG

Self-hosted RAG (Retrieval-Augmented Generation) system for technical documentation Q&A.

**Status:** 🚧 In active development — Task 3 complete, Task 4 (evaluation) next.

## Goals

A production-grade RAG system demonstrating modern MLOps practices:
- End-to-end RAG pipeline with hybrid retrieval and reranking
- Agentic workflow via LangGraph (query rewriting, relevance grading)
- Quality evaluation with Ragas, experiment tracking with MLflow
- Full observability: LLM tracing (LangFuse) + system metrics (Prometheus/Grafana)
- Multi-backend inference: Ollama for development, vLLM for production

## Tech Stack

| Layer | Technology |
|---|---|
| API | FastAPI + Pydantic |
| LLM | Qwen 2.5 7B Instruct via Ollama |
| Embeddings | BAAI/bge-small-en-v1.5 (384-dim, MPS on Apple Silicon) |
| Vector DB | Qdrant (cosine similarity) |
| Orchestration | LangChain |
| Evaluation | Ragas + MLflow *(Task 4)* |
| Observability | LangFuse, Prometheus, Grafana *(Task 7)* |
| Prod inference | vLLM *(Task 8)* |
| Packaging | Docker Compose, uv |

## Quick Start

**Prerequisites:** Docker Desktop, Python 3.12, [Ollama](https://ollama.com) (native on macOS).

```bash
# 1. Pull the LLM model into Ollama
ollama pull qwen2.5:7b-instruct-q4_K_M

# 2. Install Python dependencies
make install

# 3. Fetch FastAPI docs and index into Qdrant (one-time, ~2 min)
make fetch-docs
make up
make reindex

# 4. Warm up and query
make warmup
make ask Q='How do I define a path parameter in FastAPI?'
```

## API

The RAG API runs on `http://localhost:8000`.

### `GET /health`

```bash
make health
```

Returns Qdrant collection status, point count, and configured model names.

### `POST /ask`

```bash
make ask Q='How does dependency injection work in FastAPI?'
```

Parameters:

| Field | Type | Default | Description |
|---|---|---|---|
| `question` | string | — | Natural language question |
| `top_k` | int | 5 | Number of chunks to retrieve |
| `include_contexts` | bool | false | Include raw chunk text in response |

Response includes `answer`, `sources` (with `source_path`, `header_path`, `score`), and timing breakdown (`retrieval_ms`, `generation_ms`, `total_ms`).

## Project Structure

```
docsrag/
├── api/              # FastAPI service (Task 3)
│   ├── main.py       # /health, /ask endpoints + lifespan
│   ├── rag.py        # RAGPipeline: embed → query_points → generate
│   ├── prompts.py    # System + user prompt templates
│   ├── schemas.py    # Pydantic request/response models
│   └── config.py     # Pydantic Settings
├── indexing/         # Indexing pipeline (Task 2)
│   ├── loader.py     # Markdown loader
│   ├── chunker.py    # Hierarchical chunker (header + recursive)
│   ├── embeddings.py # EmbeddingModel (sentence-transformers)
│   └── qdrant_store.py
├── evaluation/       # Task 4 — Ragas + MLflow
├── observability/    # Task 7 — Prometheus, Grafana, LangFuse
├── benchmarks/       # Task 8 — vLLM benchmarks
├── configs/          # Experiment configs (YAML)
├── tests/
├── docker-compose.yml
└── Makefile
```

## Current State

- **Qdrant collection:** `docsrag`, 4087 chunks from 153 FastAPI docs markdown files
- **Chunk size:** 512 tokens, overlap 50 (hierarchical: header splits → recursive)
- **Retrieval:** dense vector search via `qdrant-client` directly (cosine similarity)
- **Generation:** `temperature=0.0` for determinism; answers cite sources as `[file.md]`
- **Observed scores:** top-1 ~0.85 for path parameters, ~0.79 for file uploads

## Makefile Reference

```bash
make up            # Start Qdrant + API (Ollama must be running natively)
make down          # Stop services
make build         # Build API Docker image
make health        # GET /health
make ask Q="..."   # POST /ask
make warmup        # Load LLM into Ollama RAM (run after make up)
make reindex       # Recreate Qdrant collection from scratch
make smoke         # Retrieval sanity check
make lint          # ruff
make format        # ruff --fix + black
make type-check    # mypy
make test          # pytest
```

## Roadmap

- [x] Task 1: Infrastructure setup
- [x] Task 2: Indexing pipeline (4087 chunks, smoke tests passing)
- [x] Task 3: Basic RAG API (FastAPI + LangChain + Ollama, verified end-to-end)
- [ ] Task 4: Evaluation framework (Ragas + MLflow)
- [ ] Task 5: Hybrid search + reranker
- [ ] Task 6: Agentic RAG with LangGraph
- [ ] Task 7: Observability (LangFuse + Prometheus/Grafana)
- [ ] Task 8: vLLM deployment + benchmarks
