# DocsRAG

Self-hosted RAG (Retrieval-Augmented Generation) system for technical documentation Q&A.

**Status:** 🚧 In active development — Task 4 complete, Task 5 (hybrid search) next.

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
| Evaluation | Ragas + MLflow |
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

## Evaluation

Evaluation uses [Ragas](https://docs.ragas.io) metrics over a 25-question golden dataset derived from FastAPI documentation. Results are tracked in MLflow (`http://localhost:5000`).

```bash
make eval CONFIG=configs/baseline.yaml   # run evaluation
make mlflow-ui                           # open MLflow UI
```

### Sweep results (Ragas, 25 samples, Qwen 2.5 7B)

| Config | chunk\_size | overlap | top\_k | faithfulness | answer\_relevancy | context\_precision | context\_recall |
|---|---|---|---|---|---|---|---|
| chunk\_256 | 256 | 25 | 5 | 0.646 | 0.767 | 0.417 | 0.353 |
| baseline | 512 | 50 | 5 | 0.757 | 0.849 | 0.506 | 0.431 |
| topk\_3 | 512 | 50 | 3 | 0.719 | 0.775 | 0.517 | 0.403 |
| topk\_10 | 512 | 50 | 10 | 0.818 | **0.892** | 0.526 | 0.517 |
| **chunk\_1024** ✓ | **1024** | **100** | **5** | **0.882** | 0.886 | **0.598** | **0.557** |

**Best config: `chunk_size=1024, overlap=100, top_k=5`** — best faithfulness, context precision, and context recall. All subsequent tasks (Task 5+) compare against these numbers as baseline.

> Note: chunk\_size affects indexing. To reproduce chunk\_1024 results:
> `uv run python -m indexing.run_indexing --recreate --chunk-size 1024 --overlap 100`

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
├── evaluation/       # Evaluation framework (Task 4)
│   ├── golden_dataset.json  # 25 hand-verified Q&A pairs
│   └── run_eval.py          # Ragas + MLflow eval harness
├── configs/          # Experiment configs (YAML)
│   ├── baseline.yaml
│   ├── chunk_256.yaml
│   ├── chunk_1024.yaml
│   ├── topk_3.yaml
│   └── topk_10.yaml
├── observability/    # Task 7 — Prometheus, Grafana, LangFuse
├── benchmarks/       # Task 8 — vLLM benchmarks
├── tests/
├── docker-compose.yml
└── Makefile
```

## Current State

- **Qdrant collection:** `docsrag`, 4087 chunks, chunk\_size=512, overlap=50
- **Retrieval:** dense vector search via `qdrant-client` (cosine similarity)
- **Generation:** `temperature=0.0` for determinism; answers cite sources as `[file.md]`
- **Evaluation baseline frozen** — Tasks 5+ will compare against chunk\_1024 results above

## Makefile Reference

```bash
make up            # Start Qdrant + API + MLflow (Ollama must be running natively)
make down          # Stop services
make build         # Build API Docker image
make health        # GET /health
make ask Q="..."   # POST /ask
make warmup        # Load LLM into Ollama RAM (run after make up)
make reindex       # Recreate Qdrant collection from scratch (chunk_size=512)
make smoke         # Retrieval sanity check
make eval          # Run evaluation (CONFIG=configs/baseline.yaml by default)
make mlflow-ui     # Open MLflow UI in browser
make lint          # ruff
make format        # ruff --fix + black
make type-check    # mypy
make test          # pytest
```

## Roadmap

- [x] Task 1: Infrastructure setup
- [x] Task 2: Indexing pipeline (4087 chunks, smoke tests passing)
- [x] Task 3: Basic RAG API (FastAPI + LangChain + Ollama, verified end-to-end)
- [x] Task 4: Evaluation framework (Ragas + MLflow, 5 configs swept, baseline frozen)
- [ ] Task 5: Hybrid search + reranker
- [ ] Task 6: Agentic RAG with LangGraph
- [ ] Task 7: Observability (LangFuse + Prometheus/Grafana)
- [ ] Task 8: vLLM deployment + benchmarks
