# DocsRAG

Self-hosted RAG (Retrieval-Augmented Generation) system for technical documentation Q&A.

**Status:** 🚧 In active development

## Goals

A production-grade RAG system demonstrating modern MLOps practices:
- End-to-end RAG pipeline with hybrid retrieval and reranking
- Agentic workflow via LangGraph (query rewriting, relevance grading)
- Quality evaluation with Ragas, experiment tracking with MLflow
- Full observability: LLM tracing (LangFuse) + system metrics (Prometheus/Grafana)
- Multi-backend inference: Ollama for development, vLLM for production

## Tech Stack

- **API:** FastAPI, Pydantic
- **LLM Orchestration:** LangChain, LangGraph
- **Vector DB:** Qdrant (hybrid search: dense + sparse)
- **Embeddings:** BAAI/bge-small-en-v1.5
- **Reranker:** BAAI/bge-reranker-v2-m3
- **LLM Inference:** Ollama (dev), vLLM (prod)
- **Evaluation:** Ragas + MLflow
- **Observability:** LangFuse, Prometheus, Grafana
- **Deploy:** Docker Compose, Kubernetes manifests

## Quick Start

```bash
# Prerequisites: Docker, Python 3.12, Ollama (for local dev)

# Install dependencies
make install

# Start services
make up

# Pull LLM model
ollama pull llama3.1:8b-instruct-q4_K_M
```

## Project Structure

```
docsrag/
├── api/              # FastAPI service
├── indexing/         # Document loading and indexing pipeline
├── evaluation/       # Ragas evaluation + MLflow tracking
├── observability/    # Grafana dashboards, Prometheus configs
├── benchmarks/       # Performance benchmarks (Ollama vs vLLM)
├── tests/            # Unit and integration tests
├── docs/             # Architecture docs and diagrams
└── docker-compose.yml
```

## Indexing Documentation

```bash
# Fetch FastAPI docs (one-time, ~5MB)
make fetch-docs

# Start Qdrant
make up

# Index documents into Qdrant
make index

# Smoke test
make smoke
```

## Roadmap

- [x] Task 1: Infrastructure setup
- [x] Task 2: Indexing pipeline
- [ ] Task 3: Basic RAG API (MVP)
- [ ] Task 4: Evaluation framework (Ragas + MLflow)
- [ ] Task 5: Hybrid search + reranker
- [ ] Task 6: Agentic RAG with LangGraph
- [ ] Task 7: Observability (LangFuse + Prometheus/Grafana)
- [ ] Task 8: vLLM deployment + benchmarks

