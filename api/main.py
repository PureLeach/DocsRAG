"""FastAPI entrypoint for the DocsRAG API."""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import Depends, FastAPI, HTTPException
from loguru import logger

from api.config import settings
from api.rag import RAGPipeline, get_pipeline
from api.schemas import AskRequest, AskResponse, HealthResponse


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Warm up the RAG pipeline at startup so the first request isn't slow."""
    logger.info("Starting DocsRAG API")
    get_pipeline()  # constructs and caches the singleton
    logger.info("DocsRAG API ready")
    yield
    logger.info("Shutting down DocsRAG API")


app = FastAPI(
    title="DocsRAG API",
    description="Q&A over FastAPI documentation via retrieval-augmented generation.",
    version="0.1.0",
    lifespan=lifespan,
)


PipelineDep = Annotated[RAGPipeline, Depends(get_pipeline)]


@app.get("/health", response_model=HealthResponse)
def health(pipeline: PipelineDep) -> HealthResponse:
    """Liveness + readiness check.

    Touches Qdrant to confirm the collection is reachable and reports
    configured model names. If Qdrant is down, returns 503.
    """
    try:
        points = pipeline.collection_points_count()
    except Exception as exc:  # noqa: BLE001 — we want a generic 503 here
        logger.error("Health check failed: {}", exc)
        raise HTTPException(status_code=503, detail=f"Qdrant unreachable: {exc}") from exc

    return HealthResponse(
        status="ok",
        qdrant_collection=settings.qdrant_collection,
        qdrant_points=points,
        ollama_model=settings.ollama_model,
        embedding_model=settings.embedding_model,
    )


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest, pipeline: PipelineDep) -> AskResponse:
    """Answer a question using retrieval-augmented generation."""
    try:
        answer, sources, timings = pipeline.ask(
            question=request.question,
            top_k=request.top_k,
            include_contexts=request.include_contexts,
        )
    except Exception as exc:
        logger.exception("RAG pipeline failed")
        raise HTTPException(status_code=500, detail=f"Pipeline error: {exc}") from exc

    return AskResponse(
        question=request.question,
        answer=answer,
        sources=sources,
        retrieval_ms=timings["retrieval_ms"],
        generation_ms=timings["generation_ms"],
        total_ms=timings["total_ms"],
    )