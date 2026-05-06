"""RAG pipeline: retrieve relevant chunks from Qdrant, generate an answer with Ollama."""

from __future__ import annotations

import time
from dataclasses import dataclass
from functools import lru_cache

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import ChatOllama
from loguru import logger
from qdrant_client import QdrantClient
from qdrant_client.models import ScoredPoint

from api.config import settings
from api.prompts import PROMPT
from api.schemas import Source
from indexing.embeddings import EmbeddingModel


@dataclass(slots=True)
class RetrievalHit:
    """A single retrieval result with score."""

    document: Document
    score: float


class RAGPipeline:
    """End-to-end retrieve → generate pipeline.

    Heavy resources (embedder, Qdrant client, LLM client) are constructed
    once and reused across requests. See `get_pipeline()` for the cached
    application-wide instance.
    """

    def __init__(self) -> None:
        logger.info("Initializing RAGPipeline")

        # Embedder: same model used during indexing — guarantees identical
        # vector space and pooling/normalization between index- and query-time.
        self._embedder = EmbeddingModel(model_name=settings.embedding_model)

        # Qdrant: direct client — bypasses langchain-qdrant metadata handling
        # which changed in 0.2.x. Our payload is flat: text, source_path,
        # header_path, chunk_index — we map it to Document ourselves.
        self._qdrant_client = QdrantClient(url=settings.qdrant_url)

        # LLM: ChatOllama hits OLLAMA_BASE_URL.
        # Variant A (current): host.docker.internal:11434 from container,
        # localhost:11434 from host.
        # Variant B (Ollama in Docker): http://ollama:11434 via compose network.
        self._llm = ChatOllama(
            model=settings.ollama_model,
            base_url=settings.ollama_base_url,
            temperature=0.0,  # deterministic for documentation Q&A
        )

        # Composed chain: prompt → llm → string output.
        self._chain = PROMPT | self._llm | StrOutputParser()

        logger.info(
            "RAGPipeline ready | collection={} | model={} | base_url={}",
            settings.qdrant_collection,
            settings.ollama_model,
            settings.ollama_base_url,
        )

    # ----- public API -----

    def retrieve(self, query: str, top_k: int) -> list[RetrievalHit]:
        """Vector-search the Qdrant collection."""
        query_vector = self._embedder.encode([query], show_progress=False)[0]
        response = self._qdrant_client.query_points(
            collection_name=settings.qdrant_collection,
            query=query_vector,
            limit=top_k,
            with_payload=True,
        )
        hits = [self._scored_point_to_hit(r) for r in response.points]
        logger.debug(
            "Retrieved {} hits for query={!r} (top_score={:.3f})",
            len(hits),
            query,
            hits[0].score if hits else 0.0,
        )
        return hits

    @staticmethod
    def _scored_point_to_hit(point: ScoredPoint) -> RetrievalHit:
        payload = point.payload or {}
        doc = Document(
            page_content=str(payload.get("text", "")),
            metadata={
                "source_path": payload.get("source_path", "unknown"),
                "header_path": payload.get("header_path", ""),
                "chunk_index": payload.get("chunk_index", -1),
            },
        )
        return RetrievalHit(document=doc, score=float(point.score))

    def generate(self, question: str, hits: list[RetrievalHit]) -> str:
        """Build the prompt from retrieved chunks and call the LLM."""
        context = self._format_context(hits)
        answer = self._chain.invoke({"context": context, "question": question})
        return answer.strip()

    def ask(
        self, question: str, top_k: int, include_contexts: bool
    ) -> tuple[str, list[Source], dict[str, int]]:
        """Full pipeline: retrieve → generate. Returns answer, sources, timings."""
        t0 = time.perf_counter()
        hits = self.retrieve(question, top_k=top_k)
        t1 = time.perf_counter()
        answer = self.generate(question, hits)
        t2 = time.perf_counter()

        sources = [self._hit_to_source(h, include_contexts) for h in hits]
        timings = {
            "retrieval_ms": int((t1 - t0) * 1000),
            "generation_ms": int((t2 - t1) * 1000),
            "total_ms": int((t2 - t0) * 1000),
        }
        logger.info(
            "ask | retrieval={}ms generation={}ms total={}ms | hits={} | q={!r}",
            timings["retrieval_ms"],
            timings["generation_ms"],
            timings["total_ms"],
            len(hits),
            question[:80],
        )
        return answer, sources, timings

    # ----- diagnostics -----

    def collection_points_count(self) -> int:
        """Number of points in the Qdrant collection (used by /health)."""
        info = self._qdrant_client.count(
            collection_name=settings.qdrant_collection, exact=True
        )
        return int(info.count)

    # ----- internals -----

    @staticmethod
    def _format_context(hits: list[RetrievalHit]) -> str:
        """Render retrieved chunks into a single context string for the prompt."""
        if not hits:
            return "(no relevant context found)"
        blocks: list[str] = []
        for i, hit in enumerate(hits, start=1):
            md = hit.document.metadata or {}
            source_path = md.get("source_path", "unknown")
            header_path = md.get("header_path", "")
            header_line = f" — {header_path}" if header_path else ""
            blocks.append(
                f"[{i}] {source_path}{header_line}\n{hit.document.page_content}"
            )
        return "\n\n---\n\n".join(blocks)

    @staticmethod
    def _hit_to_source(hit: RetrievalHit, include_contexts: bool) -> Source:
        md = hit.document.metadata or {}
        return Source(
            source_path=str(md.get("source_path", "unknown")),
            header_path=str(md.get("header_path", "")),
            score=hit.score,
            content=hit.document.page_content if include_contexts else None,
        )


@lru_cache(maxsize=1)
def get_pipeline() -> RAGPipeline:
    """Application-wide singleton. Lazily constructed on first call."""
    return RAGPipeline()