"""Evaluation harness: golden dataset → RAGPipeline → Ragas metrics → MLflow."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import mlflow
import yaml
from loguru import logger
from ragas import EvaluationDataset, evaluate
from ragas.dataset_schema import SingleTurnSample
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import answer_relevancy, context_precision, context_recall, faithfulness

# Ragas uses httpx internally; unset SOCKS proxy if set to avoid import errors
for _var in ("ALL_PROXY", "all_proxy", "HTTPS_PROXY", "https_proxy", "HTTP_PROXY", "http_proxy"):
    os.environ.pop(_var, None)

from langchain_ollama import ChatOllama, OllamaEmbeddings  # noqa: E402

# Make project root importable when run as a script
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from api.config import settings  # noqa: E402
from api.rag import RAGPipeline  # noqa: E402

MLFLOW_TRACKING_URI = "http://localhost:5000"
EXPERIMENT_NAME = "docsrag-rag-eval"
GOLDEN_DATASET_PATH = Path(__file__).parent / "golden_dataset.json"


# ----- data -----

def load_dataset(path: Path) -> list[dict[str, str]]:
    with path.open() as f:
        data = json.load(f)
    logger.info("Loaded {} golden samples from {}", len(data), path)
    return data


# ----- pipeline -----

def build_pipeline() -> RAGPipeline:
    return RAGPipeline()


def run_pipeline(
    pipeline: RAGPipeline,
    samples: list[dict[str, str]],
    top_k: int,
) -> list[SingleTurnSample]:
    """Run every question through the pipeline, collect answers and contexts."""
    results: list[SingleTurnSample] = []
    for i, sample in enumerate(samples, start=1):
        question = sample["question"]
        ground_truth = sample["ground_truth"]
        logger.info("[{}/{}] {}", i, len(samples), question[:80])

        answer, sources, timings = pipeline.ask(
            question=question,
            top_k=top_k,
            include_contexts=True,
        )
        contexts = [s.content for s in sources if s.content]

        results.append(
            SingleTurnSample(
                user_input=question,
                response=answer,
                retrieved_contexts=contexts,
                reference=ground_truth,
            )
        )
        logger.debug(
            "  answer_len={} contexts={} retrieval={}ms generation={}ms",
            len(answer),
            len(contexts),
            timings["retrieval_ms"],
            timings["generation_ms"],
        )

    return results


# ----- metrics -----

def build_ragas_llm(config: dict[str, Any]) -> LangchainLLMWrapper:
    llm = ChatOllama(
        model=config["llm_model"],
        base_url=settings.ollama_base_url,
        temperature=0.0,
    )
    return LangchainLLMWrapper(llm)


def build_ragas_embeddings(config: dict[str, Any]) -> LangchainEmbeddingsWrapper:
    embeddings = OllamaEmbeddings(
        model=config["llm_model"],
        base_url=settings.ollama_base_url,
    )
    return LangchainEmbeddingsWrapper(embeddings)


def compute_metrics(
    ragas_samples: list[SingleTurnSample],
    ragas_llm: LangchainLLMWrapper,
    ragas_embeddings: LangchainEmbeddingsWrapper,
) -> dict[str, float]:
    dataset = EvaluationDataset(samples=ragas_samples)

    # In ragas 0.2.x metrics are module-level singletons — set llm/embeddings in place.
    faithfulness.llm = ragas_llm
    answer_relevancy.llm = ragas_llm
    answer_relevancy.embeddings = ragas_embeddings
    context_precision.llm = ragas_llm
    context_recall.llm = ragas_llm
    metrics = [faithfulness, answer_relevancy, context_precision, context_recall]

    logger.info("Running Ragas evaluation on {} samples...", len(ragas_samples))
    result = evaluate(dataset=dataset, metrics=metrics, raise_exceptions=False)

    df = result.to_pandas()
    scores: dict[str, float] = {}
    for col in ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]:
        if col in df.columns:
            scores[col] = float(df[col].mean())

    return scores


# ----- mlflow -----

def log_to_mlflow(
    config: dict[str, Any],
    scores: dict[str, float],
    config_path: str,
    n_samples: int,
) -> str:
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run() as run:
        mlflow.log_params({
            "chunk_size": config["chunk_size"],
            "chunk_overlap": config["chunk_overlap"],
            "top_k": config["top_k"],
            "embedding_model": config["embedding_model"],
            "llm_model": config["llm_model"],
            "prompt_version": config["prompt_version"],
            "n_samples": n_samples,
            "config_file": Path(config_path).name,
        })
        mlflow.log_metrics(scores)
        run_url = (
            f"{MLFLOW_TRACKING_URI}/#/experiments/"
            f"{run.info.experiment_id}/runs/{run.info.run_id}"
        )

    return run_url


# ----- entrypoint -----

def main() -> None:
    parser = argparse.ArgumentParser(description="Run RAG evaluation")
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    args = parser.parse_args()

    config_path = Path(args.config)
    with config_path.open() as f:
        config: dict[str, Any] = yaml.safe_load(f)

    logger.info("Config: {}", config)

    samples = load_dataset(GOLDEN_DATASET_PATH)
    pipeline = build_pipeline()
    ragas_llm = build_ragas_llm(config)
    ragas_embeddings = build_ragas_embeddings(config)

    ragas_samples = run_pipeline(pipeline, samples, top_k=config["top_k"])
    scores = compute_metrics(ragas_samples, ragas_llm, ragas_embeddings)

    run_url = log_to_mlflow(config, scores, str(config_path), len(samples))

    print("\n" + "=" * 60)
    print(f"Config:  {config_path.name}")
    print(f"Samples: {len(samples)}")
    print("-" * 60)
    for metric, value in scores.items():
        print(f"  {metric:<25} {value:.4f}")
    print("=" * 60)
    print(f"MLflow run: {run_url}")


if __name__ == "__main__":
    main()
