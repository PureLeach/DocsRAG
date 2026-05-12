"""LLM factory — returns ChatOllama or ChatOpenAI depending on INFERENCE_BACKEND."""

from __future__ import annotations

from typing import TYPE_CHECKING

from api.config import settings

if TYPE_CHECKING:
    from langchain_core.language_models.chat_models import BaseChatModel

# Generation budget per call. 1024 fits the longest production answers we've
# seen from /ask (~600 tokens) with margin. Set explicitly to avoid relying
# on backend defaults — vllm-metal's default can truncate long answers and
# Ollama's num_predict default (-1 = unlimited) wastes time on runaway gens.
MAX_TOKENS = 1024


def make_llm(temperature: float = 0.0, json_mode: bool = False) -> BaseChatModel:
    """Return the configured LLM instance.

    inference_backend=ollama  → ChatOllama (local dev, Metal via Ollama)
    inference_backend=vllm    → ChatOpenAI pointing at vllm/vllm-metal endpoint
                                (OpenAI-compatible API, api_key="EMPTY" is required
                                but ignored by vLLM)

    Sampling is kept minimal and consistent across backends:
      - temperature=0.0 by default (deterministic — required for reproducible eval)
      - top_p=1.0 (no nucleus filtering on top of temperature=0)
      - max_tokens=MAX_TOKENS (explicit cap so backends don't disagree on defaults)
      - repetition control: frequency_penalty=0.3 (vllm) / repeat_penalty=1.1 (Ollama default).
        Both target the same failure mode — Qwen 2.5 at temperature=0 can fall into
        degenerate citation/phrase loops without penalty. Ollama applies repeat_penalty
        by default; we set frequency_penalty for vllm explicitly.

        History: vllm-metal 0.1.0 silently accepted but ignored frequency_penalty
        (the MLX sampler in that release didn't implement OpenAI-style penalties).
        vllm-metal 0.2.0 honors it — confirmed with a raw curl test where
        `frequency_penalty=2.0` mutates repeated tokens instead of emitting them
        verbatim. So this parameter is now active on both production CUDA vllm
        and local MLX dev. If you're on 0.1.0 the value is dead code but harmless.
    """
    if settings.inference_backend == "vllm":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=settings.vllm_model,
            base_url=settings.vllm_base_url,
            api_key="EMPTY",  # vLLM ignores the key but langchain requires it
            temperature=temperature,
            top_p=1.0,
            max_tokens=MAX_TOKENS,
            frequency_penalty=0.3,
            model_kwargs={"response_format": {"type": "json_object"}} if json_mode else {},
        )

    # Default: Ollama
    from langchain_ollama import ChatOllama

    return ChatOllama(
        model=settings.ollama_model,
        base_url=settings.ollama_base_url,
        temperature=temperature,
        top_p=1.0,
        num_predict=MAX_TOKENS,  # Ollama's name for max_tokens
        format="json" if json_mode else "",
    )
