"""LLM factory — returns ChatOllama or ChatOpenAI depending on INFERENCE_BACKEND."""

from __future__ import annotations

from langchain_core.language_models.chat_models import BaseChatModel

from api.config import settings


def make_llm(temperature: float = 0.0, json_mode: bool = False) -> BaseChatModel:
    """Return the configured LLM instance.

    inference_backend=ollama  → ChatOllama (local dev, Metal via Ollama)
    inference_backend=vllm    → ChatOpenAI pointing at vllm/vllm-metal endpoint
                                (OpenAI-compatible API, api_key="EMPTY" is required
                                but ignored by vLLM)
    """
    if settings.inference_backend == "vllm":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=settings.vllm_model,
            base_url=settings.vllm_base_url,
            api_key="EMPTY",  # vLLM ignores the key but langchain requires it
            temperature=temperature,
            max_tokens=1024,
            model_kwargs={"response_format": {"type": "json_object"}} if json_mode else {},
        )

    # Default: Ollama
    from langchain_ollama import ChatOllama
    return ChatOllama(
        model=settings.ollama_model,
        base_url=settings.ollama_base_url,
        temperature=temperature,
        format="json" if json_mode else "",
    )
