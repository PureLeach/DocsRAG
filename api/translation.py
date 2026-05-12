"""Cross-language routing for the RAG pipeline.

The corpus and embeddings are English-only (`BAAI/bge-small-en-v1.5`), so a
Russian query goes through retrieval as garbage. To support Russian users
without reindexing on a multilingual model, we:

  1. Detect Cyrillic in the user's question.
  2. If present, translate RU → EN, run RAG, then translate the EN answer
     back to RU.
  3. If absent, pass through unchanged — English path is the default and
     pays zero translation cost.

Translation uses the same LLM as the main pipeline (Qwen 2.5 is multilingual)
to avoid pulling in a separate translation model.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from langchain_core.output_parsers import StrOutputParser

from api.prompts import TRANSLATE_EN_TO_RU_PROMPT, TRANSLATE_RU_TO_EN_PROMPT

if TYPE_CHECKING:
    from langchain_core.language_models.chat_models import BaseChatModel

_CYRILLIC_RE = re.compile(r"[А-Яа-яЁё]")


def contains_cyrillic(text: str) -> bool:
    return bool(_CYRILLIC_RE.search(text))


def translate_to_english(llm: BaseChatModel, text: str, callbacks: list | None = None) -> str:
    chain = TRANSLATE_RU_TO_EN_PROMPT | llm | StrOutputParser()
    config = {"callbacks": callbacks} if callbacks else {}
    return chain.invoke({"text": text}, config=config).strip()


def translate_to_russian(llm: BaseChatModel, text: str, callbacks: list | None = None) -> str:
    chain = TRANSLATE_EN_TO_RU_PROMPT | llm | StrOutputParser()
    config = {"callbacks": callbacks} if callbacks else {}
    return chain.invoke({"text": text}, config=config).strip()
