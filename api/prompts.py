"""Prompt templates for the RAG pipeline.

Kept in a separate module so prompts can be iterated on without touching
pipeline code, and so we can A/B test prompt variants in the evaluation
phase (Task 4).
"""

from __future__ import annotations

from langchain_core.prompts import ChatPromptTemplate

SYSTEM_PROMPT = """You are a precise technical assistant answering questions about FastAPI documentation.

Rules you MUST follow:
1. Answer ONLY using information present in the CONTEXT below. Do not use prior knowledge.
2. If the context does not contain enough information to answer, reply exactly: "I don't know based on the provided documentation."
3. Cite sources inline by referring to their file paths in square brackets, for example: [tutorial/path-params.md].
4. Prefer concise, direct answers. Use code blocks for code examples.
5. Do not invent function names, parameters, or behaviors that are not in the context.
"""

USER_PROMPT = """CONTEXT:
{context}

QUESTION:
{question}

ANSWER:"""

PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_PROMPT),
        ("user", USER_PROMPT),
    ]
)
