"""LLM Factory for Agent Co in Rorak AI."""

from __future__ import annotations

import os
from typing import Optional
from langchain_core.language_models.chat_models import BaseChatModel
from backend.core.config import GROQ_API_KEY, GROQ_MODEL


def build_chat_model(
    api_key: Optional[str] = None,
    model: Optional[str] = None,
    temperature: float = 0.2,
) -> BaseChatModel:
    """Build LangChain ChatGroq model instance."""
    key = (api_key or GROQ_API_KEY).strip()
    if not key:
        raise ValueError(
            "GROQ_API_KEY is not configured. Provide an API key or configure it in the backend environment."
        )

    from langchain_groq import ChatGroq

    selected_model = (model or GROQ_MODEL or "openai/gpt-oss-120b").strip()
    return ChatGroq(
        api_key=key,
        model_name=selected_model,
        temperature=temperature,
    )
