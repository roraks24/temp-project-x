"""LLM Factory for Agent Co in Rorak AI supporting multi-provider & user custom keys."""

from __future__ import annotations

import os
from typing import Any, Optional
from langchain_core.language_models.chat_models import BaseChatModel
from backend.core.config import GROQ_API_KEY, GROQ_MODEL


def build_chat_model(
    api_key: Optional[str] = None,
    model: Optional[str] = None,
    temperature: float = 0.2,
    provider: Optional[str] = None,
    base_url: Optional[str] = None,
    custom_keys: Optional[dict[str, Any]] = None,
) -> BaseChatModel:
    """
    Build LangChain chat model with dynamic provider resolution and user-supplied custom API keys.
    Supports Groq, OpenAI, and any OpenAI-compatible endpoint (Ollama, OpenRouter, DeepSeek, LocalAI).
    """
    # Extract from custom_keys dictionary if provided
    if isinstance(custom_keys, dict):
        provider = custom_keys.get("provider") or provider
        # Try provider-specific key, then generic api_key
        prov_key = f"{provider}_api_key" if provider else None
        api_key = (
            (custom_keys.get(prov_key) if prov_key else None)
            or custom_keys.get("api_key")
            or custom_keys.get("groq_api_key")
            or custom_keys.get("openai_api_key")
            or custom_keys.get("custom_api_key")
            or api_key
        )
        model = custom_keys.get("model") or custom_keys.get("custom_model") or model
        base_url = custom_keys.get("base_url") or custom_keys.get("custom_base_url") or base_url

    norm_provider = (provider or "").lower().strip()

    # Auto-detect provider if unspecified
    if not norm_provider:
        if base_url:
            norm_provider = "custom"
        elif api_key and api_key.startswith("gsk_"):
            norm_provider = "groq"
        elif api_key and (api_key.startswith("sk-") or api_key.startswith("sess-")):
            norm_provider = "openai"
        else:
            norm_provider = "groq"

    # 1. Groq Provider
    if norm_provider == "groq":
        from langchain_groq import ChatGroq

        key = (api_key or GROQ_API_KEY).strip()
        if not key:
            raise ValueError(
                "Groq API key is not configured. Please add a custom key or set GROQ_API_KEY in the backend environment."
            )
        selected_model = (model or GROQ_MODEL or "openai/gpt-oss-120b").strip()
        return ChatGroq(
            api_key=key,
            model_name=selected_model,
            temperature=temperature,
        )

    # 2. OpenAI Provider
    elif norm_provider == "openai":
        from langchain_openai import ChatOpenAI

        key = (api_key or os.getenv("OPENAI_API_KEY", "")).strip()
        if not key:
            raise ValueError(
                "OpenAI API key is required. Please provide a valid custom OpenAI API key."
            )
        selected_model = (model or "gpt-4o").strip()
        return ChatOpenAI(
            api_key=key,
            model=selected_model,
            temperature=temperature,
        )

    # 3. Custom / OpenAI-Compatible Provider (Ollama, OpenRouter, DeepSeek, LocalAI, vLLM)
    elif norm_provider in ("custom", "openai_compatible", "ollama", "openrouter", "deepseek", "local"):
        from langchain_openai import ChatOpenAI

        url = (base_url or "http://localhost:11434/v1").strip()
        key = (api_key or os.getenv("OPENAI_API_KEY", "custom-api-key")).strip()
        selected_model = (model or "default").strip()
        return ChatOpenAI(
            base_url=url,
            api_key=key,
            model=selected_model,
            temperature=temperature,
        )

    else:
        # Fallback to Groq
        from langchain_groq import ChatGroq

        key = (api_key or GROQ_API_KEY).strip()
        if not key:
            raise ValueError(f"Unknown or unconfigured provider '{norm_provider}'. Please provide an API key.")
        selected_model = (model or GROQ_MODEL or "openai/gpt-oss-120b").strip()
        return ChatGroq(
            api_key=key,
            model_name=selected_model,
            temperature=temperature,
        )
