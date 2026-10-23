"""Tests for custom API keys, external provider configuration, and validation."""

import pytest
from fastapi.testclient import TestClient
from backend.main import app
from backend.agent.llm import build_chat_model

client = TestClient(app)


def test_build_chat_model_custom_groq():
    """Verify build_chat_model accepts custom Groq API key and model."""
    llm = build_chat_model(
        custom_keys={
            "provider": "groq",
            "api_key": "gsk_test_fake_custom_key_12345",
            "model": "llama-3.3-70b-versatile",
        }
    )
    assert llm is not None
    assert getattr(llm, "model_name", None) == "llama-3.3-70b-versatile" or getattr(llm, "model", None) == "llama-3.3-70b-versatile"


def test_build_chat_model_custom_openai():
    """Verify build_chat_model builds ChatOpenAI when provider is openai."""
    llm = build_chat_model(
        custom_keys={
            "provider": "openai",
            "api_key": "sk-test-fake-openai-key-12345",
            "model": "gpt-4o-mini",
        }
    )
    assert llm is not None
    assert getattr(llm, "model_name", None) == "gpt-4o-mini" or getattr(llm, "model", None) == "gpt-4o-mini"


def test_build_chat_model_custom_endpoint():
    """Verify build_chat_model supports custom base_url for Ollama/OpenRouter/LocalAI."""
    llm = build_chat_model(
        custom_keys={
            "provider": "custom",
            "base_url": "http://localhost:11434/v1",
            "api_key": "ollama",
            "model": "deepseek-r1",
        }
    )
    assert llm is not None
    assert getattr(llm, "model_name", None) == "deepseek-r1" or getattr(llm, "model", None) == "deepseek-r1"


def test_settings_keys_status():
    """Verify GET /api/settings/keys-status returns configured status flags."""
    resp = client.get("/api/settings/keys-status")
    assert resp.status_code == 200
    data = resp.json()
    assert "server_groq_configured" in data
    assert "server_tavily_configured" in data
    assert "default_model" in data


def test_settings_validate_empty_key():
    """Verify POST /api/settings/validate-key rejects missing/empty keys."""
    resp = client.post(
        "/api/settings/validate-key",
        json={"provider": "groq", "api_key": ""},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["valid"] is False
    assert "empty" in data["message"].lower() or "enter a valid" in data["error"].lower()


def test_settings_validate_invalid_key_fails_cleanly():
    """Verify POST /api/settings/validate-key gracefully catches authentication errors without 500 crash."""
    resp = client.post(
        "/api/settings/validate-key",
        json={"provider": "groq", "api_key": "gsk_invalid_bogus_key_9999999"},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["valid"] is False
    assert data["provider"] == "groq"
    assert data["error"] is not None


def test_chat_endpoint_accepts_custom_keys(monkeypatch):
    """Verify /chat/ endpoint parses and forwards custom_keys without schema error."""
    # Mock chat_func to verify custom_keys is received
    received_keys = {}

    def mock_chat_func(query, custom_keys=None, **kwargs):
        nonlocal received_keys
        received_keys = custom_keys or {}
        return "Response from custom key mock"

    monkeypatch.setattr("backend.routes.chat.chat_func", mock_chat_func)

    resp = client.post(
        "/chat/",
        json={
            "question": "Hello with custom key",
            "custom_keys": {
                "provider": "groq",
                "api_key": "gsk_test_custom_key",
            },
        },
    )
    assert resp.status_code == 200
    assert resp.json()["answer"] == "Response from custom key mock"
    assert received_keys.get("api_key") == "gsk_test_custom_key"
