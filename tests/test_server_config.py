"""Tests for environment-driven self-hosted server provider configuration."""

import importlib
import os
from unittest.mock import MagicMock, patch

import pytest
from fastapi import HTTPException


def _load_server_config(env_overrides: dict[str, str]):
    clean_env = {
        "AUTH_DISABLED": "true",
        "POSTGRES_PASSWORD": "test-password",
        "MEM0_TELEMETRY": "false",
        **env_overrides,
    }

    with patch.dict(os.environ, clean_env, clear=True):
        with patch("mem0.Memory.from_config", return_value=MagicMock()):
            import auth as server_auth
            import server_state

            importlib.reload(server_auth)
            # Keep the reload offline: config overrides normally come from PostgreSQL.
            with patch.object(server_state, "_load_overrides", lambda: {}):
                import server.main as server_main

                importlib.reload(server_main)
            return server_main.DEFAULT_CONFIG, server_main


def test_default_configuration_uses_anthropic_and_local_ollama():
    config, server_main = _load_server_config(
        {
            "ANTHROPIC_API_KEY": "anthropic-test-key",
            "MEM0_DEFAULT_EMBEDDER_PROVIDER": "openai",
            "MEM0_DEFAULT_LLM_PROVIDER": "openai",
            "OPENAI_API_KEY": "must-not-be-used",
        }
    )

    assert config["llm"] == {
        "provider": "anthropic",
        "config": {
            "api_key": "anthropic-test-key",
            "temperature": 0.2,
            "model": "claude-haiku-4-5-20251001",
        },
    }
    assert config["embedder"] == {
        "provider": "ollama",
        "config": {
            "model": "nomic-embed-text",
            "embedding_dims": 768,
            "ollama_base_url": "http://ollama:11434",
        },
    }
    assert config["vector_store"]["config"]["embedding_model_dims"] == 768
    assert server_main.BUNDLED_LLM_PROVIDERS == ("anthropic",)
    assert server_main.BUNDLED_EMBEDDER_PROVIDERS == ("ollama",)


@pytest.mark.parametrize(
    "config",
    [
        {"llm": {"provider": "openai"}},
        {"embedder": {"provider": "openai"}},
        {"llm": {"provider": "gemini"}},
        {"embedder": {"provider": "gemini"}},
    ],
)
def test_remote_or_non_anthropic_providers_are_rejected(config):
    _, server_main = _load_server_config({"ANTHROPIC_API_KEY": "anthropic-test-key"})

    with pytest.raises(HTTPException, match="not bundled"):
        server_main._validate_bundled_providers(config)


def test_anthropic_llm_and_ollama_embedder_configuration():
    config, server_main = _load_server_config(
        {
            "ANTHROPIC_API_KEY": "anthropic-test-key",
            "MEM0_DEFAULT_LLM_MODEL": "claude-test-model",
            "MEM0_DEFAULT_EMBEDDER_MODEL": "nomic-embed-text",
            "MEM0_EMBEDDING_DIMS": "768",
            "OLLAMA_BASE_URL": "http://ollama:11434",
        }
    )

    assert config["llm"] == {
        "provider": "anthropic",
        "config": {
            "api_key": "anthropic-test-key",
            "temperature": 0.2,
            "model": "claude-test-model",
        },
    }
    assert config["embedder"] == {
        "provider": "ollama",
        "config": {
            "model": "nomic-embed-text",
            "embedding_dims": 768,
            "ollama_base_url": "http://ollama:11434",
        },
    }
    assert config["vector_store"]["config"]["embedding_model_dims"] == 768
    assert server_main.BUNDLED_LLM_PROVIDERS == ("anthropic",)
    assert server_main.BUNDLED_EMBEDDER_PROVIDERS == ("ollama",)
