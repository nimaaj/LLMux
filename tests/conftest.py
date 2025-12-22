import pytest
import os
from unittest.mock import MagicMock

@pytest.fixture
def mock_env(monkeypatch):
    """Mock environment variables for API keys."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-openai")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-anthropic")
    monkeypatch.setenv("GOOGLE_API_KEY", "AIza-test-google")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test-deepseek")

@pytest.fixture
def mock_response():
    """Create a standard mock response for chat."""
    return {
        "text": "Hello, world!",
        "provider": "openai",
        "meta": {
            "model": "gpt-4o",
            "usage": {"total_tokens": 10},
            "latency_ms": 100
        }
    }
