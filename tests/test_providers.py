import pytest
from unittest.mock import MagicMock, AsyncMock, patch
from llmux.providers.openai import OpenAIProvider
from llmux.providers.anthropic import AnthropicProvider
from llmux.providers.gemini import GeminiProvider

class TestOpenAIProvider:
    @pytest.mark.asyncio
    @patch("llmux.providers.openai.AsyncOpenAI")
    async def test_convert_messages_text(self, mock_openai_cls):
        # Setup mock
        client_mock = MagicMock()
        mock_openai_cls.return_value = client_mock
        
        provider = OpenAIProvider(api_key="fake-key")
        
        messages = [{"role": "user", "content": "hello"}]
        converted = await provider._convert_messages(messages)
        
        assert len(converted) == 1
        assert converted[0] == {"role": "user", "content": "hello"}

    @pytest.mark.asyncio
    @patch("llmux.providers.openai.AsyncOpenAI")
    async def test_chat_call(self, mock_openai_cls):
        # Setup mock
        client_mock = AsyncMock()
        mock_openai_cls.return_value = client_mock
        
        client_mock.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content="response", tool_calls=None))],
            usage=MagicMock(prompt_tokens=10, completion_tokens=20, total_tokens=30)
        )
        
        provider = OpenAIProvider(api_key="fake-key")
        response = await provider.chat("gpt-4o", [{"role": "user", "content": "hi"}])
        
        assert response["text"] == "response"
        assert response["provider"] == "openai"
        assert response["meta"]["usage"]["total_tokens"] == 30

class TestAnthropicProvider:
    @pytest.mark.asyncio
    @patch("llmux.providers.anthropic.AsyncAnthropic")
    async def test_convert_messages_split_system(self, mock_anthropic_cls):
        client_mock = MagicMock()
        mock_anthropic_cls.return_value = client_mock
        
        provider = AnthropicProvider(api_key="fake-key")
        
        messages = [
            {"role": "system", "content": "system prompt"},
            {"role": "user", "content": "user query"}
        ]
        
        system, converted = await provider._convert_messages(messages)
        
        assert system == "system prompt"
        assert len(converted) == 1
        assert converted[0]["role"] == "user"
        assert converted[0]["content"] == "user query"

class TestGeminiProvider:
    @pytest.mark.asyncio
    @patch("llmux.providers.gemini.genai")
    async def test_convert_messages_roles(self, mock_genai):
        provider = GeminiProvider(api_key="fake-key")
        
        messages = [
            {"role": "assistant", "content": "im helper"},
            {"role": "user", "content": "hi"}
        ]
        
        system, converted = await provider._convert_messages(messages)
        
        # Gemini maps 'assistant' -> 'model'
        assert converted[0]["role"] == "model"
        # Adjusted assertion to match actual structure: [{'text': 'im helper'}]
        assert converted[0]["parts"] == [{"text": "im helper"}]
        assert converted[1]["role"] == "user"
