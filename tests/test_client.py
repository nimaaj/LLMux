import pytest
from unittest.mock import MagicMock, AsyncMock, patch
from llmux.client import UnifiedChatClient

class TestUnifiedChatClient:
    
    @pytest.mark.asyncio
    async def test_init_with_env(self, mock_env):
        """Test client initialization picking up env vars."""
        client = UnifiedChatClient()
        assert client.openai is not None
        assert client.claude is not None
        assert client.gemini is not None
        # deepseek is optional and set via env in mock_env, check usage
        assert client.deepseek is not None

    def test_init_args_override(self):
        """Test explicit args override env vars."""
        client = UnifiedChatClient(openai_api_key=None, anthropic_api_key=None)
        assert client.openai is None
        assert client.claude is None

    @pytest.mark.asyncio
    async def test_chat_delegation(self, mock_env):
        """Test that client.chat() delegates to the correct provider."""
        client = UnifiedChatClient()
        
        # Mock the providers dictionary to avoid real calls
        # We need to mock the _providers mapping which is usually built dynamically or access providers directly
        # Since providers are initialized in __init__, we need to patch the provider instance methods
        
        # Create a mock provider
        mock_provider = AsyncMock()
        mock_provider.chat.return_value = {"text": "mock response"}
        
        # Patch the internal provider usage. 
        # Since UnifiedChatClient.chat gets provider from self.providers[name], we explicitly set it
        client.providers = {"openai": mock_provider}
        
        response = await client.chat("openai", "gpt-4o", [{"role": "user", "content": "hi"}])
        
        mock_provider.chat.assert_called_once()
        assert response == {"text": "mock response"}

    @pytest.mark.asyncio
    async def test_list_models_delegation(self, mock_env):
        """Test that list_models delegates to provider."""
        client = UnifiedChatClient()
        mock_provider = AsyncMock()
        mock_provider.get_models.return_value = ["model-a", "model-b"]
        
        client.providers = {"openai": mock_provider}
        
        models = await client.list_models("openai")
        assert models == ["model-a", "model-b"]
        mock_provider.get_models.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_chat_invalid_provider(self, mock_env):
        client = UnifiedChatClient()
        with pytest.raises(ValueError, match="not configured or not supported"):
            await client.chat("invalid_provider", "model", [])

