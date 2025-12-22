from abc import ABC, abstractmethod
from typing import Dict, Any, List, AsyncIterator, Optional

from ..types import Message

class BaseLLMProvider(ABC):
    """
    Abstract base class for LLM providers.
    """
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key

    @abstractmethod
    async def chat(
        self,
        model: str,
        messages: List[Message],
        **kwargs
    ) -> Dict[str, Any]:
        """
        Send a chat request to the provider.
        """
        pass

    async def stream(
        self,
        model: str,
        messages: List[Message],
        **kwargs
    ) -> AsyncIterator[Dict[str, Any]]:
        """
        Stream a chat response from the provider.
        """
        pass

    @abstractmethod
    async def get_models(self) -> List[str]:
        """
        Get list of available models from the provider.
        """
        pass

    @staticmethod
    def normalize_usage(
        provider: str,
        *,
        input_tokens: Optional[int],
        output_tokens: Optional[int],
        total_tokens: Optional[int],
        raw: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Normalize token usage information across providers.
        """
        # Calculate total if not provided
        if total_tokens is None and input_tokens is not None and output_tokens is not None:
            total_tokens = input_tokens + output_tokens

        return {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": total_tokens,
            "raw": {
                "provider": provider,
                **(raw or {}),
            } if raw is not None else None,
        }
