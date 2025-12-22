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

        Args:
            model (str): The model identifier.
            messages (List[Message]): List of conversation messages.
            **kwargs: Additional provider-specific options.

        Returns:
            Dict[str, Any]: Standardized response dictionary.
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

        Args:
            model (str): The model identifier.
            messages (List[Message]): List of conversation messages.
            **kwargs: Additional provider-specific options.

        Yields:
            Dict[str, Any]: Stream events (tokens, errors, done).
        """
        pass

    @abstractmethod
    async def get_models(self) -> List[str]:
        """
        Get list of available models from the provider.

        Returns:
            List[str]: List of model identifiers.
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

        Creates a standardized dictionary structure for token usage statistics,
        optionally calculating totals if missing.

        Args:
            provider (str): Name of the provider.
            input_tokens (int, optional): Number of prompt tokens.
            output_tokens (int, optional): Number of generated tokens.
            total_tokens (int, optional): Total token count.
            raw (dict, optional): Raw usage data from the provider response.

        Returns:
            Dict[str, Any]: Standardized usage dictionary.
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
