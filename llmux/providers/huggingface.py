from typing import Optional
from .openai import OpenAIProvider

class HuggingFaceProvider(OpenAIProvider):
    """
    Provider for Hugging Face Inference API (OpenAI-compatible).
    """
    
    def __init__(
        self,
        api_key: Optional[str],
        base_url: str = "https://router.huggingface.co/v1/",
    ):
        """
        Initialize HuggingFaceProvider.
        """
        super().__init__(
            api_key=api_key,
            base_url=base_url,
            provider_name="huggingface"
        )
