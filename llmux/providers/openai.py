import json
import time
from typing import Dict, Any, List, AsyncIterator, Optional, Union

from openai import AsyncOpenAI

from .base import BaseLLMProvider
from ..types import Message, ToolCall
from ..utils import create_image_content, encode_image_url

class OpenAIProvider(BaseLLMProvider):
    """
    Provider for OpenAI-compatible APIs (OpenAI, DeepSeek, etc.).
    """
    
    def __init__(
        self,
        api_key: Optional[str],
        base_url: Optional[str] = None,
        provider_name: str = "openai"
    ):
        super().__init__(api_key)
        self.client = AsyncOpenAI(api_key=api_key, base_url=base_url) if api_key else None
        self.provider_name = provider_name

    async def chat(
        self,
        model: str,
        messages: List[Message],
        **kwargs
    ) -> Dict[str, Any]:
        """
        Send a chat request using the OpenAI-compatible API.

        Handles:
        - Message conversion to OpenAI format.
        - Option mapping (temperature, max_tokens, etc.).
        - Tool call parsing.
        - Token usage normalization.

        Args:
            model (str): The model identifier.
            messages (List[Message]): List of conversation messages.
            **kwargs: Additional parameters (e.g., temperature, top_p, tools).

        Returns:
            Dict[str, Any]: Standardized response dictionary containing 'text', 'tool_calls', and 'meta'.
        """
        if not self.client:
            raise RuntimeError(f"{self.provider_name.title()} client not configured")

        # Convert messages (handles multimodal content)
        converted_messages = await self._convert_messages(messages)

        # Build request kwargs
        request_kwargs = {
            "model": model,
            "messages": converted_messages,
        }

        # Map options
        optional_params = {
            "temperature": kwargs.get("temperature", 0.7),
            "max_tokens": kwargs.get("max_tokens", 1024),
            "top_p": kwargs.get("top_p"),
            "frequency_penalty": kwargs.get("frequency_penalty"),
            "presence_penalty": kwargs.get("presence_penalty"),
            "stop": kwargs.get("stop"),
            "seed": kwargs.get("seed"),
            "response_format": kwargs.get("response_format"),
            "tools": kwargs.get("tools"),
            "tool_choice": kwargs.get("tool_choice"),
            "user": kwargs.get("user"),
        }
        request_kwargs.update({k: v for k, v in optional_params.items() if v is not None})

        # Make API call with latency tracking
        start = time.perf_counter()
        resp = await self.client.chat.completions.create(**request_kwargs)
        latency_ms = (time.perf_counter() - start) * 1000.0

        choice = resp.choices[0]
        text = choice.message.content or ""
        
        # Parse tool calls
        tool_calls = self._parse_tool_calls(choice)

        # Normalize usage
        usage = None
        if resp.usage:
            usage = self.normalize_usage(
                self.provider_name,
                input_tokens=resp.usage.prompt_tokens,
                output_tokens=resp.usage.completion_tokens,
                total_tokens=resp.usage.total_tokens,
                raw={
                    "prompt_tokens": resp.usage.prompt_tokens,
                    "completion_tokens": resp.usage.completion_tokens,
                    "total_tokens": resp.usage.total_tokens,
                },
            )

        result = {
            "provider": self.provider_name,
            "text": text,
            "meta": {
                "model": resp.model,
                "usage": usage,
                "latency_ms": latency_ms,
                "finish_reason": choice.finish_reason,
            },
        }
        
        if tool_calls:
            result["tool_calls"] = tool_calls
            
        return result

    async def stream(
        self,
        model: str,
        messages: List[Message],
        **kwargs
    ) -> AsyncIterator[Dict[str, Any]]:
        """
        Stream a chat response using the OpenAI-compatible API.

        Yields events for tokens, errors, and completion.

        Args:
            model (str): The model identifier.
            messages (List[Message]): List of conversation messages.
            **kwargs: Additional parameters.

        Yields:
            Dict[str, Any]: Stream events:
                - {'type': 'token', 'text': '...', 'provider': '...'}
                - {'type': 'done', 'text': '...', 'meta': {...}}
        """
        if not self.client:
            raise RuntimeError(f"{self.provider_name.title()} client not configured")

        converted_messages = await self._convert_messages(messages)

        request_kwargs = {
            "model": model,
            "messages": converted_messages,
            "temperature": kwargs.get("temperature", 0.7),
            "max_tokens": kwargs.get("max_tokens", 1024),
            "top_p": kwargs.get("top_p"),
            "stream": True,
        }
        
        # Add other optional params if needed for streaming...

        start = time.perf_counter()
        stream = await self.client.chat.completions.create(**request_kwargs)
        
        full_text = ""
        last_chunk = None

        async for chunk in stream:
            last_chunk = chunk
            if not chunk.choices: continue
            
            delta = chunk.choices[0].delta
            content = delta.content
            
            if not content:
                continue
                
            # Handle both list and string content
            if isinstance(content, list):
                piece = "".join(part.get("text", "") for part in content if isinstance(part, dict))
            else:
                piece = str(content)

            if piece:
                full_text += piece
                yield {
                    "type": "token",
                    "provider": self.provider_name,
                    "text": piece,
                }

        latency_ms = (time.perf_counter() - start) * 1000.0

        # Extract usage
        raw_usage = None
        input_tokens = output_tokens = total_tokens = None
        if last_chunk is not None and getattr(last_chunk, "usage", None) is not None:
            u = last_chunk.usage
            raw_usage = {
                "prompt_tokens": u.prompt_tokens,
                "completion_tokens": u.completion_tokens,
                "total_tokens": u.total_tokens,
            }
            input_tokens = u.prompt_tokens
            output_tokens = u.completion_tokens
            total_tokens = u.total_tokens

        usage = self.normalize_usage(
            self.provider_name,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
            raw=raw_usage,
        )

        meta = {
            "model": getattr(last_chunk, "model", model) if last_chunk is not None else model,
            "usage": usage,
            "latency_ms": latency_ms,
        }

        yield {
            "type": "done",
            "provider": self.provider_name,
            "text": full_text,
            "meta": meta,
        }

    async def _convert_messages(self, messages: List[Message]) -> List[Dict[str, Any]]:
        """
        Convert internal message format to OpenAI's expected format.
        
        Handles:
        - Role mapping (system, user, assistant, tool)
        - Tool call results (formatted as separate messages)
        - Assistant messages with tool calls (preserving tool_calls field)
        - Multimodal content (text + images)
        - Image URL handling (converts HTTP URLs to base64 if needed for stability)

        Args:
            messages (List[Message]): Internal message list.

        Returns:
            List[Dict]: OpenAI-compatible message list.
        """
        converted = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            
            # Handle tool results: OpenAI expects these as separate messages with role "tool"
            if role == "tool":
                converted.append({
                    "role": "tool",
                    "tool_call_id": msg.get("tool_call_id", ""),
                    "content": content if isinstance(content, str) else "",
                })
                continue
            
            # Handle assistant messages that made tool calls
            # These need to preserve the tool_calls field for context
            if role == "assistant" and msg.get("tool_calls"):
                assistant_msg = {
                    "role": "assistant",
                    "content": content if isinstance(content, str) else "",
                    "tool_calls": [
                        {
                            "id": tc["id"],
                            "type": "function",
                            "function": {
                                "name": tc["name"],
                                "arguments": json.dumps(tc["arguments"]),
                            },
                        }
                        for tc in msg["tool_calls"]
                    ],
                }
                converted.append(assistant_msg)
                continue
            
            # Handle standard messages (text or multimodal)
            if isinstance(content, str):
                converted.append({"role": role, "content": content})
            else:
                # Multimodal content: list of text/image parts
                openai_content = []
                for part in content:
                    if part.get("type") == "text":
                        openai_content.append(part)
                    elif part.get("type") == "image_url":
                        url = part.get("image_url", {}).get("url", "")
                        detail = part.get("image_url", {}).get("detail")
                        
                        # Convert HTTP URLs to base64 for reliability
                        # Some OpenAI-compatible endpoints might not support fetching external URLs
                        if url.startswith(("http://", "https://")):
                            b64_data, mime_type = await encode_image_url(url)
                            url = f"data:{mime_type};base64,{b64_data}"
                        
                        image_url_obj = {"url": url}
                        if detail:
                            image_url_obj["detail"] = detail
                        openai_content.append({
                            "type": "image_url",
                            "image_url": image_url_obj
                        })
                
                converted.append({"role": role, "content": openai_content})
        
        return converted

    @staticmethod
    def _parse_tool_calls(choice) -> List[ToolCall]:
        """
        Parse tool calls from OpenAI response choice.

        Safe handling of JSON parsing for tool arguments.

        Args:
            choice: The OpenAI API response choice object.

        Returns:
            List[ToolCall]: List of parsed tool calls.
        """
        tool_calls = []
        if hasattr(choice.message, "tool_calls") and choice.message.tool_calls:
            for tc in choice.message.tool_calls:
                try:
                    args = json.loads(tc.function.arguments)
                except json.JSONDecodeError:
                    args = {"_raw": tc.function.arguments}
                tool_calls.append({
                    "id": tc.id,
                    "name": tc.function.name,
                    "arguments": args,
                })
        return tool_calls

    async def get_models(self) -> List[str]:
        """
        Get list of available models from the provider API.

        Returns:
            List[str]: List of model names/ids. Returns empty list if client not configured or API fails.
        """
        if not self.client:
            return []
        
        try:
            models = await self.client.models.list()
            return [m.id for m in models.data]
        except Exception:
            # Return empty list or raise? Returning empty list seems safer for discovery
            return []
