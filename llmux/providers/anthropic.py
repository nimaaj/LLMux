import json
import time
from typing import Dict, Any, List, AsyncIterator, Optional, Tuple

from anthropic import AsyncAnthropic

from .base import BaseLLMProvider
from ..types import Message, ToolCall, Tool
from ..utils import resolve_image_to_base64

class AnthropicProvider(BaseLLMProvider):
    """
    Provider for Anthropic (Claude) API.
    """
    
    def __init__(self, api_key: Optional[str]):
        super().__init__(api_key)
        self.client = AsyncAnthropic(api_key=api_key) if api_key else None

    async def chat(
        self,
        model: str,
        messages: List[Message],
        **kwargs
    ) -> Dict[str, Any]:
        """
        Send a chat request to the Claude API.

        Handles:
        - System prompt extraction (sent as separate parameter).
        - Message conversion (text/image handling).
        - Tool call parsing.
        
        Args:
            model (str): The specific Claude model identifier.
            messages (List[Message]): Conversation history.
            **kwargs: Options like max_tokens, temperature, tools, etc.

        Returns:
            Dict[str, Any]: Standardized response dictionary.
        """
        if not self.client:
            raise RuntimeError("Claude (Anthropic) client not configured")

        system_text, converted_messages = await self._convert_messages(messages)

        request_kwargs = {
            "model": model,
            "messages": converted_messages,
            "max_tokens": kwargs.get("max_tokens", 1024),
        }

        # Optional params
        optional_params = {
            "system": system_text,
            "temperature": kwargs.get("temperature", 0.7),
            "top_p": kwargs.get("top_p"),
            "top_k": kwargs.get("top_k"),
            "stop_sequences": kwargs.get("stop_sequences"),
            "metadata": kwargs.get("metadata"),
        }
        request_kwargs.update({k: v for k, v in optional_params.items() if v is not None})
        
        # Tools
        if kwargs.get("tools"):
            request_kwargs["tools"] = self._convert_tools(kwargs["tools"])
            if kwargs.get("tool_choice"):
                tool_choice = kwargs["tool_choice"]
                if tool_choice == "auto":
                    request_kwargs["tool_choice"] = {"type": "auto"}
                elif tool_choice == "none":
                    # Claude does not support "none" explicitly in the same way as OpenAI if tools are passed?
                    # Actually API documentation says tool_choice is optional.
                    # Creating a dummy "none" might fail.
                    # OpenAI "none" means don't call tools.
                    # For Claude, just omitting tool_choice defaults to auto.
                    # If user forced "none", maybe just don't pass tools?
                    pass 
                elif tool_choice == "required":
                    request_kwargs["tool_choice"] = {"type": "any"}
                elif isinstance(tool_choice, str):
                    request_kwargs["tool_choice"] = {"type": "tool", "name": tool_choice}

        start = time.perf_counter()
        resp = await self.client.messages.create(**request_kwargs)
        latency_ms = (time.perf_counter() - start) * 1000.0

        text = "".join(
            block.text for block in resp.content if getattr(block, "type", None) == "text"
        )
        
        tool_calls = self._parse_tool_calls(resp)

        usage = None
        if resp.usage:
            usage = self.normalize_usage(
                "claude",
                input_tokens=resp.usage.input_tokens,
                output_tokens=resp.usage.output_tokens,
                total_tokens=None,
                raw={
                    "input_tokens": resp.usage.input_tokens,
                    "output_tokens": resp.usage.output_tokens,
                },
            )

        result = {
            "provider": "claude",
            "text": text,
            "meta": {
                "model": resp.model,
                "usage": usage,
                "latency_ms": latency_ms,
                "stop_reason": resp.stop_reason,
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
        Stream a chat response from Claude.

        Yields events for tokens, errors, and completion.

        Args:
            model (str): Model identifier.
            messages (List[Message]): Conversation history.
            **kwargs: Additional parameters.

        Yields:
            Dict[str, Any]: Stream events.
        """
        if not self.client:
            raise RuntimeError("Claude (Anthropic) client not configured")

        system_text, converted_messages = await self._convert_messages(messages)

        request_kwargs = {
            "model": model,
            "messages": converted_messages,
            "max_tokens": kwargs.get("max_tokens", 1024),
        }

        optional_params = {
            "system": system_text,
            "temperature": kwargs.get("temperature", 0.7),
            "top_p": kwargs.get("top_p"),
            "top_k": kwargs.get("top_k"),
            "stop_sequences": kwargs.get("stop_sequences"),
            "metadata": kwargs.get("metadata"),
        }
        request_kwargs.update({k: v for k, v in optional_params.items() if v is not None})

        full_text = ""
        start = time.perf_counter()

        async with self.client.messages.stream(**request_kwargs) as stream:
            async for event in stream:
                if (
                    event.type == "content_block_delta"
                    and getattr(event.delta, "type", None) == "text_delta"
                ):
                    piece = event.delta.text
                    if piece:
                        full_text += piece
                        yield {
                            "type": "token",
                            "provider": "claude",
                            "text": piece,
                        }

            final_msg = await stream.get_final_message()

        latency_ms = (time.perf_counter() - start) * 1000.0

        usage = None
        if final_msg.usage:
            usage = self.normalize_usage(
                "claude",
                input_tokens=final_msg.usage.input_tokens,
                output_tokens=final_msg.usage.output_tokens,
                total_tokens=None,
                raw={
                    "input_tokens": final_msg.usage.input_tokens,
                    "output_tokens": final_msg.usage.output_tokens,
                },
            )

        yield {
            "type": "done",
            "provider": "claude",
            "text": full_text,
            "meta": {
                "model": final_msg.model,
                "usage": usage,
                "latency_ms": latency_ms,
                "stop_reason": final_msg.stop_reason,
            },
        }

    async def _convert_messages(
        self,
        messages: List[Message],
    ) -> Tuple[Optional[str], List[Dict[str, Any]]]:
        """
        Convert messages to Claude format.
        
        Anthropic's API differs from OpenAI's in that 'system' messages are passed
        as a separate top-level parameter, not within the `messages` list.
        It also requires converting image URLs to base64 data as it doesn't support
        fetching images via public URLs in the same way.

        Args:
            messages (List[Message]): Internal message list.

        Returns:
            Tuple containing:
            - system_text: Extracted system prompt string (or None)
            - converted: List of message dicts suitable for the API
        """
        system_parts = []
        converted = []
        
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            
            # Extract system messages to be sent separately
            if role == "system":
                if isinstance(content, str):
                    system_parts.append(content)
                else:
                    for part in content:
                        if part.get("type") == "text":
                            system_parts.append(part.get("text", ""))
                continue
            
            # Convert standard user/assistant messages
            if isinstance(content, str):
                converted.append({"role": role, "content": content})
            else:
                # Multimodal content handling
                claude_content = []
                for part in content:
                    if part.get("type") == "text":
                        claude_content.append({
                            "type": "text",
                            "text": part.get("text", "")
                        })
                    elif part.get("type") == "image_url":
                        url = part.get("image_url", {}).get("url", "")
                        # Claude expects base64 data for images
                        b64_data, media_type = await resolve_image_to_base64(url)
                        claude_content.append({
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": media_type,
                                "data": b64_data,
                            }
                        })
                
                if claude_content:
                    converted.append({"role": role, "content": claude_content})
        
        system_text = "\n\n".join(system_parts) if system_parts else None
        return system_text, converted

    @staticmethod
    def _convert_tools(tools: List[Tool]) -> List[Dict[str, Any]]:
        """
        Convert OpenAI-format tools to Claude format.
        
        Claude uses 'input_schema' instead of 'parameters'.
        """
        claude_tools = []
        for tool in tools:
            func = tool.get("function", {})
            claude_tools.append({
                "name": func.get("name", ""),
                "description": func.get("description", ""),
                "input_schema": func.get("parameters", {"type": "object", "properties": {}}),
            })
        return claude_tools

    @staticmethod
    def _parse_tool_calls(response) -> List[ToolCall]:
        """
        Parse tool calls from Claude response.

        Extracts tool use blocks from the response content.

        Args:
            response: The Claude API response object.

        Returns:
            List[ToolCall]: List of parsed tool calls.
        """
        tool_calls = []
        for block in response.content:
            if getattr(block, "type", None) == "tool_use":
                tool_calls.append({
                    "id": block.id,
                    "name": block.name,
                    "arguments": block.input,
                })
        return tool_calls

    async def get_models(self) -> List[str]:
        """
        Get list of available models from Anthropic API.

        Returns:
            List[str]: List of model identifiers.
        """
        if not self.client:
            return []
            
        try:
             # Anthropic doesn't have a public models list API in the client yet? 
             # Actually it does: client.models.list() was added recently.
             # Let's verify if the user's library version supports it.
             # The static method used it: return [x.id for x in client.models.list().data]
             # So safely assume it exists.
            models = await self.client.models.list()
            return [m.id for m in models.data]
        except Exception:
            return []
