import asyncio
import time
from typing import Dict, Any, List, AsyncIterator, Optional, Tuple

from google import genai
from google.genai import types

from .base import BaseLLMProvider
from ..types import Message
from ..utils import resolve_image_to_base64

class GeminiProvider(BaseLLMProvider):
    """
    Provider for Google Gemini API (using google-genai SDK).
    """
    
    def __init__(self, api_key: Optional[str]):
        super().__init__(api_key)
        if api_key:
            self.client = genai.Client(api_key=api_key)
        else:
            self.client = None

    async def chat(
        self,
        model: str,
        messages: List[Message],
        **kwargs
    ) -> Dict[str, Any]:
        """
        Send a chat request to the Gemini API.

        Handles:
        - Role mapping (assistant -> model).
        - Message structure conversion.
        - Tool configuration.
        - Safety settings.

        Args:
            model (str): Gemini model identifier.
            messages (List[Message]): Conversation history.
            **kwargs: Options like temperature, top_p, top_k, tools.

        Returns:
            Dict[str, Any]: Standardized response dictionary.
        """
        if not self.client:
            raise RuntimeError("Gemini client not configured")

        system_instruction, contents = await self._convert_messages(messages)

        config_kwargs = {
            "temperature": kwargs.get("temperature", 0.7),
            "max_output_tokens": kwargs.get("max_tokens", 1024),
            "system_instruction": system_instruction,
        }
        if kwargs.get("top_p") is not None:
            config_kwargs["top_p"] = kwargs["top_p"]
        if kwargs.get("top_k") is not None:
            config_kwargs["top_k"] = kwargs["top_k"]
        
        # Tools
        if kwargs.get("tools"):
            # New SDK tool configuration might differ slightly; 
            # generally accepts OpenAI-like tools or FunctionDeclarations.
            # Assuming it can handle the dicts or we need to convert.
            # The _convert_tools method returns a config dict with function_declarations.
            tools_config = self._convert_tools(kwargs["tools"])
            config_kwargs["tools"] = tools_config

        # Create config object if needed or pass as kwargs
        # google-genai SDK uses client.models.generate_content(model=..., config=..., contents=...)
        # config can be a GenerateContentConfig object or dict?
        # Trying to pass flat arguments + config object is usually required.
        
        # Construct config
        # Note: 'system_instruction' is part of config in new SDK usually
        config = types.GenerateContentConfig(**config_kwargs)

        start = time.perf_counter()
        
        # Execute (async)
        # client.aio.models.generate_content is the async version
        resp = await self.client.aio.models.generate_content(
            model=model,
            contents=contents,
            config=config,
        )
        
        latency_ms = (time.perf_counter() - start) * 1000.0

        try:
            text = resp.text or ""
        except ValueError:
            text = ""
        
        # Parse tool calls from response
        tool_calls = []
        if resp.candidates and resp.candidates[0].content and resp.candidates[0].content.parts:
            for part in resp.candidates[0].content.parts:
                if part.function_call:
                    fc = part.function_call
                    tool_calls.append({
                        "id": f"gemini_{fc.name}_{len(tool_calls)}", # Gemini doesn't always provide call IDs
                        "name": fc.name,
                        "arguments": fc.args, # New SDK usually returns dict for args
                    })

        # Usage
        usage = None
        if resp.usage_metadata:
            um = resp.usage_metadata
            usage = self.normalize_usage(
                "gemini",
                input_tokens=um.prompt_token_count,
                output_tokens=um.candidates_token_count,
                total_tokens=um.total_token_count,
                raw={
                    "prompt_token_count": um.prompt_token_count,
                    "candidates_token_count": um.candidates_token_count,
                    "total_token_count": um.total_token_count,
                },
            )

        result = {
            "provider": "gemini",
            "text": text,
            "meta": {
                "model": model,
                "usage": usage,
                "latency_ms": latency_ms,
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
        Stream a chat response from Gemini.

        Yields events for tokens and completion.

        Args:
            model (str): Model identifier.
            messages (List[Message]): Conversation history.
            **kwargs: Additional parameters.

        Yields:
            Dict[str, Any]: Stream events.
        """
        if not self.client:
            raise RuntimeError("Gemini client not configured")

        system_instruction, contents = await self._convert_messages(messages)

        config_kwargs = {
            "temperature": kwargs.get("temperature", 0.7),
            "max_output_tokens": kwargs.get("max_tokens", 1024),
            "system_instruction": system_instruction,
        }
        if kwargs.get("top_p") is not None:
            config_kwargs["top_p"] = kwargs["top_p"]
        if kwargs.get("top_k") is not None:
            config_kwargs["top_k"] = kwargs["top_k"]

        config = types.GenerateContentConfig(**config_kwargs)

        full_text = ""
        start = time.perf_counter()

        async for chunk in await self.client.aio.models.generate_content_stream(
            model=model,
            contents=contents,
            config=config,
        ):
            if chunk.text:
                full_text += chunk.text
                yield {
                    "type": "token",
                    "provider": "gemini",
                    "text": chunk.text,
                }

        latency_ms = (time.perf_counter() - start) * 1000.0
        
        # Usage metadata availability in stream chunks varies; 
        # usually last chunk has it or we track it.
        # Assuming we can't easily get it for now without aggregating.
        usage = None

        yield {
            "type": "done",
            "provider": "gemini",
            "text": full_text,
            "meta": {
                "model": model,
                "usage": usage,
                "latency_ms": latency_ms,
            },
        }

    async def _convert_messages(
        self,
        messages: List[Message],
    ) -> Tuple[Optional[str], List[Dict[str, Any]]]:
        """
        Convert messages to Gemini format (google-genai SDK).
        """
        system_instruction = None
        contents = []
        
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            
            # Extract system instruction
            if role == "system":
                if isinstance(content, str):
                    system_instruction = content
                else:
                    texts = [p.get("text", "") for p in content if p.get("type") == "text"]
                    system_instruction = "\n".join(texts)
                continue
            
            # Map roles: OpenAI role -> Gemini role
            # "assistant" -> "model"
            gemini_role = "model" if role == "assistant" else "user"
            
            # Build content parts
            parts = []
            if isinstance(content, str):
                parts.append({"text": content})
            else:
                for part in content:
                    if part.get("type") == "text":
                        parts.append({"text": part.get("text", "")})
                    elif part.get("type") == "image_url":
                        url = part.get("image_url", {}).get("url", "")
                        b64_data, mime_type = await resolve_image_to_base64(url)
                        parts.append({
                            "inline_data": {
                                "mime_type": mime_type,
                                "data": b64_data,
                            }
                        })
            
            if parts:
                contents.append(types.Content(role=gemini_role, parts=parts))
        
        return system_instruction, contents

    @staticmethod
    def _convert_tools(tools: List[Dict[str, Any]]) -> List[types.Tool]:
        """
        Convert OpenAI-format tools to Gemini format (google-genai SDK).
        """
        function_declarations = []
        for tool in tools:
            func = tool.get("function", {})
            # google-genai SDK handles OpenAI JSON schema for parameters generally well
            function_declarations.append(types.FunctionDeclaration(
                name=func.get("name", ""),
                description=func.get("description", ""),
                parameters=func.get("parameters"),
            ))
        return [types.Tool(function_declarations=function_declarations)]

    async def get_models(self) -> List[str]:
        """
        Get list of available models from Gemini API.

        Uses the new google-genai SDK logic.
        """
        if not self.client:
            return []
            
        def _list():
            try:
                # Based on user request/docs for new SDK
                msg = []
                for m in self.client.models.list():
                    # Check supported actions if available/needed
                    # The user snippet checked for 'generateContent'
                    if hasattr(m, "supported_actions") and m.supported_actions:
                         if "generateContent" in m.supported_actions:
                             msg.append(m.name)
                    else:
                        # Fallback if supported_actions isn't explicitly there or needed
                        msg.append(m.name)
                return msg
            except Exception:
                return []

        # models.list is sync in the new SDK? Or async?
        # The docs say `client.models.list()`. It returns an iterator.
        # It handles pagination automatically.
        # Since we use `asyncio.to_thread` for the previous sync implementation, 
        # check if new SDK list is sync. Usually the top-level client is sync unless using .aio.
        # self.client is sync Client? 
        # Wait, I initialized `self.client = genai.Client(...)`. This is the top-level client.
        # It supports `client.aio` for async.
        # `client.models.list` is synchronous. `client.aio.models.list` might be async.
        
        # Let's use the async client for listing if possible, or wrap sync in thread.
        # User snippet: `for m in client.models.list():` (Sync loop)
        return await asyncio.to_thread(_list)
