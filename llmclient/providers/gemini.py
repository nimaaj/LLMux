import asyncio
import time
from typing import Dict, Any, List, AsyncIterator, Optional, Tuple

import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold

from .base import BaseLLMProvider
from ..types import Message
from ..utils import resolve_image_to_base64

class GeminiProvider(BaseLLMProvider):
    """
    Provider for Google Gemini API.
    """
    
    def __init__(self, api_key: Optional[str]):
        super().__init__(api_key)
        if api_key:
            genai.configure(api_key=api_key)
            self.genai = genai
        else:
            self.genai = None

    async def chat(
        self,
        model: str,
        messages: List[Message],
        **kwargs
    ) -> Dict[str, Any]:
        if not self.genai:
            raise RuntimeError("Gemini client not configured")

        system_instruction, contents = await self._convert_messages(messages)

        generation_config = {
            "temperature": kwargs.get("temperature", 0.7),
            "max_output_tokens": kwargs.get("max_tokens", 1024),
        }
        if kwargs.get("top_p") is not None:
            generation_config["top_p"] = kwargs["top_p"]
        if kwargs.get("top_k") is not None:
            generation_config["top_k"] = kwargs["top_k"]

        gemini_tools = None
        if kwargs.get("tools"):
            gemini_tools = self._convert_tools(kwargs["tools"])

        model_obj = self.genai.GenerativeModel(
            model,
            system_instruction=system_instruction,
            tools=gemini_tools,
        )

        def _call():
            # Pass generic safety settings to avoid blocking harmless headers
            safety_settings = {
                HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
            }
            return model_obj.generate_content(
                contents,
                generation_config=generation_config,
                safety_settings=safety_settings,
            )

        start = time.perf_counter()
        resp = await asyncio.to_thread(_call)
        latency_ms = (time.perf_counter() - start) * 1000.0

        try:
            text = resp.text
        except ValueError:
            text = ""
        
        tool_calls = []
        for candidate in resp.candidates:
            for part in candidate.content.parts:
                if hasattr(part, "function_call") and part.function_call:
                    fc = part.function_call
                    tool_calls.append({
                        "id": f"gemini_{fc.name}_{len(tool_calls)}",
                        "name": fc.name,
                        "arguments": dict(fc.args) if fc.args else {},
                    })

        um = getattr(resp, "usage_metadata", None)
        raw_usage = None
        input_tokens = output_tokens = total_tokens = None
        if um is not None:
            raw_usage = {
                "prompt_token_count": um.prompt_token_count,
                "candidates_token_count": um.candidates_token_count,
                "total_token_count": um.total_token_count,
            }
            input_tokens = um.prompt_token_count
            output_tokens = um.candidates_token_count
            total_tokens = um.total_token_count

        usage = self.normalize_usage(
            "gemini",
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
            raw=raw_usage,
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
        if not self.genai:
            raise RuntimeError("Gemini client not configured")

        system_instruction, contents = await self._convert_messages(messages)

        generation_config = {
            "temperature": kwargs.get("temperature", 0.7),
            "max_output_tokens": kwargs.get("max_tokens", 1024),
        }
        if kwargs.get("top_p") is not None:
            generation_config["top_p"] = kwargs["top_p"]
        if kwargs.get("top_k") is not None:
            generation_config["top_k"] = kwargs["top_k"]
        if kwargs.get("stop_sequences") is not None:
            generation_config["stop_sequences"] = kwargs["stop_sequences"]

        gemini_model = self.genai.GenerativeModel(
            model,
            system_instruction=system_instruction,
        )

        full_text = ""
        start = time.perf_counter()

        response = await gemini_model.generate_content_async(
            contents,
            generation_config=generation_config,
            stream=True,
        )

        async for chunk in response:
            if chunk.text:
                full_text += chunk.text
                yield {
                    "type": "token",
                    "provider": "gemini",
                    "text": chunk.text,
                }

        latency_ms = (time.perf_counter() - start) * 1000.0

        usage = None
        if hasattr(response, "usage_metadata") and response.usage_metadata:
            um = response.usage_metadata
            usage = self.normalize_usage(
                "gemini",
                input_tokens=getattr(um, "prompt_token_count", None),
                output_tokens=getattr(um, "candidates_token_count", None),
                total_tokens=getattr(um, "total_token_count", None),
                raw={
                    "prompt_token_count": getattr(um, "prompt_token_count", None),
                    "candidates_token_count": getattr(um, "candidates_token_count", None),
                    "total_token_count": getattr(um, "total_token_count", None),
                },
            )

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
        Convert messages to Gemini format.
        
        Gemini uses different role names:
        - "user" -> "user"
        - "assistant" -> "model"
        
        System prompts are passed separately, similar to Anthropic.
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
            gemini_role = "model" if role == "assistant" else "user"
            
            # Build content parts
            if isinstance(content, str):
                contents.append({
                    "role": gemini_role,
                    "parts": [{"text": content}],
                })
            else:
                parts = []
                for part in content:
                    if part.get("type") == "text":
                        parts.append({"text": part.get("text", "")})
                    elif part.get("type") == "image_url":
                        url = part.get("image_url", {}).get("url", "")
                        # Gemini supports inline data
                        b64_data, mime_type = await resolve_image_to_base64(url)
                        parts.append({
                            "inline_data": {
                                "mime_type": mime_type,
                                "data": b64_data,
                            }
                        })
                
                if parts:
                    contents.append({"role": gemini_role, "parts": parts})
        
        return system_instruction, contents

    @staticmethod
    def _convert_tools(tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Convert OpenAI-format tools to Gemini format.
        """
        function_declarations = []
        for tool in tools:
            func = tool.get("function", {})
            function_declarations.append({
                "name": func.get("name", ""),
                "description": func.get("description", ""),
                "parameters": func.get("parameters", {"type": "object", "properties": {}}),
            })
        return [{"function_declarations": function_declarations}]

    async def get_models(self) -> List[str]:
        """
        Get list of available models.
        """
        if not self.genai:
            return []
            
        def _list():
            try:
                msg = []
                for m in self.genai.list_models():
                    if 'generateContent' in m.supported_actions:
                        msg.append(m.name)
                return msg
            except Exception:
                return []

        return await asyncio.to_thread(_list)
