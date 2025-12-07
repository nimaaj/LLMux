import asyncio
import os
import time
from typing import Literal, List, Dict, Any, Optional, AsyncIterator

from openai import AsyncOpenAI, OpenAI
from anthropic import AsyncAnthropic, Anthropic
from google import genai as genai_client
import google.generativeai as genai
import dotenv
dotenv.load_dotenv()



Provider = Literal["openai", "claude", "gemini","deepseek"]
Message = Dict[str, str]  # {"role": "system"|"user"|"assistant", "content": "..."}


class UnifiedChatClient:
    def __init__(
        self,
        openai_api_key: Optional[str] = dotenv.get_key(".env", "OPENAI_API_KEY"),
        anthropic_api_key: Optional[str] = dotenv.get_key(".env", "ANTHROPIC_API_KEY"),
        gemini_api_key: Optional[str] = dotenv.get_key(".env", "GOOGLE_API_KEY"),
        deepseek_api_key: Optional[str] = dotenv.get_key(".env", "DEEPSEEK_API_KEY"),
    ):
        self.openai = AsyncOpenAI(api_key=openai_api_key) if openai_api_key else None
        self.deepseek = AsyncOpenAI(api_key=deepseek_api_key,base_url="https://api.deepseek.com") if deepseek_api_key else None
        self.claude = AsyncAnthropic(api_key=anthropic_api_key) if anthropic_api_key else None
        if gemini_api_key:
            genai.configure(api_key=gemini_api_key)
            self.gemini = genai
        else:
            self.gemini = None

    # --------------------------
    # Helpers
    # --------------------------
    @staticmethod
    def _split_system_messages(messages: List[Message]):
        """Split system vs non-system messages."""
        system_msgs = [m["content"] for m in messages if m["role"] == "system"]
        other_msgs = [m for m in messages if m["role"] != "system"]
        system_text = "\n\n".join(system_msgs) if system_msgs else None
        return system_text, other_msgs

    @staticmethod
    def _normalize_usage(
        provider: str,
        *,
        input_tokens: Optional[int],
        output_tokens: Optional[int],
        total_tokens: Optional[int],
        raw: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Return a normalized usage dict with provider-specific raw info."""
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

    # --------------------------
    # Non-streaming provider-specific methods
    # --------------------------
    async def _openai_chat(
        self,
        model: str,
        messages: List[Message],
        **opts,
    ) -> Dict[str, Any]:
        if not self.openai:
            raise RuntimeError("OpenAI client not configured")

        # Required params
        kwargs = {
            "model": model,
            "messages": messages,
        }

        # Optional params - only include if explicitly set or have non-None defaults
        optional_params = {
            "temperature": opts.get("temperature", 0.7),
            "max_tokens": opts.get("max_tokens", 1024),
            "top_p": opts.get("top_p"),
            "frequency_penalty": opts.get("frequency_penalty"),
            "presence_penalty": opts.get("presence_penalty"),
            "stop": opts.get("stop"),
            "seed": opts.get("seed"),
            "response_format": opts.get("response_format"),
            "tools": opts.get("tools"),
            "tool_choice": opts.get("tool_choice"),
            "user": opts.get("user"),
        }
        kwargs.update({k: v for k, v in optional_params.items() if v is not None})

        start = time.perf_counter()
        resp = await self.openai.chat.completions.create(**kwargs)
        latency_ms = (time.perf_counter() - start) * 1000.0

        choice = resp.choices[0]
        text = choice.message.content or ""

        usage = None
        if resp.usage:
            usage = self._normalize_usage(
                "openai",
                input_tokens=resp.usage.prompt_tokens,
                output_tokens=resp.usage.completion_tokens,
                total_tokens=resp.usage.total_tokens,
                raw={
                    "prompt_tokens": resp.usage.prompt_tokens,
                    "completion_tokens": resp.usage.completion_tokens,
                    "total_tokens": resp.usage.total_tokens,
                },
            )

        return {
            "provider": "openai",
            "text": text,
            "meta": {
                "model": resp.model,
                "usage": usage,
                "latency_ms": latency_ms,
                "finish_reason": choice.finish_reason,
            },
        }
    async def _deepseek_chat(
        self,
        model: str,
        messages: List[Message],
        **opts,
    ) -> Dict[str, Any]:
        if not self.deepseek:
            raise RuntimeError("DeepSeek client not configured")

        # Required params
        kwargs = {
            "model": model,
            "messages": messages,
        }

        # Optional params - only include if explicitly set or have non-None defaults
        optional_params = {
            "temperature": opts.get("temperature", 0.7),
            "max_tokens": opts.get("max_tokens", 1024),
            "top_p": opts.get("top_p"),
            "frequency_penalty": opts.get("frequency_penalty"),
            "presence_penalty": opts.get("presence_penalty"),
            "stop": opts.get("stop"),
            "seed": opts.get("seed"),
            "response_format": opts.get("response_format"),
            "tools": opts.get("tools"),
            "tool_choice": opts.get("tool_choice"),
            "user": opts.get("user"),
        }
        kwargs.update({k: v for k, v in optional_params.items() if v is not None})

        start = time.perf_counter()
        resp = await self.deepseek.chat.completions.create(**kwargs)
        latency_ms = (time.perf_counter() - start) * 1000.0

        choice = resp.choices[0]
        text = choice.message.content or ""

        usage = None
        if resp.usage:
            usage = self._normalize_usage(
                "deepseek",
                input_tokens=resp.usage.prompt_tokens,
                output_tokens=resp.usage.completion_tokens,
                total_tokens=resp.usage.total_tokens,
                raw={
                    "prompt_tokens": resp.usage.prompt_tokens,
                    "completion_tokens": resp.usage.completion_tokens,
                    "total_tokens": resp.usage.total_tokens,
                },
            )

        return {
            "provider": "deepseek",
            "text": text,
            "meta": {
                "model": resp.model,
                "usage": usage,
                "latency_ms": latency_ms,
                "finish_reason": choice.finish_reason,
            },
        }

    async def _claude_chat(
        self,
        model: str,
        messages: List[Message],
        **opts,
    ) -> Dict[str, Any]:
        if not self.claude:
            raise RuntimeError("Claude (Anthropic) client not configured")

        system_text, user_assistant_msgs = self._split_system_messages(messages)

        # Required params
        kwargs = {
            "model": model,
            "messages": user_assistant_msgs,
            "max_tokens": opts.get("max_tokens", 1024),
        }

        # Optional params - only include if explicitly set or have non-None defaults
        optional_params = {
            "system": system_text,
            "temperature": opts.get("temperature", 0.7),
            "top_p": opts.get("top_p"),
            "top_k": opts.get("top_k"),
            "stop_sequences": opts.get("stop_sequences"),
            "metadata": opts.get("metadata"),
        }
        kwargs.update({k: v for k, v in optional_params.items() if v is not None})

        start = time.perf_counter()
        resp = await self.claude.messages.create(**kwargs)
        latency_ms = (time.perf_counter() - start) * 1000.0

        text = "".join(
            block.text for block in resp.content if getattr(block, "type", None) == "text"
        )

        usage = None
        if resp.usage:
            usage = self._normalize_usage(
                "claude",
                input_tokens=resp.usage.input_tokens,
                output_tokens=resp.usage.output_tokens,
                total_tokens=None,
                raw={
                    "input_tokens": resp.usage.input_tokens,
                    "output_tokens": resp.usage.output_tokens,
                },
            )

        return {
            "provider": "claude",
            "text": text,
            "meta": {
                "model": resp.model,
                "usage": usage,
                "latency_ms": latency_ms,
            },
        }

    async def _gemini_chat(
        self,
        model: str,
        messages: List[Message],
        **opts,
    ) -> Dict[str, Any]:
        if not self.gemini:
            raise RuntimeError("Gemini client not configured")

        temperature = opts.get("temperature", 0.7)
        max_tokens = opts.get("max_tokens", 1024)
        top_p = opts.get("top_p", None)
        top_k = opts.get("top_k", None)

        system_text, other_msgs = self._split_system_messages(messages)

        contents = []
        for m in other_msgs:
            role = "model" if m["role"] == "assistant" else m["role"]
            contents.append({
                "role": role,
                "parts": [{"text": m["content"]}],
            })

        generation_config = {
            "temperature": temperature,
            "max_output_tokens": max_tokens,
        }
        if top_p is not None:
            generation_config["top_p"] = top_p
        if top_k is not None:
            generation_config["top_k"] = top_k

        model_obj = genai.GenerativeModel(
            model,
            system_instruction=system_text if system_text else None,
        )

        def _call():
            return model_obj.generate_content(
                contents,
                generation_config=generation_config,
            )

        start = time.perf_counter()
        resp = await asyncio.to_thread(_call)
        latency_ms = (time.perf_counter() - start) * 1000.0

        text = resp.text

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

        usage = self._normalize_usage(
            "gemini",
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
            raw=raw_usage,
        )

        return {
            "provider": "gemini",
            "text": text,
            "meta": {
                "model": model,
                "usage": usage,
                "latency_ms": latency_ms,
            },
        }

    # --------------------------
    # Streaming provider-specific methods
    # --------------------------
    async def _openai_stream(
        self,
        model: str,
        messages: List[Message],
        **opts,
    ) -> AsyncIterator[Dict[str, Any]]:
        """
        True streaming for OpenAI.

        Yields:
            {"type": "token", "provider": "openai", "text": "..."}
            ...
            {"type": "done", "provider": "openai", "text": full_text, "meta": {...}}
        """
        if not self.openai:
            raise RuntimeError("OpenAI client not configured")

        temperature = opts.get("temperature", 0.7)
        max_tokens = opts.get("max_tokens", 1024)
        top_p = opts.get("top_p", None)

        start = time.perf_counter()
        # Async stream of ChatCompletionChunk objects
        stream = await self.openai.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            stream=True,
        )

        full_text = ""
        last_chunk = None

        async for chunk in stream:
            last_chunk = chunk
            # Each chunk.choices[0].delta.content may be a list or string
            delta = chunk.choices[0].delta
            content = delta.content
            if not content:
                continue
            # The SDK often returns a list of content parts; handle both
            if isinstance(content, list):
                piece = "".join(part.get("text", "") for part in content if isinstance(part, dict))
            else:
                piece = str(content)

            if piece:
                full_text += piece
                yield {
                    "type": "token",
                    "provider": "openai",
                    "text": piece,
                }

        latency_ms = (time.perf_counter() - start) * 1000.0

        raw_usage = None
        input_tokens = output_tokens = total_tokens = None
        # Usage is often only present on the final chunk
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

        usage = self._normalize_usage(
            "openai",
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
            "provider": "openai",
            "text": full_text,
            "meta": meta,
        }

    async def _deepseek_stream(
            self,
            model: str,
            messages: List[Message],
            **opts,
        ) -> AsyncIterator[Dict[str, Any]]:
            """
            True streaming for DeepSeek.

            Yields:
                {"type": "token", "provider": "deepseek", "text": "..."}
                ...
                {"type": "done", "provider": "deepseek", "text": full_text, "meta": {...}}
            """
            if not self.deepseek:
                raise RuntimeError("DeepSeek client not configured")

            temperature = opts.get("temperature", 0.7)
            max_tokens = opts.get("max_tokens", 1024)
            top_p = opts.get("top_p", None)

            start = time.perf_counter()
            # Async stream of ChatCompletionChunk objects
            stream = await self.deepseek.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
                stream=True,
            )

            full_text = ""
            last_chunk = None

            async for chunk in stream:
                last_chunk = chunk
                # Each chunk.choices[0].delta.content may be a list or string
                delta = chunk.choices[0].delta
                content = delta.content
                if not content:
                    continue
                # The SDK often returns a list of content parts; handle both
                if isinstance(content, list):
                    piece = "".join(part.get("text", "") for part in content if isinstance(part, dict))
                else:
                    piece = str(content)

                if piece:
                    full_text += piece
                    yield {
                        "type": "token",
                        "provider": "deepseek",
                        "text": piece,
                    }

            latency_ms = (time.perf_counter() - start) * 1000.0

            raw_usage = None
            input_tokens = output_tokens = total_tokens = None
            # Usage is often only present on the final chunk
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

            usage = self._normalize_usage(
                "deepseek",
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
                "provider": "deepseek",
                "text": full_text,
                "meta": meta,
            }

    async def _claude_stream(
        self,
        model: str,
        messages: List[Message],
        **opts,
    ) -> AsyncIterator[Dict[str, Any]]:
        """
        True streaming for Claude using AsyncAnthropic.messages.stream.
        Yields:
            {"type": "token", "provider": "claude", "text": "..."}
            ...
            {"type": "done", "provider": "claude", "text": full_text, "meta": {...}}
        """
        if not self.claude:
            raise RuntimeError("Claude (Anthropic) client not configured")

        system_text, user_assistant_msgs = self._split_system_messages(messages)

        # Required params
        kwargs = {
            "model": model,
            "messages": user_assistant_msgs,
            "max_tokens": opts.get("max_tokens", 1024),
        }

        # Optional params - only include if explicitly set or have non-None defaults
        optional_params = {
            "system": system_text,
            "temperature": opts.get("temperature", 0.7),
            "top_p": opts.get("top_p"),
            "top_k": opts.get("top_k"),
            "stop_sequences": opts.get("stop_sequences"),
            "metadata": opts.get("metadata"),
        }
        kwargs.update({k: v for k, v in optional_params.items() if v is not None})

        full_text = ""
        start = time.perf_counter()

        async with self.claude.messages.stream(**kwargs) as stream:
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
            usage = self._normalize_usage(
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
    def _convert_messages_to_gemini(
        self, messages: List[Message]
    ) -> tuple[str | None, list[dict]]:
        """
        Convert OpenAI-style messages to Gemini format.
        
        Returns:
            (system_instruction, contents) tuple
        
        Gemini expects:
            - system_instruction: str (optional, passed separately to GenerativeModel)
            - contents: list of {"role": "user"|"model", "parts": [{"text": "..."}]}
        """
        system_instruction = None
        contents = []

        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")

            # Handle system messages
            if role == "system":
                # Gemini uses system_instruction parameter instead of a message
                system_instruction = content
                continue

            # Map roles: OpenAI/Claude use "assistant", Gemini uses "model"
            gemini_role = "model" if role == "assistant" else "user"

            # Handle string content
            if isinstance(content, str):
                contents.append({
                    "role": gemini_role,
                    "parts": [{"text": content}],
                })
            # Handle multimodal content (list of parts)
            elif isinstance(content, list):
                parts = []
                for part in content:
                    if part.get("type") == "text":
                        parts.append({"text": part.get("text", "")})
                    elif part.get("type") == "image_url":
                        # Handle base64 images
                        url = part.get("image_url", {}).get("url", "")
                        if url.startswith("data:"):
                            # Parse data URI: data:image/jpeg;base64,<data>
                            header, b64_data = url.split(",", 1)
                            mime_type = header.split(":")[1].split(";")[0]
                            parts.append({
                                "inline_data": {
                                    "mime_type": mime_type,
                                    "data": b64_data,
                                }
                            })
                if parts:
                    contents.append({"role": gemini_role, "parts": parts})

        return system_instruction, contents
    async def _gemini_stream(
    self,
    model: str,
    messages: List[Message],
    **opts,
    ) -> AsyncIterator[Dict[str, Any]]:
        if not self.gemini:
            raise RuntimeError("Gemini client not configured")

        system_instruction, contents = self._convert_messages_to_gemini(messages)

        generation_config = {
            "temperature": opts.get("temperature", 0.7),
            "max_output_tokens": opts.get("max_tokens", 1024),
        }
        if opts.get("top_p") is not None:
            generation_config["top_p"] = opts["top_p"]
        if opts.get("top_k") is not None:
            generation_config["top_k"] = opts["top_k"]
        if opts.get("stop_sequences") is not None:
            generation_config["stop_sequences"] = opts["stop_sequences"]

        # Pass system_instruction to model constructor
        gemini_model = self.gemini.GenerativeModel(
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

        # Get usage from final response if available
        usage = None
        if hasattr(response, "usage_metadata") and response.usage_metadata:
            um = response.usage_metadata
            usage = self._normalize_usage(
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
    # --------------------------
    # Unified public entrypoints
    # --------------------------
    async def chat(
        self,
        provider: Provider,
        model: str,
        messages: List[Message],
        **opts,
    ) -> Dict[str, Any]:
        """
        Unified non-streaming chat interface.

        Returns:
            {
                "provider": "openai" | "claude" | "gemini"| "deepseek",
                "text": "<assistant reply>",
                "meta": {
                    "model": "<model name>",
                    "usage": {
                        "input_tokens": int | None,
                        "output_tokens": int | None,
                        "total_tokens": int | None,
                        "raw": { ... provider-specific ... } | None,
                    },
                    "latency_ms": float,
                },
            }
        """
        if provider == "openai":
            return await self._openai_chat(model, messages, **opts)
        if provider == "deepseek":
            return await self._deepseek_chat(model, messages, **opts)
        elif provider == "claude":
            return await self._claude_chat(model, messages, **opts)
        elif provider == "gemini":
            return await self._gemini_chat(model, messages, **opts)
        else:
            raise ValueError(f"Unknown provider: {provider!r}")

    async def astream(
        self,
        provider: Provider,
        model: str,
        messages: List[Message],
        **opts,
    ) -> AsyncIterator[Dict[str, Any]]:
        """
        Unified streaming chat interface.

        Yields events of the form:
            {"type": "token", "provider": "...", "text": "..."}
            ...
            {"type": "done", "provider": "...", "text": full_text, "meta": {...}}
        """
        if provider == "openai":
            async for ev in self._openai_stream(model, messages, **opts):
                yield ev

        elif provider == "deepseek":
            async for ev in self._deepseek_stream(model, messages, **opts):
                yield ev
        elif provider == "claude":
            async for ev in self._claude_stream(model, messages, **opts):
                yield ev
        elif provider == "gemini":
            async for ev in self._gemini_stream(model, messages, **opts):
                yield ev
        else:
            raise ValueError(f"Unknown provider: {provider!r}")

    @staticmethod
    def get_models(provider: str) -> list[str]:
        """Get list of model names for a given provider.
        
        Args:
            provider: One of 'gemini', 'openai', 'deepseek', 'anthropic'
        
        Returns:
            List of model name strings
        """
        match provider.lower():
            case "gemini":
                client = genai_client.Client(api_key=dotenv.get_key(".env", "GOOGLE_API_KEY"))
                return [
                    m.name for m in client.models.list()
                    if "generateContent" in m.supported_actions
                ]
            case "openai":
                client = OpenAI(api_key=dotenv.get_key(".env", "OPENAI_API_KEY"))
                return [x.id for x in client.models.list().data]
            case "deepseek":
                client = OpenAI(api_key=dotenv.get_key(".env", "DEEPSEEK_API_KEY"), base_url="https://api.deepseek.com")
                return [x.id for x in client.models.list().data]
            case "anthropic":
                client = Anthropic(api_key=dotenv.get_key(".env", "ANTHROPIC_API_KEY"))
                return [x.id for x in client.models.list().data]
            case _:
                raise ValueError(f"Unknown provider: {provider}. Use 'gemini', 'openai', 'deepseek', or 'anthropic'")