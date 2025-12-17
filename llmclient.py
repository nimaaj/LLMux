import asyncio
import base64
import json
import os
import time
from pathlib import Path
from typing import Callable, Literal, List, Dict, Any, Optional, AsyncIterator, Union, TypedDict

import httpx
from openai import AsyncOpenAI, OpenAI
from anthropic import AsyncAnthropic, Anthropic
from google import genai as genai_client
import google.generativeai as genai
import dotenv
dotenv.load_dotenv()


# =============================================================================
# Type Definitions
# =============================================================================

Provider = Literal["openai", "claude", "gemini", "deepseek"]


class TextContent(TypedDict, total=False):
    """Text content part."""
    type: Literal["text"]
    text: str


class ImageUrlDetail(TypedDict, total=False):
    """Image URL with optional detail level."""
    url: str
    detail: Literal["auto", "low", "high"]  # OpenAI-specific


class ImageContent(TypedDict, total=False):
    """Image content part (OpenAI format - used as standard)."""
    type: Literal["image_url"]
    image_url: ImageUrlDetail


# Content can be a string or list of content parts
ContentPart = Union[TextContent, ImageContent]
MessageContent = Union[str, List[ContentPart]]


# =============================================================================
# Tool Calling Type Definitions
# =============================================================================

class FunctionParameters(TypedDict, total=False):
    """JSON Schema for function parameters."""
    type: Literal["object"]
    properties: Dict[str, Any]
    required: List[str]


class FunctionDefinition(TypedDict, total=False):
    """Function definition for tools."""
    name: str
    description: str
    parameters: FunctionParameters


class Tool(TypedDict):
    """Tool definition (OpenAI format as standard)."""
    type: Literal["function"]
    function: FunctionDefinition


class ToolCall(TypedDict, total=False):
    """Tool call from LLM response."""
    id: str
    name: str
    arguments: Dict[str, Any]  # Parsed JSON arguments


class ToolResult(TypedDict):
    """Tool result to send back to LLM."""
    tool_call_id: str
    content: str


# =============================================================================
# Message Type (depends on ToolCall)
# =============================================================================

class Message(TypedDict, total=False):
    """Chat message with optional multimodal content."""
    role: Literal["system", "user", "assistant", "tool"]
    content: MessageContent
    tool_call_id: str  # For tool result messages
    tool_calls: List[ToolCall]  # For assistant messages with tool calls


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

    # ==========================================================================
    # Image Helpers (Static/Class Methods)
    # ==========================================================================
    
    @staticmethod
    def encode_image_file(image_path: Union[str, Path]) -> tuple[str, str]:
        """
        Encode a local image file to base64.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Tuple of (base64_data, mime_type)
        """
        path = Path(image_path)
        if not path.exists():
            raise FileNotFoundError(f"Image file not found: {image_path}")
        
        mime_types = {
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".png": "image/png",
            ".gif": "image/gif",
            ".webp": "image/webp",
            ".bmp": "image/bmp",
        }
        mime_type = mime_types.get(path.suffix.lower(), "image/jpeg")
        
        with open(path, "rb") as f:
            b64_data = base64.b64encode(f.read()).decode("utf-8")
        
        return b64_data, mime_type

    @staticmethod
    async def encode_image_url(url: str) -> tuple[str, str]:
        """
        Download an image from URL and encode to base64.
        
        Args:
            url: HTTP(S) URL to the image
            
        Returns:
            Tuple of (base64_data, mime_type)
        """
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        }
        async with httpx.AsyncClient(timeout=30.0, headers=headers, follow_redirects=True) as client:
            response = await client.get(url)
            response.raise_for_status()
            content_type = response.headers.get("content-type", "image/jpeg")
            mime_type = content_type.split(";")[0].strip()
            b64_data = base64.b64encode(response.content).decode("utf-8")
        return b64_data, mime_type

    @staticmethod
    def create_image_content(
        source: str,
        *,
        mime_type: Optional[str] = None,
        detail: Optional[Literal["auto", "low", "high"]] = None,
    ) -> ImageContent:
        """
        Create an image content part from various sources.
        
        Args:
            source: One of:
                - Local file path (e.g., "./image.jpg")
                - HTTP(S) URL (e.g., "https://example.com/image.jpg")
                - Base64 data URI (e.g., "data:image/jpeg;base64,...")
                - Raw base64 string (requires mime_type)
            mime_type: MIME type (required for raw base64, auto-detected otherwise)
            detail: Image detail level for OpenAI ("auto", "low", "high")
            
        Returns:
            ImageContent dict in OpenAI format (standard format)
        """
        # Already a data URI
        if source.startswith("data:"):
            url = source
        # HTTP(S) URL - keep as-is for OpenAI, will be converted for others
        elif source.startswith(("http://", "https://")):
            url = source
        # Raw base64 string (if mime_type provided, assume it's base64)
        elif mime_type:
            url = f"data:{mime_type};base64,{source}"
        # Local file path - check length first to avoid OS errors with long strings
        elif len(source) < 260 and Path(source).exists():
            b64_data, detected_mime = UnifiedChatClient.encode_image_file(source)
            url = f"data:{detected_mime};base64,{b64_data}"
        else:
            raise ValueError(
                f"Cannot determine image source type for: {source[:50]}... "
                "Provide mime_type for raw base64 data."
            )
        
        image_url: ImageUrlDetail = {"url": url}
        if detail:
            image_url["detail"] = detail
            
        return {"type": "image_url", "image_url": image_url}

    @staticmethod
    def create_text_content(text: str) -> TextContent:
        """Create a text content part."""
        return {"type": "text", "text": text}

    @classmethod
    def create_message(
        cls,
        role: Literal["system", "user", "assistant"],
        content: Union[str, List[Union[str, ContentPart]]],
    ) -> Message:
        """
        Create a message with text and/or images.
        
        Args:
            role: Message role ("system", "user", "assistant")
            content: Either:
                - A string (text-only message)
                - A list of content parts (text/image dicts or strings)
                
        Returns:
            Message dict
            
        Example:
            # Text-only
            client.create_message("user", "Hello!")
            
            # With image
            client.create_message("user", [
                "What's in this image?",
                client.create_image_content("./photo.jpg"),
            ])
        """
        if isinstance(content, str):
            return {"role": role, "content": content}
        
        # Normalize list content - convert plain strings to TextContent
        normalized: List[ContentPart] = []
        for item in content:
            if isinstance(item, str):
                normalized.append(cls.create_text_content(item))
            else:
                normalized.append(item)
        
        return {"role": role, "content": normalized}

    # ==========================================================================
    # Tool Calling Helpers
    # ==========================================================================
    
    @staticmethod
    def create_tool(
        name: str,
        description: str,
        parameters: Dict[str, Any],
        required: Optional[List[str]] = None,
    ) -> Tool:
        """
        Create a tool definition.
        
        Args:
            name: Function name (e.g., "get_weather")
            description: Description of what the function does
            parameters: JSON Schema properties for parameters
            required: List of required parameter names
            
        Returns:
            Tool definition in OpenAI format
            
        Example:
            tool = client.create_tool(
                name="get_weather",
                description="Get the current weather for a location",
                parameters={
                    "location": {
                        "type": "string",
                        "description": "City name, e.g., 'Paris'"
                    },
                    "unit": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"],
                        "description": "Temperature unit"
                    }
                },
                required=["location"]
            )
        """
        return {
            "type": "function",
            "function": {
                "name": name,
                "description": description,
                "parameters": {
                    "type": "object",
                    "properties": parameters,
                    "required": required or [],
                },
            },
        }

    @staticmethod
    def create_tool_result(tool_call_id: str, content: str) -> Message:
        """
        Create a tool result message to send back to the LLM.
        
        Args:
            tool_call_id: ID from the tool call
            content: Result of executing the tool (string)
            
        Returns:
            Message with role "tool"
        """
        return {
            "role": "tool",
            "tool_call_id": tool_call_id,
            "content": content,
        }

    @staticmethod
    def create_assistant_message_with_tool_calls(
        content: str,
        tool_calls: List[ToolCall],
    ) -> Message:
        """
        Create an assistant message with tool calls (for conversation history).
        
        Args:
            content: Text content from the assistant (can be empty)
            tool_calls: List of tool calls from the response
            
        Returns:
            Message with role "assistant" and tool_calls
        """
        return {
            "role": "assistant",
            "content": content,
            "tool_calls": tool_calls,
        }

    @staticmethod
    def _parse_tool_calls_openai(choice) -> List[ToolCall]:
        """Parse tool calls from OpenAI/DeepSeek response."""
        import json
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

    @staticmethod
    def _parse_tool_calls_claude(response) -> List[ToolCall]:
        """Parse tool calls from Claude response."""
        tool_calls = []
        for block in response.content:
            if getattr(block, "type", None) == "tool_use":
                tool_calls.append({
                    "id": block.id,
                    "name": block.name,
                    "arguments": block.input,
                })
        return tool_calls

    @staticmethod
    def _convert_tools_for_claude(tools: List[Tool]) -> List[Dict[str, Any]]:
        """Convert OpenAI-format tools to Claude format."""
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
    def _convert_tools_for_gemini(tools: List[Tool]) -> List[Dict[str, Any]]:
        """Convert OpenAI-format tools to Gemini format."""
        function_declarations = []
        for tool in tools:
            func = tool.get("function", {})
            function_declarations.append({
                "name": func.get("name", ""),
                "description": func.get("description", ""),
                "parameters": func.get("parameters", {"type": "object", "properties": {}}),
            })
        return [{"function_declarations": function_declarations}]

    # ==========================================================================
    # Message Conversion for Each Provider
    # ==========================================================================
    
    @staticmethod
    def _is_multimodal_message(message: Message) -> bool:
        """Check if a message contains multimodal content."""
        return isinstance(message.get("content"), list)

    async def _convert_messages_for_openai(
        self,
        messages: List[Message],
    ) -> List[Dict[str, Any]]:
        """
        Convert messages to OpenAI format.
        
        OpenAI natively supports:
        - Text content: {"type": "text", "text": "..."}
        - Image URLs: {"type": "image_url", "image_url": {"url": "..."}}
        - Base64 images: {"type": "image_url", "image_url": {"url": "data:..."}}
        - Tool calls and tool results
        
        Note: We convert HTTP URLs to base64 for reliability, as OpenAI's servers
        may fail to download from certain sources.
        """
        converted = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            
            # Handle tool result messages
            if role == "tool":
                converted.append({
                    "role": "tool",
                    "tool_call_id": msg.get("tool_call_id", ""),
                    "content": content if isinstance(content, str) else "",
                })
                continue
            
            # Handle assistant messages with tool_calls
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
            
            if isinstance(content, str):
                converted.append({"role": role, "content": content})
            else:
                # Process multimodal content
                openai_content = []
                for part in content:
                    if part.get("type") == "text":
                        openai_content.append(part)
                    elif part.get("type") == "image_url":
                        url = part.get("image_url", {}).get("url", "")
                        detail = part.get("image_url", {}).get("detail")
                        
                        # Convert HTTP URLs to base64 for reliability
                        if url.startswith(("http://", "https://")):
                            b64_data, mime_type = await self.encode_image_url(url)
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

    async def _convert_messages_for_claude(
        self,
        messages: List[Message],
    ) -> tuple[Optional[str], List[Dict[str, Any]]]:
        """
        Convert messages to Claude format and extract system message.
        
        Claude format for images:
        {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": "image/jpeg",
                "data": "<base64>"
            }
        }
        
        Returns:
            (system_text, converted_messages)
        """
        system_parts = []
        converted = []
        
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            
            # Collect system messages
            if role == "system":
                if isinstance(content, str):
                    system_parts.append(content)
                else:
                    # Extract text from multimodal system message
                    for part in content:
                        if part.get("type") == "text":
                            system_parts.append(part.get("text", ""))
                continue
            
            # Convert content
            if isinstance(content, str):
                converted.append({"role": role, "content": content})
            else:
                claude_content = []
                for part in content:
                    if part.get("type") == "text":
                        claude_content.append({
                            "type": "text",
                            "text": part.get("text", "")
                        })
                    elif part.get("type") == "image_url":
                        url = part.get("image_url", {}).get("url", "")
                        
                        # Convert to Claude format
                        if url.startswith("data:"):
                            # Parse data URI
                            header, b64_data = url.split(",", 1)
                            media_type = header.split(":")[1].split(";")[0]
                            claude_content.append({
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": media_type,
                                    "data": b64_data,
                                }
                            })
                        elif url.startswith(("http://", "https://")):
                            # Download and convert to base64 for Claude
                            b64_data, mime_type = await self.encode_image_url(url)
                            claude_content.append({
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": mime_type,
                                    "data": b64_data,
                                }
                            })
                
                if claude_content:
                    converted.append({"role": role, "content": claude_content})
        
        system_text = "\n\n".join(system_parts) if system_parts else None
        return system_text, converted

    async def _convert_messages_for_gemini(
        self,
        messages: List[Message],
    ) -> tuple[Optional[str], List[Dict[str, Any]]]:
        """
        Convert messages to Gemini format.
        
        Gemini format:
        - system_instruction: str (passed separately to GenerativeModel)
        - contents: [{"role": "user"|"model", "parts": [{"text": "..."} | {"inline_data": {...}}]}]
        
        Returns:
            (system_instruction, contents)
        """
        system_instruction = None
        contents = []
        
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            
            # Handle system messages
            if role == "system":
                if isinstance(content, str):
                    system_instruction = content
                else:
                    # Extract text from multimodal content
                    texts = [p.get("text", "") for p in content if p.get("type") == "text"]
                    system_instruction = "\n".join(texts)
                continue
            
            # Map roles: OpenAI/Claude use "assistant", Gemini uses "model"
            gemini_role = "model" if role == "assistant" else "user"
            
            # Handle string content
            if isinstance(content, str):
                contents.append({
                    "role": gemini_role,
                    "parts": [{"text": content}],
                })
            # Handle multimodal content
            else:
                parts = []
                for part in content:
                    if part.get("type") == "text":
                        parts.append({"text": part.get("text", "")})
                    elif part.get("type") == "image_url":
                        url = part.get("image_url", {}).get("url", "")
                        
                        if url.startswith("data:"):
                            # Parse data URI
                            header, b64_data = url.split(",", 1)
                            mime_type = header.split(":")[1].split(";")[0]
                            parts.append({
                                "inline_data": {
                                    "mime_type": mime_type,
                                    "data": b64_data,
                                }
                            })
                        elif url.startswith(("http://", "https://")):
                            # Download and convert for Gemini
                            b64_data, mime_type = await self.encode_image_url(url)
                            parts.append({
                                "inline_data": {
                                    "mime_type": mime_type,
                                    "data": b64_data,
                                }
                            })
                
                if parts:
                    contents.append({"role": gemini_role, "parts": parts})
        
        return system_instruction, contents

    # ==========================================================================
    # Legacy Helper (for backward compatibility)
    # ==========================================================================
    
    @staticmethod
    def _split_system_messages(messages: List[Message]):
        """Split system vs non-system messages (legacy helper)."""
        system_msgs = []
        other_msgs = []
        for m in messages:
            if m["role"] == "system":
                content = m.get("content", "")
                if isinstance(content, str):
                    system_msgs.append(content)
                else:
                    # Extract text from multimodal
                    texts = [p.get("text", "") for p in content if p.get("type") == "text"]
                    system_msgs.extend(texts)
            else:
                other_msgs.append(m)
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

        # Convert messages for OpenAI (handles multimodal content)
        converted_messages = await self._convert_messages_for_openai(messages)

        # Required params
        kwargs = {
            "model": model,
            "messages": converted_messages,
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
        
        # Parse tool calls if present
        tool_calls = self._parse_tool_calls_openai(choice)

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

        result = {
            "provider": "openai",
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
    async def _deepseek_chat(
        self,
        model: str,
        messages: List[Message],
        **opts,
    ) -> Dict[str, Any]:
        if not self.deepseek:
            raise RuntimeError("DeepSeek client not configured")

        # Convert messages (DeepSeek uses OpenAI format)
        converted_messages = await self._convert_messages_for_openai(messages)

        # Required params
        kwargs = {
            "model": model,
            "messages": converted_messages,
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
        
        # Parse tool calls if present
        tool_calls = self._parse_tool_calls_openai(choice)

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

        result = {
            "provider": "deepseek",
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

    async def _claude_chat(
        self,
        model: str,
        messages: List[Message],
        **opts,
    ) -> Dict[str, Any]:
        if not self.claude:
            raise RuntimeError("Claude (Anthropic) client not configured")

        # Convert messages for Claude (handles multimodal content)
        system_text, converted_messages = await self._convert_messages_for_claude(messages)

        # Required params
        kwargs = {
            "model": model,
            "messages": converted_messages,
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
        
        # Add tools if provided (convert to Claude format)
        if opts.get("tools"):
            kwargs["tools"] = self._convert_tools_for_claude(opts["tools"])
            if opts.get("tool_choice"):
                tool_choice = opts["tool_choice"]
                if tool_choice == "auto":
                    kwargs["tool_choice"] = {"type": "auto"}
                elif tool_choice == "none":
                    kwargs["tool_choice"] = {"type": "none"}  
                elif tool_choice == "required":
                    kwargs["tool_choice"] = {"type": "any"}
                elif isinstance(tool_choice, str):
                    kwargs["tool_choice"] = {"type": "tool", "name": tool_choice}

        start = time.perf_counter()
        resp = await self.claude.messages.create(**kwargs)
        latency_ms = (time.perf_counter() - start) * 1000.0

        text = "".join(
            block.text for block in resp.content if getattr(block, "type", None) == "text"
        )
        
        # Parse tool calls if present
        tool_calls = self._parse_tool_calls_claude(resp)

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

        # Convert messages for Gemini (handles multimodal content)
        system_instruction, contents = await self._convert_messages_for_gemini(messages)

        generation_config = {
            "temperature": temperature,
            "max_output_tokens": max_tokens,
        }
        if top_p is not None:
            generation_config["top_p"] = top_p
        if top_k is not None:
            generation_config["top_k"] = top_k

        # Prepare tools if provided
        gemini_tools = None
        if opts.get("tools"):
            gemini_tools = self._convert_tools_for_gemini(opts["tools"])

        model_obj = genai.GenerativeModel(
            model,
            system_instruction=system_instruction,
            tools=gemini_tools,
        )

        def _call():
            return model_obj.generate_content(
                contents,
                generation_config=generation_config,
            )

        start = time.perf_counter()
        resp = await asyncio.to_thread(_call)
        latency_ms = (time.perf_counter() - start) * 1000.0

        # Get text (may be empty if tool call)
        try:
            text = resp.text
        except ValueError:
            text = ""
        
        # Parse tool calls from Gemini response
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

        usage = self._normalize_usage(
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

        # Convert messages for OpenAI (handles multimodal content)
        converted_messages = await self._convert_messages_for_openai(messages)

        temperature = opts.get("temperature", 0.7)
        max_tokens = opts.get("max_tokens", 1024)
        top_p = opts.get("top_p", None)

        start = time.perf_counter()
        # Async stream of ChatCompletionChunk objects
        stream = await self.openai.chat.completions.create(
            model=model,
            messages=converted_messages,
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

            # Convert messages (DeepSeek uses OpenAI format)
            converted_messages = await self._convert_messages_for_openai(messages)

            temperature = opts.get("temperature", 0.7)
            max_tokens = opts.get("max_tokens", 1024)
            top_p = opts.get("top_p", None)

            start = time.perf_counter()
            # Async stream of ChatCompletionChunk objects
            stream = await self.deepseek.chat.completions.create(
                model=model,
                messages=converted_messages,
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

        # Convert messages for Claude (handles multimodal content)
        system_text, converted_messages = await self._convert_messages_for_claude(messages)

        # Required params
        kwargs = {
            "model": model,
            "messages": converted_messages,
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

    async def _gemini_stream(
        self,
        model: str,
        messages: List[Message],
        **opts,
    ) -> AsyncIterator[Dict[str, Any]]:
        if not self.gemini:
            raise RuntimeError("Gemini client not configured")

        # Convert messages for Gemini (handles multimodal content)
        system_instruction, contents = await self._convert_messages_for_gemini(messages)

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

    # ==========================================================================
    # MCP Integration Methods
    # ==========================================================================
    
    async def chat_with_tools(
        self,
        provider: Provider,
        model: str,
        messages: List[Message],
        tools: Optional[List[Tool]] = None,
        mcp_executor: Optional[Any] = None,
        tool_handlers: Optional[Dict[str, Callable]] = None,
        auto_execute: bool = True,
        max_iterations: int = 10,
        **opts,
    ) -> Dict[str, Any]:
        """
        Chat with automatic tool execution support.
        
        This method provides a higher-level interface that:
        1. Combines native tools with MCP tools
        2. Automatically executes tool calls (if auto_execute=True)
        3. Continues the conversation until completion
        
        Args:
            provider: LLM provider
            model: Model name
            messages: Conversation messages
            tools: Native tool definitions (OpenAI format)
            mcp_executor: MCPToolExecutor instance for MCP tools
            tool_handlers: Dict mapping tool names to handler functions
                          for native tools. Signature: (arguments: dict) -> str
            auto_execute: Whether to automatically execute tools
            max_iterations: Maximum tool execution iterations
            **opts: Additional options for chat()
            
        Returns:
            Final response with "text", "meta", and optional "tool_history"
            
        Example:
            >>> # With native tools
            >>> def get_weather(location: str) -> str:
            ...     return f"Sunny in {location}"
            >>> 
            >>> tools = [client.create_tool("get_weather", "Get weather", {"location": {"type": "string"}})]
            >>> response = await client.chat_with_tools(
            ...     provider="openai",
            ...     model="gpt-4o",
            ...     messages=[{"role": "user", "content": "What's the weather in Paris?"}],
            ...     tools=tools,
            ...     tool_handlers={"get_weather": lambda args: get_weather(**args)},
            ... )
            
            >>> # With MCP tools
            >>> from mcp_client import MCPToolExecutor
            >>> executor = MCPToolExecutor()
            >>> await executor.connect_stdio("fs", "npx", ["-y", "@modelcontextprotocol/server-filesystem"])
            >>> response = await client.chat_with_tools(
            ...     provider="openai",
            ...     model="gpt-4o",
            ...     messages=[{"role": "user", "content": "List files in /tmp"}],
            ...     mcp_executor=executor,
            ... )
        """
        # Combine native tools with MCP tools
        all_tools = list(tools) if tools else []
        if mcp_executor:
            mcp_tools = mcp_executor.get_tools()
            # Remove internal metadata before sending to LLM
            for tool in mcp_tools:
                clean_tool = {k: v for k, v in tool.items() if not k.startswith("_")}
                all_tools.append(clean_tool)
        
        tool_handlers = tool_handlers or {}
        tool_history = []
        working_messages = list(messages)
        
        for iteration in range(max_iterations):
            # Make the chat request
            response = await self.chat(
                provider=provider,
                model=model,
                messages=working_messages,
                tools=all_tools if all_tools else None,
                **opts,
            )
            
            # Check for tool calls
            tool_calls = response.get("tool_calls", [])
            
            if not tool_calls or not auto_execute:
                # No tool calls or auto-execute disabled, return response
                response["tool_history"] = tool_history
                return response
            
            # Execute tool calls
            tool_results = []
            for tc in tool_calls:
                tool_name = tc.get("name", "")
                arguments = tc.get("arguments", {})
                tool_id = tc.get("id", "")
                
                try:
                    # Check native handlers first
                    if tool_name in tool_handlers:
                        result = tool_handlers[tool_name](arguments)
                        if asyncio.iscoroutine(result):
                            result = await result
                    # Then check MCP executor
                    elif mcp_executor and tool_name in mcp_executor.tool_names:
                        result = await mcp_executor.call_tool(tool_name, arguments)
                    else:
                        result = f"Error: No handler for tool '{tool_name}'"
                    
                    tool_results.append({
                        "role": "tool",
                        "tool_call_id": tool_id,
                        "content": str(result),
                    })
                    tool_history.append({
                        "tool": tool_name,
                        "arguments": arguments,
                        "result": str(result),
                    })
                except Exception as e:
                    error_msg = f"Error executing tool '{tool_name}': {str(e)}"
                    tool_results.append({
                        "role": "tool",
                        "tool_call_id": tool_id,
                        "content": error_msg,
                    })
                    tool_history.append({
                        "tool": tool_name,
                        "arguments": arguments,
                        "error": str(e),
                    })
            
            # Add assistant message with tool calls and tool results
            assistant_msg = self.create_assistant_message_with_tool_calls(
                response.get("text", ""),
                tool_calls,
            )
            working_messages.append(assistant_msg)
            working_messages.extend(tool_results)
        
        # Max iterations reached
        response["tool_history"] = tool_history
        response["warning"] = f"Max iterations ({max_iterations}) reached"
        return response
    
    async def astream_with_tools(
        self,
        provider: Provider,
        model: str,
        messages: List[Message],
        tools: Optional[List[Tool]] = None,
        mcp_executor: Optional[Any] = None,
        tool_handlers: Optional[Dict[str, Callable]] = None,
        auto_execute: bool = True,
        max_iterations: int = 10,
        **opts,
    ) -> AsyncIterator[Dict[str, Any]]:
        """
        Stream chat with automatic tool execution support.
        
        Similar to chat_with_tools but yields streaming events.
        
        Yields events of the form:
            {"type": "token", "provider": "...", "text": "..."}
            {"type": "tool_call", "provider": "...", "tool_calls": [...]}
            {"type": "tool_result", "provider": "...", "results": [...]}
            {"type": "done", "provider": "...", "text": full_text, "meta": {...}}
        """
        # Combine native tools with MCP tools
        all_tools = list(tools) if tools else []
        if mcp_executor:
            mcp_tools = mcp_executor.get_tools()
            for tool in mcp_tools:
                clean_tool = {k: v for k, v in tool.items() if not k.startswith("_")}
                all_tools.append(clean_tool)
        
        tool_handlers = tool_handlers or {}
        working_messages = list(messages)
        
        for iteration in range(max_iterations):
            # Stream the response
            full_text = ""
            tool_calls = []
            final_event = None
            
            async for event in self.astream(
                provider=provider,
                model=model,
                messages=working_messages,
                tools=all_tools if all_tools else None,
                **opts,
            ):
                if event["type"] == "token":
                    full_text += event["text"]
                    yield event
                elif event["type"] == "done":
                    final_event = event
                    tool_calls = event.get("tool_calls", [])
            
            if not tool_calls or not auto_execute:
                if final_event:
                    yield final_event
                return
            
            # Yield tool call event
            yield {
                "type": "tool_call",
                "provider": provider,
                "tool_calls": tool_calls,
            }
            
            # Execute tool calls
            tool_results = []
            for tc in tool_calls:
                tool_name = tc.get("name", "")
                arguments = tc.get("arguments", {})
                tool_id = tc.get("id", "")
                
                try:
                    if tool_name in tool_handlers:
                        result = tool_handlers[tool_name](arguments)
                        if asyncio.iscoroutine(result):
                            result = await result
                    elif mcp_executor and tool_name in mcp_executor.tool_names:
                        result = await mcp_executor.call_tool(tool_name, arguments)
                    else:
                        result = f"Error: No handler for tool '{tool_name}'"
                    
                    tool_results.append({
                        "role": "tool",
                        "tool_call_id": tool_id,
                        "content": str(result),
                    })
                except Exception as e:
                    tool_results.append({
                        "role": "tool",
                        "tool_call_id": tool_id,
                        "content": f"Error: {str(e)}",
                    })
            
            # Yield tool results event
            yield {
                "type": "tool_result",
                "provider": provider,
                "results": tool_results,
            }
            
            # Add to messages for next iteration
            assistant_msg = self.create_assistant_message_with_tool_calls(
                full_text,
                tool_calls,
            )
            working_messages.append(assistant_msg)
            working_messages.extend(tool_results)