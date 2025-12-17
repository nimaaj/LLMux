"""
LLMux - Unified LLM Client Library

This module provides a unified interface for interacting with multiple LLM providers
(OpenAI, Claude/Anthropic, Gemini, DeepSeek) through a single, consistent API.

Why LLMux?
----------
Each LLM provider has its own SDK, message formats, and quirks. LLMux abstracts
these differences, allowing you to:
- Switch between providers with a single parameter change
- Use the same message format across all providers
- Handle multimodal content (images) uniformly
- Use tool calling with the same interface
- Integrate MCP (Model Context Protocol) tools seamlessly

Architecture Overview:
---------------------
1. **Message Format**: Uses OpenAI's format as the standard. Messages are
   automatically converted to each provider's native format.
   
2. **Image Handling**: Images can be provided as URLs, local files, or base64.
   They're automatically converted to each provider's required format.
   
3. **Tool Calling**: Uses OpenAI's tool format as the standard. Tools are
   automatically converted for Claude and Gemini.
   
4. **Streaming**: All providers support streaming with a unified event format.

Quick Start:
-----------
```python
from llmclient import UnifiedChatClient
from rich_llm_printer import RichPrinter

client = UnifiedChatClient()
printer = RichPrinter()

# Simple chat
response = await client.chat(
    provider="openai",
    model="gpt-4o",
    messages=[{"role": "user", "content": "Hello!"}]
)
printer.print_chat(response)

# Switch to Claude - same interface!
response = await client.chat(
    provider="claude",
    model="claude-sonnet-4-20250514",
    messages=[{"role": "user", "content": "Hello!"}]
)
```
"""

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

# Load environment variables from .env file for API keys
dotenv.load_dotenv()


# =============================================================================
# Type Definitions
# =============================================================================
# These TypedDicts provide type hints for the data structures used throughout
# the client. They help with IDE autocompletion and catch type errors.

# Supported LLM providers
Provider = Literal["openai", "claude", "gemini", "deepseek"]


class TextContent(TypedDict, total=False):
    """
    Text content part for multimodal messages.
    
    This is used when a message contains multiple content types (text + images).
    For text-only messages, you can just use a string for content.
    
    Example:
        {"type": "text", "text": "What's in this image?"}
    """
    type: Literal["text"]
    text: str


class ImageUrlDetail(TypedDict, total=False):
    """
    Image URL specification with optional detail level.
    
    The 'detail' parameter is OpenAI-specific and controls how the model
    processes the image:
    - "auto": Let the model decide
    - "low": Faster processing, less detail
    - "high": Slower processing, more detail
    
    Example:
        {"url": "https://example.com/image.jpg", "detail": "high"}
    """
    url: str
    detail: Literal["auto", "low", "high"]  # OpenAI-specific


class ImageContent(TypedDict, total=False):
    """
    Image content part for multimodal messages (OpenAI format).
    
    This format is used as the standard throughout LLMux. Images are
    automatically converted to provider-specific formats when needed.
    
    The URL can be:
    - HTTP(S) URL: "https://example.com/image.jpg"
    - Data URI: "data:image/jpeg;base64,/9j/4AAQ..."
    
    Example:
        {
            "type": "image_url",
            "image_url": {"url": "https://example.com/cat.jpg"}
        }
    """
    type: Literal["image_url"]
    image_url: ImageUrlDetail


# Content can be a simple string or a list of content parts (text + images)
ContentPart = Union[TextContent, ImageContent]
MessageContent = Union[str, List[ContentPart]]


# =============================================================================
# Tool Calling Type Definitions
# =============================================================================
# These types define the structure for function calling / tool use.
# We use OpenAI's format as the standard, converting for other providers.

class FunctionParameters(TypedDict, total=False):
    """
    JSON Schema for function parameters.
    
    This follows JSON Schema specification for defining the expected
    structure of arguments passed to a tool/function.
    
    Example:
        {
            "type": "object",
            "properties": {
                "location": {"type": "string", "description": "City name"},
                "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}
            },
            "required": ["location"]
        }
    """
    type: Literal["object"]
    properties: Dict[str, Any]
    required: List[str]


class FunctionDefinition(TypedDict, total=False):
    """
    Function definition for tools.
    
    Describes what a tool/function does and what parameters it accepts.
    
    Example:
        {
            "name": "get_weather",
            "description": "Get the current weather for a location",
            "parameters": {...}
        }
    """
    name: str
    description: str
    parameters: FunctionParameters


class Tool(TypedDict):
    """
    Tool definition in OpenAI format.
    
    This is the standard format used throughout LLMux. Tools are
    automatically converted to Claude and Gemini formats when needed.
    
    Example:
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get weather for a location",
                "parameters": {...}
            }
        }
    """
    type: Literal["function"]
    function: FunctionDefinition


class ToolCall(TypedDict, total=False):
    """
    Tool call from an LLM response.
    
    When an LLM decides to use a tool, it returns one or more tool calls
    with this structure. You then execute the tool and send the result back.
    
    Example:
        {
            "id": "call_abc123",
            "name": "get_weather",
            "arguments": {"location": "Paris", "unit": "celsius"}
        }
    """
    id: str
    name: str
    arguments: Dict[str, Any]  # Parsed JSON arguments


class ToolResult(TypedDict):
    """
    Tool result to send back to the LLM.
    
    After executing a tool, send the result back to the LLM using this format.
    The tool_call_id must match the id from the original ToolCall.
    
    Example:
        {
            "tool_call_id": "call_abc123",
            "content": "The weather in Paris is sunny, 22°C"
        }
    """
    tool_call_id: str
    content: str


# =============================================================================
# Message Type (depends on ToolCall)
# =============================================================================

class Message(TypedDict, total=False):
    """
    Chat message with optional multimodal content and tool support.
    
    This is the main message type used throughout LLMux. It supports:
    - Text messages (content as string)
    - Multimodal messages (content as list of text/image parts)
    - Tool results (role="tool" with tool_call_id)
    - Assistant messages with tool calls
    
    Roles:
    - "system": System prompt / instructions
    - "user": User message
    - "assistant": Model response
    - "tool": Tool execution result
    
    Examples:
        # Simple text message
        {"role": "user", "content": "Hello!"}
        
        # Multimodal message with image
        {"role": "user", "content": [
            {"type": "text", "text": "What's in this image?"},
            {"type": "image_url", "image_url": {"url": "..."}}
        ]}
        
        # Tool result
        {"role": "tool", "tool_call_id": "call_123", "content": "Result..."}
    """
    role: Literal["system", "user", "assistant", "tool"]
    content: MessageContent
    tool_call_id: str  # For tool result messages
    tool_calls: List[ToolCall]  # For assistant messages with tool calls


class UnifiedChatClient:
    """
    Unified client for interacting with multiple LLM providers.
    
    This class provides a single interface for OpenAI, Claude (Anthropic),
    Gemini (Google), and DeepSeek. It handles:
    
    - Message format conversion between providers
    - Image/multimodal content handling
    - Tool calling with automatic format conversion
    - Streaming and non-streaming responses
    - Token usage tracking
    
    Provider Configuration:
    ----------------------
    API keys are loaded from environment variables (via .env file):
    - OPENAI_API_KEY: For OpenAI models
    - ANTHROPIC_API_KEY: For Claude models
    - GOOGLE_API_KEY: For Gemini models
    - DEEPSEEK_API_KEY: For DeepSeek models
    
    You can also pass API keys directly to the constructor.
    
    Example Usage:
    -------------
    ```python
    # Create client (loads API keys from .env)
    client = UnifiedChatClient()
    
    # Non-streaming chat
    response = await client.chat(
        provider="openai",
        model="gpt-4o",
        messages=[{"role": "user", "content": "Hello!"}],
        temperature=0.7
    )
    print(response["text"])
    
    # Streaming chat
    async for event in client.astream(
        provider="claude",
        model="claude-sonnet-4-20250514",
        messages=messages
    ):
        if event["type"] == "token":
            print(event["text"], end="")
    
    # With tools
    response = await client.chat_with_tools(
        provider="openai",
        model="gpt-4o",
        messages=messages,
        tools=my_tools,
        tool_handlers=my_handlers
    )
    ```
    """
    
    def __init__(
        self,
        openai_api_key: Optional[str] = dotenv.get_key(".env", "OPENAI_API_KEY"),
        anthropic_api_key: Optional[str] = dotenv.get_key(".env", "ANTHROPIC_API_KEY"),
        gemini_api_key: Optional[str] = dotenv.get_key(".env", "GOOGLE_API_KEY"),
        deepseek_api_key: Optional[str] = dotenv.get_key(".env", "DEEPSEEK_API_KEY"),
    ):
        """
        Initialize the UnifiedChatClient with API keys.
        
        Args:
            openai_api_key: OpenAI API key (defaults to OPENAI_API_KEY env var)
            anthropic_api_key: Anthropic API key (defaults to ANTHROPIC_API_KEY env var)
            gemini_api_key: Google API key (defaults to GOOGLE_API_KEY env var)
            deepseek_api_key: DeepSeek API key (defaults to DEEPSEEK_API_KEY env var)
            
        Note:
            API keys are loaded from .env file by default. You only need to pass
            them explicitly if you want to override the environment variables.
        """
        # Initialize provider clients only if API keys are available
        # This allows using the client with only some providers configured
        
        # OpenAI client - used for GPT models
        self.openai = AsyncOpenAI(api_key=openai_api_key) if openai_api_key else None
        
        # DeepSeek client - uses OpenAI-compatible API with different base URL
        self.deepseek = AsyncOpenAI(
            api_key=deepseek_api_key,
            base_url="https://api.deepseek.com"
        ) if deepseek_api_key else None
        
        # Claude/Anthropic client
        self.claude = AsyncAnthropic(api_key=anthropic_api_key) if anthropic_api_key else None
        
        # Gemini client - uses google.generativeai module
        if gemini_api_key:
            genai.configure(api_key=gemini_api_key)
            self.gemini = genai
        else:
            self.gemini = None

    # ==========================================================================
    # Image Helpers (Static/Class Methods)
    # ==========================================================================
    # These methods help create multimodal messages with images.
    # Images can come from URLs, local files, or base64 data.
    
    @staticmethod
    def encode_image_file(image_path: Union[str, Path]) -> tuple[str, str]:
        """
        Encode a local image file to base64.
        
        This method reads an image file from disk and encodes it as base64,
        which is the format required by most LLM providers for inline images.
        
        Args:
            image_path: Path to the image file (string or Path object)
            
        Returns:
            Tuple of (base64_data, mime_type):
            - base64_data: The image encoded as a base64 string
            - mime_type: Detected MIME type (e.g., "image/jpeg")
            
        Raises:
            FileNotFoundError: If the image file doesn't exist
            
        Example:
            b64_data, mime_type = client.encode_image_file("./photo.jpg")
            # b64_data: "/9j/4AAQ..."
            # mime_type: "image/jpeg"
        """
        path = Path(image_path)
        if not path.exists():
            raise FileNotFoundError(f"Image file not found: {image_path}")
        
        # Map file extensions to MIME types
        mime_types = {
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".png": "image/png",
            ".gif": "image/gif",
            ".webp": "image/webp",
            ".bmp": "image/bmp",
        }
        mime_type = mime_types.get(path.suffix.lower(), "image/jpeg")
        
        # Read and encode the file
        with open(path, "rb") as f:
            b64_data = base64.b64encode(f.read()).decode("utf-8")
        
        return b64_data, mime_type

    @staticmethod
    async def encode_image_url(url: str) -> tuple[str, str]:
        """
        Download an image from a URL and encode it to base64.
        
        This is useful when you want to ensure the image is embedded in the
        request rather than requiring the LLM provider to fetch it. Some
        providers have trouble fetching from certain URLs.
        
        Args:
            url: HTTP(S) URL of the image
            
        Returns:
            Tuple of (base64_data, mime_type)
            
        Example:
            b64_data, mime_type = await client.encode_image_url(
                "https://example.com/cat.jpg"
            )
        """
        # Use a browser-like User-Agent to avoid being blocked
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        }
        
        async with httpx.AsyncClient(timeout=30.0, headers=headers, follow_redirects=True) as http_client:
            response = await http_client.get(url)
            response.raise_for_status()
            
            # Extract MIME type from Content-Type header
            content_type = response.headers.get("content-type", "image/jpeg")
            mime_type = content_type.split(";")[0].strip()
            
            # Encode the image data
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
        
        This is the main helper for adding images to messages. It automatically
        detects the source type and creates the appropriate content structure.
        
        Source Types (auto-detected):
        - Data URI: "data:image/jpeg;base64,..."
        - HTTP(S) URL: "https://example.com/image.jpg"
        - Local file path: "./photo.jpg" or "/path/to/image.png"
        - Raw base64: Requires mime_type parameter
        
        Args:
            source: Image source (URL, file path, data URI, or base64)
            mime_type: MIME type (required for raw base64 data)
            detail: Image detail level for OpenAI ("auto", "low", "high")
            
        Returns:
            ImageContent dict ready to use in a message
            
        Examples:
            # From URL
            img = client.create_image_content("https://example.com/cat.jpg")
            
            # From local file (auto-encoded to base64)
            img = client.create_image_content("./photo.jpg")
            
            # From base64 with mime type
            img = client.create_image_content(b64_data, mime_type="image/png")
            
            # With detail level (OpenAI only)
            img = client.create_image_content(url, detail="high")
        """
        # Already a data URI - use as-is
        if source.startswith("data:"):
            url = source
        # HTTP(S) URL - keep as-is (will be converted to base64 for some providers)
        elif source.startswith(("http://", "https://")):
            url = source
        # Raw base64 string (if mime_type provided)
        elif mime_type:
            url = f"data:{mime_type};base64,{source}"
        # Local file path - check if it exists and encode
        elif len(source) < 260 and Path(source).exists():
            b64_data, detected_mime = UnifiedChatClient.encode_image_file(source)
            url = f"data:{detected_mime};base64,{b64_data}"
        else:
            raise ValueError(
                f"Cannot determine image source type for: {source[:50]}... "
                "Provide mime_type for raw base64 data."
            )
        
        # Build the image_url structure
        image_url: ImageUrlDetail = {"url": url}
        if detail:
            image_url["detail"] = detail
            
        return {"type": "image_url", "image_url": image_url}

    @staticmethod
    def create_text_content(text: str) -> TextContent:
        """
        Create a text content part for multimodal messages.
        
        This is used when building messages with mixed content (text + images).
        For text-only messages, you can just use a string directly.
        
        Args:
            text: The text content
            
        Returns:
            TextContent dict
            
        Example:
            content = [
                client.create_text_content("What's in this image?"),
                client.create_image_content("./photo.jpg")
            ]
        """
        return {"type": "text", "text": text}

    @classmethod
    def create_message(
        cls,
        role: Literal["system", "user", "assistant"],
        content: Union[str, List[Union[str, ContentPart]]],
    ) -> Message:
        """
        Create a message with text and/or images.
        
        This is the recommended way to create multimodal messages. It handles
        both simple text and complex multimodal content.
        
        Args:
            role: Message role ("system", "user", or "assistant")
            content: Either:
                - A string for text-only messages
                - A list of content parts (text strings and image dicts)
                
        Returns:
            Message dict ready to use in the messages array
            
        Examples:
            # Text-only message
            msg = client.create_message("user", "Hello!")
            
            # Multimodal message with text and image
            msg = client.create_message("user", [
                "What's in this image?",
                client.create_image_content("./photo.jpg"),
            ])
            
            # Multiple images
            msg = client.create_message("user", [
                "Compare these images:",
                client.create_image_content("./cat.jpg"),
                client.create_image_content("./dog.jpg"),
            ])
        """
        # Simple text content - no transformation needed
        if isinstance(content, str):
            return {"role": role, "content": content}
        
        # List content - normalize strings to TextContent dicts
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
    # These methods help create and parse tool definitions and tool call results.
    # Tools allow LLMs to perform actions like calling APIs, querying databases,
    # or executing code. All tools use OpenAI's format as the standard.
    #
    # Tool Calling Flow:
    # 1. Define tools with create_tool() - describes what functions the LLM can call
    # 2. Send messages with tools to chat() - LLM may request tool calls
    # 3. Execute the requested tool calls with your handler functions
    # 4. Create tool results with create_tool_result() - tells LLM the results
    # 5. Continue conversation with results - LLM formulates final response
    
    @staticmethod
    def create_tool(
        name: str,
        description: str,
        parameters: Dict[str, Any],
        required: Optional[List[str]] = None,
    ) -> Tool:
        """
        Create a tool definition for function calling.
        
        Tools describe functions that the LLM can call. The LLM doesn't execute
        the function itself - it returns a request to call it with specific
        arguments, and your code executes the function.
        
        Args:
            name: Function name (use lowercase_with_underscores by convention)
            description: Clear description of what the function does and when
                        to use it. This helps the LLM decide when to call it.
            parameters: JSON Schema for the function parameters. Each key is
                       a parameter name, value is its schema (type, description).
            required: List of required parameter names. Omitted = all optional.
            
        Returns:
            Tool definition dict in OpenAI format (works with all providers)
            
        Example:
            >>> tool = client.create_tool(
            ...     name="get_weather",
            ...     description="Get the current weather for a location",
            ...     parameters={
            ...         "location": {
            ...             "type": "string",
            ...             "description": "City name, e.g., 'Paris, France'"
            ...         },
            ...         "unit": {
            ...             "type": "string",
            ...             "enum": ["celsius", "fahrenheit"],
            ...             "description": "Temperature unit (default: celsius)"
            ...         }
            ...     },
            ...     required=["location"]
            ... )
        
        Note:
            The parameters follow JSON Schema specification. Common types:
            - "string": Text values
            - "number": Numeric values (int or float)
            - "boolean": true/false
            - "array": Lists (use "items" to specify element type)
            - "object": Nested objects
            
            Use "enum" to restrict values to a specific set.
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
        
        After executing a tool call, you need to tell the LLM what the result
        was. This creates the message with the proper format.
        
        Args:
            tool_call_id: The ID from the tool call (tc["id"]). This links
                         the result to the specific call request.
            content: The result of executing the tool (must be a string).
                    For structured data, JSON-encode it.
            
        Returns:
            Message with role "tool" ready to add to messages array
            
        Example:
            >>> # After LLM requests: {"name": "get_weather", "id": "call_123", ...}
            >>> weather_data = {"temp": 22, "condition": "sunny"}
            >>> result_msg = client.create_tool_result(
            ...     tool_call_id="call_123",
            ...     content=json.dumps(weather_data)
            ... )
            >>> messages.append(result_msg)
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
        Create an assistant message with tool calls for conversation history.
        
        When manually managing tool calling (auto_execute=False), you need to
        include the assistant's tool call requests in the message history
        before adding tool results. This creates that message.
        
        Args:
            content: Text content from the assistant (often empty for tool calls)
            tool_calls: List of tool calls from response["tool_calls"]
            
        Returns:
            Message with role "assistant" containing the tool_calls
            
        Example:
            >>> response = await client.chat(..., tools=tools)
            >>> if response.get("tool_calls"):
            ...     # Save the assistant's request
            ...     messages.append(client.create_assistant_message_with_tool_calls(
            ...         response["text"],
            ...         response["tool_calls"]
            ...     ))
            ...     # Execute tools and add results
            ...     for tc in response["tool_calls"]:
            ...         result = execute_my_tool(tc["name"], tc["arguments"])
            ...         messages.append(client.create_tool_result(tc["id"], result))
            ...     # Continue conversation
            ...     response = await client.chat(..., messages=messages)
        """
        return {
            "role": "assistant",
            "content": content,
            "tool_calls": tool_calls,
        }

    # --------------------------------------------------------------------------
    # Internal Tool Parsing Methods
    # --------------------------------------------------------------------------
    # These parse tool calls from provider-specific response formats into
    # our standard ToolCall format.
    
    @staticmethod
    def _parse_tool_calls_openai(choice) -> List[ToolCall]:
        """
        Parse tool calls from OpenAI/DeepSeek response.
        
        OpenAI returns tool calls as objects with function.name and 
        function.arguments (JSON string). We parse the arguments and
        return a standardized format.
        """
        import json
        tool_calls = []
        if hasattr(choice.message, "tool_calls") and choice.message.tool_calls:
            for tc in choice.message.tool_calls:
                try:
                    # Arguments come as JSON string - parse to dict
                    args = json.loads(tc.function.arguments)
                except json.JSONDecodeError:
                    # If parsing fails, preserve the raw string
                    args = {"_raw": tc.function.arguments}
                tool_calls.append({
                    "id": tc.id,
                    "name": tc.function.name,
                    "arguments": args,
                })
        return tool_calls

    @staticmethod
    def _parse_tool_calls_claude(response) -> List[ToolCall]:
        """
        Parse tool calls from Claude response.
        
        Claude uses "tool_use" content blocks. Unlike OpenAI, the arguments
        are already parsed as a dict, not a JSON string.
        """
        tool_calls = []
        for block in response.content:
            if getattr(block, "type", None) == "tool_use":
                tool_calls.append({
                    "id": block.id,
                    "name": block.name,
                    "arguments": block.input,  # Already a dict
                })
        return tool_calls

    # --------------------------------------------------------------------------
    # Internal Tool Format Conversion
    # --------------------------------------------------------------------------
    # Convert from our standard OpenAI format to provider-specific formats.
    
    @staticmethod
    def _convert_tools_for_claude(tools: List[Tool]) -> List[Dict[str, Any]]:
        """
        Convert OpenAI-format tools to Claude format.
        
        Claude format:
        {
            "name": "...",
            "description": "...",
            "input_schema": { JSON Schema }  # Note: "input_schema" not "parameters"
        }
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
    def _convert_tools_for_gemini(tools: List[Tool]) -> List[Dict[str, Any]]:
        """
        Convert OpenAI-format tools to Gemini format.
        
        Gemini format wraps tools in a "function_declarations" array:
        [{"function_declarations": [
            {"name": "...", "description": "...", "parameters": {...}}
        ]}]
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

    # ==========================================================================
    # Message Conversion for Each Provider
    # ==========================================================================
    # Each provider has a different format for messages, especially for:
    # - System messages (some treat as separate parameter, others as regular message)
    # - Multimodal content (different image format structures)
    # - Tool calls and results (different response/request formats)
    #
    # These methods convert our standard OpenAI-style format to each provider's
    # specific requirements. This is where the "unified" magic happens.
    
    @staticmethod
    def _is_multimodal_message(message: Message) -> bool:
        """Check if a message contains multimodal content (images)."""
        return isinstance(message.get("content"), list)

    async def _convert_messages_for_openai(
        self,
        messages: List[Message],
    ) -> List[Dict[str, Any]]:
        """
        Convert messages to OpenAI format.
        
        OpenAI's format is our "standard" format, so this is mostly pass-through
        with some transformations for reliability:
        
        1. Tool result messages: Ensures proper structure with tool_call_id
        2. Assistant messages with tool_calls: Converts arguments back to JSON strings
        3. Image URLs: Downloads and converts to base64 for reliability
           (OpenAI's servers sometimes fail to fetch from certain URLs)
        
        OpenAI natively supports:
        - Text content: {"type": "text", "text": "..."}
        - Image URLs: {"type": "image_url", "image_url": {"url": "..."}}
        - Base64 images: {"type": "image_url", "image_url": {"url": "data:..."}}
        - Tool calls and tool results
        """
        converted = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            
            # Handle tool result messages - pass through with correct structure
            if role == "tool":
                converted.append({
                    "role": "tool",
                    "tool_call_id": msg.get("tool_call_id", ""),
                    "content": content if isinstance(content, str) else "",
                })
                continue
            
            # Handle assistant messages with tool_calls
            # Need to convert arguments dict back to JSON string
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
                                # OpenAI expects arguments as JSON string, not dict
                                "arguments": json.dumps(tc["arguments"]),
                            },
                        }
                        for tc in msg["tool_calls"]
                    ],
                }
                converted.append(assistant_msg)
                continue
            
            # Handle simple text content - no transformation needed
            if isinstance(content, str):
                converted.append({"role": role, "content": content})
            else:
                # Process multimodal content (text + images)
                openai_content = []
                for part in content:
                    if part.get("type") == "text":
                        openai_content.append(part)
                    elif part.get("type") == "image_url":
                        url = part.get("image_url", {}).get("url", "")
                        detail = part.get("image_url", {}).get("detail")
                        
                        # Convert HTTP URLs to base64 for reliability
                        # (OpenAI may fail to fetch from some URLs)
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
        
        Key differences from OpenAI:
        1. System message is passed as a separate parameter, not in messages array
        2. Image format is different - uses "source" object with base64 data
        3. Claude requires base64 for all images (doesn't fetch URLs)
        
        Claude image format:
        {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": "image/jpeg",
                "data": "<base64 string without data: prefix>"
            }
        }
        
        Returns:
            Tuple of (system_text, converted_messages):
            - system_text: Combined system messages as single string (or None)
            - converted_messages: Messages in Claude format (without system)
        """
        system_parts = []
        converted = []
        
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            
            # Collect system messages separately - Claude takes system as parameter
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
                # Convert multimodal content to Claude format
                claude_content = []
                for part in content:
                    if part.get("type") == "text":
                        claude_content.append({
                            "type": "text",
                            "text": part.get("text", "")
                        })
                    elif part.get("type") == "image_url":
                        url = part.get("image_url", {}).get("url", "")
                        # Claude requires base64 for all images
                        b64_data, media_type = await self._resolve_image_to_base64(url)
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
        
        # Combine all system messages into one string
        system_text = "\n\n".join(system_parts) if system_parts else None
        return system_text, converted

    async def _convert_messages_for_gemini(
        self,
        messages: List[Message],
    ) -> tuple[Optional[str], List[Dict[str, Any]]]:
        """
        Convert messages to Gemini format.
        
        Key differences from OpenAI:
        1. System instruction is passed to GenerativeModel constructor, not in contents
        2. Role names differ: "assistant" → "model"
        3. Image format uses "inline_data" with mime_type and data fields
        4. Content structure uses "parts" array
        
        Gemini format:
        - system_instruction: str (passed separately to GenerativeModel)
        - contents: [
            {
                "role": "user"|"model",
                "parts": [
                    {"text": "..."},
                    {"inline_data": {"mime_type": "...", "data": "..."}}
                ]
            }
          ]
        
        Returns:
            Tuple of (system_instruction, contents):
            - system_instruction: System message text (or None)
            - contents: Messages in Gemini format
        """
        system_instruction = None
        contents = []
        
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            
            # Handle system messages - Gemini takes this as model parameter
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
                        # Gemini requires inline base64 data
                        b64_data, mime_type = await self._resolve_image_to_base64(url)
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
    # Helper Methods
    # ==========================================================================

    async def _execute_tool_call(
        self,
        tool_call: Dict[str, Any],
        tool_handlers: Dict[str, Callable],
        mcp_executor: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """
        Execute a single tool call and return the result message.
        
        Args:
            tool_call: Dict with 'id', 'name', 'arguments' keys
            tool_handlers: Dict mapping tool names to handler functions
            mcp_executor: Optional MCP executor for MCP tools
            
        Returns:
            Dict with 'role': 'tool', 'tool_call_id', 'content' keys
        """
        tool_name = tool_call.get("name", "")
        arguments = tool_call.get("arguments", {})
        tool_id = tool_call.get("id", "")
        
        try:
            # Check native handlers first, then MCP
            if tool_name in tool_handlers:
                result = tool_handlers[tool_name](arguments)
                if asyncio.iscoroutine(result):
                    result = await result
            elif mcp_executor and tool_name in mcp_executor.tool_names:
                result = await mcp_executor.call_tool(tool_name, arguments)
            else:
                result = f"Error: No handler for tool '{tool_name}'"
            
            return {
                "role": "tool",
                "tool_call_id": tool_id,
                "content": str(result),
            }
        except Exception as e:
            return {
                "role": "tool",
                "tool_call_id": tool_id,
                "content": f"Error executing tool '{tool_name}': {str(e)}",
            }

    @staticmethod
    def _normalize_usage(
        provider: str,
        *,
        input_tokens: Optional[int],
        output_tokens: Optional[int],
        total_tokens: Optional[int],
        raw: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Normalize token usage information across providers.
        
        Each provider reports token usage differently. This creates a
        consistent format while preserving the raw provider-specific data.
        
        Returns:
            {
                "input_tokens": int | None,    # Tokens in the prompt
                "output_tokens": int | None,   # Tokens in the response
                "total_tokens": int | None,    # Sum of input + output
                "raw": {                       # Provider-specific raw data
                    "provider": "...",
                    ... original fields ...
                }
            }
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

    # ==========================================================================
    # Provider-Specific Chat Methods (Non-Streaming)
    # ==========================================================================
    # These are the internal methods that make actual API calls to each provider.
    # They:
    # 1. Convert messages to provider-specific format
    # 2. Make the API call with appropriate parameters
    # 3. Parse the response (including tool calls if present)
    # 4. Normalize the response to our standard format
    #
    # Users don't call these directly - use chat() instead.
    
    async def _openai_compatible_chat(
        self,
        client: AsyncOpenAI,
        provider_name: str,
        model: str,
        messages: List[Message],
        **opts,
    ) -> Dict[str, Any]:
        """
        Shared implementation for OpenAI-compatible APIs (OpenAI, DeepSeek).
        
        Both OpenAI and DeepSeek use the same API format, so this method
        handles both with the provider name as a parameter.
        """
        if not client:
            raise RuntimeError(f"{provider_name.title()} client not configured")

        # Convert messages (handles multimodal content)
        converted_messages = await self._convert_messages_for_openai(messages)

        # Build request kwargs
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

        # Make API call with latency tracking
        start = time.perf_counter()
        resp = await client.chat.completions.create(**kwargs)
        latency_ms = (time.perf_counter() - start) * 1000.0

        choice = resp.choices[0]
        text = choice.message.content or ""
        
        # Parse tool calls if present
        tool_calls = self._parse_tool_calls_openai(choice)

        # Normalize usage statistics
        usage = None
        if resp.usage:
            usage = self._normalize_usage(
                provider_name,
                input_tokens=resp.usage.prompt_tokens,
                output_tokens=resp.usage.completion_tokens,
                total_tokens=resp.usage.total_tokens,
                raw={
                    "prompt_tokens": resp.usage.prompt_tokens,
                    "completion_tokens": resp.usage.completion_tokens,
                    "total_tokens": resp.usage.total_tokens,
                },
            )

        # Build standardized response
        result = {
            "provider": provider_name,
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

    async def _openai_chat(
        self,
        model: str,
        messages: List[Message],
        **opts,
    ) -> Dict[str, Any]:
        """Make a non-streaming chat request to OpenAI."""
        return await self._openai_compatible_chat(self.openai, "openai", model, messages, **opts)

    async def _deepseek_chat(
        self,
        model: str,
        messages: List[Message],
        **opts,
    ) -> Dict[str, Any]:
        """Make a non-streaming chat request to DeepSeek."""
        return await self._openai_compatible_chat(self.deepseek, "deepseek", model, messages, **opts)

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
    async def _openai_compatible_stream(
        self,
        client: AsyncOpenAI,
        provider_name: str,
        model: str,
        messages: List[Message],
        **opts,
    ) -> AsyncIterator[Dict[str, Any]]:
        """
        Shared streaming implementation for OpenAI-compatible APIs (OpenAI, DeepSeek).

        Yields:
            {"type": "token", "provider": "...", "text": "..."}
            ...
            {"type": "done", "provider": "...", "text": full_text, "meta": {...}}
        """
        if not client:
            raise RuntimeError(f"{provider_name.title()} client not configured")

        # Convert messages (handles multimodal content)
        converted_messages = await self._convert_messages_for_openai(messages)

        temperature = opts.get("temperature", 0.7)
        max_tokens = opts.get("max_tokens", 1024)
        top_p = opts.get("top_p", None)

        start = time.perf_counter()
        stream = await client.chat.completions.create(
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
                    "provider": provider_name,
                    "text": piece,
                }

        latency_ms = (time.perf_counter() - start) * 1000.0

        # Extract usage from final chunk if available
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

        usage = self._normalize_usage(
            provider_name,
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
            "provider": provider_name,
            "text": full_text,
            "meta": meta,
        }

    async def _openai_stream(
        self,
        model: str,
        messages: List[Message],
        **opts,
    ) -> AsyncIterator[Dict[str, Any]]:
        """Streaming for OpenAI."""
        async for event in self._openai_compatible_stream(self.openai, "openai", model, messages, **opts):
            yield event

    async def _deepseek_stream(
        self,
        model: str,
        messages: List[Message],
        **opts,
    ) -> AsyncIterator[Dict[str, Any]]:
        """Streaming for DeepSeek."""
        async for event in self._openai_compatible_stream(self.deepseek, "deepseek", model, messages, **opts):
            yield event

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

    # ==========================================================================
    # Unified Public Entrypoints
    # ==========================================================================
    # These are the main methods users should call. They provide a unified
    # interface that works identically across all providers.
    
    async def chat(
        self,
        provider: Provider,
        model: str,
        messages: List[Message],
        **opts,
    ) -> Dict[str, Any]:
        """
        Make a non-streaming chat request to any provider.
        
        This is the main method for simple chat completions. It automatically
        handles message format conversion, image encoding, and response
        normalization for each provider.
        
        Args:
            provider: Which LLM provider to use ("openai", "claude", "gemini", "deepseek")
            model: Model name (e.g., "gpt-4o", "claude-3-5-sonnet-20241022", "gemini-1.5-flash")
            messages: Conversation messages in OpenAI format
            **opts: Provider-specific options:
                - temperature (float): Randomness, 0.0-1.0 (default: 0.7)
                - max_tokens (int): Max response tokens (default: 1024)
                - top_p (float): Nucleus sampling threshold
                - tools (List[Tool]): Tool definitions for function calling
                - tool_choice (str): "auto", "none", "required", or specific tool name
                - stop (List[str]): Stop sequences
                - seed (int): Random seed for reproducibility (OpenAI/DeepSeek)
                
        Returns:
            {
                "provider": "openai" | "claude" | "gemini" | "deepseek",
                "text": "<assistant reply>",
                "tool_calls": [...] | None,  # If tools were used
                "meta": {
                    "model": "<actual model name>",
                    "usage": {
                        "input_tokens": int | None,
                        "output_tokens": int | None,
                        "total_tokens": int | None,
                        "raw": { ... provider-specific ... }
                    },
                    "latency_ms": float,
                    "finish_reason": "..." | None,
                },
            }
            
        Example:
            >>> response = await client.chat(
            ...     provider="openai",
            ...     model="gpt-4o",
            ...     messages=[
            ...         {"role": "system", "content": "You are a helpful assistant."},
            ...         {"role": "user", "content": "Hello!"}
            ...     ],
            ...     temperature=0.5,
            ...     max_tokens=500,
            ... )
            >>> print(response["text"])
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
        Make a streaming chat request to any provider.
        
        Streaming allows you to display the response as it's generated,
        providing a better user experience for long responses.
        
        Args:
            provider: Which LLM provider to use ("openai", "claude", "gemini", "deepseek")
            model: Model name
            messages: Conversation messages in OpenAI format
            **opts: Provider-specific options (same as chat())
            
        Yields:
            Token events as they arrive:
            {"type": "token", "provider": "...", "text": "partial text..."}
            
            Final event with complete response and metadata:
            {"type": "done", "provider": "...", "text": "full response", "meta": {...}}
            
        Example:
            >>> async for event in client.astream(
            ...     provider="openai",
            ...     model="gpt-4o",
            ...     messages=[{"role": "user", "content": "Write a haiku"}],
            ... ):
            ...     if event["type"] == "token":
            ...         print(event["text"], end="", flush=True)  # Stream to console
            ...     elif event["type"] == "done":
            ...         print()  # Newline at end
            ...         print(f"Total tokens: {event['meta']['usage']['total_tokens']}")
            
        Note:
            - Tool calls are NOT supported in streaming mode
            - Use chat() with tools for function calling
        """
        # Route to provider-specific streaming implementation
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
        """
        Get list of available model names for a given provider.
        
        This queries the provider's API to get current model availability.
        Requires the appropriate API key to be set in environment variables.
        
        Args:
            provider: One of 'gemini', 'openai', 'deepseek', 'anthropic'
        
        Returns:
            List of model name strings
            
        Example:
            >>> models = UnifiedChatClient.get_models("openai")
            >>> print(models)
            ['gpt-4o', 'gpt-4o-mini', 'gpt-4-turbo', ...]
            
        Note:
            For Gemini, only models that support 'generateContent' are returned.
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
    # These methods provide high-level integration with the Model Context Protocol
    # (MCP). They allow combining native tool definitions with MCP tools from
    # connected servers, with automatic tool execution loop handling.
    #
    # The key advantage of these methods over manual tool handling:
    # 1. Automatic tool execution - no manual call/result loop
    # 2. Seamless MCP integration - MCP tools work like native tools
    # 3. Multi-iteration support - handles chains of tool calls
    # 4. Unified interface - same API for native and MCP tools
    
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
        
        This is the recommended method for function calling and MCP integration.
        It handles the entire tool call/result loop automatically, allowing the
        LLM to make multiple tool calls before returning the final response.
        
        How it works:
        1. Sends messages + tool definitions to the LLM
        2. If LLM requests tool calls, executes them automatically
        3. Sends results back to LLM
        4. Repeats until LLM provides a final response (no more tool calls)
        
        Args:
            provider: LLM provider ("openai", "claude", "gemini", "deepseek")
            model: Model name
            messages: Conversation messages in OpenAI format
            tools: Native tool definitions (created with create_tool())
            mcp_executor: MCPToolExecutor instance with connected MCP servers.
                         Tools from all connected servers are automatically available.
            tool_handlers: Dict mapping tool names to handler functions for native
                          tools. Functions receive (arguments: dict) and return str.
                          Can be sync or async functions.
            auto_execute: If True (default), automatically execute tool calls and
                         continue conversation. If False, return on first tool call
                         for manual handling.
            max_iterations: Safety limit on tool execution iterations (default: 10)
            **opts: Additional options passed to chat() (temperature, max_tokens, etc.)
            
        Returns:
            Final response dict with:
            - "provider": Provider name
            - "text": Final assistant response text
            - "tool_calls": List of tool calls (only if auto_execute=False)
            - "tool_history": List of all tool executions (name, arguments, result)
            - "meta": Model info, usage, latency
            - "warning": Set if max_iterations reached
            
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
            
            # Execute tool calls using shared helper
            tool_results = []
            for tc in tool_calls:
                result_msg = await self._execute_tool_call(tc, tool_handlers, mcp_executor)
                tool_results.append(result_msg)
                # Track history
                tool_history.append({
                    "tool": tc.get("name", ""),
                    "arguments": tc.get("arguments", {}),
                    "result": result_msg["content"],
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
        
        This combines streaming output with automatic tool execution, yielding
        events for both text tokens and tool execution progress.
        
        Event Flow:
        1. Token events as response streams in
        2. tool_call event when LLM requests tools
        3. tool_result event after tools execute
        4. Repeat until done event with final response
        
        Args:
            (Same as chat_with_tools)
            
        Yields:
            Token events (streamed response text):
            {"type": "token", "provider": "...", "text": "partial..."}
            
            Tool call events (when LLM requests tools):
            {"type": "tool_call", "provider": "...", "tool_calls": [{"name": "...", "arguments": {...}}]}
            
            Tool result events (after execution):
            {"type": "tool_result", "provider": "...", "results": [{"tool_call_id": "...", "content": "..."}]}
            
            Final done event:
            {"type": "done", "provider": "...", "text": "full response", "meta": {...}}
            
        Example:
            >>> async for event in client.astream_with_tools(
            ...     provider="openai",
            ...     model="gpt-4o",
            ...     messages=[{"role": "user", "content": "Get the weather in Paris"}],
            ...     tools=tools,
            ...     tool_handlers=handlers,
            ... ):
            ...     if event["type"] == "token":
            ...         print(event["text"], end="", flush=True)
            ...     elif event["type"] == "tool_call":
            ...         print(f"\\n[Calling tools: {[tc['name'] for tc in event['tool_calls']]}]")
            ...     elif event["type"] == "tool_result":
            ...         print(f"[Got {len(event['results'])} results]")
            ...     elif event["type"] == "done":
            ...         print(f"\\n[Done]")
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
        
        # Tool execution loop - continues until LLM stops requesting tools
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
            
            # No tool calls or auto-execute disabled - we're done
            if not tool_calls or not auto_execute:
                if final_event:
                    yield final_event
                return
            
            # Yield tool call event so caller can track progress
            yield {
                "type": "tool_call",
                "provider": provider,
                "tool_calls": tool_calls,
            }
            
            # Execute tool calls using shared helper
            tool_results = []
            for tc in tool_calls:
                result_msg = await self._execute_tool_call(tc, tool_handlers, mcp_executor)
                tool_results.append(result_msg)
            
            # Yield tool results event so caller can track progress
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