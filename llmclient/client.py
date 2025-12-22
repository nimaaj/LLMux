
import asyncio
import dotenv
import os
from pathlib import Path
from typing import Optional, Dict, List, Any, Callable, AsyncIterator, Literal, Union

from openai import OpenAI
from anthropic import Anthropic
from google import genai as genai_client

from .types import Message, Tool, ToolCall, ContentPart, ImageContent, TextContent
from .utils import (
    create_message, create_image_content, create_text_content, 
    create_tool, create_tool_result, create_assistant_message_with_tool_calls,
    encode_image_file, encode_image_url
)
from .providers.base import BaseLLMProvider
from .providers.openai import OpenAIProvider
from .providers.anthropic import AnthropicProvider
from .providers.anthropic import AnthropicProvider
from .providers.gemini import GeminiProvider
from .providers.huggingface import HuggingFaceProvider

# Load environment variables
dotenv.load_dotenv()

class UnifiedChatClient:
    """
    Unified client for interacting with multiple LLM providers.
    
    This class provides a single interface for OpenAI, Claude (Anthropic),
    Gemini (Google), and DeepSeek.
    """
    
    def __init__(
        self,
        openai_api_key: Optional[str] = dotenv.get_key(".env", "OPENAI_API_KEY"),
        anthropic_api_key: Optional[str] = dotenv.get_key(".env", "ANTHROPIC_API_KEY"),
        gemini_api_key: Optional[str] = dotenv.get_key(".env", "GOOGLE_API_KEY"),
        deepseek_api_key: Optional[str] = dotenv.get_key(".env", "DEEPSEEK_API_KEY"),
        huggingface_api_key: Optional[str] = dotenv.get_key(".env", "HUGGINGFACE_API_KEY"),
    ):
        """
        Initialize the UnifiedChatClient with API keys.

        Args:
            openai_api_key: API key for OpenAI. Defaults to env var OPENAI_API_KEY.
            anthropic_api_key: API key for Anthropic (Claude). Defaults to env var ANTHROPIC_API_KEY.
            gemini_api_key: API key for Google Gemini. Defaults to env var GOOGLE_API_KEY.
            deepseek_api_key: API key for DeepSeek. Defaults to env var DEEPSEEK_API_KEY.
            huggingface_api_key: API key for Hugging Face. Defaults to env var HUGGINGFACE_API_KEY.
        """
        self.providers: Dict[str, BaseLLMProvider] = {}
        
        # Initialize providers based on available keys
        # We check each key independently so the client can work with a subset of providers
        if openai_api_key:
            self.providers["openai"] = OpenAIProvider(api_key=openai_api_key)
            
        if anthropic_api_key:
            self.providers["claude"] = AnthropicProvider(api_key=anthropic_api_key)
            
        if gemini_api_key:
            self.providers["gemini"] = GeminiProvider(api_key=gemini_api_key)
            
        if deepseek_api_key:
            # DeepSeek is OpenAI-compatible, so we reuse the OpenAIProvider
            # with a custom base_url
            self.providers["deepseek"] = OpenAIProvider(
                api_key=deepseek_api_key, 
                base_url="https://api.deepseek.com",
                provider_name="deepseek"
            )
            
        if huggingface_api_key:
            self.providers["huggingface"] = HuggingFaceProvider(api_key=huggingface_api_key)

        # Expose provider clients for direct access if needed
        # This maintains some backward compatibility with older versions of the library
        # where users might expect to access `client.openai` directly.
        self.openai = getattr(self.providers.get("openai"), "client", None)
        self.claude = getattr(self.providers.get("claude"), "client", None)
        self.gemini = getattr(self.providers.get("gemini"), "genai", None)
        self.deepseek = getattr(self.providers.get("deepseek"), "client", None)
        self.huggingface = getattr(self.providers.get("huggingface"), "client", None)

    # ==========================================================================
    # Image Helpers (Static/Class Methods) - Re-exported from utils
    # ==========================================================================
    
    @staticmethod
    def encode_image_file(image_path: Union[str, Path]) -> tuple[str, str]:
        return encode_image_file(image_path)

    @staticmethod
    async def encode_image_url(url: str) -> tuple[str, str]:
        return await encode_image_url(url)

    @staticmethod
    def create_image_content(
        source: str, 
        *, 
        mime_type: Optional[str] = None, 
        detail: Optional[Literal["auto", "low", "high"]] = None
    ) -> ImageContent:
        return create_image_content(source, mime_type=mime_type, detail=detail)

    @staticmethod
    def create_text_content(text: str) -> TextContent:
        return create_text_content(text)

    @classmethod
    def create_message(
        cls,
        role: Literal["system", "user", "assistant"],
        content: Union[str, List[Union[str, ContentPart]]],
    ) -> Message:
        return create_message(role, content)

    # ==========================================================================
    # Tool Calling Helpers - Re-exported from utils
    # ==========================================================================

    @staticmethod
    def create_tool(
        name: str,
        description: str,
        parameters: Dict[str, Any],
        required: Optional[List[str]] = None,
    ) -> Tool:
        return create_tool(name, description, parameters, required)

    @staticmethod
    def create_tool_result(tool_call_id: str, content: str) -> Message:
        return create_tool_result(tool_call_id, content)

    @staticmethod
    def create_assistant_message_with_tool_calls(
        content: str,
        tool_calls: List[ToolCall],
    ) -> Message:
        return create_assistant_message_with_tool_calls(content, tool_calls)

    # ==========================================================================
    # Unified Chat Methods
    # ==========================================================================

    async def list_models(self, provider: str) -> List[str]:
        """
        Get list of available models for a configured provider.
        
        This uses the configured credentials in the client instance.
        """
        # Normalize provider names
        provider = provider.lower()
        if provider == "anthropic":
             provider = "claude"
        elif provider == "google":
             provider = "gemini"

        if provider not in self.providers:
            raise ValueError(f"Provider '{provider}' not configured or not supported.")
        
        return await self.providers[provider].get_models()

    async def chat(
        self,
        provider: str,
        model: str,
        messages: List[Message],
        **opts,
    ) -> Dict[str, Any]:
        """
        Send a chat request to the specified provider.
        """
        if provider not in self.providers:
            # Try to handle case where user requests a provider that wasn't configured
            raise ValueError(f"Provider '{provider}' not configured or not supported.")
        
        return await self.providers[provider].chat(model, messages, **opts)

    async def astream(
        self,
        provider: str,
        model: str,
        messages: List[Message],
        **opts,
    ) -> AsyncIterator[Dict[str, Any]]:
        """
        Stream response from the specified provider.
        """
        if provider not in self.providers:
            raise ValueError(f"Provider '{provider}' not configured or not supported.")
        
        async for event in self.providers[provider].stream(model, messages, **opts):
            yield event

    async def chat_with_tools(
        self,
        provider: str,
        model: str,
        messages: List[Message],
        tools: Optional[List[Tool]] = None,
        mcp_executor: Optional[Any] = None,
        tool_handlers: Optional[Dict[str, Callable]] = None,
        auto_execute: bool = True,
        max_iterations: int = 10,
        **opts
    ) -> Dict[str, Any]:
        """
        Chat with automatic tool execution loop.
        """
        current_messages = list(messages)
        tool_handlers = tool_handlers or {}
        
        # Combine native tools and MCP tools (if provided)
        # Note: MCP tools are already in OpenAI format
        all_tools = list(tools) if tools else []
        if mcp_executor:
            all_tools.extend(mcp_executor.get_tools())
        
        opts["tools"] = all_tools
        tool_history = []
        
        for _ in range(max_iterations):
            # 1. Call LLM
            response = await self.chat(
                provider=provider,
                model=model,
                messages=current_messages,
                **opts
            )
            
            # 2. Check for tool calls
            tool_calls = response.get("tool_calls")
            if not tool_calls or not auto_execute:
                # No tools called or auto-execute disabled -> return final response
                if tool_history:
                    response["tool_history"] = tool_history
                return response
            
            # 3. Add assistant message to history
            current_messages.append(
                self.create_assistant_message_with_tool_calls(
                    response.get("text", ""), 
                    tool_calls
                )
            )
            
            # 4. Execute tools
            for tc in tool_calls:
                tool_name = tc.get("name", "")
                
                # Execute and capture result
                tool_result_msg = await self._execute_tool_call(
                    tc, tool_handlers, mcp_executor
                )
                
                # Add result to history
                current_messages.append(tool_result_msg)
                
                # Track history
                tool_history.append({
                    "tool": tool_name,
                    "arguments": tc.get("arguments"),
                    "result": tool_result_msg["content"]
                })
        
        # If max iterations reached
        return response

    async def _execute_tool_call(
        self,
        tool_call: Dict[str, Any],
        tool_handlers: Dict[str, Callable],
        mcp_executor: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """
        Execute a single tool call and return the result message.
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
            case "huggingface":
                client = OpenAI(
                    api_key=dotenv.get_key(".env", "HUGGINGFACE_API_KEY"), 
                    base_url="https://router.huggingface.co/v1/"
                )
                return [x.id for x in client.models.list().data]
            case "anthropic":
                client = Anthropic(api_key=dotenv.get_key(".env", "ANTHROPIC_API_KEY"))
                return [x.id for x in client.models.list().data]
            case _:
                raise ValueError(f"Unknown provider: {provider}. Use 'gemini', 'openai', 'deepseek', 'anthropic', or 'huggingface'")

