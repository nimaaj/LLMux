from typing import Literal, List, Dict, Any, Union, TypedDict, Optional

# =============================================================================
# Type Definitions
# =============================================================================

# Supported LLM providers
Provider = Literal["openai", "claude", "gemini", "deepseek"]


class TextContent(TypedDict, total=False):
    """
    Text content part for multimodal messages.
    """
    type: Literal["text"]
    text: str


class ImageUrlDetail(TypedDict, total=False):
    """
    Image URL specification with optional detail level.
    """
    url: str
    detail: Literal["auto", "low", "high"]  # OpenAI-specific


class ImageContent(TypedDict, total=False):
    """
    Image content part for multimodal messages (OpenAI format).
    """
    type: Literal["image_url"]
    image_url: ImageUrlDetail


# Content can be a simple string or a list of content parts (text + images)
ContentPart = Union[TextContent, ImageContent]
MessageContent = Union[str, List[ContentPart]]


# =============================================================================
# Tool Calling Type Definitions
# =============================================================================

class FunctionParameters(TypedDict, total=False):
    """
    JSON Schema for function parameters.
    """
    type: Literal["object"]
    properties: Dict[str, Any]
    required: List[str]


class FunctionDefinition(TypedDict, total=False):
    """
    Function definition for tools.
    """
    name: str
    description: str
    parameters: FunctionParameters


class Tool(TypedDict):
    """
    Tool definition in OpenAI format.
    """
    type: Literal["function"]
    function: FunctionDefinition


class ToolCall(TypedDict, total=False):
    """
    Tool call from an LLM response.
    """
    id: str
    name: str
    arguments: Dict[str, Any]  # Parsed JSON arguments


class ToolResult(TypedDict):
    """
    Tool result to send back to the LLM.
    """
    tool_call_id: str
    content: str


# =============================================================================
# Message Type (depends on ToolCall)
# =============================================================================

class Message(TypedDict, total=False):
    """
    Chat message with optional multimodal content and tool support.
    
    Roles:
    - "system": System prompt / instructions
    - "user": User message
    - "assistant": Model response
    - "tool": Tool execution result
    """
    role: Literal["system", "user", "assistant", "tool"]
    content: MessageContent
    tool_call_id: str  # For tool result messages
    tool_calls: List[ToolCall]  # For assistant messages with tool calls
