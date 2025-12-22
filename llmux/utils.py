
import base64
import json
import httpx
from pathlib import Path
from typing import Union, List, Optional, Dict, Any, Literal, Tuple

from .types import (
    Message, ContentPart, TextContent, ImageContent, Tool, 
    ToolCall, FunctionParameters, ToolResult, ImageUrlDetail
)

# =============================================================================
# Image Helpers
# =============================================================================

def encode_image_file(image_path: Union[str, Path]) -> Tuple[str, str]:
    """
    Encode a local image file to base64 for LLM usage.

    Reads the file from the given path, determines its MIME type based on extension,
    and returns a tuple of the base64-encoded data and the MIME type.

    Args:
        image_path (Union[str, Path]): Absolute path to the image file.

    Returns:
        Tuple[str, str]: A tuple containing:
            - b64_data (str): The base64-encoded string of the image content.
            - mime_type (str): The MIME type (e.g., 'image/png').

    Raises:
        FileNotFoundError: If the file does not exist.
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


async def encode_image_url(url: str) -> Tuple[str, str]:
    """
    Fetch an image from a URL and encode it to base64.

    This function downloads the image content using an async HTTP client,
    extracts the MIME type from the response headers, and encodes the content.

    Args:
        url (str): The publicly accessible URL of the image.

    Returns:
        Tuple[str, str]: A tuple containing:
            - b64_data (str): The base64-encoded string of the downloaded content.
            - mime_type (str): The MIME type from the Content-Type header.

    Raises:
        httpx.HTTPError: If the download fails (timeout, 404, etc.).
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


async def resolve_image_to_base64(url: str) -> Tuple[str, str]:
    """
    Resolve an arbitrary image reference (URL or data URI) to base64 data.

    This is a helper to handle both remote URLs and inline data URIs uniformly.

    Args:
        url (str): HTTP/HTTPS URL or Data URI (data:image/...).

    Returns:
        Tuple[str, str]: A tuple containing (base64_data, mime_type).
    """
    if url.startswith("data:"):
        # Parse data URI: data:[<mediatype>][;base64],<data>
        header, data = url.split(",", 1)
        mime_type = header.split(":")[1].split(";")[0]
        return data, mime_type
    else:
        # Download from URL
        return await encode_image_url(url)


def create_image_content(
    source: str,
    *,
    mime_type: Optional[str] = None,
    detail: Optional[Literal["auto", "low", "high"]] = None,
) -> ImageContent:
    """
    Create a standardized image content part for multimodal messages.

    This helper accepts various image sources and formats them into the
    expected structure for the Unified Chat Client.

    Args:
        source (str): Can be:
            - A local file path (e.g., "/path/to/image.png")
            - A remote URL (e.g., "https://example.com/image.jpg")
            - A data URI (e.g., "data:image/png;base64,...")
            - Raw base64 data (requires `mime_type` kwarg)
        mime_type (str, optional): Required if `source` is raw base64 data.
                                   Example: "image/png".
        detail (str, optional): Detail level for OpenAI vision ('auto', 'low', 'high').

    Returns:
        ImageContent: A dictionary adhering to the internal ImageContent type.

    Raises:
        ValueError: If the source type cannot be determined or requires explicit mime_type.
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
        b64_data, detected_mime = encode_image_file(source)
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


def create_text_content(text: str) -> TextContent:
    """
    Create a standardized simple text content part.

    Args:
        text (str): The text message content.

    Returns:
        TextContent: A dictionary {"type": "text", "text": text}.
    """
    return {"type": "text", "text": text}


def create_message(
    role: Literal["system", "user", "assistant"],
    content: Union[str, List[Union[str, ContentPart]]],
) -> Message:
    """
    Create a standardized Message object.

    Handles both simple string content and lists of content parts (multimodal).
    Automatically normalizes string elements within a list to TextContent objects.

    Args:
        role (str): The role of the message sender ('system', 'user', 'assistant').
        content (Union[str, List]): The content of the message.

    Returns:
        Message: A dictionary matching the Message type definition.
    """
    # Simple text content - no transformation needed
    if isinstance(content, str):
        return {"role": role, "content": content}
    
    # List content - normalize strings to TextContent dicts
    normalized: List[ContentPart] = []
    for item in content:
        if isinstance(item, str):
            normalized.append(create_text_content(item))
        else:
            normalized.append(item)
    
    return {"role": role, "content": normalized}


# =============================================================================
# Tool Calling Helpers
# =============================================================================

def create_tool(
    name: str,
    description: str,
    parameters: Dict[str, Any],
    required: Optional[List[str]] = None,
) -> Tool:
    """
    Create a standardized Tool definition for function calling.

    Formats the tool definition according to the OpenAI function calling schema,
    which is widely adopted by other providers.

    Args:
        name (str): The name of the function/tool to be called.
        description (str): A clear description of what the tool does.
        parameters (Dict): A JSON Schema dictionary defining the expected arguments.
        required (List[str], optional): A list of parameter names that are required.

    Returns:
        Tool: A dictionary representing the tool definition.
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


def create_tool_result(tool_call_id: str, content: str) -> Message:
    """
    Create a tool result message to send back to the LLM.

    This message marks the completion of a tool execution and provides
    the output to the model.

    Args:
        tool_call_id (str): The ID of the tool call this result corresponds to.
        content (str): The stringified result of the tool execution.

    Returns:
        Message: A message dictionary with role='tool'.
    """
    return {
        "role": "tool",
        "tool_call_id": tool_call_id,
        "content": content,
    }


def create_assistant_message_with_tool_calls(
    content: str,
    tool_calls: List[ToolCall],
) -> Message:
    """
    Create an assistant message that includes tool calls.

    This represents a model's request to execute one or more tools.

    Args:
        content (str): Optional text content accompanying the tool calls (can be empty).
        tool_calls (List[ToolCall]): List of tool call objects.

    Returns:
        Message: A message dictionary with role='assistant'.
    """
    return {
        "role": "assistant",
        "content": content,
        "tool_calls": tool_calls,
    }
