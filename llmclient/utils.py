
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
    Encode a local image file to base64.
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
    Download an image from a URL and encode it to base64.
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
    Resolve an image URL (HTTP or data URI) to base64 data and mime type.
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
    Create an image content part from various sources.
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
    Create a text content part for multimodal messages.
    """
    return {"type": "text", "text": text}


def create_message(
    role: Literal["system", "user", "assistant"],
    content: Union[str, List[Union[str, ContentPart]]],
) -> Message:
    """
    Create a message with text and/or images.
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
    Create a tool definition for function calling.
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
    Create an assistant message with tool calls for conversation history.
    """
    return {
        "role": "assistant",
        "content": content,
        "tool_calls": tool_calls,
    }
