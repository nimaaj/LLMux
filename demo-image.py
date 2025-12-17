"""
Demo: Including images in context with UnifiedChatClient.

This demonstrates how to send images to vision-capable models using:
1. Local file paths
2. Image URLs
3. Base64-encoded images (inline)
4. Multiple images in one message

Supported providers for vision:
- OpenAI: gpt-4o, gpt-4o-mini, gpt-4-turbo, gpt-4-vision-preview
- Claude: claude-3-opus, claude-3-sonnet, claude-3-haiku, claude-3-5-sonnet
- Gemini: gemini-1.5-pro, gemini-1.5-flash, gemini-2.0-flash
- DeepSeek: deepseek-chat (vision support may vary)
"""
import asyncio
from pathlib import Path
from typing import List

from rich_llm_printer import RichPrinter,RichStreamPrinter
from llmclient import UnifiedChatClient, Message
from rich.console import Console
console = Console()

# =============================================================================
# Demo Functions
# =============================================================================

# Use more reliable image URLs (Unsplash provides stable URLs)
CAT_IMAGE_URL = "https://images.unsplash.com/photo-1514888286974-6c03e2ca1dba?w=400"
DOG_IMAGE_URL = "https://images.unsplash.com/photo-1587300003388-59208cc962cb?w=400"


async def demo_with_url():
    """Demo: Analyze an image from URL (OpenAI)."""
    print("\n" + "=" * 60)
    print("Demo: Image from URL (OpenAI)")
    print("=" * 60)

    client = UnifiedChatClient()
    printer = RichPrinter()

    # Using the unified create_message helper
    messages: List[Message] = [
        {"role": "system", "content": "You are a helpful assistant that can analyze images."},
        client.create_message("user", [
            "What do you see in this image? Describe it briefly.",
            client.create_image_content(CAT_IMAGE_URL),  # URL is handled automatically
        ]),
    ]

    resp = await client.chat(
        provider="openai",
        model="gpt-4o-mini",
        messages=messages,
        max_tokens=300,
    )
    printer.print_chat(resp)


async def demo_with_gemini():
    """Demo: Image analysis with Gemini (auto-converts URL to base64)."""
    print("\n" + "=" * 60)
    print("Demo: Image with Gemini")
    print("=" * 60)

    client = UnifiedChatClient()
    printer = RichPrinter()

    # Gemini doesn't support URLs directly, but our client auto-converts
    messages: List[Message] = [
        {"role": "system", "content": "You are a helpful assistant."},
        client.create_message("user", [
            "Describe what you see in this image in one sentence.",
            client.create_image_content(CAT_IMAGE_URL),
        ]),
    ]

    resp = await client.chat(
        provider="gemini",
        model="gemini-2.0-flash",
        messages=messages,
        max_tokens=200,
    )
    printer.print_chat(resp)


async def demo_with_claude():
    """Demo: Image analysis with Claude."""
    print("\n" + "=" * 60)
    print("Demo: Image with Claude")
    print("=" * 60)

    client = UnifiedChatClient()
    printer = RichPrinter()

    # Claude also auto-converts URLs to base64
    messages: List[Message] = [
        {"role": "system", "content": "You are a helpful assistant that analyzes images concisely."},
        client.create_message("user", [
            "What animal is in this image? Answer in one word.",
            client.create_image_content(CAT_IMAGE_URL),
        ]),
    ]

    resp = await client.chat(
        provider="claude",
        model="claude-sonnet-4-20250514",
        messages=messages,
        max_tokens=100,
    )
    printer.print_chat(resp)


async def demo_streaming_with_image():
    """Demo: Streaming response while analyzing an image."""
    print("\n" + "=" * 60)
    print("Demo: Streaming with Image (OpenAI)")
    print("=" * 60)

    client = UnifiedChatClient()
    printer = RichStreamPrinter()

    messages: List[Message] = [
        {"role": "system", "content": "You are a helpful assistant."},
        client.create_message("user", [
            "Describe this image in detail, including colors, composition, and mood.",
            client.create_image_content(CAT_IMAGE_URL),
        ]),
    ]

    stream = client.astream(
        provider="openai",
        model="gpt-4o-mini",
        messages=messages,
        max_tokens=500,
    )

    await printer.print_stream(stream)


async def demo_local_file():
    """Demo: Analyze a local image file."""
    print("\n" + "=" * 60)
    print("Demo: Local Image File")
    print("=" * 60)

    # Check for a sample image
    sample_paths = ["sample.jpg", "sample.png", "test.jpg", "test.png"]
    image_path = None
    for p in sample_paths:
        if Path(p).exists():
            image_path = p
            break

    if not image_path:
        print("No local image found. To test with a local file:")
        print("  1. Add an image file named 'sample.jpg' or 'sample.png'")
        print("  2. Run this demo again")
        print("\nSkipping local file demo...")
        return

    client = UnifiedChatClient()
    printer = RichPrinter()

    # create_image_content auto-detects local files
    messages: List[Message] = [
        {"role": "system", "content": "You are a helpful assistant."},
        client.create_message("user", [
            "What do you see in this image?",
            client.create_image_content(image_path),  # Local path handled automatically
        ]),
    ]

    resp = await client.chat(
        provider="openai",
        model="gpt-4o-mini",
        messages=messages,
        max_tokens=300,
    )
    printer.print_chat(resp)


async def demo_multiple_images():
    """Demo: Send multiple images in one message."""
    print("\n" + "=" * 60)
    print("Demo: Multiple Images")
    print("=" * 60)

    client = UnifiedChatClient()
    printer = RichPrinter()

    # Multiple images in a single message
    messages: List[Message] = [
        {"role": "system", "content": "You are a helpful assistant that can compare images."},
        client.create_message("user", [
            "Compare these two images. What animals are shown and how do they differ?",
            client.create_image_content(CAT_IMAGE_URL),
            client.create_image_content(DOG_IMAGE_URL),
        ]),
    ]

    resp = await client.chat(
        provider="openai",
        model="gpt-4o-mini",
        messages=messages,
        max_tokens=400,
    )
    printer.print_chat(resp)


async def demo_base64_direct():
    """Demo: Using base64 data directly."""
    print("\n" + "=" * 60)
    print("Demo: Direct Base64 Image")
    print("=" * 60)

    client = UnifiedChatClient()
    printer = RichPrinter()

    # Download and encode image manually (to demonstrate base64 usage)
    b64_data, mime_type = await client.encode_image_url(CAT_IMAGE_URL)
    
    print(f"Encoded image: {len(b64_data)} bytes base64, {mime_type}")

    # Use raw base64 with mime_type
    messages: List[Message] = [
        {"role": "system", "content": "You are a helpful assistant."},
        client.create_message("user", [
            "What color is this animal?",
            client.create_image_content(b64_data, mime_type=mime_type),
        ]),
    ]

    resp = await client.chat(
        provider="openai",
        model="gpt-4o-mini",
        messages=messages,
        max_tokens=100,
    )
    printer.print_chat(resp)


async def main():
    """Run all demos."""
    print("=" * 60)
    print("LLMux Image Demo - Unified API")
    print("=" * 60)
    print("\nThis demo shows how to include images in messages.")
    print("All providers use the same unified message format.")
    print("Make sure you have API keys configured in .env")

    # Run demos (comment out any you don't want to run)
    try:
        await demo_with_url()
    except Exception as e:
        print(f"OpenAI URL demo failed: {e}")

    try:
        await demo_with_gemini()
    except Exception as e:
        print(f"Gemini demo failed: {e}")

    try:
        await demo_with_claude()
    except Exception as e:
        print(f"Claude demo failed: {e}")

    try:
        await demo_streaming_with_image()
    except Exception as e:
        print(f"Streaming demo failed: {e}")

    try:
        await demo_local_file()
    except Exception as e:
        print(f"Local file demo failed: {e}")

    try:
        await demo_multiple_images()
    except Exception as e:
        print(f"Multiple images demo failed: {e}")

    try:
        await demo_base64_direct()
    except Exception as e:
        print(f"Base64 demo failed: {e}")

    print("\n" + "=" * 60)
    print("All demos completed!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
