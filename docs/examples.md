# Examples

Practical examples for using LLM Client.

## Basic Non-Streaming

```python
import asyncio
from llmclient import UnifiedChatClient
from rich_llm_printer import RichPrinter

async def main():
    client = UnifiedChatClient()
    
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Explain Python decorators briefly."},
    ]
    
    response = await client.chat(
        provider="openai",
        model="gpt-4o",
        messages=messages,
        temperature=0.7,
        max_tokens=500,
    )
    
    # Pretty print with Rich
    printer = RichPrinter()
    printer.print_chat(response)

asyncio.run(main())
```

## Basic Streaming

```python
import asyncio
from llmclient import UnifiedChatClient
from rich_llm_printer import RichStreamPrinter

async def main():
    client = UnifiedChatClient()
    
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Write a haiku about programming."},
    ]
    
    event_stream = client.astream(
        provider="claude",
        model="claude-sonnet-4-20250514",
        messages=messages,
        temperature=0.7,
        max_tokens=500,
    )
    
    printer = RichStreamPrinter(
        title="Streaming Response",
        show_metadata=True,
        code_theme="dracula",
    )
    
    final_event = await printer.print_stream(event_stream)

asyncio.run(main())
```

## Multi-Provider Comparison

Compare responses from different providers:

```python
import asyncio
from llmclient import UnifiedChatClient
from rich_llm_printer import RichPrinter

async def compare_providers():
    client = UnifiedChatClient()
    printer = RichPrinter(show_metadata=True)
    
    messages = [
        {"role": "user", "content": "What is the capital of France?"},
    ]
    
    providers = [
        ("openai", "gpt-4o-mini"),
        ("claude", "claude-3-5-haiku-latest"),
        ("gemini", "gemini-1.5-flash"),
        ("deepseek", "deepseek-chat"),
    ]
    
    for provider, model in providers:
        try:
            response = await client.chat(
                provider=provider,
                model=model,
                messages=messages,
            )
            printer.print_chat(response)
        except Exception as e:
            print(f"{provider}: Error - {e}")

asyncio.run(compare_providers())
```

## Streaming Without Rich Printer

Handle streaming manually for custom implementations:

```python
import asyncio
from llmclient import UnifiedChatClient

async def manual_stream():
    client = UnifiedChatClient()
    
    messages = [
        {"role": "user", "content": "Count from 1 to 5."},
    ]
    
    full_text = ""
    
    async for event in client.astream(
        provider="openai",
        model="gpt-4o-mini",
        messages=messages,
    ):
        if event["type"] == "token":
            print(event["text"], end="", flush=True)
            full_text += event["text"]
        elif event["type"] == "done":
            print("\n")
            print(f"Total tokens: {event['meta']['usage']['total_tokens']}")
            print(f"Latency: {event['meta']['latency_ms']:.2f}ms")

asyncio.run(manual_stream())
```

## Conversation with History

Maintain conversation context:

```python
import asyncio
from llmclient import UnifiedChatClient
from rich_llm_printer import RichPrinter

async def conversation():
    client = UnifiedChatClient()
    printer = RichPrinter()
    
    messages = [
        {"role": "system", "content": "You are a helpful math tutor."},
    ]
    
    # First turn
    messages.append({"role": "user", "content": "What is 2 + 2?"})
    response = await client.chat(
        provider="openai",
        model="gpt-4o-mini",
        messages=messages,
    )
    printer.print_chat(response)
    messages.append({"role": "assistant", "content": response["text"]})
    
    # Second turn
    messages.append({"role": "user", "content": "Now multiply that by 3."})
    response = await client.chat(
        provider="openai",
        model="gpt-4o-mini",
        messages=messages,
    )
    printer.print_chat(response)

asyncio.run(conversation())
```

## Custom Printer Configuration

Customize the Rich printer appearance:

```python
import asyncio
from llmclient import UnifiedChatClient
from rich_llm_printer import RichPrinter, RichStreamPrinter

async def custom_printers():
    client = UnifiedChatClient()
    
    messages = [
        {"role": "user", "content": "Write a Python hello world."},
    ]
    
    # Non-streaming with custom style
    printer = RichPrinter(
        title="Code Example",
        show_metadata=False,
        code_theme="monokai",
        border_style="cyan",
        show_provider_info=False,
    )
    
    response = await client.chat(
        provider="openai",
        model="gpt-4o-mini",
        messages=messages,
    )
    printer.print_chat(response)
    
    # Streaming with custom style
    stream_printer = RichStreamPrinter(
        title="Live Response",
        show_metadata=True,
        code_theme="dracula",
        refresh_rate=60,
        show_final_title=True,
        border_style="magenta",
    )
    
    event_stream = client.astream(
        provider="claude",
        model="claude-3-5-haiku-latest",
        messages=messages,
    )
    await stream_printer.print_stream(event_stream)

asyncio.run(custom_printers())
```

## Error Handling

Handle provider errors gracefully:

```python
import asyncio
from llmclient import UnifiedChatClient
from rich_llm_printer import RichPrinter

async def with_error_handling():
    client = UnifiedChatClient()
    printer = RichPrinter()
    
    messages = [
        {"role": "user", "content": "Hello!"},
    ]
    
    try:
        response = await client.chat(
            provider="openai",
            model="gpt-4o",
            messages=messages,
        )
        printer.print_chat(response)
    except RuntimeError as e:
        print(f"Client not configured: {e}")
    except ValueError as e:
        print(f"Invalid provider: {e}")
    except Exception as e:
        print(f"API error: {e}")

asyncio.run(with_error_handling())
```
