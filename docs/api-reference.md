# API Reference

Complete API documentation for LLM Client.

## UnifiedChatClient

The main client class for interacting with LLM providers.

### Constructor

```python
from llmclient import UnifiedChatClient

client = UnifiedChatClient(
    openai_api_key=None,      # Optional: Override env variable
    anthropic_api_key=None,   # Optional: Override env variable
    gemini_api_key=None,      # Optional: Override env variable
    deepseek_api_key=None,    # Optional: Override env variable
)
```

By default, API keys are loaded from the `.env` file.

### Providers & Models

| Provider | Example Models |
|----------|----------------|
| `openai` | `gpt-4o`, `gpt-4o-mini`, `gpt-4-turbo` |
| `claude` | `claude-sonnet-4-20250514`, `claude-3-5-haiku-latest` |
| `gemini` | `gemini-1.5-pro`, `gemini-1.5-flash` |
| `deepseek` | `deepseek-chat`, `deepseek-reasoner` |

---

## `chat()` - Non-Streaming

Make a standard request/response call to an LLM.

### Signature

```python
async def chat(
    provider: str,
    model: str,
    messages: List[Dict[str, str]],
    **opts
) -> Dict[str, Any]
```

### Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `provider` | `str` | Yes | `"openai"`, `"claude"`, `"gemini"`, or `"deepseek"` |
| `model` | `str` | Yes | Model name (e.g., `"gpt-4o"`) |
| `messages` | `List[Message]` | Yes | Conversation messages |
| `temperature` | `float` | No | Sampling temperature (default: 0.7) |
| `max_tokens` | `int` | No | Max output tokens (default: 1024) |
| `top_p` | `float` | No | Nucleus sampling parameter |
| `top_k` | `int` | No | Top-k sampling (Claude/Gemini only) |

### Message Format

```python
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Hello!"},
    {"role": "assistant", "content": "Hi there!"},
    {"role": "user", "content": "How are you?"},
]
```

### Response Format

```python
{
    "provider": "openai",
    "text": "The assistant's response...",
    "meta": {
        "model": "gpt-4o",
        "usage": {
            "input_tokens": 25,
            "output_tokens": 150,
            "total_tokens": 175,
            "raw": { ... }  # Provider-specific usage data
        },
        "latency_ms": 1234.56,
        "finish_reason": "stop"
    }
}
```

### Example

```python
response = await client.chat(
    provider="openai",
    model="gpt-4o",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Explain Python decorators briefly."},
    ],
    temperature=0.7,
    max_tokens=500,
)

print(response["text"])
```

---

## `astream()` - Streaming

Stream tokens in real-time from an LLM.

### Signature

```python
async def astream(
    provider: str,
    model: str,
    messages: List[Dict[str, str]],
    **opts
) -> AsyncIterator[Dict[str, Any]]
```

### Parameters

Same as `chat()`.

### Event Types

#### Token Event (multiple)

Emitted for each token/chunk received:

```python
{
    "type": "token",
    "provider": "openai",
    "text": "Hello"
}
```

#### Done Event (once)

Emitted when streaming is complete:

```python
{
    "type": "done",
    "provider": "openai",
    "text": "Full response text...",
    "meta": {
        "model": "gpt-4o",
        "usage": { ... },
        "latency_ms": 1234.56
    }
}
```

### Example

```python
async for event in client.astream(
    provider="claude",
    model="claude-sonnet-4-20250514",
    messages=messages,
):
    if event["type"] == "token":
        print(event["text"], end="", flush=True)
    elif event["type"] == "done":
        print(f"\n\nTokens used: {event['meta']['usage']['total_tokens']}")
```

---

## RichPrinter

Display non-streaming responses with beautiful Rich formatting.

### Constructor

```python
from rich_llm_printer import RichPrinter

printer = RichPrinter(
    title="Response",           # Panel title
    show_metadata=True,         # Show usage stats
    code_theme="coffee",        # Syntax highlighting theme
    inline_code_theme="monokai",
    show_provider_info=True,    # Show provider in title
    border_style="green",       # Panel border color
)
```

### Methods

#### `print_chat(response)`

Display a response from `UnifiedChatClient.chat()`.

```python
response = await client.chat(...)
printer.print_chat(response)
```

#### `get_response()`

Get the last printed response.

#### `get_text()`

Get the text from the last printed response.

#### `get_provider()`

Get the provider name from the last response.

---

## RichStreamPrinter

Display streaming responses with real-time Rich formatting.

### Constructor

```python
from rich_llm_printer import RichStreamPrinter

printer = RichStreamPrinter(
    title="Streaming Response",
    show_metadata=True,
    code_theme="dracula",
    inline_code_theme="monokai",
    refresh_rate=30,            # Display refresh rate (Hz)
    show_final_title=True,      # Change title when done
    show_provider_info=True,
    border_style="blue",
)
```

### Methods

#### `print_stream(event_stream)`

Process and display a streaming event iterator.

```python
event_stream = client.astream(...)
final_event = await printer.print_stream(event_stream)
```

#### `get_full_text()`

Get the full assembled text.

#### `get_final_event()`

Get the final event if available.

#### `get_provider()`

Get the provider name.

---

## Provider Feature Matrix

| Feature | OpenAI | Claude | Gemini | DeepSeek |
|---------|--------|--------|--------|----------|
| Streaming | ✅ | ✅ | ✅ | ✅ |
| Non-Streaming | ✅ | ✅ | ✅ | ✅ |
| System Messages | ✅ | ✅ | ✅ | ✅ |
| Temperature | ✅ | ✅ | ✅ | ✅ |
| Top P | ✅ | ✅ | ✅ | ✅ |
| Top K | ❌ | ✅ | ✅ | ❌ |
| Token Usage | ✅ | ✅ | ✅ | ✅ |
