# LLMux

A unified Python client for interacting with multiple LLM providers (OpenAI, Anthropic Claude, Google Gemini, and DeepSeek) with beautiful terminal output using Rich.

## ✨ Features

- **Unified API** - Single interface for OpenAI, Claude, Gemini, and DeepSeek
- **Streaming & Non-Streaming** - Both real-time streaming and traditional request/response
- **Beautiful Output** - Rich terminal formatting with Markdown rendering
- **Normalized Metrics** - Consistent token usage and latency stats across providers
- **Model Discovery** - Dynamically fetch available models from providers

## 📦 Installation

**Prerequisites:** Python 3.12+

```bash
# With uv (recommended)
uv sync

# Or with pip
pip install openai anthropic google-generativeai rich python-dotenv
```

## 🔧 Configuration

Create a `.env` file with your API keys (only configure providers you need):

```env
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
GOOGLE_API_KEY=...
DEEPSEEK_API_KEY=sk-...
```

## 🚀 Quick Start

```python
import asyncio
from llmclient import UnifiedChatClient
from rich_llm_printer import RichPrinter, RichStreamPrinter

async def main():
    client = UnifiedChatClient()
    messages = [{"role": "user", "content": "Hello!"}]
    
    # Non-streaming
    response = await client.chat("openai", "gpt-4o", messages)
    RichPrinter().print_chat(response)
    
    # Streaming
    stream = client.astream("claude", "claude-sonnet-4-20250514", messages)
    await RichStreamPrinter().print_stream(stream)

asyncio.run(main())
```

## 📚 Documentation

- **[API Reference](docs/api-reference.md)** - Complete API documentation
- **[Examples](docs/examples.md)** - Practical usage examples

## 🗂️ Project Structure

```
├── llmclient.py          # UnifiedChatClient - core LLM interface
├── rich_llm_printer.py   # RichPrinter & RichStreamPrinter
├── main.py               # Non-streaming demo
├── demo streaming.py     # Streaming demo
├── docs/                 # Documentation
│   ├── api-reference.md
│   └── examples.md
└── pyproject.toml        # Dependencies
```

## 📄 License

MIT

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
