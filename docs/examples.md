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
from rich_llm_printer import RichPrinter

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
    
    printer = RichPrinter()
    final_event = await printer.print_stream(event_stream)

asyncio.run(main())
```

## Vision / Multimodal

Send images to vision-capable models. The same message format works across all providers.

### Image from URL

```python
import asyncio
from llmclient import UnifiedChatClient
from rich_llm_printer import RichPrinter

async def analyze_image_url():
    client = UnifiedChatClient()
    printer = RichPrinter()
    
    image_url = "https://images.unsplash.com/photo-1514888286974-6c03e2ca1dba?w=400"
    
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        client.create_message("user", [
            "What do you see in this image?",
            client.create_image_content(image_url),
        ]),
    ]
    
    response = await client.chat(
        provider="openai",
        model="gpt-4o-mini",
        messages=messages,
        max_tokens=300,
    )
    printer.print_chat(response)

asyncio.run(analyze_image_url())
```

### Image from Local File

```python
import asyncio
from llmclient import UnifiedChatClient
from rich_llm_printer import RichPrinter

async def analyze_local_image():
    client = UnifiedChatClient()
    printer = RichPrinter()
    
    # Local file paths are automatically detected and encoded
    messages = [
        client.create_message("user", [
            "Describe this image.",
            client.create_image_content("./photo.jpg"),
        ]),
    ]
    
    response = await client.chat(
        provider="gemini",
        model="gemini-2.0-flash",
        messages=messages,
    )
    printer.print_chat(response)

asyncio.run(analyze_local_image())
```

### Multiple Images

```python
import asyncio
from llmclient import UnifiedChatClient
from rich_llm_printer import RichPrinter

async def compare_images():
    client = UnifiedChatClient()
    printer = RichPrinter()
    
    cat_url = "https://images.unsplash.com/photo-1514888286974-6c03e2ca1dba?w=400"
    dog_url = "https://images.unsplash.com/photo-1587300003388-59208cc962cb?w=400"
    
    messages = [
        client.create_message("user", [
            "Compare these two images. What animals are shown?",
            client.create_image_content(cat_url),
            client.create_image_content(dog_url),
        ]),
    ]
    
    response = await client.chat(
        provider="openai",
        model="gpt-4o-mini",
        messages=messages,
        max_tokens=400,
    )
    printer.print_chat(response)

asyncio.run(compare_images())
```

### Using Base64 Directly

```python
import asyncio
from llmclient import UnifiedChatClient
from rich_llm_printer import RichPrinter

async def use_base64():
    client = UnifiedChatClient()
    printer = RichPrinter()
    
    # Download and encode manually
    b64_data, mime_type = await client.encode_image_url(
        "https://images.unsplash.com/photo-1514888286974-6c03e2ca1dba?w=400"
    )
    
    messages = [
        client.create_message("user", [
            "What color is this animal?",
            client.create_image_content(b64_data, mime_type=mime_type),
        ]),
    ]
    
    response = await client.chat(
        provider="claude",
        model="claude-sonnet-4-20250514",
        messages=messages,
    )
    printer.print_chat(response)

asyncio.run(use_base64())
```

### Streaming with Images

```python
import asyncio
from llmclient import UnifiedChatClient
from rich_llm_printer import RichPrinter

async def stream_with_image():
    client = UnifiedChatClient()
    printer = RichPrinter()
    
    image_url = "https://images.unsplash.com/photo-1514888286974-6c03e2ca1dba?w=400"
    
    messages = [
        client.create_message("user", [
            "Describe this image in detail.",
            client.create_image_content(image_url),
        ]),
    ]
    
    stream = client.astream(
        provider="openai",
        model="gpt-4o-mini",
        messages=messages,
        max_tokens=500,
    )
    
    await printer.print_stream(stream)

asyncio.run(stream_with_image())
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

## Tool Calling

### Basic Tool Calling

```python
import asyncio
from llmclient import UnifiedChatClient
from rich_llm_printer import RichPrinter

async def basic_tools():
    client = UnifiedChatClient()
    printer = RichPrinter()
    
    # Define a tool
    tools = [
        client.create_tool(
            name="get_weather",
            description="Get the current weather for a location",
            parameters={
                "location": {
                    "type": "string",
                    "description": "City name"
                }
            },
            required=["location"]
        )
    ]
    
    messages = [
        {"role": "user", "content": "What's the weather in Tokyo?"}
    ]
    
    # First call - LLM decides to use the tool
    response = await client.chat(
        provider="openai",
        model="gpt-4o",
        messages=messages,
        tools=tools,
    )
    
    if response.get("tool_calls"):
        for tc in response["tool_calls"]:
            print(f"Tool called: {tc['name']}")
            print(f"Arguments: {tc['arguments']}")
            
            # Execute the tool (your implementation)
            result = f"Sunny, 25°C in {tc['arguments']['location']}"
            
            # Add assistant message and tool result
            messages.append(
                client.create_assistant_message_with_tool_calls("", response["tool_calls"])
            )
            messages.append(
                client.create_tool_result(tc["id"], result)
            )
        
        # Second call - LLM responds with tool result
        response = await client.chat(
            provider="openai",
            model="gpt-4o",
            messages=messages,
            tools=tools,
        )
    
    printer.print_chat(response)

asyncio.run(basic_tools())
```

### Automatic Tool Execution

```python
import asyncio
from llmclient import UnifiedChatClient
from rich_llm_printer import RichPrinter

async def auto_tools():
    client = UnifiedChatClient()
    printer = RichPrinter()
    
    # Define tools
    tools = [
        client.create_tool(
            name="calculate",
            description="Evaluate a math expression",
            parameters={
                "expression": {"type": "string", "description": "Math expression"}
            },
            required=["expression"]
        ),
        client.create_tool(
            name="get_time",
            description="Get current time",
            parameters={}
        )
    ]
    
    # Define handlers
    def calculate(args):
        try:
            return str(eval(args["expression"]))
        except:
            return "Error evaluating expression"
    
    def get_time(args):
        from datetime import datetime
        return datetime.now().strftime("%H:%M:%S")
    
    handlers = {
        "calculate": calculate,
        "get_time": get_time,
    }
    
    # chat_with_tools handles the loop automatically
    response = await client.chat_with_tools(
        provider="openai",
        model="gpt-4o",
        messages=[
            {"role": "user", "content": "What's 123 * 456? And what time is it?"}
        ],
        tools=tools,
        tool_handlers=handlers,
    )
    
    # Show tool history
    if response.get("tool_history"):
        print("Tools executed:")
        for entry in response["tool_history"]:
            print(f"  - {entry['tool']}: {entry['result']}")
    
    printer.print_chat(response)

asyncio.run(auto_tools())
```

## MCP (Model Context Protocol)

### Connect to MCP Server

```python
import asyncio
from mcp_client import mcp_executor

async def discover_mcp_tools():
    async with mcp_executor() as executor:
        # Connect to filesystem server
        conn = await executor.connect_stdio(
            name="filesystem",
            command="npx",
            args=["-y", "@modelcontextprotocol/server-filesystem", "/tmp"]
        )
        
        print(f"Connected to: {conn.name}")
        print(f"Available tools: {len(conn.tools)}")
        
        for tool in conn.tools:
            print(f"  - {tool['function']['name']}")

asyncio.run(discover_mcp_tools())
```

### Use MCP Tools with LLM

```python
import asyncio
from llmclient import UnifiedChatClient
from mcp_client import mcp_executor
from rich_llm_printer import RichPrinter

async def mcp_with_llm():
    client = UnifiedChatClient()
    printer = RichPrinter()
    
    async with mcp_executor() as executor:
        # Connect to filesystem server
        await executor.connect_stdio(
            name="filesystem",
            command="npx",
            args=["-y", "@modelcontextprotocol/server-filesystem", "/tmp"]
        )
        
        # Use MCP tools with LLM
        response = await client.chat_with_tools(
            provider="openai",
            model="gpt-4o",
            messages=[
                {"role": "user", "content": "List files in /tmp directory"}
            ],
            mcp_executor=executor,
        )
        
        printer.print_chat(response)

asyncio.run(mcp_with_llm())
```

### Combine Native and MCP Tools

```python
import asyncio
from llmclient import UnifiedChatClient
from mcp_client import mcp_executor
from rich_llm_printer import RichPrinter

async def combined_tools():
    client = UnifiedChatClient()
    printer = RichPrinter()
    
    # Native tool
    native_tools = [
        client.create_tool(
            name="get_time",
            description="Get current time",
            parameters={}
        )
    ]
    
    handlers = {
        "get_time": lambda _: __import__("datetime").datetime.now().strftime("%H:%M:%S")
    }
    
    async with mcp_executor() as executor:
        # MCP tools (optional - won't fail if npx not available)
        try:
            await executor.connect_stdio(
                name="filesystem",
                command="npx",
                args=["-y", "@modelcontextprotocol/server-filesystem", "/tmp"]
            )
        except:
            pass
        
        response = await client.chat_with_tools(
            provider="openai",
            model="gpt-4o",
            messages=[
                {"role": "user", "content": "What time is it? Also list files in /tmp."}
            ],
            tools=native_tools,
            tool_handlers=handlers,
            mcp_executor=executor if executor.tool_names else None,
        )
        
        printer.print_chat(response)

asyncio.run(combined_tools())
```

### Multi-Server MCP Setup

```python
import asyncio
from mcp_client import MCPToolExecutor

async def multi_server():
    executor = MCPToolExecutor()
    
    try:
        # Connect to multiple servers
        await executor.connect({
            "name": "filesystem",
            "transport": "stdio",
            "command": "npx",
            "args": ["-y", "@modelcontextprotocol/server-filesystem", "/tmp"]
        })
        
        await executor.connect({
            "name": "fetch",
            "transport": "stdio",
            "command": "npx",
            "args": ["-y", "@modelcontextprotocol/server-fetch"]
        })
        
        print(f"Total tools: {len(executor.tool_names)}")
        
        # Use tools from any server
        result = await executor.call_tool("list_directory", {"path": "/tmp"})
        print(result)
        
    finally:
        await executor.disconnect_all()

asyncio.run(multi_server())
```
