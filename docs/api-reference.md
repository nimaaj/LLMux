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

### `list_models()`

Get a list of available models for a configured provider.

```python
async def list_models(provider: str) -> List[str]
```

**Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `provider` | `str` | Yes | `"openai"`, `"claude"`, `"gemini"`, or `"deepseek"` |

**Returns:**
List of model name strings.

**Example:**
```python
models = await client.list_models("gemini")
print(models)
# ['gemini-1.5-flash-latest', 'gemini-1.5-pro-latest', ...]
```

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

#### Text-Only Messages

```python
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Hello!"},
    {"role": "assistant", "content": "Hi there!"},
    {"role": "user", "content": "How are you?"},
]
```

#### Multimodal Messages (with Images)

Use the helper methods for multimodal content:

```python
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    client.create_message("user", [
        "What's in this image?",
        client.create_image_content("https://example.com/image.jpg"),
    ]),
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

## Image / Multimodal Helpers

Helper methods for creating multimodal messages with images.

### `create_message()`

Create a message with text and/or images.

```python
message = client.create_message(
    role="user",  # "system", "user", or "assistant"
    content=[
        "Describe this image:",
        client.create_image_content("./photo.jpg"),
    ]
)
```

### `create_image_content()`

Create an image content part from various sources.

```python
# From URL (auto-downloaded for all providers)
client.create_image_content("https://example.com/image.jpg")

# From local file (auto-detected and encoded)
client.create_image_content("./photo.jpg")

# From base64 data (requires mime_type)
client.create_image_content(base64_string, mime_type="image/jpeg")

# With detail level (OpenAI only)
client.create_image_content(url, detail="high")  # "auto", "low", "high"
```

### `create_text_content()`

Create a text content part (used internally by `create_message`).

```python
text_part = client.create_text_content("What's in this image?")
```

### `encode_image_file()`

Encode a local image file to base64. Returns `(base64_data, mime_type)`.

```python
b64_data, mime_type = client.encode_image_file("./photo.jpg")
```

### `encode_image_url()`

Download and encode an image URL to base64. Returns `(base64_data, mime_type)`.

```python
b64_data, mime_type = await client.encode_image_url("https://example.com/image.jpg")
```

### Supported Image Formats

- JPEG (`.jpg`, `.jpeg`)
- PNG (`.png`)
- GIF (`.gif`)
- WebP (`.webp`)
- BMP (`.bmp`)

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
| Vision/Images | ✅ | ✅ | ✅ | ⚠️ |
| System Messages | ✅ | ✅ | ✅ | ✅ |
| Temperature | ✅ | ✅ | ✅ | ✅ |
| Top P | ✅ | ✅ | ✅ | ✅ |
| Top K | ❌ | ✅ | ✅ | ❌ |
| Token Usage | ✅ | ✅ | ✅ | ✅ |
| Tool Calling | ✅ | ✅ | ✅ | ✅ |
| MCP Integration | ✅ | ✅ | ✅ | ✅ |

⚠️ = Limited or model-dependent support

---

## Tool Calling

### `create_tool()`

Create a tool definition for function calling.

```python
tool = client.create_tool(
    name="get_weather",
    description="Get the current weather for a location",
    parameters={
        "location": {
            "type": "string",
            "description": "City name, e.g., 'Paris'"
        },
        "unit": {
            "type": "string",
            "enum": ["celsius", "fahrenheit"],
            "description": "Temperature unit"
        }
    },
    required=["location"]
)
```

### `create_tool_result()`

Create a tool result message to send back to the LLM.

```python
result_msg = client.create_tool_result(
    tool_call_id="call_abc123",
    content="The weather in Paris is sunny, 22°C"
)
```

### `create_assistant_message_with_tool_calls()`

Create an assistant message that includes tool calls (for conversation history).

```python
assistant_msg = client.create_assistant_message_with_tool_calls(
    text="I'll check the weather for you.",
    tool_calls=[
        {"id": "call_abc123", "name": "get_weather", "arguments": {"location": "Paris"}}
    ]
)
```

### Tool Response Format

When tools are provided, responses may include tool calls:

```python
{
    "provider": "openai",
    "text": "",  # May be empty when tool is called
    "tool_calls": [
        {
            "id": "call_abc123",
            "name": "get_weather",
            "arguments": {"location": "Paris"}
        }
    ],
    "meta": { ... }
}
```

---

## `chat_with_tools()` - Automatic Tool Execution

High-level method that automatically executes tools and continues the conversation.

### Signature

```python
async def chat_with_tools(
    provider: str,
    model: str,
    messages: List[Message],
    tools: Optional[List[Tool]] = None,
    mcp_executor: Optional[MCPToolExecutor] = None,
    tool_handlers: Optional[Dict[str, Callable]] = None,
    auto_execute: bool = True,
    max_iterations: int = 10,
    **opts
) -> Dict[str, Any]
```

### Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `tools` | `List[Tool]` | No | Native tool definitions |
| `mcp_executor` | `MCPToolExecutor` | No | MCP executor for MCP tools |
| `tool_handlers` | `Dict[str, Callable]` | No | Functions to handle native tools |
| `auto_execute` | `bool` | No | Auto-execute tools (default: True) |
| `max_iterations` | `int` | No | Max tool execution loops (default: 10) |

### Example

```python
# Define tools and handlers
tools = [
    client.create_tool(
        name="calculate",
        description="Evaluate a math expression",
        parameters={"expression": {"type": "string"}},
        required=["expression"]
    )
]

def calculate(args):
    return str(eval(args["expression"]))

# Use chat_with_tools
response = await client.chat_with_tools(
    provider="openai",
    model="gpt-4o",
    messages=[{"role": "user", "content": "What is 25 * 4?"}],
    tools=tools,
    tool_handlers={"calculate": calculate},
)

print(response["text"])  # "25 * 4 = 100"
print(response["tool_history"])  # [{tool, arguments, result}, ...]
```

---

## MCP (Model Context Protocol)

MCP (Model Context Protocol) enables LLMs to interact with external tools and resources through a standardized protocol. LLMux provides a complete MCP client implementation.

### MCP Type Definitions

#### `TransportType`

Supported transport protocols for MCP connections:

```python
TransportType = Literal["stdio", "sse", "streamable-http", "websocket"]
```

| Transport | Description | Use Case |
|-----------|-------------|----------|
| `stdio` | Subprocess communication | Local MCP servers (npx, python, etc.) |
| `sse` | Server-Sent Events over HTTP | Remote servers with SSE support |
| `streamable-http` | Streamable HTTP transport | HTTP-based MCP servers |
| `websocket` | WebSocket transport | Real-time bidirectional servers |

#### `MCPServerConfig`

Configuration dictionary for MCP server connections:

```python
class MCPServerConfig(TypedDict, total=False):
    name: str                    # Required: Unique connection name
    transport: TransportType     # Transport type (default: "stdio")
    # For stdio transport
    command: str                 # Command to run (e.g., "npx", "python")
    args: List[str]              # Command arguments
    env: Dict[str, str]          # Environment variables
    # For HTTP-based transports
    url: str                     # Server URL
    headers: Dict[str, str]      # HTTP headers (e.g., for authentication)
```

#### `MCPConnection`

Represents an active MCP server connection:

```python
@dataclass
class MCPConnection:
    name: str                           # Connection identifier
    session: ClientSession              # MCP client session
    tools: List[Dict[str, Any]]         # Discovered tools (OpenAI format)
    resources: List[Dict[str, Any]]     # Discovered resources
```

---

### MCPToolExecutor

The main class for managing MCP server connections and tool execution.

**IMPORTANT**: `MCPToolExecutor` must be used as an async context manager to ensure proper cleanup of connections.

```python
from mcp_client import MCPToolExecutor, mcp_executor

# Method 1: Using the mcp_executor() helper (recommended)
async with mcp_executor() as executor:
    await executor.connect_stdio(
        name="filesystem",
        command="npx",
        args=["-y", "@modelcontextprotocol/server-filesystem", "/tmp"]
    )
    tools = executor.get_tools()
    result = await executor.call_tool("list_directory", {"path": "/tmp"})
# Connections are automatically cleaned up when exiting the context

# Method 2: Using MCPToolExecutor directly
async with MCPToolExecutor() as executor:
    await executor.connect_stdio(...)
    # ... use executor ...
# Automatic cleanup on context exit
```

---

### Connection Methods

#### `connect_stdio()`

Connect to an MCP server via subprocess (stdio transport). Best for local MCP servers.

```python
async def connect_stdio(
    name: str,
    command: str,
    args: Optional[List[str]] = None,
    env: Optional[Dict[str, str]] = None,
) -> MCPConnection
```

**Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `name` | `str` | Yes | Unique identifier for this connection |
| `command` | `str` | Yes | Command to run (e.g., `"npx"`, `"python"`, `"uv"`) |
| `args` | `List[str]` | No | Command arguments |
| `env` | `Dict[str, str]` | No | Environment variables to set |

**Returns:** `MCPConnection` object with active session and discovered tools.

**Example:**

```python
async with mcp_executor() as executor:
    # Connect to filesystem server
    conn = await executor.connect_stdio(
        name="filesystem",
        command="npx",
        args=["-y", "@modelcontextprotocol/server-filesystem", "/tmp", "/home"],
        env={"NODE_ENV": "production"}
    )
    print(f"Connected! Found {len(conn.tools)} tools")
```

---

#### `connect_sse()`

Connect to an MCP server via Server-Sent Events (SSE) transport. Best for remote HTTP servers.

```python
async def connect_sse(
    name: str,
    url: str,
    headers: Optional[Dict[str, str]] = None,
) -> MCPConnection
```

**Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `name` | `str` | Yes | Unique identifier for this connection |
| `url` | `str` | Yes | Server URL (e.g., `"http://localhost:8000/mcp"`) |
| `headers` | `Dict[str, str]` | No | HTTP headers for authentication |

**Example:**

```python
async with mcp_executor() as executor:
    conn = await executor.connect_sse(
        name="remote-api",
        url="https://api.example.com/mcp",
        headers={"Authorization": "Bearer your-token-here"}
    )
```

---

#### `connect_http()`

Connect to an MCP server via streamable HTTP transport.

```python
async def connect_http(
    name: str,
    url: str,
    headers: Optional[Dict[str, str]] = None,
) -> MCPConnection
```

**Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `name` | `str` | Yes | Unique identifier for this connection |
| `url` | `str` | Yes | Server URL |
| `headers` | `Dict[str, str]` | No | HTTP headers |

**Example:**

```python
async with mcp_executor() as executor:
    conn = await executor.connect_http(
        name="http-server",
        url="http://localhost:8080/mcp",
    )
```

---

#### `connect()`

Generic connection method using a configuration dictionary. Useful for dynamic configuration.

```python
async def connect(config: MCPServerConfig) -> MCPConnection
```

**Example:**

```python
async with mcp_executor() as executor:
    # Connect using config dict
    conn = await executor.connect({
        "name": "filesystem",
        "transport": "stdio",
        "command": "npx",
        "args": ["-y", "@modelcontextprotocol/server-filesystem", "/tmp"]
    })
    
    # Connect to multiple servers from config
    configs = [
        {"name": "fs", "transport": "stdio", "command": "npx", "args": [...]},
        {"name": "api", "transport": "sse", "url": "http://localhost:8000/mcp"},
    ]
    for config in configs:
        await executor.connect(config)
```

---

### Tool Discovery Methods

#### `get_tools()`

Get all available tools from all connected servers in OpenAI-compatible format.

```python
def get_tools() -> List[Dict[str, Any]]
```

**Returns:** List of tool definitions compatible with `UnifiedChatClient`.

**Example:**

```python
async with mcp_executor() as executor:
    await executor.connect_stdio("fs", "npx", ["-y", "@modelcontextprotocol/server-filesystem", "/tmp"])
    
    tools = executor.get_tools()
    for tool in tools:
        print(f"- {tool['function']['name']}: {tool['function']['description']}")
```

---

#### `get_tools_for_server()`

Get tools for a specific connected server.

```python
def get_tools_for_server(server_name: str) -> List[Dict[str, Any]]
```

**Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `server_name` | `str` | Yes | Name of the server connection |

**Returns:** List of tools for that server, or empty list if server not found.

**Example:**

```python
async with mcp_executor() as executor:
    await executor.connect_stdio("fs", "npx", [...])
    await executor.connect_stdio("memory", "npx", [...])
    
    # Get tools only from filesystem server
    fs_tools = executor.get_tools_for_server("fs")
```

---

### Properties

#### `tool_names`

List of all available tool names across all connected servers.

```python
@property
def tool_names() -> List[str]
```

**Example:**

```python
async with mcp_executor() as executor:
    await executor.connect_stdio("fs", "npx", [...])
    
    print(executor.tool_names)
    # ['read_file', 'write_file', 'list_directory', ...]
```

---

#### `connections`

Dictionary of all active connections, keyed by connection name.

```python
@property
def connections() -> Dict[str, MCPConnection]
```

**Example:**

```python
async with mcp_executor() as executor:
    await executor.connect_stdio("fs", "npx", [...])
    await executor.connect_stdio("memory", "npx", [...])
    
    for name, conn in executor.connections.items():
        print(f"{name}: {len(conn.tools)} tools")
```

---

### Tool Execution Methods

#### `call_tool()`

Execute a single tool on the appropriate MCP server.

```python
async def call_tool(
    tool_name: str,
    arguments: Dict[str, Any],
) -> str
```

**Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `tool_name` | `str` | Yes | Name of the tool to execute |
| `arguments` | `Dict[str, Any]` | Yes | Tool arguments |

**Returns:** Tool result as a string.

**Raises:** `ValueError` if tool is not found.

**Example:**

```python
async with mcp_executor() as executor:
    await executor.connect_stdio("fs", "npx", ["-y", "@modelcontextprotocol/server-filesystem", "/tmp"])
    
    # Call a tool directly
    result = await executor.call_tool(
        "list_directory",
        {"path": "/tmp"}
    )
    print(result)
```

---

#### `execute_tool_calls()`

Execute multiple tool calls from an LLM response and return formatted results.

```python
async def execute_tool_calls(
    tool_calls: List[Dict[str, Any]],
) -> List[Dict[str, Any]]
```

**Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `tool_calls` | `List[Dict]` | Yes | List of tool calls from LLM response |

**Returns:** List of tool result messages ready to send back to the LLM.

**Example:**

```python
# Tool calls from LLM response
tool_calls = [
    {"id": "call_1", "name": "list_directory", "arguments": {"path": "/tmp"}},
    {"id": "call_2", "name": "read_file", "arguments": {"path": "/tmp/test.txt"}},
]

# Execute all tool calls
results = await executor.execute_tool_calls(tool_calls)
# Returns:
# [
#     {"role": "tool", "tool_call_id": "call_1", "content": "..."},
#     {"role": "tool", "tool_call_id": "call_2", "content": "..."},
# ]
```

---

### Resource Methods

MCP servers can expose resources (files, data, etc.) in addition to tools.

#### `list_resources()`

List available resources from connected MCP servers.

```python
async def list_resources(
    server_name: Optional[str] = None
) -> List[Dict[str, Any]]
```

**Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `server_name` | `str` | No | Filter by server name (all servers if None) |

**Returns:** List of resource definitions with `uri`, `name`, and `server` fields.

**Example:**

```python
async with mcp_executor() as executor:
    await executor.connect_stdio("fs", "npx", [...])
    
    # List all resources
    resources = await executor.list_resources()
    for r in resources:
        print(f"{r['name']}: {r['uri']} (from {r['server']})")
    
    # List resources from specific server
    fs_resources = await executor.list_resources("fs")
```

---

#### `read_resource()`

Read content from an MCP resource.

```python
async def read_resource(
    uri: str,
    server_name: Optional[str] = None
) -> str
```

**Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `uri` | `str` | Yes | Resource URI |
| `server_name` | `str` | No | Server to read from (auto-detect if None) |

**Returns:** Resource content as a string.

**Raises:** `ValueError` if resource not found.

**Example:**

```python
async with mcp_executor() as executor:
    await executor.connect_stdio("fs", "npx", [...])
    
    # Read a resource
    content = await executor.read_resource("file:///tmp/config.json")
    print(content)
```

---

### Helper Functions

#### `mcp_executor()`

Async context manager for easy MCPToolExecutor usage with automatic cleanup.

```python
@asynccontextmanager
async def mcp_executor() -> AsyncIterator[MCPToolExecutor]
```

**Example:**

```python
from mcp_client import mcp_executor

async with mcp_executor() as executor:
    await executor.connect_stdio("fs", "npx", [...])
    tools = executor.get_tools()
    result = await executor.call_tool("read_file", {"path": "/tmp/test.txt"})
# All connections automatically cleaned up
```

---

### MCPToolExecutor Quick Reference

| Method/Property | Description |
|----------------|-------------|
| `connect_stdio(name, command, args, env)` | Connect via subprocess |
| `connect_sse(name, url, headers)` | Connect via SSE |
| `connect_http(name, url, headers)` | Connect via HTTP |
| `connect(config)` | Connect using config dict |
| `get_tools()` | Get all tools (OpenAI format) |
| `get_tools_for_server(name)` | Get tools for specific server |
| `tool_names` | List of all tool names |
| `connections` | Dict of active connections |
| `call_tool(name, args)` | Execute a single tool |
| `execute_tool_calls(calls)` | Execute multiple tool calls |
| `list_resources(server_name)` | List available resources |
| `read_resource(uri, server_name)` | Read a resource |

---

### Using MCP with chat_with_tools()

```python
from llmclient import UnifiedChatClient
from mcp_client import mcp_executor

client = UnifiedChatClient()

async with mcp_executor() as executor:
    # Connect to MCP server
    await executor.connect_stdio(
        name="filesystem",
        command="npx",
        args=["-y", "@modelcontextprotocol/server-filesystem", "/tmp"]
    )
    
    # Chat with MCP tools automatically available
    response = await client.chat_with_tools(
        provider="openai",
        model="gpt-4o",
        messages=[
            {"role": "user", "content": "List files in /tmp"}
        ],
        mcp_executor=executor,
    )
    
    print(response["text"])
```

### Combining Native and MCP Tools

```python
# Native tools with handlers
native_tools = [client.create_tool("calculate", ...)]
handlers = {"calculate": lambda args: eval(args["expression"])}

# MCP tools
async with mcp_executor() as executor:
    await executor.connect_stdio("fs", "npx", [...])
    
    response = await client.chat_with_tools(
        provider="openai",
        model="gpt-4o",
        messages=messages,
        tools=native_tools,           # Native tools
        tool_handlers=handlers,       # Native handlers
        mcp_executor=executor,        # MCP tools auto-discovered
    )
```

---

## `get_models()` - List Available Models

Static method to retrieve available models for a provider.

### Signature

```python
@staticmethod
def get_models(provider: str) -> list[str]
```

### Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `provider` | `str` | Yes | `"openai"`, `"anthropic"`, `"gemini"`, or `"deepseek"` |

### Returns

A list of model name strings available for the specified provider.

### Example

```python
from llmclient import UnifiedChatClient

# Get all available OpenAI models
openai_models = UnifiedChatClient.get_models("openai")
print(openai_models)

# Get all available Gemini models
gemini_models = UnifiedChatClient.get_models("gemini")
for model in gemini_models:
    print(model)
```

### Notes

- This is a static method and does not require an instance of `UnifiedChatClient`.
- API keys are read from the `.env` file.
- For Claude/Anthropic, use `"anthropic"` as the provider name.
- For Gemini, only models supporting `generateContent` are returned.

---

## Type Definitions

The following TypedDicts are used for type hints:

### `TextContent`

```python
class TextContent(TypedDict):
    type: Literal["text"]
    text: str
```

### `ImageUrlDetail`

```python
class ImageUrlDetail(TypedDict, total=False):
    url: str
    detail: Literal["auto", "low", "high"]
```

### `ImageContent`

```python
class ImageContent(TypedDict):
    type: Literal["image_url"]
    image_url: ImageUrlDetail
```

### `Message`

```python
class Message(TypedDict):
    role: Literal["system", "user", "assistant"]
    content: MessageContent
```

### `ContentPart`

```python
ContentPart = Union[TextContent, ImageContent]
```

### `MessageContent`

```python
MessageContent = Union[str, List[ContentPart]]
```

---

## Tool Type Definitions

### `Tool`

```python
class Tool(TypedDict):
    type: Literal["function"]
    function: FunctionDefinition
```

### `FunctionDefinition`

```python
class FunctionDefinition(TypedDict, total=False):
    name: str
    description: str
    parameters: FunctionParameters
```

### `FunctionParameters`

```python
class FunctionParameters(TypedDict, total=False):
    type: Literal["object"]
    properties: Dict[str, Any]
    required: List[str]
```

### `ToolCall`

```python
class ToolCall(TypedDict, total=False):
    id: str
    name: str
    arguments: Dict[str, Any]
```

### `ToolResult`

```python
class ToolResult(TypedDict):
    tool_call_id: str
    content: str
```