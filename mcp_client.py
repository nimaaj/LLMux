"""
MCP (Model Context Protocol) Client Wrapper for LLMux.

This module provides a unified interface for connecting to MCP servers
and integrating their tools with the UnifiedChatClient.

What is MCP?
------------
MCP (Model Context Protocol) is an open standard that enables LLMs to interact 
with external tools, data sources, and services through a standardized protocol.
Think of it as a "USB for AI" - a universal way for AI models to connect to 
any tool or data source.

Key Concepts:
- **MCP Server**: A program that exposes tools/resources via the MCP protocol
- **MCP Client**: Connects to servers and invokes their tools (this module)
- **Transport**: How client/server communicate (stdio, HTTP, WebSocket)
- **Tools**: Functions the server exposes (e.g., read_file, search_web)
- **Resources**: Data the server exposes (e.g., file contents, database records)

Supported transports:
- stdio: Connect to local MCP servers via subprocess (most common)
- sse: Connect to HTTP/SSE-based MCP servers
- streamable-http: Connect to streamable HTTP MCP servers
- websocket: Connect to WebSocket-based MCP servers (not yet implemented)

Example Usage:
-------------
```python
from mcp_client import mcp_executor

async with mcp_executor() as executor:
    # Connect to a filesystem MCP server
    await executor.connect_stdio(
        name="fs",
        command="npx",
        args=["-y", "@modelcontextprotocol/server-filesystem", "/tmp"]
    )
    
    # Get available tools
    tools = executor.get_tools()
    
    # Call a tool directly
    result = await executor.call_tool("list_directory", {"path": "/tmp"})
```
"""

import asyncio
from contextlib import asynccontextmanager, AsyncExitStack
from dataclasses import dataclass, field
from typing import (
    Any,
    AsyncIterator,
    Callable,
    Dict,
    List,
    Literal,
    Optional,
    TypedDict,
    Union,
)

# MCP SDK imports - the official Python SDK for Model Context Protocol
from mcp import ClientSession  # Core session class for MCP communication
from mcp.client.stdio import StdioServerParameters, stdio_client  # For subprocess-based servers
from mcp.client.sse import sse_client  # For Server-Sent Events transport
from mcp.client.streamable_http import streamablehttp_client  # For HTTP streaming transport
from mcp.types import Tool as MCPTool, TextContent as MCPTextContent  # Type definitions


# =============================================================================
# Type Definitions
# =============================================================================

# TransportType defines the communication method between client and server.
# Each transport has different use cases:
# - "stdio": Best for local servers, spawns a subprocess
# - "sse": HTTP-based, good for remote servers with Server-Sent Events
# - "streamable-http": Modern HTTP transport with streaming support
# - "websocket": Real-time bidirectional communication (future support)
TransportType = Literal["stdio", "sse", "streamable-http", "websocket"]


class MCPServerConfig(TypedDict, total=False):
    """
    Configuration dictionary for connecting to an MCP server.
    
    This TypedDict defines the shape of configuration objects used with
    the `connect()` method. Using `total=False` makes all fields optional,
    allowing flexibility in configuration.
    
    Fields:
        name: Unique identifier for this connection (required for connect())
        transport: Communication protocol ("stdio", "sse", "streamable-http")
        
        For stdio transport (subprocess-based):
            command: Executable to run (e.g., "npx", "python", "uv")
            args: Command-line arguments
            env: Environment variables to set for the subprocess
            
        For HTTP-based transports (sse, streamable-http):
            url: Server endpoint URL
            headers: HTTP headers (useful for authentication)
    
    Example:
        config = {
            "name": "filesystem",
            "transport": "stdio",
            "command": "npx",
            "args": ["-y", "@modelcontextprotocol/server-filesystem", "/tmp"]
        }
    """
    name: str
    transport: TransportType
    # For stdio transport
    command: str
    args: List[str]
    env: Dict[str, str]
    # For HTTP-based transports
    url: str
    # For authentication
    headers: Dict[str, str]


@dataclass
class MCPConnection:
    """
    Represents an active connection to an MCP server.
    
    This dataclass holds all state for a single MCP server connection,
    including the communication session and discovered capabilities.
    
    Attributes:
        name: Unique identifier for this connection
        session: The MCP ClientSession for communication
        tools: List of tools discovered from this server (in OpenAI format)
        resources: List of resources exposed by this server
    
    Note:
        Tools are automatically discovered and converted to OpenAI format
        when a connection is established. This allows seamless integration
        with the UnifiedChatClient's tool calling system.
    """
    name: str
    session: ClientSession
    tools: List[Dict[str, Any]] = field(default_factory=list)
    resources: List[Dict[str, Any]] = field(default_factory=list)


class MCPToolExecutor:
    """
    Manages MCP server connections and tool execution.
    
    This is the main class for interacting with MCP servers. It provides a
    unified interface for:
    - Connecting to multiple MCP servers simultaneously
    - Discovering available tools from each server
    - Executing tools and returning results
    - Converting between MCP and OpenAI tool formats
    
    Architecture Overview:
    ---------------------
    The MCPToolExecutor manages multiple connections using an AsyncExitStack,
    which ensures proper cleanup of all async resources when the context exits.
    
    Connection Flow:
    1. Create executor within async context: `async with MCPToolExecutor() as executor:`
    2. Connect to servers: `await executor.connect_stdio(...)`
    3. Tools are automatically discovered and registered
    4. Execute tools: `await executor.call_tool("tool_name", {...})`
    5. Cleanup happens automatically when context exits
    
    Tool Registration:
    -----------------
    When you connect to a server, its tools are:
    1. Fetched via the MCP protocol
    2. Converted to OpenAI format (the standard format used by LLMux)
    3. Registered in `_tool_map` for routing calls to the correct server
    
    IMPORTANT: Always use as an async context manager!
    -------------------------------------------------
    The MCP SDK uses anyio which requires proper async context management.
    Always use this pattern:
    
        async with MCPToolExecutor() as executor:
            await executor.connect_stdio(...)
            # ... use executor ...
        # Connections are automatically cleaned up here
    
    Or use the convenience function:
    
        async with mcp_executor() as executor:
            ...
    """
    
    def __init__(self):
        """
        Initialize the MCPToolExecutor.
        
        Note: The executor is NOT ready to use until you enter the async context.
        Internal state:
        - _connections: Maps server names to their MCPConnection objects
        - _tool_map: Maps tool names to their source server names
        - _exit_stack: Manages nested async context managers for cleanup
        """
        # Dictionary mapping connection names to MCPConnection objects
        # Example: {"filesystem": MCPConnection(...), "web": MCPConnection(...)}
        self._connections: Dict[str, MCPConnection] = {}
        
        # Dictionary mapping tool names to their source server names
        # This allows routing tool calls to the correct server
        # Example: {"read_file": "filesystem", "list_directory": "filesystem"}
        self._tool_map: Dict[str, str] = {}
        
        # AsyncExitStack manages multiple async context managers
        # It ensures proper cleanup of all resources in reverse order
        # Set to None until we enter the async context
        self._exit_stack: Optional[AsyncExitStack] = None
    
    async def __aenter__(self) -> "MCPToolExecutor":
        """
        Enter the async context manager.
        
        This method is called when you use `async with MCPToolExecutor() as executor:`.
        It initializes the AsyncExitStack which will manage all our nested
        context managers (transport connections, sessions, etc.).
        
        Returns:
            self: The executor instance, ready for use
        """
        self._exit_stack = AsyncExitStack()
        await self._exit_stack.__aenter__()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """
        Exit the async context manager and cleanup all connections.
        
        This method is called when exiting the `async with` block. It:
        1. Closes all transport connections via the AsyncExitStack
        2. Clears all connection and tool registrations
        
        The AsyncExitStack handles cleanup in reverse order, ensuring
        sessions are closed before their transports.
        
        Args:
            exc_type: Exception type if an error occurred, None otherwise
            exc_val: Exception value if an error occurred, None otherwise
            exc_tb: Exception traceback if an error occurred, None otherwise
        """
        if self._exit_stack:
            await self._exit_stack.__aexit__(exc_type, exc_val, exc_tb)
            self._exit_stack = None
        self._connections.clear()
        self._tool_map.clear()
    
    @property
    def connections(self) -> Dict[str, MCPConnection]:
        """
        Get all active connections.
        
        Returns:
            Dictionary mapping connection names to MCPConnection objects
        
        Example:
            for name, conn in executor.connections.items():
                print(f"{name}: {len(conn.tools)} tools")
        """
        return self._connections
    
    @property
    def tool_names(self) -> List[str]:
        """
        Get all available tool names from all connected servers.
        
        Returns:
            List of tool names (strings)
        
        Example:
            if "read_file" in executor.tool_names:
                result = await executor.call_tool("read_file", {"path": "/tmp/test.txt"})
        """
        return list(self._tool_map.keys())
    
    def get_tools(self) -> List[Dict[str, Any]]:
        """
        Get all available tools in OpenAI format.
        
        This method aggregates tools from all connected servers and returns
        them in OpenAI's tool format, which is used as the standard format
        throughout LLMux. This makes MCP tools directly compatible with
        the UnifiedChatClient's tool calling system.
        
        Returns:
            List of tool definitions in OpenAI format:
            [
                {
                    "type": "function",
                    "function": {
                        "name": "tool_name",
                        "description": "What the tool does",
                        "parameters": {...}  # JSON Schema
                    }
                },
                ...
            ]
        
        Example:
            tools = executor.get_tools()
            response = await client.chat(
                provider="openai",
                model="gpt-4o",
                messages=messages,
                tools=tools  # Pass MCP tools directly!
            )
        """
        all_tools = []
        for conn in self._connections.values():
            all_tools.extend(conn.tools)
        return all_tools
    
    def get_tools_for_server(self, server_name: str) -> List[Dict[str, Any]]:
        """
        Get tools for a specific server.
        
        Useful when you have multiple servers connected and want to
        inspect or use tools from a specific one.
        
        Args:
            server_name: Name of the connection (as specified in connect_*)
            
        Returns:
            List of tool definitions, or empty list if server not found
        """
        if server_name in self._connections:
            return self._connections[server_name].tools
        return []
    
    def _ensure_context(self):
        """
        Ensure we're inside an async context.
        
        This is a safety check that raises an error if someone tries to
        use the executor without the proper async context manager pattern.
        
        Raises:
            RuntimeError: If not inside an async context
        """
        if self._exit_stack is None:
            raise RuntimeError(
                "MCPToolExecutor must be used as an async context manager: "
                "async with MCPToolExecutor() as executor: ..."
            )
    
    async def connect_stdio(
        self,
        name: str,
        command: str,
        args: Optional[List[str]] = None,
        env: Optional[Dict[str, str]] = None,
    ) -> MCPConnection:
        """
        Connect to an MCP server via stdio transport.
        
        This is the most common transport for local MCP servers. It works by:
        1. Spawning a subprocess running the MCP server
        2. Communicating via the process's stdin/stdout
        3. Using JSON-RPC messages over these streams
        
        The stdio transport is ideal for:
        - Local development and testing
        - Running MCP servers packaged as npm packages (via npx)
        - Running Python-based MCP servers
        - Any executable that implements the MCP protocol
        
        Args:
            name: Unique name for this connection (used to identify tools)
            command: Command to run (e.g., "npx", "python", "uv", "node")
            args: Command arguments (e.g., ["run", "mcp-server-name"])
            env: Environment variables to set for the subprocess
            
        Returns:
            MCPConnection object with active session and discovered tools
            
        Raises:
            ValueError: If a connection with this name already exists
            RuntimeError: If not inside an async context
            
        Example:
            # Connect to the official filesystem MCP server via npx
            async with MCPToolExecutor() as executor:
                conn = await executor.connect_stdio(
                    name="filesystem",
                    command="npx",
                    args=["-y", "@modelcontextprotocol/server-filesystem", "/tmp"]
                )
                print(f"Connected! {len(conn.tools)} tools available")
                
            # Connect to a Python-based MCP server
            async with MCPToolExecutor() as executor:
                conn = await executor.connect_stdio(
                    name="my-server",
                    command="python",
                    args=["-m", "my_mcp_server"],
                    env={"DEBUG": "true"}
                )
        """
        # Ensure we're inside the async context manager
        self._ensure_context()
        
        # Check for duplicate connection names
        if name in self._connections:
            raise ValueError(f"Connection '{name}' already exists")
        
        # Create parameters for the stdio transport
        # StdioServerParameters configures how to spawn the subprocess
        server_params = StdioServerParameters(
            command=command,
            args=args or [],
            env=env,
        )
        
        # Use the exit stack to properly manage the context managers.
        # This is crucial for proper cleanup - the AsyncExitStack ensures
        # that when we exit our context, all nested contexts are properly
        # cleaned up in reverse order (session before transport).
        
        # Step 1: Start the subprocess and get communication streams
        # stdio_client is an async context manager that spawns the process
        read_stream, write_stream = await self._exit_stack.enter_async_context(
            stdio_client(server_params)
        )
        
        # Step 2: Create and initialize the MCP session
        # ClientSession handles the MCP protocol (JSON-RPC messages)
        session = await self._exit_stack.enter_async_context(
            ClientSession(read_stream, write_stream)
        )
        
        # Step 3: Initialize the session (performs MCP handshake)
        # This exchanges capabilities between client and server
        await session.initialize()
        
        # Step 4: Discover available tools from the server
        # The server responds with a list of tools it provides
        tools_response = await session.list_tools()
        
        # Step 5: Convert MCP tools to OpenAI format for compatibility
        tools = self._convert_mcp_tools(tools_response.tools, name)
        
        # Step 6: Register tools in our routing map
        # This allows call_tool() to route requests to the correct server
        for tool in tools:
            tool_name = tool["function"]["name"]
            self._tool_map[tool_name] = name
        
        # Step 7: Create and store the connection object
        connection = MCPConnection(
            name=name,
            session=session,
            tools=tools,
        )
        self._connections[name] = connection
        
        return connection
    
    async def connect_sse(
        self,
        name: str,
        url: str,
        headers: Optional[Dict[str, str]] = None,
    ) -> MCPConnection:
        """
        Connect to an MCP server via SSE (Server-Sent Events) transport.
        
        SSE is an HTTP-based transport where:
        - Client sends requests via HTTP POST
        - Server pushes responses via Server-Sent Events stream
        
        This transport is ideal for:
        - Remote MCP servers deployed as web services
        - Servers behind firewalls (HTTP is usually allowed)
        - Servers that need authentication headers
        
        Args:
            name: Unique name for this connection
            url: Server URL endpoint (e.g., "http://localhost:8000/mcp")
            headers: Optional HTTP headers (e.g., {"Authorization": "Bearer token"})
            
        Returns:
            MCPConnection object with active session and discovered tools
            
        Example:
            async with MCPToolExecutor() as executor:
                conn = await executor.connect_sse(
                    name="remote-api",
                    url="https://api.example.com/mcp",
                    headers={"Authorization": "Bearer your-api-key"}
                )
        """
        self._ensure_context()
        
        if name in self._connections:
            raise ValueError(f"Connection '{name}' already exists")
        
        # SSE client establishes HTTP connection with Server-Sent Events
        # The server will push messages to us over this persistent connection
        read_stream, write_stream = await self._exit_stack.enter_async_context(
            sse_client(url, headers=headers)
        )
        
        # Create and initialize MCP session over the SSE transport
        session = await self._exit_stack.enter_async_context(
            ClientSession(read_stream, write_stream)
        )
        await session.initialize()
        
        # Discover and register tools (same as stdio transport)
        tools_response = await session.list_tools()
        tools = self._convert_mcp_tools(tools_response.tools, name)
        
        for tool in tools:
            tool_name = tool["function"]["name"]
            self._tool_map[tool_name] = name
        
        connection = MCPConnection(
            name=name,
            session=session,
            tools=tools,
        )
        self._connections[name] = connection
        
        return connection
    
    async def connect_http(
        self,
        name: str,
        url: str,
        headers: Optional[Dict[str, str]] = None,
    ) -> MCPConnection:
        """
        Connect to an MCP server via streamable HTTP transport.
        
        This is a modern HTTP-based transport that supports streaming
        responses. It's similar to SSE but uses a different protocol
        for bidirectional streaming over HTTP.
        
        Args:
            name: Unique name for this connection
            url: Server URL endpoint
            headers: Optional HTTP headers
            
        Returns:
            MCPConnection object with active session and discovered tools
            
        Example:
            async with MCPToolExecutor() as executor:
                conn = await executor.connect_http(
                    name="streaming-server",
                    url="http://localhost:8080/mcp",
                )
        """
        self._ensure_context()
        
        if name in self._connections:
            raise ValueError(f"Connection '{name}' already exists")
        
        # Streamable HTTP client returns three values (third is metadata)
        read_stream, write_stream, _ = await self._exit_stack.enter_async_context(
            streamablehttp_client(url, headers=headers)
        )
        
        session = await self._exit_stack.enter_async_context(
            ClientSession(read_stream, write_stream)
        )
        await session.initialize()
        
        # Discover and register tools
        tools_response = await session.list_tools()
        tools = self._convert_mcp_tools(tools_response.tools, name)
        
        for tool in tools:
            tool_name = tool["function"]["name"]
            self._tool_map[tool_name] = name
        
        connection = MCPConnection(
            name=name,
            session=session,
            tools=tools,
        )
        self._connections[name] = connection
        
        return connection
    
    async def connect(self, config: MCPServerConfig) -> MCPConnection:
        """
        Connect to an MCP server using a configuration dictionary.
        
        This is a generic connection method that routes to the appropriate
        transport-specific method based on the config. It's useful for:
        - Dynamic configuration (e.g., from a config file)
        - Connecting to multiple servers from a list of configs
        
        Args:
            config: Server configuration dictionary with keys:
                - name: (required) Unique connection name
                - transport: "stdio", "sse", or "streamable-http" (default: "stdio")
                - Other keys depending on transport type
            
        Returns:
            MCPConnection object with active session
            
        Example:
            # Connect from config
            async with MCPToolExecutor() as executor:
                conn = await executor.connect({
                    "name": "filesystem",
                    "transport": "stdio",
                    "command": "npx",
                    "args": ["-y", "@modelcontextprotocol/server-filesystem", "/tmp"]
                })
                
            # Connect multiple servers from config list
            configs = load_mcp_configs("mcp_servers.json")
            async with MCPToolExecutor() as executor:
                for config in configs:
                    await executor.connect(config)
        """
        # Get transport type, defaulting to stdio (most common)
        transport = config.get("transport", "stdio")
        name = config["name"]
        
        # Route to the appropriate transport-specific method
        if transport == "stdio":
            return await self.connect_stdio(
                name=name,
                command=config["command"],
                args=config.get("args"),
                env=config.get("env"),
            )
        elif transport == "sse":
            return await self.connect_sse(
                name=name,
                url=config["url"],
                headers=config.get("headers"),
            )
        elif transport == "streamable-http":
            return await self.connect_http(
                name=name,
                url=config["url"],
                headers=config.get("headers"),
            )
        else:
            raise ValueError(f"Unsupported transport: {transport}")
    
    async def call_tool(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
    ) -> str:
        """
        Execute a tool on the appropriate MCP server.
        
        This method:
        1. Looks up which server provides the requested tool
        2. Sends the tool call to that server via MCP
        3. Extracts and returns the text content from the response
        
        The routing is automatic - you don't need to know which server
        provides which tool. Just call by tool name.
        
        Args:
            tool_name: Name of the tool to execute (e.g., "read_file")
            arguments: Tool arguments as a dictionary (e.g., {"path": "/tmp/test.txt"})
            
        Returns:
            Tool result as a string. For structured data, this is typically
            JSON or formatted text depending on the tool.
            
        Raises:
            ValueError: If the tool is not found in any connected server
            
        Example:
            # List directory contents
            result = await executor.call_tool(
                "list_directory",
                {"path": "/tmp"}
            )
            print(result)  # Shows directory listing
            
            # Read a file
            content = await executor.call_tool(
                "read_file",
                {"path": "/tmp/config.json"}
            )
        """
        # Look up which server provides this tool
        if tool_name not in self._tool_map:
            raise ValueError(f"Tool '{tool_name}' not found in any connected server")
        
        # Get the connection for this tool's server
        server_name = self._tool_map[tool_name]
        conn = self._connections[server_name]
        
        # Execute the tool via MCP protocol
        # The session handles serializing arguments and parsing the response
        result = await conn.session.call_tool(tool_name, arguments)
        
        # Extract text content from the MCP result
        # MCP tools can return multiple content blocks (text, images, etc.)
        # Here we extract and join all text content
        if result.content:
            texts = []
            for content in result.content:
                # Check for MCPTextContent type
                if isinstance(content, MCPTextContent):
                    texts.append(content.text)
                # Fallback for other content types with text attribute
                elif hasattr(content, "text"):
                    texts.append(content.text)
                # Last resort: convert to string
                else:
                    texts.append(str(content))
            return "\n".join(texts)
        
        return ""
    
    async def execute_tool_calls(
        self,
        tool_calls: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Execute multiple tool calls and return results formatted for LLM.
        
        This method is designed to process tool calls from an LLM response
        and return results in the format expected by the LLM for the next
        turn of conversation.
        
        Each tool call is executed and the result is formatted as a
        "tool" message with the appropriate tool_call_id.
        
        Args:
            tool_calls: List of tool calls from LLM response, each with:
                - id: Unique identifier for this tool call
                - name: Name of the tool to execute
                - arguments: Dictionary of arguments for the tool
            
        Returns:
            List of tool result messages, each with:
                - role: "tool"
                - tool_call_id: ID from the original tool call
                - content: Result string (or error message)
                
        Example:
            # Process tool calls from LLM response
            tool_calls = response.get("tool_calls", [])
            # [{"id": "call_123", "name": "read_file", "arguments": {"path": "/tmp/x"}}]
            
            results = await executor.execute_tool_calls(tool_calls)
            # [{"role": "tool", "tool_call_id": "call_123", "content": "file contents..."}]
            
            # Add results to messages and continue conversation
            messages.extend(results)
        """
        results = []
        for tc in tool_calls:
            tool_name = tc.get("name", "")
            arguments = tc.get("arguments", {})
            tool_id = tc.get("id", "")
            
            try:
                # Execute the tool
                result = await self.call_tool(tool_name, arguments)
                results.append({
                    "role": "tool",
                    "tool_call_id": tool_id,
                    "content": result,
                })
            except Exception as e:
                # On error, return the error message as the tool result
                # This allows the LLM to see what went wrong and potentially retry
                results.append({
                    "role": "tool",
                    "tool_call_id": tool_id,
                    "content": f"Error: {str(e)}",
                })
        
        return results
    
    def _convert_mcp_tools(
        self,
        mcp_tools: List[MCPTool],
        server_name: str,
    ) -> List[Dict[str, Any]]:
        """
        Convert MCP tool definitions to OpenAI format.
        
        MCP and OpenAI use different formats for tool definitions. This method
        converts MCP's format to OpenAI's format, which is used as the standard
        throughout LLMux.
        
        MCP Tool Format:
            {
                "name": "tool_name",
                "description": "What the tool does",
                "inputSchema": {...}  # JSON Schema for parameters
            }
            
        OpenAI Tool Format:
            {
                "type": "function",
                "function": {
                    "name": "tool_name",
                    "description": "What the tool does",
                    "parameters": {...}  # JSON Schema
                }
            }
        
        Args:
            mcp_tools: List of MCP tool definitions from server
            server_name: Name of the source server (stored as metadata)
            
        Returns:
            List of tools in OpenAI format
        """
        converted = []
        for tool in mcp_tools:
            converted.append({
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description or "",
                    # Use inputSchema if provided, otherwise empty object schema
                    "parameters": tool.inputSchema if tool.inputSchema else {
                        "type": "object",
                        "properties": {},
                    },
                },
                # Internal metadata - tracks which server provides this tool
                # This is stripped before sending to LLMs
                "_mcp_server": server_name,
            })
        return converted
    
    async def list_resources(self, server_name: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        List available resources from MCP servers.
        
        MCP servers can expose "resources" in addition to tools. Resources
        are data items that can be read, such as:
        - Files and directories
        - Database records
        - API responses
        - Configuration data
        
        Args:
            server_name: Optional server name to filter by. If None, lists
                        resources from all connected servers.
            
        Returns:
            List of resource definitions, each with:
                - uri: Resource identifier (e.g., "file:///tmp/config.json")
                - name: Human-readable name
                - server: Which server provides this resource
                
        Example:
            # List all resources
            resources = await executor.list_resources()
            for r in resources:
                print(f"{r['name']}: {r['uri']}")
                
            # List resources from specific server
            fs_resources = await executor.list_resources("filesystem")
        """
        resources = []
        
        if server_name:
            # Filter to specific server
            if server_name in self._connections:
                conn = self._connections[server_name]
                result = await conn.session.list_resources()
                resources.extend([
                    {"uri": str(r.uri), "name": r.name, "server": server_name}
                    for r in result.resources
                ])
        else:
            # List from all servers
            for name, conn in self._connections.items():
                try:
                    result = await conn.session.list_resources()
                    resources.extend([
                        {"uri": str(r.uri), "name": r.name, "server": name}
                        for r in result.resources
                    ])
                except Exception:
                    # Server may not support resources - skip silently
                    pass
        
        return resources
    
    async def read_resource(self, uri: str, server_name: Optional[str] = None) -> str:
        """
        Read content from an MCP resource.
        
        Resources are identified by URIs (e.g., "file:///path/to/file").
        If you don't specify which server to use, this method will try
        all connected servers until one successfully returns the resource.
        
        Args:
            uri: Resource URI (e.g., "file:///tmp/config.json")
            server_name: Optional server name. If not provided, tries all servers.
            
        Returns:
            Resource content as a string
            
        Raises:
            ValueError: If resource not found on any server
            
        Example:
            # Read a specific resource
            content = await executor.read_resource("file:///tmp/config.json")
            
            # Read from a specific server
            content = await executor.read_resource(
                "db://users/123",
                server_name="database"
            )
        """
        from mcp.types import AnyUrl
        
        # Determine which connections to try
        connections_to_try = (
            [self._connections[server_name]] if server_name
            else list(self._connections.values())
        )
        
        # Try each connection until we find the resource
        for conn in connections_to_try:
            try:
                result = await conn.session.read_resource(AnyUrl(uri))
                if result.contents:
                    content = result.contents[0]
                    # Extract text content
                    if hasattr(content, "text"):
                        return content.text
                    return str(content)
            except Exception:
                # Resource not found on this server, try next
                continue
        
        raise ValueError(f"Resource '{uri}' not found")


# =============================================================================
# Context Manager Helper
# =============================================================================

@asynccontextmanager
async def mcp_executor() -> AsyncIterator[MCPToolExecutor]:
    """
    Convenience context manager for MCPToolExecutor.
    
    This is a simple wrapper that provides a cleaner API for the common
    use case of creating an executor, using it, and cleaning up.
    
    This function:
        async with mcp_executor() as executor:
            ...
            
    Is equivalent to:
        async with MCPToolExecutor() as executor:
            ...
    
    Example:
        async with mcp_executor() as executor:
            # Connect to servers
            await executor.connect_stdio("fs", "npx", ["-y", "@mcp/server-filesystem"])
            
            # Get tools for LLM
            tools = executor.get_tools()
            
            # Call tools directly
            result = await executor.call_tool("read_file", {"path": "/tmp/test.txt"})
            
        # All connections automatically cleaned up here
    """
    async with MCPToolExecutor() as executor:
        yield executor


# =============================================================================
# Factory Functions
# =============================================================================

async def create_mcp_executor_from_configs(
    configs: List[MCPServerConfig],
) -> MCPToolExecutor:
    """
    Create an MCPToolExecutor and connect to multiple servers.
    
    ⚠️ WARNING: This function returns an executor that has already entered
    its async context. You MUST ensure proper cleanup by calling
    `await executor.__aexit__(None, None, None)` when done, or preferably
    use the context manager pattern instead.
    
    RECOMMENDED: Use MCPToolExecutor directly as a context manager instead:
    
        async with MCPToolExecutor() as executor:
            for config in configs:
                await executor.connect(config)
            # ... use executor ...
        # Automatic cleanup
    
    Args:
        configs: List of server configurations
        
    Returns:
        MCPToolExecutor with all servers connected
        
    Example (not recommended):
        executor = await create_mcp_executor_from_configs([...])
        try:
            # use executor
        finally:
            await executor.__aexit__(None, None, None)
            
    Example (recommended):
        async with MCPToolExecutor() as executor:
            for config in configs:
                await executor.connect(config)
    """
    executor = MCPToolExecutor()
    await executor.__aenter__()
    
    try:
        for config in configs:
            await executor.connect(config)
    except:
        # If any connection fails, clean up and re-raise
        await executor.__aexit__(None, None, None)
        raise
    
    return executor
