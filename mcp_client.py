"""
MCP (Model Context Protocol) Client Wrapper for LLMux.

This module provides a unified interface for connecting to MCP servers
and integrating their tools with the UnifiedChatClient.

Supported transports:
- stdio: Connect to local MCP servers via subprocess
- sse: Connect to HTTP/SSE-based MCP servers
- streamable-http: Connect to streamable HTTP MCP servers
- websocket: Connect to WebSocket-based MCP servers
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

from mcp import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client
from mcp.client.sse import sse_client
from mcp.client.streamable_http import streamablehttp_client
from mcp.types import Tool as MCPTool, TextContent as MCPTextContent


# =============================================================================
# Type Definitions
# =============================================================================

TransportType = Literal["stdio", "sse", "streamable-http", "websocket"]


class MCPServerConfig(TypedDict, total=False):
    """Configuration for an MCP server connection."""
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
    """Represents an active MCP server connection."""
    name: str
    session: ClientSession
    tools: List[Dict[str, Any]] = field(default_factory=list)
    resources: List[Dict[str, Any]] = field(default_factory=list)


class MCPToolExecutor:
    """
    Manages MCP server connections and tool execution.
    
    This class provides a unified interface for:
    - Connecting to multiple MCP servers
    - Discovering available tools
    - Executing tools and returning results
    - Converting between MCP and LLMux tool formats
    
    IMPORTANT: Use the async context manager pattern:
        async with MCPToolExecutor() as executor:
            await executor.connect_stdio(...)
    """
    
    def __init__(self):
        self._connections: Dict[str, MCPConnection] = {}
        self._tool_map: Dict[str, str] = {}  # tool_name -> server_name
        self._exit_stack: Optional[AsyncExitStack] = None
    
    async def __aenter__(self) -> "MCPToolExecutor":
        """Enter async context."""
        self._exit_stack = AsyncExitStack()
        await self._exit_stack.__aenter__()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Exit async context and cleanup all connections."""
        if self._exit_stack:
            await self._exit_stack.__aexit__(exc_type, exc_val, exc_tb)
            self._exit_stack = None
        self._connections.clear()
        self._tool_map.clear()
    
    @property
    def connections(self) -> Dict[str, MCPConnection]:
        """Get all active connections."""
        return self._connections
    
    @property
    def tool_names(self) -> List[str]:
        """Get all available tool names."""
        return list(self._tool_map.keys())
    
    def get_tools(self) -> List[Dict[str, Any]]:
        """
        Get all available tools in OpenAI format.
        
        Returns:
            List of tool definitions compatible with UnifiedChatClient
        """
        all_tools = []
        for conn in self._connections.values():
            all_tools.extend(conn.tools)
        return all_tools
    
    def get_tools_for_server(self, server_name: str) -> List[Dict[str, Any]]:
        """Get tools for a specific server."""
        if server_name in self._connections:
            return self._connections[server_name].tools
        return []
    
    def _ensure_context(self):
        """Ensure we're inside an async context."""
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
        
        Args:
            name: Unique name for this connection
            command: Command to run (e.g., "uv", "python", "npx")
            args: Command arguments (e.g., ["run", "mcp-server"])
            env: Environment variables to set
            
        Returns:
            MCPConnection object with active session
            
        Example:
            >>> async with MCPToolExecutor() as executor:
            ...     conn = await executor.connect_stdio(
            ...         name="filesystem",
            ...         command="npx",
            ...         args=["-y", "@modelcontextprotocol/server-filesystem", "/tmp"]
            ...     )
        """
        self._ensure_context()
        
        if name in self._connections:
            raise ValueError(f"Connection '{name}' already exists")
        
        server_params = StdioServerParameters(
            command=command,
            args=args or [],
            env=env,
        )
        
        # Use the exit stack to properly manage the context managers
        read_stream, write_stream = await self._exit_stack.enter_async_context(
            stdio_client(server_params)
        )
        session = await self._exit_stack.enter_async_context(
            ClientSession(read_stream, write_stream)
        )
        await session.initialize()
        
        # Discover tools
        tools_response = await session.list_tools()
        tools = self._convert_mcp_tools(tools_response.tools, name)
        
        # Register tools
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
    
    async def connect_sse(
        self,
        name: str,
        url: str,
        headers: Optional[Dict[str, str]] = None,
    ) -> MCPConnection:
        """
        Connect to an MCP server via SSE transport.
        
        Args:
            name: Unique name for this connection
            url: Server URL (e.g., "http://localhost:8000/mcp")
            headers: Optional HTTP headers
            
        Returns:
            MCPConnection object with active session
        """
        self._ensure_context()
        
        if name in self._connections:
            raise ValueError(f"Connection '{name}' already exists")
        
        # Use the exit stack to properly manage the context managers
        read_stream, write_stream = await self._exit_stack.enter_async_context(
            sse_client(url, headers=headers)
        )
        session = await self._exit_stack.enter_async_context(
            ClientSession(read_stream, write_stream)
        )
        await session.initialize()
        
        # Discover tools
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
        
        Args:
            name: Unique name for this connection
            url: Server URL (e.g., "http://localhost:8000/mcp")
            headers: Optional HTTP headers
            
        Returns:
            MCPConnection object with active session
        """
        self._ensure_context()
        
        if name in self._connections:
            raise ValueError(f"Connection '{name}' already exists")
        
        # Use the exit stack to properly manage the context managers
        read_stream, write_stream, _ = await self._exit_stack.enter_async_context(
            streamablehttp_client(url, headers=headers)
        )
        session = await self._exit_stack.enter_async_context(
            ClientSession(read_stream, write_stream)
        )
        await session.initialize()
        
        # Discover tools
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
        Connect to an MCP server using the provided configuration.
        
        Args:
            config: Server configuration dictionary
            
        Returns:
            MCPConnection object with active session
            
        Example:
            >>> async with MCPToolExecutor() as executor:
            ...     conn = await executor.connect({
            ...         "name": "filesystem",
            ...         "transport": "stdio",
            ...         "command": "npx",
            ...         "args": ["-y", "@modelcontextprotocol/server-filesystem", "/tmp"]
            ...     })
        """
        transport = config.get("transport", "stdio")
        name = config["name"]
        
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
        
        Args:
            tool_name: Name of the tool to call
            arguments: Tool arguments
            
        Returns:
            Tool result as a string
            
        Raises:
            ValueError: If tool is not found
        """
        if tool_name not in self._tool_map:
            raise ValueError(f"Tool '{tool_name}' not found in any connected server")
        
        server_name = self._tool_map[tool_name]
        conn = self._connections[server_name]
        
        result = await conn.session.call_tool(tool_name, arguments)
        
        # Extract text content from result
        if result.content:
            texts = []
            for content in result.content:
                if isinstance(content, MCPTextContent):
                    texts.append(content.text)
                elif hasattr(content, "text"):
                    texts.append(content.text)
                else:
                    texts.append(str(content))
            return "\n".join(texts)
        
        return ""
    
    async def execute_tool_calls(
        self,
        tool_calls: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Execute multiple tool calls and return results.
        
        Args:
            tool_calls: List of tool calls from LLM response
            
        Returns:
            List of tool result messages for sending back to LLM
        """
        results = []
        for tc in tool_calls:
            tool_name = tc.get("name", "")
            arguments = tc.get("arguments", {})
            tool_id = tc.get("id", "")
            
            try:
                result = await self.call_tool(tool_name, arguments)
                results.append({
                    "role": "tool",
                    "tool_call_id": tool_id,
                    "content": result,
                })
            except Exception as e:
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
        Convert MCP tools to OpenAI format.
        
        Args:
            mcp_tools: List of MCP tool definitions
            server_name: Name of the server (for namespacing)
            
        Returns:
            List of tools in OpenAI format
        """
        converted = []
        for tool in mcp_tools:
            # Use server_name prefix if tool name might conflict
            # For now, use raw tool name
            converted.append({
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description or "",
                    "parameters": tool.inputSchema if tool.inputSchema else {
                        "type": "object",
                        "properties": {},
                    },
                },
                "_mcp_server": server_name,  # Internal metadata
            })
        return converted
    
    async def list_resources(self, server_name: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        List available resources from MCP servers.
        
        Args:
            server_name: Optional server name to filter by
            
        Returns:
            List of resource definitions
        """
        resources = []
        
        if server_name:
            if server_name in self._connections:
                conn = self._connections[server_name]
                result = await conn.session.list_resources()
                resources.extend([
                    {"uri": str(r.uri), "name": r.name, "server": server_name}
                    for r in result.resources
                ])
        else:
            for name, conn in self._connections.items():
                try:
                    result = await conn.session.list_resources()
                    resources.extend([
                        {"uri": str(r.uri), "name": r.name, "server": name}
                        for r in result.resources
                    ])
                except Exception:
                    pass  # Server may not support resources
        
        return resources
    
    async def read_resource(self, uri: str, server_name: Optional[str] = None) -> str:
        """
        Read a resource from an MCP server.
        
        Args:
            uri: Resource URI
            server_name: Optional server name (auto-detected if not provided)
            
        Returns:
            Resource content as string
        """
        from mcp.types import AnyUrl
        
        # If server not specified, try all connections
        connections_to_try = (
            [self._connections[server_name]] if server_name
            else list(self._connections.values())
        )
        
        for conn in connections_to_try:
            try:
                result = await conn.session.read_resource(AnyUrl(uri))
                if result.contents:
                    content = result.contents[0]
                    if hasattr(content, "text"):
                        return content.text
                    return str(content)
            except Exception:
                continue
        
        raise ValueError(f"Resource '{uri}' not found")


# =============================================================================
# Context Manager Alias for Easy Usage
# =============================================================================

@asynccontextmanager
async def mcp_executor() -> AsyncIterator[MCPToolExecutor]:
    """
    Context manager for MCPToolExecutor with automatic cleanup.
    
    Example:
        async with mcp_executor() as executor:
            await executor.connect_stdio("fs", "npx", ["-y", "@modelcontextprotocol/server-filesystem"])
            tools = executor.get_tools()
            result = await executor.call_tool("read_file", {"path": "/tmp/test.txt"})
    """
    async with MCPToolExecutor() as executor:
        yield executor


# =============================================================================
# Convenience Functions
# =============================================================================

async def create_mcp_executor_from_configs(
    configs: List[MCPServerConfig],
) -> MCPToolExecutor:
    """
    Create an MCPToolExecutor and connect to multiple servers.
    
    NOTE: The returned executor must be used within an async context.
    Prefer using MCPToolExecutor directly as a context manager.
    
    Args:
        configs: List of server configurations
        
        
    Example:
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
        await executor.__aexit__(None, None, None)
        raise
    
    return executor
