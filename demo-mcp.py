"""
Demo: MCP (Model Context Protocol) Integration with LLMux

This demonstrates how to use MCP servers with the UnifiedChatClient.
You can connect to multiple MCP servers and use their tools alongside
native tools in a unified way.

Requirements:
- npm/npx installed (for running MCP servers)
- Or any other MCP server you want to connect to

Example MCP servers:
- @modelcontextprotocol/server-filesystem (file operations)
- @modelcontextprotocol/server-fetch (web fetching)
- @modelcontextprotocol/server-memory (key-value store)
"""

import asyncio
from llmclient import UnifiedChatClient
from mcp_client import MCPToolExecutor, mcp_executor
from rich_llm_printer import RichPrinter


# =============================================================================
# Demo 1: Basic MCP Tool Discovery
# =============================================================================

async def demo_tool_discovery():
    """Demonstrate connecting to an MCP server and discovering tools."""
    print("=" * 60)
    print("Demo 1: MCP Tool Discovery")
    print("=" * 60)
    
    async with mcp_executor() as executor:
        try:
            # Connect to the filesystem server
            # This requires npx to be installed
            conn = await executor.connect_stdio(
                name="filesystem",
                command="npx",
                args=["-y", "@modelcontextprotocol/server-filesystem", "/tmp"]
            )
            
            print(f"\n✓ Connected to: {conn.name}")
            print(f"\nDiscovered {len(conn.tools)} tools:")
            for tool in conn.tools:
                name = tool["function"]["name"]
                desc = tool["function"]["description"][:60] + "..." if len(tool["function"]["description"]) > 60 else tool["function"]["description"]
                print(f"  - {name}: {desc}")
            
        except FileNotFoundError:
            print("\n⚠️  npx not found. Install Node.js to run MCP servers.")
        except Exception as e:
            print(f"\n⚠️  Error: {e}")


# =============================================================================
# Demo 2: Using MCP Tools with LLM
# =============================================================================

async def demo_mcp_with_llm(provider: str = "openai", model: str = "gpt-4o-mini"):
    """Use MCP tools with an LLM for natural language file operations."""
    print("\n" + "=" * 60)
    print("Demo 2: MCP Tools with LLM")
    print("=" * 60)
    
    client = UnifiedChatClient()
    printer = RichPrinter(title="MCP Demo", show_metadata=True)
    
    async with mcp_executor() as executor:
        try:
            # Connect to filesystem server
            await executor.connect_stdio(
                name="filesystem",
                command="npx",
                args=["-y", "@modelcontextprotocol/server-filesystem", "/tmp"]
            )
            
            print(f"\n✓ Connected to MCP server with {len(executor.tool_names)} tools")
            
            # Use chat_with_tools for automatic tool execution
            messages = [
                {
                    "role": "system",
                    "content": "You are a helpful assistant with access to file system tools. Use them to help the user.",
                },
                {
                    "role": "user",
                    "content": "List the files in /tmp directory and tell me what you find.",
                },
            ]
            
            print(f"\nUser: {messages[-1]['content']}")
            
            response = await client.chat_with_tools(
                provider=provider,
                model=model,
                messages=messages,
                mcp_executor=executor,
            )
            
            # Show tool execution history
            if response.get("tool_history"):
                print("\n📧 Tool Execution History:")
                for entry in response["tool_history"]:
                    print(f"  - {entry['tool']}({entry.get('arguments', {})})")
                    if "error" in entry:
                        print(f"    Error: {entry['error']}")
            
            print("\n")
            printer.print_chat(response)
            
        except FileNotFoundError:
            print("\n⚠️  npx not found. Install Node.js to run MCP servers.")
        except Exception as e:
            print(f"\n⚠️  Error: {e}")
            import traceback
            traceback.print_exc()


# =============================================================================
# Demo 3: Combining Native Tools with MCP Tools
# =============================================================================

async def demo_combined_tools(provider: str = "openai", model: str = "gpt-4o-mini"):
    """Combine native tools and MCP tools in a single conversation."""
    print("\n" + "=" * 60)
    print("Demo 3: Combined Native + MCP Tools")
    print("=" * 60)
    
    client = UnifiedChatClient()
    printer = RichPrinter(title="Combined Tools Demo")
    
    # Define a native tool
    native_tools = [
        client.create_tool(
            name="get_current_time",
            description="Get the current date and time",
            parameters={},
        ),
        client.create_tool(
            name="calculate",
            description="Perform a mathematical calculation",
            parameters={
                "expression": {
                    "type": "string",
                    "description": "Mathematical expression to evaluate (e.g., '2 + 2')"
                }
            },
            required=["expression"],
        ),
    ]
    
    # Define handlers for native tools
    def get_current_time(args):
        from datetime import datetime
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    def calculate(args):
        expression = args.get("expression", "")
        try:
            # Simple safe eval for basic math
            allowed = set("0123456789+-*/.() ")
            if all(c in allowed for c in expression):
                return str(eval(expression))
            return "Error: Invalid expression"
        except Exception as e:
            return f"Error: {e}"
    
    tool_handlers = {
        "get_current_time": get_current_time,
        "calculate": calculate,
    }
    
    async with mcp_executor() as executor:
        try:
            # Try to connect to MCP server (optional)
            try:
                await executor.connect_stdio(
                    name="filesystem",
                    command="npx",
                    args=["-y", "@modelcontextprotocol/server-filesystem", "/tmp"]
                )
                print(f"\n✓ Connected to MCP server")
            except:
                print("\n⚠️  MCP server not available, using only native tools")
            
            all_tools = native_tools + executor.get_tools()
            print(f"Total tools available: {len(all_tools)}")
            
            messages = [
                {
                    "role": "system",
                    "content": "You are a helpful assistant with access to various tools.",
                },
                {
                    "role": "user",
                    "content": "What time is it? Also calculate 15 * 24 for me.",
                },
            ]
            
            print(f"\nUser: {messages[-1]['content']}")
            
            response = await client.chat_with_tools(
                provider=provider,
                model=model,
                messages=messages,
                tools=native_tools,
                mcp_executor=executor if executor.tool_names else None,
                tool_handlers=tool_handlers,
            )
            
            if response.get("tool_history"):
                print("\n📧 Tool Execution History:")
                for entry in response["tool_history"]:
                    print(f"  - {entry['tool']}: {entry.get('result', entry.get('error', 'N/A'))[:50]}")
            
            print("\n")
            printer.print_chat(response)
            
        except Exception as e:
            print(f"\n⚠️  Error: {e}")
            import traceback
            traceback.print_exc()


# =============================================================================
# Demo 4: Multi-Server MCP Setup
# =============================================================================

async def demo_multi_server():
    """Connect to multiple MCP servers simultaneously."""
    print("\n" + "=" * 60)
    print("Demo 4: Multi-Server MCP Setup")
    print("=" * 60)
    
    async with mcp_executor() as executor:
        try:
            # Try connecting to multiple servers
            servers_to_try = [
                {
                    "name": "filesystem",
                    "transport": "stdio",
                    "command": "npx",
                    "args": ["-y", "@modelcontextprotocol/server-filesystem", "/tmp"]
                },
                # Add more servers as needed
                # {
                #     "name": "fetch",
                #     "transport": "stdio",
                #     "command": "npx",
                #     "args": ["-y", "@modelcontextprotocol/server-fetch"]
                # },
            ]
            
            for config in servers_to_try:
                try:
                    conn = await executor.connect(config)
                    print(f"\n✓ {config['name']}: {len(conn.tools)} tools")
                    for tool in conn.tools[:3]:  # Show first 3 tools
                        print(f"    - {tool['function']['name']}")
                    if len(conn.tools) > 3:
                        print(f"    ... and {len(conn.tools) - 3} more")
                except Exception as e:
                    print(f"\n⚠️  {config['name']}: Failed - {e}")
            
            print(f"\n📊 Total connected servers: {len(executor.connections)}")
            print(f"📧 Total available tools: {len(executor.tool_names)}")
            
        except Exception as e:
            print(f"\n⚠️  Error: {e}")


# =============================================================================
# Demo 5: Manual Tool Execution
# =============================================================================

async def demo_manual_execution():
    """Manually execute MCP tools without LLM."""
    print("\n" + "=" * 60)
    print("Demo 5: Manual MCP Tool Execution")
    print("=" * 60)
    
    async with mcp_executor() as executor:
        try:
            await executor.connect_stdio(
                name="filesystem",
                command="npx",
                args=["-y", "@modelcontextprotocol/server-filesystem", "/tmp"]
            )
            
            print("\n✓ Connected to filesystem server")
            
            # Manually call a tool
            print("\nCalling 'list_directory' tool...")
            result = await executor.call_tool(
                "list_directory",
                {"path": "/tmp"}
            )
            
            print(f"\nResult:\n{result[:500]}..." if len(result) > 500 else f"\nResult:\n{result}")
            
        except FileNotFoundError:
            print("\n⚠️  npx not found. Install Node.js to run MCP servers.")
        except Exception as e:
            print(f"\n⚠️  Error: {e}")


# =============================================================================
# Main
# =============================================================================

async def main():
    """Run all demos."""
    print("\n🚀 LLMux MCP Integration Demo\n")
    
    # Run demos
    await demo_tool_discovery()
    await demo_manual_execution()
    await demo_mcp_with_llm()
    await demo_combined_tools()
    await demo_multi_server()
    
    print("\n" + "=" * 60)
    print("✅ All demos completed!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
