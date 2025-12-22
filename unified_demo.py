#!/usr/bin/env python3
"""
LLMux Unified Demo
==================

This is a monolithic demo file that combines all functionalities of the LLMux project
into a single, self-contained script.

It includes:
1. Streaming & Interaction Demos
2. Image & Vision Demos
3. Tool Calling (Function Calling) Demos
4. MCP (Model Context Protocol) Integration

Usage:
    python unified_demo.py
"""

import asyncio
import json
import os
import sys
from pathlib import Path
from typing import List, Dict, Any, Callable, Optional

# Third-party imports
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.table import Table
from rich import inspect

# Local imports
from llmux import UnifiedChatClient, Message, RichPrinter, RichStreamPrinter, mcp_executor

# Initialize global console
console = Console()


# =============================================================================
# PART 1: STREAMING DEMOS
# From: demo streaming.py
# =============================================================================

async def demo_basic_stream():
    """Basic demo using UnifiedChatClient directly."""
    console.print("[bold cyan]=== Basic Stream Demo ===")
    
    client = UnifiedChatClient()
    
    messages: List[Message] = [
        {"role": "system", "content": "You are a concise assistant."},
        {"role": "user", "content": "introduce yourself in one sentence using markdown syntax."},
    ]
    
    # Create event stream using your UnifiedChatClient
    event_stream = client.astream(
        provider="deepseek",
        model="deepseek-chat",
        messages=messages,
        temperature=0.7,
        max_tokens=500
    )
    
    # Create and use printer
    printer = RichStreamPrinter(
        title="Markdown Demo",
        show_metadata=True,
        code_theme="dracula",
        refresh_rate=40
    )
    
    final_event = await printer.print_stream(event_stream)
    inspect(final_event)


async def demo_interactive_conversation():
    """Demo interactive conversation with streaming."""
    console.print("[bold cyan]\n=== Interactive Conversation ===")
    console.print("\n type exit to return to main menu")
    
    client = UnifiedChatClient()
    printer = RichStreamPrinter(
        title="Assistant",
        show_metadata=False,
        border_style="cyan"
    )
    
    conversation_history: List[Message] = [
        {"role": "system", "content": "You are a friendly and knowledgeable assistant."}
    ]
    
    while True:
        # Get user input
        console.print("\n[bold yellow]You:[/bold yellow]", end=" ")
        # Using standard input() here for simplicity in async context
        user_input = await asyncio.to_thread(input)
        user_input = user_input.strip()
        
        if user_input.lower() in ['exit', 'quit', 'bye']:
            console.print("[green]Goodbye![/green]")
            break
        
        # Add to conversation
        conversation_history.append({"role": "user", "content": user_input})
        
        # Stream response
        event_stream = client.astream(
            provider="deepseek",
            model="deepseek-chat",
            messages=conversation_history,
            temperature=0.7,
            max_tokens=1000
        )
        
        final_event = await printer.print_stream(event_stream)
        
        # Add assistant response to conversation
        if "full_text" in final_event:
            conversation_history.append({
                "role": "assistant", 
                "content": final_event["full_text"]
            })


# =============================================================================
# PART 2: IMAGE & VISION DEMOS
# From: demo-image.py
# =============================================================================

CAT_IMAGE_URL = "https://images.unsplash.com/photo-1514888286974-6c03e2ca1dba?w=400"
DOG_IMAGE_URL = "https://images.unsplash.com/photo-1587300003388-59208cc962cb?w=400"

async def demo_with_url():
    """Demo: Analyze an image from URL (OpenAI)."""
    console.print(Panel("Demo: Image from URL (OpenAI)", style="magenta"))

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
    console.print(Panel("Demo: Image with Gemini", style="magenta"))

    client = UnifiedChatClient()
    printer = RichPrinter()

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
    console.print(Panel("Demo: Image with Claude", style="magenta"))

    client = UnifiedChatClient()
    printer = RichPrinter()

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
    console.print(Panel("Demo: Streaming with Image (OpenAI)", style="magenta"))

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
    console.print(Panel("Demo: Local Image File", style="magenta"))

    # Check for a sample image
    sample_paths = ["sample.jpg", "sample.png", "test.jpg", "test.png"]
    image_path = None
    for p in sample_paths:
        if Path(p).exists():
            image_path = p
            break

    if not image_path:
        console.print("[yellow]No local image found.[/yellow] To test with a local file:")
        console.print("  1. Add an image file named 'sample.jpg', 'test.jpg' etc.")
        console.print("  2. Run this demo again")
        console.print("\nSkipping local file demo...")
        return

    client = UnifiedChatClient()
    printer = RichPrinter()

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
    console.print(Panel("Demo: Multiple Images", style="magenta"))

    client = UnifiedChatClient()
    printer = RichPrinter()

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
    console.print(Panel("Demo: Direct Base64 Image", style="magenta"))

    client = UnifiedChatClient()
    printer = RichPrinter()

    # Download and encode image manually (to demonstrate base64 usage)
    b64_data, mime_type = await client.encode_image_url(CAT_IMAGE_URL)
    
    console.print(f"Encoded image: {len(b64_data)} bytes base64, {mime_type}")

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


# =============================================================================
# PART 3: TOOL CALLING DEMOS
# From: demo-tools.py
# =============================================================================

# --- Mock Tools ---

def get_weather(location: str, unit: str = "celsius") -> dict:
    """Mock weather function - returns fake weather data."""
    weather_data = {
        "Paris": {"temp": 18, "condition": "Partly cloudy"},
        "London": {"temp": 14, "condition": "Rainy"},
        "Tokyo": {"temp": 22, "condition": "Sunny"},
        "New York": {"temp": 20, "condition": "Clear"},
    }
    
    data = weather_data.get(location, {"temp": 15, "condition": "Unknown"})
    temp = data["temp"]
    if unit == "fahrenheit":
        temp = int(temp * 9/5 + 32)
    
    return {
        "location": location,
        "temperature": temp,
        "unit": unit,
        "condition": data["condition"],
    }

def calculate(expression: str) -> dict:
    """Mock calculator - evaluates simple math expressions."""
    try:
        # WARNING: eval is dangerous in production - this is just for demo
        result = eval(expression, {"__builtins__": {}}, {})
        return {"expression": expression, "result": result}
    except Exception as e:
        return {"expression": expression, "error": str(e)}

def search_web(query: str) -> dict:
    """Mock web search - returns fake search results."""
    return {
        "query": query,
        "results": [
            {"title": f"Result 1 for '{query}'", "url": "https://example.com/1"},
            {"title": f"Result 2 for '{query}'", "url": "https://example.com/2"},
        ],
    }

TOOL_FUNCTIONS = {
    "get_weather": get_weather,
    "calculate": calculate,
    "search_web": search_web,
}

def execute_tool(name: str, arguments: dict) -> str:
    """Execute a tool and return the result as a string."""
    if name not in TOOL_FUNCTIONS:
        return json.dumps({"error": f"Unknown tool: {name}"})
    
    try:
        result = TOOL_FUNCTIONS[name](**arguments)
        return json.dumps(result)
    except Exception as e:
        return json.dumps({"error": str(e)})

# --- Tool Demos ---

async def demo_single_tool_call(client: UnifiedChatClient, provider: str, model: str):
    """Demo: Single tool call (weather)"""
    console.print(Panel(f"Demo: Single Tool Call ({provider})", style="green"))
    
    weather_tool = client.create_tool(
        name="get_weather",
        description="Get the current weather for a location",
        parameters={
            "location": {"type": "string", "description": "City name"},
            "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
        },
        required=["location"],
    )
    
    messages = [{"role": "user", "content": "What's the weather like in Paris?"}]
    console.print(f"\nUser: {messages[0]['content']}")
    
    # First call
    response = await client.chat(
        provider=provider, model=model, messages=messages, tools=[weather_tool]
    )
    
    console.print(f"\nLLM Response: {response['text']}")
    
    if response.get("tool_calls"):
        console.print("[bold]  Tool Calls:[/bold]")
        messages.append(client.create_assistant_message_with_tool_calls(
            response["text"] or "", response["tool_calls"]
        ))
        
        for tc in response["tool_calls"]:
            console.print(f"    - {tc['name']}({tc['arguments']})")
            result = execute_tool(tc["name"], tc["arguments"])
            console.print(f"    - Result: {result}")
            messages.append(client.create_tool_result(tc["id"], result))
        
        final_response = await client.chat(
            provider=provider, model=model, messages=messages, tools=[weather_tool]
        )
        console.print(f"\nFinal Response: {final_response['text']}")


async def demo_multiple_tools(client: UnifiedChatClient, provider: str, model: str):
    """Demo: Multiple tools available"""
    console.print(Panel(f"Demo: Multiple Tools ({provider})", style="green"))
    
    tools = [
        client.create_tool(name="get_weather", description="Get weather", 
                           parameters={"location": {"type": "string"}}, required=["location"]),
        client.create_tool(name="calculate", description="Evaluate math", 
                           parameters={"expression": {"type": "string"}}, required=["expression"]),
        client.create_tool(name="search_web", description="Search web", 
                           parameters={"query": {"type": "string"}}, required=["query"]),
    ]
    
    messages = [{"role": "user", "content": "What's 15% of 200, and what's the weather in Tokyo?"}]
    console.print(f"\nUser: {messages[0]['content']}")
    
    response = await client.chat(
        provider=provider, model=model, messages=messages, tools=tools
    )
    
    if response.get("tool_calls"):
        tool_results = []
        messages.append(client.create_assistant_message_with_tool_calls(
            response["text"] or "", response["tool_calls"]
        ))
        for tc in response["tool_calls"]:
            console.print(f"  Tool Call: {tc['name']}({tc['arguments']})")
            result = execute_tool(tc["name"], tc["arguments"])
            tool_results.append(client.create_tool_result(tc["id"], result))
        
        messages.extend(tool_results)
        final_response = await client.chat(
            provider=provider, model=model, messages=messages, tools=tools
        )
        console.print(f"\nFinal Response: {final_response['text']}")


async def demo_tool_choice(client: UnifiedChatClient, provider: str, model: str):
    """Demo: Forcing tool use"""
    console.print(Panel(f"Demo: Tool Choice ({provider})", style="green"))
    
    calc_tool = client.create_tool(name="calculate", description="Evaluate math", 
                                   parameters={"expression": {"type": "string"}}, required=["expression"])
    
    messages = [{"role": "user", "content": "What is 7 times 8?"}]
    console.print(f"\nUser: {messages[0]['content']}")
    console.print("(Forcing tool use with tool_choice='required')")
    
    response = await client.chat(
        provider=provider, model=model, messages=messages, 
        tools=[calc_tool], tool_choice="required"
    )
    
    if response.get("tool_calls"):
        for tc in response["tool_calls"]:
            console.print(f"  Tool Call: {tc['name']}({tc['arguments']})")
            result = execute_tool(tc["name"], tc["arguments"])
            console.print(f"  Result: {result}")


async def demo_no_tool_needed(client: UnifiedChatClient, provider: str, model: str):
    """Demo: LLM chooses not to use tools"""
    console.print(Panel(f"Demo: No Tool Needed ({provider})", style="green"))
    
    weather_tool = client.create_tool(name="get_weather", description="Get weather",
                                      parameters={"location": {"type": "string"}}, required=["location"])
    
    messages = [{"role": "user", "content": "What is the capital of France?"}]
    
    response = await client.chat(
        provider=provider, model=model, messages=messages, tools=[weather_tool]
    )
    console.print(f"\nLLM Response: {response['text']}")


# =============================================================================
# PART 4: MCP DEMOS
# From: demo-mcp.py
# =============================================================================

async def demo_tool_discovery():
    """Demonstrate connecting to an MCP server and discovering tools."""
    console.print(Panel("Demo 1: MCP Tool Discovery", style="blue"))
    
    async with mcp_executor() as executor:
        try:
            conn = await executor.connect_stdio(
                name="filesystem",
                command="npx",
                args=["-y", "@modelcontextprotocol/server-filesystem", "/tmp"]
            )
            console.print(f"\n[green]Connected to: {conn.name}[/green]")
            console.print(f"Discovered {len(conn.tools)} tools:")
            for tool in conn.tools:
                console.print(f"  - {tool['function']['name']}")
        except FileNotFoundError:
            console.print("[red]npx not found. Install Node.js to run MCP servers.[/red]")
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")


async def demo_mcp_with_llm(provider: str = "openai", model: str = "gpt-4o-mini"):
    """Use MCP tools with an LLM for natural language file operations."""
    console.print(Panel("Demo 2: MCP Tools with LLM", style="blue"))
    
    client = UnifiedChatClient()
    printer = RichPrinter(title="MCP Demo", show_metadata=True)
    
    async with mcp_executor() as executor:
        try:
            await executor.connect_stdio(
                name="filesystem",
                command="npx",
                args=["-y", "@modelcontextprotocol/server-filesystem", "/tmp"]
            )
            
            messages = [
                {"role": "system", "content": "You are a helpful assistant with access to file system tools."},
                {"role": "user", "content": "List the files in /tmp directory and tell me what you find."},
            ]
            
            resp = await client.chat_with_tools(
                provider=provider, model=model, messages=messages, mcp_executor=executor,
            )
            printer.print_chat(resp)
            
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")


async def demo_combined_tools(provider: str = "openai", model: str = "gpt-4o-mini"):
    """Combine native tools and MCP tools."""
    console.print(Panel("Demo 3: Combined Native + MCP Tools", style="blue"))
    
    client = UnifiedChatClient()
    printer = RichPrinter(title="Combined Tools Demo")
    
    # Native tool defs
    native_tools = [
        client.create_tool(name="get_current_time", description="Get current time", parameters={}),
        client.create_tool(name="calculate", description="Calculate", parameters={"expression": {"type": "string"}}, required=["expression"]),
    ]
    
    # Native tool handlers
    def get_current_time(args):
        from datetime import datetime
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Reuse execution logic from above for calculate
    tool_handlers = {"get_current_time": get_current_time, "calculate": lambda args: str(calculate(args.get("expression", "")))}

    async with mcp_executor() as executor:
        try:
            try:
                await executor.connect_stdio(name="filesystem", command="npx", args=["-y", "@modelcontextprotocol/server-filesystem", "/tmp"])
            except:
                console.print("[yellow]MCP server not available, using only native tools[/yellow]")
            
            messages = [
                {"role": "system", "content": "You are a helpful assistant with tools."},
                {"role": "user", "content": "What time is it? Also calculate 15 * 24."},
            ]
            
            resp = await client.chat_with_tools(
                provider=provider, model=model, messages=messages, tools=native_tools,
                mcp_executor=executor if executor.tool_names else None, tool_handlers=tool_handlers
            )
            printer.print_chat(resp)
            
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")


async def demo_multi_server():
    """Connect to multiple MCP servers."""
    console.print(Panel("Demo 4: Multi-Server MCP Setup", style="blue"))
    
    async with mcp_executor() as executor:
        try:
            servers = [
                {"name": "filesystem", "transport": "stdio", "command": "npx", "args": ["-y", "@modelcontextprotocol/server-filesystem", "/tmp"]}
            ]
            for config in servers:
                try:
                    conn = await executor.connect(config)
                    console.print(f"[green]✓ {config['name']}: {len(conn.tools)} tools[/green]")
                except Exception as e:
                    console.print(f"[red]Failed to connect to {config['name']}: {e}[/red]")
        except Exception as e:
             console.print(f"[red]Error: {e}[/red]")


async def demo_manual_execution():
    """Manually execute MCP tools."""
    console.print(Panel("Demo 5: Manual MCP Tool Execution", style="blue"))
    
    async with mcp_executor() as executor:
        try:
            await executor.connect_stdio(name="filesystem", command="npx", args=["-y", "@modelcontextprotocol/server-filesystem", "/tmp"])
            
            console.print("Calling 'list_directory' tool...")
            result = await executor.call_tool("list_directory", {"path": "/tmp"})
            console.print(f"Result len: {len(str(result))}")
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")


# =============================================================================
# PART 5: UTILITIES
# =============================================================================

async def demo_list_models():
    """List available models for configured providers."""
    from rich.columns import Columns
    from rich.text import Text
    
    console.print(Panel("Utilities: List Available Models", style="cyan"))
    
    client = UnifiedChatClient()
    providers = ["gemini", "openai", "deepseek", "anthropic", "huggingface"]
    
    with console.status("[bold green]Fetching models...[/bold green]"):
        results = {}
        for provider in providers:
            try:
                models = await client.list_models(provider)
                results[provider] = models
            except Exception as e:
                # Don't fail completely if one provider is missing/fails
                results[provider] = f"Error: {str(e)}"

    for provider, data in results.items():
        if isinstance(data, list) and data:
            # Sort models for better readability
            data.sort()
            
            # Create a simple list of Text objects for Columns
            items = [Text(m, style="green") for m in data]
            
            console.print(Panel(
                Columns(items, equal=True, expand=True), 
                title=f"[bold]{provider.title()}[/bold] ({len(data)} models)",
                border_style="green"
            ))
        elif isinstance(data, list) and not data:
             console.print(Panel(
                "[yellow]No models found (empty list returned).[/yellow]", 
                title=f"[bold]{provider.title()}[/bold]",
                border_style="yellow"
            ))
        else:
            console.print(Panel(
                f"[red]{data}[/red]", 
                title=f"[bold]{provider.title()}[/bold] (Error/Not Configured)",
                border_style="red"
            ))

# =============================================================================
# PART 6: MAIN MENU RUNNER
# =============================================================================

async def run_menu():
    """Main application loop."""
    
    # Setup for Tool Demos
    client = UnifiedChatClient()
    defaults = {"provider": "openai", "model": "gpt-4o-mini"}
    
    async def tools_wrapper(func):
        await func(client, defaults["provider"], defaults["model"])

    while True:
        console.clear()
        console.print(Panel.fit(
            "[bold white]LLMux[/bold white] [bold yellow]Unified Demo[/bold yellow]", 
            subtitle="Monolithic Demo File",
            border_style="yellow"
        ))
        
        # Define menu structure
        menu_structure = {
            "1": ("Streaming & Interaction", [
                ("1", "Basic Stream", demo_basic_stream),
                ("2", "Interactive Chat", demo_interactive_conversation),
            ]),
            "2": ("Image & Vision", [
                ("1", "Image from URL", demo_with_url),
                ("2", "Gemini Vision", demo_with_gemini),
                ("3", "Claude Vision", demo_with_claude),
                ("4", "Streaming Image", demo_streaming_with_image),
                ("5", "Local File", demo_local_file),
                ("6", "Multiple Images", demo_multiple_images),
                ("7", "Base64 Direct", demo_base64_direct),
            ]),
            "3": ("Tool Calling", [
                ("1", "Single Tool (Weather)", lambda: tools_wrapper(demo_single_tool_call)),
                ("2", "Multiple Tools", lambda: tools_wrapper(demo_multiple_tools)),
                ("3", "Forced Tool Choice", lambda: tools_wrapper(demo_tool_choice)),
                ("4", "No Tool Needed", lambda: tools_wrapper(demo_no_tool_needed)),
            ]),
            "4": ("MCP Integration", [
                ("1", "Tool Discovery", demo_tool_discovery),
                ("2", "Manual Execution", demo_manual_execution),
                ("3", "MCP with LLM", demo_mcp_with_llm),
                ("4", "Combined Tools", demo_combined_tools),
                ("5", "Multi-Server", demo_multi_server),
            ]),
            "5": ("Utilities", [
                ("1", "List Models", demo_list_models),
            ]),
            "0": ("Exit", None)
        }
        
        # Display Main Menu
        table = Table(show_header=False, box=None)
        table.add_column("Key", style="bold cyan", width=4)
        table.add_column("Description")
        
        for key, (desc, _) in menu_structure.items():
            table.add_row(f"[{key}]", desc)
        
        console.print(table)
        main_choice = Prompt.ask("Select a category", choices=list(menu_structure.keys()), default="0")
        
        if main_choice == "0":
            console.print("[green]Goodbye![/green]")
            break
            
        # Handle Sub-menu
        category_name, sub_items = menu_structure[main_choice]
        
        while True:
            console.clear()
            console.print(Panel.fit(f"[bold]{category_name}[/bold]", border_style="cyan"))
            
            sub_table = Table(show_header=False, box=None)
            sub_table.add_column("Key", style="bold cyan", width=4)
            sub_table.add_column("Description")
            
            for key, desc, _ in sub_items:
                sub_table.add_row(f"[{key}]", desc)
            sub_table.add_row("[0]", "Back to Main Menu")
            
            console.print(sub_table)
            
            sub_choice_keys = [item[0] for item in sub_items] + ["0"]
            sub_choice = Prompt.ask("Select an option", choices=sub_choice_keys, default="0")
            
            if sub_choice == "0":
                break
                
            selected_func = next(item[2] for item in sub_items if item[0] == sub_choice)
            
            # Run Function
            console.clear()
            console.print(f"[bold]Running demo...[/bold]\n")
            try:
                if asyncio.iscoroutinefunction(selected_func) or (hasattr(selected_func, '__call__') and asyncio.iscoroutinefunction(selected_func.__call__)):
                    await selected_func()
                else:
                    res = selected_func()
                    if asyncio.iscoroutine(res):
                        await res
            except Exception as e:
                console.print(f"\n[bold red]Error:[/bold red] {e}")
                import traceback
                traceback.print_exc()
            
            console.print("\n[dim]Press Enter to continue...[/dim]")
            await asyncio.to_thread(input)

if __name__ == "__main__":
    try:
        asyncio.run(run_menu())
    except KeyboardInterrupt:
        console.print("\n[yellow]Exiting...[/yellow]")
