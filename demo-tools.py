"""
Demo: Tool Calling with LLMux

This demo shows how to use native tool calling (function calling) with
different LLM providers through the unified interface.
"""

import asyncio
import json
from llmclient import UnifiedChatClient


# =============================================================================
# Define some simple tools (mock implementations)
# =============================================================================

def get_weather(location: str, unit: str = "celsius") -> dict:
    """Mock weather function - returns fake weather data."""
    # In a real app, this would call a weather API
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


# =============================================================================
# Map tool names to functions
# =============================================================================

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


# =============================================================================
# Demo functions
# =============================================================================

async def demo_single_tool_call(client: UnifiedChatClient, provider: str, model: str):
    """Demo: Single tool call (weather)"""
    print(f"\n{'='*60}")
    print(f"Demo: Single Tool Call ({provider})")
    print(f"{'='*60}")
    
    # Define the weather tool
    weather_tool = client.create_tool(
        name="get_weather",
        description="Get the current weather for a location",
        parameters={
            "location": {
                "type": "string",
                "description": "City name, e.g., 'Paris', 'London', 'Tokyo'",
            },
            "unit": {
                "type": "string",
                "enum": ["celsius", "fahrenheit"],
                "description": "Temperature unit (default: celsius)",
            },
        },
        required=["location"],
    )
    
    messages = [
        {"role": "user", "content": "What's the weather like in Paris?"},
    ]
    
    print(f"\nUser: {messages[0]['content']}")
    
    # First call - LLM decides to use the tool
    response = await client.chat(
        provider=provider,
        model=model,
        messages=messages,
        tools=[weather_tool],
    )
    
    print(f"\nLLM Response:")
    print(f"  Text: {response['text']}")
    
    if response.get("tool_calls"):
        print(f"  Tool Calls:")
        
        # Add assistant message with tool calls to history (once)
        messages.append(client.create_assistant_message_with_tool_calls(
            response["text"] or "",
            response["tool_calls"]
        ))
        
        # Execute all tools and add results
        for tc in response["tool_calls"]:
            print(f"    - {tc['name']}({tc['arguments']})")
            result = execute_tool(tc["name"], tc["arguments"])
            print(f"    - Result: {result}")
            messages.append(client.create_tool_result(tc["id"], result))
        
        # Get final response with tool results
        final_response = await client.chat(
            provider=provider,
            model=model,
            messages=messages,
            tools=[weather_tool],
        )
        
        print(f"\nFinal Response: {final_response['text']}")
    else:
        print("  (No tool calls made)")
    
    print(f"\nLatency: {response['meta']['latency_ms']:.0f}ms")


async def demo_multiple_tools(client: UnifiedChatClient, provider: str, model: str):
    """Demo: Multiple tools available"""
    print(f"\n{'='*60}")
    print(f"Demo: Multiple Tools ({provider})")
    print(f"{'='*60}")
    
    # Define multiple tools
    tools = [
        client.create_tool(
            name="get_weather",
            description="Get the current weather for a location",
            parameters={
                "location": {
                    "type": "string",
                    "description": "City name",
                },
            },
            required=["location"],
        ),
        client.create_tool(
            name="calculate",
            description="Evaluate a mathematical expression",
            parameters={
                "expression": {
                    "type": "string",
                    "description": "Math expression to evaluate, e.g., '2 + 2 * 3'",
                },
            },
            required=["expression"],
        ),
        client.create_tool(
            name="search_web",
            description="Search the web for information",
            parameters={
                "query": {
                    "type": "string",
                    "description": "Search query",
                },
            },
            required=["query"],
        ),
    ]
    
    messages = [
        {"role": "user", "content": "What's 15% of 200, and what's the weather in Tokyo?"},
    ]
    
    print(f"\nUser: {messages[0]['content']}")
    
    response = await client.chat(
        provider=provider,
        model=model,
        messages=messages,
        tools=tools,
    )
    
    print(f"\nLLM Response:")
    print(f"  Text: {response['text'] or '(thinking...)'}")
    
    if response.get("tool_calls"):
        print(f"  Tool Calls:")
        tool_results = []
        for tc in response["tool_calls"]:
            print(f"    - {tc['name']}({tc['arguments']})")
            result = execute_tool(tc["name"], tc["arguments"])
            print(f"      Result: {result}")
            tool_results.append(client.create_tool_result(tc["id"], result))
        
        # Add assistant message with tool calls
        messages.append(client.create_assistant_message_with_tool_calls(
            response["text"] or "",
            response["tool_calls"]
        ))
        
        # Add tool results
        messages.extend(tool_results)
        
        # Get final response
        final_response = await client.chat(
            provider=provider,
            model=model,
            messages=messages,
            tools=tools,
        )
        
        print(f"\nFinal Response: {final_response['text']}")


async def demo_tool_choice(client: UnifiedChatClient, provider: str, model: str):
    """Demo: Forcing tool use with tool_choice"""
    print(f"\n{'='*60}")
    print(f"Demo: Tool Choice ({provider})")
    print(f"{'='*60}")
    
    calc_tool = client.create_tool(
        name="calculate",
        description="Evaluate a mathematical expression",
        parameters={
            "expression": {
                "type": "string",
                "description": "Math expression to evaluate",
            },
        },
        required=["expression"],
    )
    
    # This question could be answered without a tool, but we force tool use
    messages = [
        {"role": "user", "content": "What is 7 times 8?"},
    ]
    
    print(f"\nUser: {messages[0]['content']}")
    print("(Forcing tool use with tool_choice='required')")
    
    response = await client.chat(
        provider=provider,
        model=model,
        messages=messages,
        tools=[calc_tool],
        tool_choice="required",  # Force the LLM to use a tool
    )
    
    print(f"\nLLM Response:")
    print(f"  Text: {response['text'] or '(using tool)'}")
    
    if response.get("tool_calls"):
        for tc in response["tool_calls"]:
            print(f"  Tool Call: {tc['name']}({tc['arguments']})")
            result = execute_tool(tc["name"], tc["arguments"])
            print(f"  Result: {result}")


async def demo_no_tool_needed(client: UnifiedChatClient, provider: str, model: str):
    """Demo: LLM chooses not to use tools"""
    print(f"\n{'='*60}")
    print(f"Demo: No Tool Needed ({provider})")
    print(f"{'='*60}")
    
    weather_tool = client.create_tool(
        name="get_weather",
        description="Get the current weather for a location",
        parameters={
            "location": {"type": "string", "description": "City name"},
        },
        required=["location"],
    )
    
    messages = [
        {"role": "user", "content": "What is the capital of France?"},
    ]
    
    print(f"\nUser: {messages[0]['content']}")
    print("(Weather tool available but not needed)")
    
    response = await client.chat(
        provider=provider,
        model=model,
        messages=messages,
        tools=[weather_tool],
    )
    
    print(f"\nLLM Response: {response['text']}")
    print(f"Tool Calls: {response.get('tool_calls', [])}")


# =============================================================================
# Main
# =============================================================================

async def main():
    client = UnifiedChatClient()
    
    # Test with different providers
    # Uncomment the provider you want to test
    
    # OpenAI
    provider, model = "openai", "gpt-4o-mini"
    
    # Claude
    # provider, model = "claude", "claude-3-5-haiku-latest"
    
    # Gemini
    # provider, model = "gemini", "gemini-1.5-flash"
    
    # DeepSeek
    # provider, model = "deepseek", "deepseek-chat"
    
    print(f"\n🔧 Tool Calling Demo with {provider.upper()} ({model})")
    print("=" * 60)
    
    # Run demos
    await demo_single_tool_call(client, provider, model)
    await demo_multiple_tools(client, provider, model)
    await demo_tool_choice(client, provider, model)
    await demo_no_tool_needed(client, provider, model)
    
    print("\n✅ All demos completed!")


if __name__ == "__main__":
    asyncio.run(main())
