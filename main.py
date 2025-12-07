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
        ("openai", "gpt-4"),
        ("claude", "claude-3-5-haiku-latest"),
        ("gemini", "models/gemini-2.5-flash-lite"),
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