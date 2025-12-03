"""
Main application demonstrating the use of rich stream printer with UnifiedChatClient.
"""
import asyncio
from typing import List
from rich_llm_printer import RichPrinter
from llmclient import UnifiedChatClient, Message


async def main():
    client = UnifiedChatClient()
    
    messages: List[Message] = [
        {"role": "system", "content": "You are a concise assistant."},
        {"role": "user", "content": "introduce yourself in one sentence using markdown syntax."},
    ]
    
    # Create event stream using your UnifiedChatClient
    resp = await client.chat(
        provider="deepseek",
        model="deepseek-chat",
        messages=messages,
        temperature=0.7,
        max_tokens=500,
    )
    printer = RichPrinter()
    printer.print_chat(resp)


if __name__ == "__main__":
    asyncio.run(main())