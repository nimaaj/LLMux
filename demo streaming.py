"""
Main application demonstrating the use of rich stream printer with UnifiedChatClient.
"""
import asyncio
from typing import List
from rich.console import Console
from rich import inspect
from rich_llm_printer import RichStreamPrinter
from llmclient import UnifiedChatClient, Message

console = Console()


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
    console.print("[bold cyan]\\n=== Interactive Conversation ===")
    
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
        user_input = input().strip()
        
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



async def main():
    """Run all demos."""
    try:
        # Run demos
        await demo_basic_stream()

        
        #  Uncomment to run interactive demo
        #await demo_interactive_conversation()
        
        console.print("\n[bold green] All demos completed successfully![/bold green]")
        
    except KeyboardInterrupt:
        console.print("\n[yellow]\\nDemo interrupted by user[/yellow]")
    except Exception as e:
        console.print(f"\n[bold red]Error in main:[/bold red] {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())