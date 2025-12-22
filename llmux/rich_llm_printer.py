"""
Rich stream printer module for displaying streaming LLM responses.
"""
from typing import Dict, Any, AsyncIterator, Optional
from rich.console import Console
from rich.markdown import Markdown
from rich.syntax import Syntax
from rich.panel import Panel
from rich.live import Live
from rich.console import Group
from rich.text import Text
import json

console = Console()

class RichStreamPrinter:
    """
    A class for beautifully displaying streaming LLM responses using rich.
    
    Attributes:
        title: Title for the display panel
        show_metadata: Whether to show metadata at the end
        code_theme: Theme for code blocks
        inline_code_theme: Theme for inline code
        refresh_rate: Refresh rate for Live display
        show_final_title: Whether to change title to "Final Response" at the end
        show_provider_info: Whether to show provider information in title
    """
    
    def __init__(
        self,
        title: str = "Streaming Response",
        show_metadata: bool = True,
        code_theme: str = "coffee",
        inline_code_theme: str = "monokai",
        refresh_rate: int = 30,
        show_final_title: bool = True,
        show_provider_info: bool = True,
        border_style: str = "blue"
    ):
        self.title = title
        self.show_metadata = show_metadata
        self.code_theme = code_theme
        self.inline_code_theme = inline_code_theme
        self.refresh_rate = refresh_rate
        self.show_final_title = show_final_title
        self.show_provider_info = show_provider_info
        self.border_style = border_style
        self._full_text = ""
        self._final_event: Optional[Dict[str, Any]] = None
        self._provider: Optional[str] = None
    
    async def print_stream(
        self, 
        event_stream: AsyncIterator[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Process and display streaming events with rich formatting.
        
        Args:
            event_stream: Async iterator yielding event dictionaries
            
        Returns:
            The final event dictionary containing the complete response
        """
        self._full_text = ""
        self._final_event = None
        self._provider = None
        panel = Panel("", border_style=self.border_style)
        
        with Live(panel, refresh_per_second=self.refresh_rate, console=console) as live:
            async for event in event_stream:
                await self._process_event(event, live)
        
        return self._final_event or {}
    
    async def _process_event(
        self, 
        event: Dict[str, Any], 
        live: Live
    ) -> None:
        """Process a single event and update the display."""
        if event["type"] == "token":
            self._full_text += event["text"]
            # Capture provider from first token event
            if self._provider is None and "provider" in event:
                self._provider = event["provider"]
            self._update_display(live, is_final=False)
        elif event["type"] == "done":
            self._final_event = event
            if "full_text" not in self._final_event:
                self._final_event["full_text"] = self._full_text
            # Ensure provider is set
            if self._provider is None and "provider" in event:
                self._provider = event["provider"]
            self._update_display(live, is_final=True)
    
    def _update_display(
        self, 
        live: Live, 
        is_final: bool = False
    ) -> None:
        """Update the Live display with current content."""
        title = self._build_title(is_final)
        content = self._build_content(is_final)
        
        live.update(
            Panel(
                content,
                title=title,
                border_style="green" if is_final else self.border_style,
                padding=(1, 2)
            )
        )
    
    def _build_title(self, is_final: bool) -> str:
        """Build the panel title."""
        title_parts = []
        
        if is_final and self.show_final_title:
            title_parts.append("[bold]Final Response[/bold]")
        else:
            title_parts.append(f"[bold]{self.title}[/bold]")
        
        if self.show_provider_info and self._provider:
            title_parts.append(f"[dim]({self._provider})[/dim]")
        
        return " ".join(title_parts)
    
    def _build_content(self, is_final: bool) -> Any:
        """Build the panel content."""
        if not self._full_text.strip():
            return Text("(waiting for response...)", style="dim italic")
        
        # Create markdown content
        markdown = Markdown(
            self._full_text, 
            code_theme=self.code_theme, 
            inline_code_theme=self.inline_code_theme
        )
        
        # Add metadata if this is the final event
        if is_final and self.show_metadata and self._final_event:
            meta = self._final_event.get("meta", {})
            if meta:
                metadata_json = json.dumps(meta, indent=2, default=str)
                metadata_display = Syntax(
                    metadata_json, 
                    "json", 
                    theme="lightbulb",
                    background_color="default"
                )
                
                # Create info panel for metadata
                metadata_panel = Panel(
                    metadata_display,
                    title="[bold]Metadata[/bold]",
                    border_style="dim"
                )
                
                return Group(markdown, metadata_panel)
        
        return markdown
    
    def get_full_text(self) -> str:
        """Get the full assembled text."""
        return self._full_text
    
    def get_final_event(self) -> Optional[Dict[str, Any]]:
        """Get the final event if available."""
        return self._final_event
    
    def get_provider(self) -> Optional[str]:
        """Get the provider name if available."""
        return self._provider


class RichPrinter:
    """
    A class for beautifully displaying non-streaming LLM responses using rich.
    
    Designed to work with the `chat` method output from UnifiedChatClient.
    
    Attributes:
        title: Title for the display panel
        show_metadata: Whether to show metadata at the end
        code_theme: Theme for code blocks
        inline_code_theme: Theme for inline code
        show_provider_info: Whether to show provider information in title
        border_style: Border style for the panel
    """
    
    def __init__(
        self,
        title: str = "Response",
        show_metadata: bool = True,
        code_theme: str = "coffee",
        inline_code_theme: str = "monokai",
        show_provider_info: bool = True,
        border_style: str = "green"
    ):
        self.title = title
        self.show_metadata = show_metadata
        self.code_theme = code_theme
        self.inline_code_theme = inline_code_theme
        self.show_provider_info = show_provider_info
        self.border_style = border_style
        self._response: Optional[Dict[str, Any]] = None
    
    def print_chat(self, response: Dict[str, Any]) -> Dict[str, Any]:
        """
        Display a non-streaming chat response with rich formatting.
        
        Args:
            response: Response dictionary from UnifiedChatClient.chat()
                Expected format:
                {
                    "provider": "openai" | "claude" | "gemini" | "deepseek",
                    "text": "<assistant reply>",
                    "meta": {
                        "model": "<model name>",
                        "usage": {...},
                        "latency_ms": float,
                    },
                }
            
        Returns:
            The same response dictionary for chaining
        """
        self._response = response
        
        provider = response.get("provider", "")
        text = response.get("text", "")
        meta = response.get("meta", {})
        
        # Build title
        title = self._build_title(provider)
        
        # Build content
        content = self._build_content(text, meta)
        
        # Print the panel
        console.print(
            Panel(
                content,
                title=title,
                border_style=self.border_style,
                padding=(1, 2)
            )
        )
        
        return response
    
    def _build_title(self, provider: str) -> str:
        """Build the panel title."""
        title_parts = [f"[bold]{self.title}[/bold]"]
        
        if self.show_provider_info and provider:
            title_parts.append(f"[dim]({provider})[/dim]")
        
        return " ".join(title_parts)
    
    def _build_content(self, text: str, meta: Dict[str, Any]) -> Any:
        """Build the panel content."""
        if not text.strip():
            return Text("(empty response)", style="dim italic")
        
        # Create markdown content
        markdown = Markdown(
            text, 
            code_theme=self.code_theme, 
            inline_code_theme=self.inline_code_theme
        )
        
        # Add metadata if enabled
        if self.show_metadata and meta:
            metadata_json = json.dumps(meta, indent=2, default=str)
            metadata_display = Syntax(
                metadata_json, 
                "json", 
                theme="lightbulb",
                background_color="default"
            )
            
            # Create info panel for metadata
            metadata_panel = Panel(
                metadata_display,
                title="[bold]Metadata[/bold]",
                border_style="dim"
            )
            
            return Group(markdown, metadata_panel)
        
        return markdown
    
    def get_response(self) -> Optional[Dict[str, Any]]:
        """Get the last printed response."""
        return self._response
    
    def get_text(self) -> str:
        """Get the text from the last printed response."""
        if self._response:
            return self._response.get("text", "")
        return ""
    
    def get_provider(self) -> Optional[str]:
        """Get the provider from the last printed response."""
        if self._response:
            return self._response.get("provider")
        return None
