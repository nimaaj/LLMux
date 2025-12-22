import asyncio
import dotenv
from llmclient import UnifiedChatClient
from rich import print

async def main():
    dotenv.load_dotenv()
    client = UnifiedChatClient()
    
    print("Fetching models...")
    
    providers = ["gemini", "openai", "deepseek", "anthropic", "huggingface"]
    results = {}
    
    for provider in providers:
        try:
            print(f"Querying {provider}...")
            # We use the new async instance method list_models
            # It handles aliases (anthropic -> claude)
            models = await client.list_models(provider)
            results[provider] = models
        except ValueError as e:
            results[provider] = f"Error: {e}"
        except Exception as e:
            results[provider] = f"Error: {e}"

    print(results)

if __name__ == "__main__":
    asyncio.run(main())