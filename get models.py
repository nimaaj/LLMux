from llmclient import UnifiedChatClient
from rich import print
import dotenv
dotenv.load_dotenv()
client = UnifiedChatClient()
print([client.get_models("gemini"),      # Returns Gemini models
client.get_models("openai"),      # Returns OpenAI models
client.get_models("deepseek") ,   # Returns DeepSeek models
client.get_models("anthropic") ])  # Returns Anthropic models