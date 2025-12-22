import pytest
from unittest.mock import patch, mock_open, MagicMock
from llmclient.client import UnifiedChatClient

class TestUtils:
    
    def test_create_text_content(self):
        content = UnifiedChatClient.create_text_content("Hello")
        assert content == {"type": "text", "text": "Hello"}

    def test_create_image_content_from_url(self):
        content = UnifiedChatClient.create_image_content("https://example.com/img.jpg")
        assert content["type"] == "image_url"
        assert content["image_url"]["url"] == "https://example.com/img.jpg"

    def test_create_image_content_from_base64(self):
        b64 = "SGVsbG8="
        content = UnifiedChatClient.create_image_content(b64, mime_type="image/png")
        assert content["image_url"]["url"] == "data:image/png;base64,SGVsbG8="

    def test_create_message_text(self):
        msg = UnifiedChatClient.create_message("user", "Hello world")
        assert msg == {"role": "user", "content": "Hello world"}

    def test_create_message_multimodal(self):
        content = [
            "Look at this",
            UnifiedChatClient.create_image_content("https://example.com/cat.jpg")
        ]
        msg = UnifiedChatClient.create_message("user", content)
        assert msg["role"] == "user"
        assert len(msg["content"]) == 2
        assert msg["content"][0] == {"type": "text", "text": "Look at this"}

    @patch("pathlib.Path.exists")
    @patch("builtins.open", new_callable=mock_open, read_data=b"image data")
    def test_encode_image_file(self, mock_file, mock_exists):
        mock_exists.return_value = True
        
        # Test with a mock file path
        b64_data, mime_type = UnifiedChatClient.encode_image_file("test.jpg")
        
        assert mime_type == "image/jpeg"
        # "image data" in base64 is "aW1hZ2UgZGF0YQ=="
        assert b64_data == "aW1hZ2UgZGF0YQ=="

    def test_create_tool(self):
        tool = UnifiedChatClient.create_tool(
            name="get_weather",
            description="Get weather",
            parameters={"location": {"type": "string"}},
            required=["location"]
        )
        assert tool["type"] == "function"
        assert tool["function"]["name"] == "get_weather"
        assert tool["function"]["parameters"]["required"] == ["location"]

    def test_create_tool_result(self):
        result = UnifiedChatClient.create_tool_result("call_123", "result content")
        assert result["role"] == "tool"
        assert result["tool_call_id"] == "call_123"
        assert result["content"] == "result content"
