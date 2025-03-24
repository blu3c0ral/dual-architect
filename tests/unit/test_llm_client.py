# test_llm_interface.py

import unittest
import json
from unittest.mock import patch, MagicMock, call
import os
import pytest
import responses
import requests

# Import the modules to test
from src.utils.llm_client import (
    Message,
    Conversation,
    LLMProvider,
    OpenAIProvider,
    GeminiProvider,
    ClaudeProvider,
    LLMInterface,
)


class TestMessage(unittest.TestCase):
    """Tests for the Message class."""

    def test_init(self):
        """Test Message initialization."""
        message = Message("user", "Hello world")
        self.assertEqual(message.role, "user")
        self.assertEqual(message.content, "Hello world")

    def test_to_dict(self):
        """Test converting Message to dictionary."""
        message = Message("system", "You are a helpful assistant")
        expected = {"role": "system", "content": "You are a helpful assistant"}
        self.assertEqual(message.to_dict(), expected)

    def test_with_empty_content(self):
        """Test Message with empty content."""
        message = Message("assistant", "")
        self.assertEqual(message.content, "")
        self.assertEqual(message.to_dict(), {"role": "assistant", "content": ""})


class TestConversation(unittest.TestCase):
    """Tests for the Conversation class."""

    def test_init_without_system_message(self):
        """Test Conversation initialization without system message."""
        conversation = Conversation()
        self.assertEqual(len(conversation.messages), 0)

    def test_init_with_system_message(self):
        """Test Conversation initialization with system message."""
        system_message = "You are a helpful assistant"
        conversation = Conversation(system_message)
        self.assertEqual(len(conversation.messages), 1)
        self.assertEqual(conversation.messages[0].role, "system")
        self.assertEqual(conversation.messages[0].content, system_message)

    def test_set_system_message_new(self):
        """Test setting a new system message."""
        conversation = Conversation()
        system_message = "You are a helpful assistant"
        conversation.set_system_message(system_message)
        self.assertEqual(len(conversation.messages), 1)
        self.assertEqual(conversation.messages[0].role, "system")
        self.assertEqual(conversation.messages[0].content, system_message)

    def test_set_system_message_update(self):
        """Test updating an existing system message."""
        old_message = "You are a helpful assistant"
        new_message = "You are a coding assistant"
        conversation = Conversation(old_message)
        conversation.set_system_message(new_message)
        self.assertEqual(len(conversation.messages), 1)
        self.assertEqual(conversation.messages[0].role, "system")
        self.assertEqual(conversation.messages[0].content, new_message)

    def test_add_user_message(self):
        """Test adding a user message."""
        conversation = Conversation()
        conversation.add_user_message("Hello")
        self.assertEqual(len(conversation.messages), 1)
        self.assertEqual(conversation.messages[0].role, "user")
        self.assertEqual(conversation.messages[0].content, "Hello")

    def test_add_assistant_message(self):
        """Test adding an assistant message."""
        conversation = Conversation()
        conversation.add_assistant_message("Hello there!")
        self.assertEqual(len(conversation.messages), 1)
        self.assertEqual(conversation.messages[0].role, "assistant")
        self.assertEqual(conversation.messages[0].content, "Hello there!")

    def test_get_messages(self):
        """Test getting messages in dictionary format."""
        conversation = Conversation("You are a helpful assistant")
        conversation.add_user_message("Hello")
        conversation.add_assistant_message("Hi there!")

        expected = [
            {"role": "system", "content": "You are a helpful assistant"},
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
        ]

        self.assertEqual(conversation.get_messages(), expected)

    def test_clear(self):
        """Test clearing all messages except system message."""
        conversation = Conversation("You are a helpful assistant")
        conversation.add_user_message("Hello")
        conversation.add_assistant_message("Hi there!")

        conversation.clear()

        # Should only have system message left
        self.assertEqual(len(conversation.messages), 1)
        self.assertEqual(conversation.messages[0].role, "system")

    def test_clear_without_system_message(self):
        """Test clearing conversation without system message."""
        conversation = Conversation()
        conversation.add_user_message("Hello")
        conversation.add_assistant_message("Hi there!")

        conversation.clear()

        # Should have no messages
        self.assertEqual(len(conversation.messages), 0)

    def test_conversation_complex_flow(self):
        """Test a complex conversation flow with multiple operations."""
        conversation = Conversation()

        # Add messages
        conversation.add_user_message("Hello")
        conversation.add_assistant_message("Hi there!")

        # Add system message after other messages
        conversation.set_system_message("You are a helpful assistant")

        # Verify system message is at the beginning
        self.assertEqual(conversation.messages[0].role, "system")

        # Add more messages
        conversation.add_user_message("How are you?")
        conversation.add_assistant_message("I'm doing well, thanks!")

        # Update system message
        conversation.set_system_message("You are a coding assistant")

        # Verify correct order and content
        messages = conversation.get_messages()
        self.assertEqual(len(messages), 5)
        self.assertEqual(messages[0]["role"], "system")
        self.assertEqual(messages[0]["content"], "You are a coding assistant")
        self.assertEqual(messages[1]["role"], "user")
        self.assertEqual(messages[1]["content"], "Hello")
        self.assertEqual(messages[2]["role"], "assistant")
        self.assertEqual(messages[2]["content"], "Hi there!")
        self.assertEqual(messages[3]["role"], "user")
        self.assertEqual(messages[3]["content"], "How are you?")
        self.assertEqual(messages[4]["role"], "assistant")
        self.assertEqual(messages[4]["content"], "I'm doing well, thanks!")


class TestLLMProvider(unittest.TestCase):
    """Tests for the LLMProvider abstract base class."""

    def test_instantiation(self):
        """Test that LLMProvider cannot be instantiated directly."""
        with self.assertRaises(TypeError):
            LLMProvider()

    def test_subclass_without_query_method(self):
        """Test that subclasses must implement the query method."""

        class InvalidProvider(LLMProvider):
            pass

        with self.assertRaises(TypeError):
            InvalidProvider()

    def test_valid_subclass(self):
        """Test a valid subclass implementation."""

        class ValidProvider(LLMProvider):
            def query(self, prompt, conversation=None):
                return "Test response"

        provider = ValidProvider()
        self.assertEqual(provider.query("Hello"), "Test response")


@responses.activate
class TestOpenAIProvider(unittest.TestCase):
    """Tests for the OpenAIProvider class."""

    def setUp(self):
        """Set up for each test."""
        self.api_key = "test_openai_key"
        self.provider = OpenAIProvider(api_key=self.api_key)

    def test_init(self):
        """Test initialization with explicit API key."""
        self.assertEqual(self.provider.api_key, self.api_key)
        self.assertEqual(self.provider.model, "gpt-4")
        self.assertEqual(
            self.provider.api_url, "https://api.openai.com/v1/chat/completions"
        )

    def test_init_with_env_var(self):
        """Test initialization with API key from environment variable."""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "env_test_key"}):
            provider = OpenAIProvider()
            self.assertEqual(provider.api_key, "env_test_key")

    def test_init_without_api_key(self):
        """Test initialization without API key."""
        with patch.dict(os.environ, {}, clear=True):
            provider = OpenAIProvider()
            self.assertIsNone(provider.api_key)

    def test_query_without_api_key(self):
        """Test query behavior without an API key (should simulate response)."""
        provider = OpenAIProvider(api_key=None)

        with patch.object(provider, "_simulate_response") as mock_simulate:
            mock_simulate.return_value = '{"simulated": true}'
            response = provider.query("Hello")
            mock_simulate.assert_called_once_with("Hello")
            self.assertEqual(response, '{"simulated": true}')

    @responses.activate
    def test_query_with_conversation(self):
        """Test query with a conversation context."""
        # Mock the OpenAI API response
        responses.add(
            responses.POST,
            "https://api.openai.com/v1/chat/completions",
            json={"choices": [{"message": {"content": "Hello there!"}}]},
            status=200,
        )

        conversation = Conversation("You are a helpful assistant")
        response = self.provider.query("Hello", conversation)

        self.assertEqual(response, "Hello there!")
        self.assertEqual(len(conversation.messages), 3)  # system, user, assistant
        self.assertEqual(conversation.messages[1].content, "Hello")
        self.assertEqual(conversation.messages[2].content, "Hello there!")

        # Check request payload
        request = responses.calls[0].request
        payload = json.loads(request.body)
        self.assertEqual(payload["model"], "gpt-4")
        self.assertEqual(
            len(payload["messages"]), 2
        )  # system, user (assistant not sent)
        self.assertEqual(payload["messages"][0]["role"], "system")
        self.assertEqual(payload["messages"][1]["role"], "user")
        self.assertEqual(payload["messages"][1]["content"], "Hello")

    @responses.activate
    def test_query_without_conversation(self):
        """Test query without a conversation context."""
        # Mock the OpenAI API response
        responses.add(
            responses.POST,
            "https://api.openai.com/v1/chat/completions",
            json={"choices": [{"message": {"content": "Hello there!"}}]},
            status=200,
        )

        response = self.provider.query("Hello")

        self.assertEqual(response, "Hello there!")

        # Check request payload
        request = responses.calls[0].request
        payload = json.loads(request.body)
        self.assertEqual(len(payload["messages"]), 1)  # only user message
        self.assertEqual(payload["messages"][0]["role"], "user")
        self.assertEqual(payload["messages"][0]["content"], "Hello")

    @responses.activate
    def test_api_error(self):
        """Test handling of API errors."""
        # Mock an API error response
        responses.add(
            responses.POST,
            "https://api.openai.com/v1/chat/completions",
            json={"error": {"message": "Invalid API key"}},
            status=401,
        )

        with patch.object(self.provider, "_error_response") as mock_error:
            mock_error.return_value = '{"error": true}'
            response = self.provider.query("Hello")
            self.assertEqual(response, '{"error": true}')

    def test_connection_error(self):
        """Test handling of connection errors."""
        with patch("requests.post") as mock_post:
            mock_post.side_effect = Exception("Connection error")

            with patch.object(self.provider, "_error_response") as mock_error:
                mock_error.return_value = '{"error": true}'
                response = self.provider.query("Hello")
                self.assertEqual(response, '{"error": true}')
                mock_error.assert_called_once_with("Connection error")

    def test_simulate_response(self):
        """Test simulated response generation."""
        provider = OpenAIProvider(api_key=None)
        response = provider._simulate_response("Hello")

        # Verify response is valid JSON and has expected fields
        data = json.loads(response)
        self.assertIn("task_type", data)
        self.assertIn("technologies", data)
        self.assertIn("components", data)
        self.assertIn("dependencies", data)
        self.assertIn("complexity_estimate", data)
        self.assertIn("key_features", data)
        self.assertIn("concerns", data)

    def test_error_response(self):
        """Test error response generation."""
        error_response = self.provider._error_response("Test error")

        # Verify response is valid JSON and has expected fields
        data = json.loads(error_response)
        self.assertEqual(data["task_type"], "unknown")
        self.assertEqual(
            data["concerns"], ["Error connecting to OpenAI service: Test error"]
        )


@responses.activate
class TestGeminiProvider(unittest.TestCase):
    """Tests for the GeminiProvider class."""

    def setUp(self):
        """Set up for each test."""
        self.api_key = "test_gemini_key"
        self.provider = GeminiProvider(api_key=self.api_key)

    def test_init(self):
        """Test initialization with explicit API key."""
        self.assertEqual(self.provider.api_key, self.api_key)
        self.assertEqual(self.provider.model, "gemini-pro")
        self.assertEqual(
            self.provider.api_url,
            "https://generativelanguage.googleapis.com/v1/models/gemini-pro:generateContent",
        )

    def test_init_with_env_var(self):
        """Test initialization with API key from environment variable."""
        with patch.dict(os.environ, {"GEMINI_API_KEY": "env_test_key"}):
            provider = GeminiProvider()
            self.assertEqual(provider.api_key, "env_test_key")

    def test_init_custom_model(self):
        """Test initialization with custom model."""
        provider = GeminiProvider(model="gemini-ultra")
        self.assertEqual(provider.model, "gemini-ultra")
        self.assertEqual(
            provider.api_url,
            "https://generativelanguage.googleapis.com/v1/models/gemini-ultra:generateContent",
        )

    def test_query_without_api_key(self):
        """Test query behavior without an API key (should simulate response)."""
        provider = GeminiProvider(api_key=None)

        with patch.object(provider, "_simulate_response") as mock_simulate:
            mock_simulate.return_value = '{"simulated": true}'
            response = provider.query("Hello")
            mock_simulate.assert_called_once_with("Hello")
            self.assertEqual(response, '{"simulated": true}')

    @responses.activate
    def test_query_with_conversation_role_mapping(self):
        """Test query with conversation focusing on role mapping for Gemini."""
        # Mock the Gemini API response
        url = f"https://generativelanguage.googleapis.com/v1/models/gemini-pro:generateContent?key={self.api_key}"
        responses.add(
            responses.POST,
            url,
            json={"candidates": [{"content": {"parts": [{"text": "Hello there!"}]}}]},
            status=200,
        )

        # Create a conversation with system, user, and assistant messages
        conversation = Conversation("You are a helpful assistant")
        conversation.add_user_message("Initial message")
        conversation.add_assistant_message("Initial response")

        # Query with additional user message
        response = self.provider.query("Hello", conversation)

        self.assertEqual(response, "Hello there!")

        # Verify last messages in conversation
        self.assertEqual(conversation.messages[-2].role, "user")
        self.assertEqual(conversation.messages[-2].content, "Hello")
        self.assertEqual(conversation.messages[-1].role, "assistant")
        self.assertEqual(conversation.messages[-1].content, "Hello there!")

        # Check request payload for correct role mappings
        request = responses.calls[0].request
        payload = json.loads(request.body)

        # First message should be system transformed to user with prefix
        self.assertEqual(payload["contents"][0]["role"], "user")
        self.assertIn("System instruction:", payload["contents"][0]["parts"][0]["text"])

        # Second message should be user
        self.assertEqual(payload["contents"][1]["role"], "user")
        self.assertEqual(payload["contents"][1]["parts"][0]["text"], "Initial message")

        # Third message should be model (not assistant)
        self.assertEqual(payload["contents"][2]["role"], "model")
        self.assertEqual(payload["contents"][2]["parts"][0]["text"], "Initial response")

        # Fourth message should be the new user message
        self.assertEqual(payload["contents"][3]["role"], "user")
        self.assertEqual(payload["contents"][3]["parts"][0]["text"], "Hello")

    @responses.activate
    def test_query_without_conversation(self):
        """Test query without conversation context."""
        url = f"https://generativelanguage.googleapis.com/v1/models/gemini-pro:generateContent?key={self.api_key}"
        responses.add(
            responses.POST,
            url,
            json={"candidates": [{"content": {"parts": [{"text": "Hello there!"}]}}]},
            status=200,
        )

        response = self.provider.query("Hello")

        self.assertEqual(response, "Hello there!")

        # Check request payload
        request = responses.calls[0].request
        payload = json.loads(request.body)
        self.assertEqual(len(payload["contents"]), 1)
        self.assertEqual(payload["contents"][0]["role"], "user")
        self.assertEqual(payload["contents"][0]["parts"][0]["text"], "Hello")

    @responses.activate
    def test_api_error(self):
        """Test handling of API errors."""
        url = f"https://generativelanguage.googleapis.com/v1/models/gemini-pro:generateContent?key={self.api_key}"
        responses.add(
            responses.POST,
            url,
            json={"error": {"message": "Invalid API key"}},
            status=401,
        )

        with patch.object(self.provider, "_error_response") as mock_error:
            mock_error.return_value = '{"error": true}'
            response = self.provider.query("Hello")
            self.assertEqual(response, '{"error": true}')

    @responses.activate
    def test_empty_candidates(self):
        """Test handling of response with empty candidates."""
        url = f"https://generativelanguage.googleapis.com/v1/models/gemini-pro:generateContent?key={self.api_key}"
        responses.add(responses.POST, url, json={"candidates": []}, status=200)

        with patch.object(self.provider, "_error_response") as mock_error:
            mock_error.return_value = '{"error": true}'
            response = self.provider.query("Hello")
            self.assertEqual(response, '{"error": true}')

    def test_simulate_response(self):
        """Test simulated response generation."""
        provider = GeminiProvider(api_key=None)
        response = provider._simulate_response("Hello")

        # Verify response is valid JSON and has expected fields
        data = json.loads(response)
        self.assertIn("task_type", data)
        self.assertIn("technologies", data)
        self.assertIn("components", data)
        self.assertIn("dependencies", data)
        self.assertIn("complexity_estimate", data)
        self.assertIn("key_features", data)
        self.assertIn("concerns", data)

    def test_error_response(self):
        """Test error response generation."""
        error_response = self.provider._error_response("Test error")

        # Verify response is valid JSON and has expected fields
        data = json.loads(error_response)
        self.assertEqual(data["task_type"], "unknown")
        self.assertEqual(
            data["concerns"], ["Error connecting to Gemini service: Test error"]
        )


@responses.activate
class TestClaudeProvider(unittest.TestCase):
    """Tests for the ClaudeProvider class."""

    def setUp(self):
        """Set up for each test."""
        self.api_key = "test_claude_key"
        self.provider = ClaudeProvider(api_key=self.api_key)

    def test_init(self):
        """Test initialization with explicit API key."""
        self.assertEqual(self.provider.api_key, self.api_key)
        self.assertEqual(self.provider.model, "claude-3-opus-20240229")
        self.assertEqual(self.provider.api_url, "https://api.anthropic.com/v1/messages")

    def test_init_with_env_var(self):
        """Test initialization with API key from environment variable."""
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "env_test_key"}):
            provider = ClaudeProvider()
            self.assertEqual(provider.api_key, "env_test_key")

    def test_query_without_api_key(self):
        """Test query behavior without an API key (should simulate response)."""
        provider = ClaudeProvider(api_key=None)

        with patch.object(provider, "_simulate_response") as mock_simulate:
            mock_simulate.return_value = '{"simulated": true}'
            response = provider.query("Hello")
            mock_simulate.assert_called_once_with("Hello")
            self.assertEqual(response, '{"simulated": true}')

    @responses.activate
    def test_query_with_conversation_system_handling(self):
        """Test query with conversation focusing on system message handling for Claude."""
        # Mock the Claude API response
        responses.add(
            responses.POST,
            "https://api.anthropic.com/v1/messages",
            json={"content": [{"text": "Hello there!"}]},
            status=200,
        )

        # Create a conversation with system message
        conversation = Conversation("You are a helpful assistant")
        conversation.add_user_message("Initial message")
        conversation.add_assistant_message("Initial response")

        # Query with additional user message
        response = self.provider.query("Hello", conversation)

        self.assertEqual(response, "Hello there!")

        # Verify last messages in conversation
        self.assertEqual(conversation.messages[-2].role, "user")
        self.assertEqual(conversation.messages[-2].content, "Hello")
        self.assertEqual(conversation.messages[-1].role, "assistant")
        self.assertEqual(conversation.messages[-1].content, "Hello there!")

        # Check request payload for correct system message handling
        request = responses.calls[0].request
        payload = json.loads(request.body)

        # Should have system parameter separately (not in messages)
        self.assertEqual(payload["system"], "You are a helpful assistant")

        # Messages should only include user and assistant, not system
        self.assertEqual(len(payload["messages"]), 3)
        self.assertEqual(payload["messages"][0]["role"], "user")
        self.assertEqual(payload["messages"][0]["content"], "Initial message")
        self.assertEqual(payload["messages"][1]["role"], "assistant")
        self.assertEqual(payload["messages"][1]["content"], "Initial response")
        self.assertEqual(payload["messages"][2]["role"], "user")
        self.assertEqual(payload["messages"][2]["content"], "Hello")

    @responses.activate
    def test_query_without_system_message(self):
        """Test query with conversation but without system message."""
        # Mock the Claude API response
        responses.add(
            responses.POST,
            "https://api.anthropic.com/v1/messages",
            json={"content": [{"text": "Hello there!"}]},
            status=200,
        )

        # Create a conversation without system message
        conversation = Conversation()
        conversation.add_user_message("Initial message")

        # Query with additional user message
        response = self.provider.query("Hello", conversation)

        # Check request payload - should not have system parameter
        request = responses.calls[0].request
        payload = json.loads(request.body)
        self.assertNotIn("system", payload)

    @responses.activate
    def test_query_without_conversation(self):
        """Test query without conversation context."""
        responses.add(
            responses.POST,
            "https://api.anthropic.com/v1/messages",
            json={"content": [{"text": "Hello there!"}]},
            status=200,
        )

        response = self.provider.query("Hello")

        self.assertEqual(response, "Hello there!")

        # Check request payload
        request = responses.calls[0].request
        payload = json.loads(request.body)
        self.assertEqual(len(payload["messages"]), 1)
        self.assertEqual(payload["messages"][0]["role"], "user")
        self.assertEqual(payload["messages"][0]["content"], "Hello")
        self.assertNotIn("system", payload)

    @responses.activate
    def test_api_error(self):
        """Test handling of API errors."""
        responses.add(
            responses.POST,
            "https://api.anthropic.com/v1/messages",
            json={"error": {"message": "Invalid API key"}},
            status=401,
        )

        with patch.object(self.provider, "_error_response") as mock_error:
            mock_error.return_value = '{"error": true}'
            response = self.provider.query("Hello")
            self.assertEqual(response, '{"error": true}')

    def test_simulate_response(self):
        """Test simulated response generation."""
        provider = ClaudeProvider(api_key=None)
        response = provider._simulate_response("Hello")

        # Verify response is valid JSON and has expected fields
        data = json.loads(response)
        self.assertIn("task_type", data)
        self.assertIn("technologies", data)
        self.assertIn("components", data)
        self.assertIn("dependencies", data)
        self.assertIn("complexity_estimate", data)
        self.assertIn("key_features", data)
        self.assertIn("concerns", data)

    def test_error_response(self):
        """Test error response generation."""
        error_response = self.provider._error_response("Test error")

        # Verify response is valid JSON and has expected fields
        data = json.loads(error_response)
        self.assertEqual(data["task_type"], "unknown")
        self.assertEqual(
            data["concerns"], ["Error connecting to Claude service: Test error"]
        )


class TestLLMInterface(unittest.TestCase):
    """Tests for the LLMInterface class."""

    def test_init_default(self):
        """Test initialization with default parameters."""
        with patch("src.utils.llm_client.OpenAIProvider") as mock_provider:
            mock_provider.return_value = MagicMock()
            interface = LLMInterface()

            self.assertEqual(interface.provider_name, "openai")
            mock_provider.assert_called_once_with(
                api_key=None, model="gpt-4", temperature=0.5
            )

    def test_init_custom_provider(self):
        """Test initialization with custom provider."""
        with patch("src.utils.llm_client.GeminiProvider") as mock_provider:
            mock_provider.return_value = MagicMock()
            interface = LLMInterface(
                provider="gemini", api_key="test_key", model="gemini-pro"
            )

            self.assertEqual(interface.provider_name, "gemini")
            mock_provider.assert_called_once_with(
                api_key="test_key", model="gemini-pro", temperature=0.5
            )

    def test_init_invalid_provider(self):
        """Test initialization with invalid provider."""
        with self.assertRaises(ValueError):
            LLMInterface(provider="invalid_provider")

    def test_start_conversation(self):
        """Test starting a new conversation."""
        interface = LLMInterface(provider="openai")
        interface.start_conversation("You are a helpful assistant")

        self.assertIsNotNone(interface.conversation)
        self.assertEqual(len(interface.conversation.messages), 1)
        self.assertEqual(interface.conversation.messages[0].role, "system")
        self.assertEqual(
            interface.conversation.messages[0].content, "You are a helpful assistant"
        )

    def test_start_conversation_without_system(self):
        """Test starting a conversation without system message."""
        interface = LLMInterface(provider="openai")
        interface.start_conversation()

        self.assertIsNotNone(interface.conversation)
        self.assertEqual(len(interface.conversation.messages), 0)

    def test_update_system_message(self):
        """Test updating system message in a conversation."""
        interface = LLMInterface(provider="openai")
        interface.start_conversation("You are a helpful assistant")
        interface.update_system_message("You are a coding assistant")

        self.assertEqual(interface.conversation.messages[0].role, "system")
        self.assertEqual(
            interface.conversation.messages[0].content, "You are a coding assistant"
        )

    def test_update_system_no_conversation(self):
        """Test updating system message without starting conversation."""
        interface = LLMInterface(provider="openai")

        with self.assertRaises(ValueError):
            interface.update_system_message("You are a helpful assistant")

    def test_query_without_conversation(self):
        """Test querying without using conversation."""
        mock_provider = MagicMock()
        mock_provider.query.return_value = "Test response"

        with patch("src.utils.llm_client.OpenAIProvider", return_value=mock_provider):
            interface = LLMInterface(provider="openai")
            response = interface.query("Hello")

            self.assertEqual(response, "Test response")
            mock_provider.query.assert_called_once_with("Hello", None)

    def test_query_with_conversation(self):
        """Test querying using conversation context."""
        mock_provider = MagicMock()
        mock_provider.query.return_value = "Test response"

        with patch("src.utils.llm_client.OpenAIProvider", return_value=mock_provider):
            interface = LLMInterface(provider="openai")
            interface.start_conversation("You are a helpful assistant")

            response = interface.query("Hello", use_conversation=True)

            self.assertEqual(response, "Test response")
            mock_provider.query.assert_called_once()
            # The conversation object should have been passed to provider.query
            self.assertEqual(mock_provider.query.call_args[0][0], "Hello")
            self.assertEqual(
                mock_provider.query.call_args[0][1], interface.conversation
            )

    def test_query_with_conversation_not_started(self):
        """Test querying with conversation when no conversation is started."""
        interface = LLMInterface(provider="openai")

        with self.assertRaises(ValueError):
            interface.query("Hello", use_conversation=True)

    def test_end_conversation(self):
        """Test ending a conversation."""
        interface = LLMInterface(provider="openai")
        interface.start_conversation("You are a helpful assistant")
        interface.end_conversation()

        self.assertIsNone(interface.conversation)

    def test_end_conversation_not_started(self):
        """Test ending a conversation that wasn't started."""
        interface = LLMInterface(provider="openai")
        # Should not raise an error
        interface.end_conversation()
        self.assertIsNone(interface.conversation)

    def test_get_conversation_history(self):
        """Test getting conversation history."""
        interface = LLMInterface(provider="openai")
        interface.start_conversation("You are a helpful assistant")

        # Mock the conversation's get_messages method
        interface.conversation.get_messages = MagicMock(
            return_value=[
                {"role": "system", "content": "You are a helpful assistant"},
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there!"},
            ]
        )

        history = interface.get_conversation_history()

        self.assertEqual(len(history), 3)
        interface.conversation.get_messages.assert_called_once()

    def test_get_conversation_history_not_started(self):
        """Test getting conversation history when no conversation is started."""
        interface = LLMInterface(provider="openai")

        with self.assertRaises(ValueError):
            interface.get_conversation_history()

    def test_switch_provider(self):
        """Test switching provider."""
        mock_openai = MagicMock()
        mock_gemini = MagicMock()

        with patch(
            "src.utils.llm_client.OpenAIProvider", return_value=mock_openai
        ), patch("src.utils.llm_client.GeminiProvider", return_value=mock_gemini):

            interface = LLMInterface(provider="openai")
            self.assertEqual(interface.provider_name, "openai")
            self.assertEqual(interface.provider, mock_openai)

            interface.switch_provider("gemini", api_key="test_key")
            self.assertEqual(interface.provider_name, "gemini")
            self.assertEqual(interface.provider, mock_gemini)

    def test_integration_flow(self):
        """Test a complete integration flow across the interface."""
        # Create mocks
        mock_openai = MagicMock()
        mock_openai.query.return_value = "OpenAI response"

        mock_claude = MagicMock()
        mock_claude.query.return_value = "Claude response"

        with patch(
            "src.utils.llm_client.OpenAIProvider", return_value=mock_openai
        ), patch("src.utils.llm_client.ClaudeProvider", return_value=mock_claude):

            # Initialize with OpenAI
            interface = LLMInterface(provider="openai")

            # Start conversation with system message
            interface.start_conversation("You are a helpful assistant")

            # Send query using conversation
            response1 = interface.query("Hello", use_conversation=True)
            self.assertEqual(response1, "OpenAI response")

            # Switch to Claude
            interface.switch_provider("claude")

            # Update system message
            interface.update_system_message("You are a coding assistant")

            # Send query using conversation
            response2 = interface.query(
                "Can you help with Python?", use_conversation=True
            )
            self.assertEqual(response2, "Claude response")

            # Get conversation history
            # Mock the history for testing
            interface.conversation.get_messages = MagicMock(
                return_value=[
                    {"role": "system", "content": "You are a coding assistant"},
                    {"role": "user", "content": "Hello"},
                    {"role": "assistant", "content": "OpenAI response"},
                    {"role": "user", "content": "Can you help with Python?"},
                    {"role": "assistant", "content": "Claude response"},
                ]
            )

            history = interface.get_conversation_history()
            self.assertEqual(len(history), 5)

            # End conversation
            interface.end_conversation()
            self.assertIsNone(interface.conversation)


@responses.activate
class TestIntegration(unittest.TestCase):
    """Integration tests across multiple components."""

    def test_openai_real_api_flow(self):
        """Test a complete flow with mocked OpenAI API."""
        # Mock the OpenAI API response for two consecutive calls
        responses.add(
            responses.POST,
            "https://api.openai.com/v1/chat/completions",
            json={"choices": [{"message": {"content": "Hello there!"}}]},
            status=200,
        )
        responses.add(
            responses.POST,
            "https://api.openai.com/v1/chat/completions",
            json={"choices": [{"message": {"content": "I can help with Python!"}}]},
            status=200,
        )

        # Create interface with OpenAI
        interface = LLMInterface(provider="openai", api_key="test_key")

        # Start conversation
        interface.start_conversation("You are a coding assistant")

        # Send first query
        response1 = interface.query("Hello", use_conversation=True)
        self.assertEqual(response1, "Hello there!")

        # Send second query
        response2 = interface.query("Can you help with Python?", use_conversation=True)
        self.assertEqual(response2, "I can help with Python!")

        # Check conversation history
        history = interface.get_conversation_history()
        self.assertEqual(len(history), 5)  # system + 2 user + 2 assistant

        # Verify the API was called twice with expected payloads
        self.assertEqual(len(responses.calls), 2)

        # First call should have system and user message
        payload1 = json.loads(responses.calls[0].request.body)
        self.assertEqual(len(payload1["messages"]), 2)
        self.assertEqual(payload1["messages"][0]["role"], "system")
        self.assertEqual(
            payload1["messages"][0]["content"], "You are a coding assistant"
        )
        self.assertEqual(payload1["messages"][1]["role"], "user")
        self.assertEqual(payload1["messages"][1]["content"], "Hello")

        # Second call should have system, user, assistant, and second user
        payload2 = json.loads(responses.calls[1].request.body)
        self.assertEqual(len(payload2["messages"]), 4)
        self.assertEqual(payload2["messages"][0]["role"], "system")
        self.assertEqual(payload2["messages"][1]["role"], "user")
        self.assertEqual(payload2["messages"][1]["content"], "Hello")
        self.assertEqual(payload2["messages"][2]["role"], "assistant")
        self.assertEqual(payload2["messages"][2]["content"], "Hello there!")
        self.assertEqual(payload2["messages"][3]["role"], "user")
        self.assertEqual(
            payload2["messages"][3]["content"], "Can you help with Python?"
        )

    def test_provider_switching_with_conversation(self):
        """Test switching providers while maintaining conversation context."""
        # Mock OpenAI API response
        responses.add(
            responses.POST,
            "https://api.openai.com/v1/chat/completions",
            json={"choices": [{"message": {"content": "OpenAI response"}}]},
            status=200,
        )

        # Mock Claude API response
        responses.add(
            responses.POST,
            "https://api.anthropic.com/v1/messages",
            json={"content": [{"text": "Claude response"}]},
            status=200,
        )

        # Create interface with OpenAI
        interface = LLMInterface(provider="openai", api_key="test_key")

        # Start conversation
        interface.start_conversation("You are a helpful assistant")

        # Query OpenAI
        response1 = interface.query("Hello from OpenAI", use_conversation=True)
        self.assertEqual(response1, "OpenAI response")

        # Switch to Claude
        interface.switch_provider("claude", api_key="test_key")

        # Query Claude with same conversation
        response2 = interface.query("Hello from Claude", use_conversation=True)
        self.assertEqual(response2, "Claude response")

        # Check conversation history - should include both interactions
        history = interface.get_conversation_history()
        self.assertEqual(len(history), 5)  # system + 2 user + 2 assistant

        # Verify the correct API formatting for each provider
        openai_request = json.loads(responses.calls[0].request.body)
        claude_request = json.loads(responses.calls[1].request.body)

        # OpenAI puts system in messages array
        self.assertEqual(openai_request["messages"][0]["role"], "system")

        # Claude has separate system parameter
        self.assertEqual(claude_request["system"], "You are a helpful assistant")
        self.assertNotIn("system", [msg["role"] for msg in claude_request["messages"]])


class TestEdgeCases(unittest.TestCase):
    """Tests for edge cases and error handling."""

    def test_empty_prompt(self):
        """Test handling of empty prompt."""
        mock_provider = MagicMock()
        mock_provider.query.return_value = "Response to empty prompt"

        with patch("src.utils.llm_client.OpenAIProvider", return_value=mock_provider):
            interface = LLMInterface(provider="openai")

            # Query with empty string
            response = interface.query("")

            # Provider should be called with empty string
            mock_provider.query.assert_called_once_with("", None)
            self.assertEqual(response, "Response to empty prompt")

    def test_unicode_handling(self):
        """Test handling of Unicode characters in prompt and response."""
        unicode_prompt = "你好，世界! Здравствуй, мир! مرحبا بالعالم!"
        unicode_response = (
            "Hello in multiple languages: 你好，世界! Здравствуй, мир! مرحبا بالعالم!"
        )

        mock_provider = MagicMock()
        mock_provider.query.return_value = unicode_response

        with patch("src.utils.llm_client.OpenAIProvider", return_value=mock_provider):
            interface = LLMInterface(provider="openai")
            response = interface.query(unicode_prompt)

            mock_provider.query.assert_called_once_with(unicode_prompt, None)
            self.assertEqual(response, unicode_response)

    def test_extremely_long_prompt(self):
        """Test handling of extremely long prompt."""
        long_prompt = "test " * 10000  # 50,000 characters

        mock_provider = MagicMock()
        mock_provider.query.return_value = "Response to long prompt"

        with patch("src.utils.llm_client.OpenAIProvider", return_value=mock_provider):
            interface = LLMInterface(provider="openai")
            response = interface.query(long_prompt)

            mock_provider.query.assert_called_once()
            self.assertEqual(mock_provider.query.call_args[0][0], long_prompt)
            self.assertEqual(response, "Response to long prompt")

    def test_timeout_handling(self):
        """Test handling of request timeout."""
        with patch("requests.post") as mock_post:
            mock_post.side_effect = requests.Timeout("Request timed out")

            # Set up interface with OpenAI
            interface = LLMInterface(provider="openai", api_key="test_key")

            # Call should not raise an exception but return error response
            response = interface.query("Hello")

            # Ensure response contains error information
            self.assertIn("Error connecting to OpenAI service", response)

    def test_rate_limit_handling(self):
        """Test handling of rate limit errors."""
        # Mock a rate limit response from OpenAI
        rate_limit_response = {
            "error": {
                "message": "Rate limit exceeded",
                "type": "rate_limit_exceeded",
                "param": None,
                "code": "rate_limit_exceeded",
            }
        }

        with patch("requests.post") as mock_post:
            mock_response = MagicMock()
            mock_response.status_code = 429
            mock_response.json.return_value = rate_limit_response
            mock_response.text = json.dumps(rate_limit_response)
            mock_post.return_value = mock_response

            # Set up interface with OpenAI
            interface = LLMInterface(provider="openai", api_key="test_key")

            # Call should not raise an exception but return error response
            response = interface.query("Hello")

            # Ensure response contains error information
            self.assertIn("Error connecting to OpenAI service", response)


if __name__ == "__main__":
    unittest.main()
