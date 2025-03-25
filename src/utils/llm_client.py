import os
import json
from typing import Dict, List, Any, Optional, Union
import requests
from abc import ABC, abstractmethod

# Configure logging
# Set up logging for the module
import src.utils.logger as setup_logger

logger = setup_logger.setup_logger("llm_client")


class Message:
    """
    Represents a message in a conversation with roles and content.
    """

    def __init__(self, role: str, content: str):
        """
        Initialize a message.

        Args:
            role: The role of the message sender (system, user, assistant)
            content: The content of the message
        """
        self.role = role
        self.content = content

    def to_dict(self) -> Dict[str, str]:
        """Convert message to dictionary format for API requests."""
        return {"role": self.role, "content": self.content}


class Conversation:
    """
    Maintains the state of a conversation with an LLM.
    """

    def __init__(self, system_message: Optional[str] = None):
        """
        Initialize a conversation.

        Args:
            system_message: Optional system message to set the behavior of the LLM
        """
        self.messages: List[Message] = []
        if system_message:
            self.set_system_message(system_message)

    def set_system_message(self, content: str) -> None:
        """
        Set or update the system message for the conversation.

        Args:
            content: The system message content
        """
        # Check if a system message already exists and update it
        for i, message in enumerate(self.messages):
            if message.role == "system":
                self.messages[i] = Message("system", content)
                return

        # Otherwise insert a new system message at the beginning
        self.messages.insert(0, Message("system", content))

    def add_user_message(self, content: str) -> None:
        """
        Add a user message to the conversation.

        Args:
            content: The user message content
        """
        self.messages.append(Message("user", content))

    def add_assistant_message(self, content: str) -> None:
        """
        Add an assistant message to the conversation.

        Args:
            content: The assistant message content
        """
        self.messages.append(Message("assistant", content))

    def get_messages(self) -> List[Dict[str, str]]:
        """
        Get all messages in dictionary format for API requests.

        Returns:
            List of message dictionaries
        """
        return [message.to_dict() for message in self.messages]

    def clear(self) -> None:
        """Clear all messages except the system message."""
        system_message = None
        for message in self.messages:
            if message.role == "system":
                system_message = message
                break

        self.messages = []
        if system_message:
            self.messages.append(system_message)


class LLMProvider(ABC):
    """
    Abstract base class for LLM providers.
    """

    def __init__(self, temperature: float = 0.5):
        """
        Initialize the LLM provider with configurable temperature.

        Args:
            temperature: Controls randomness in response generation.
                         Higher values (e.g., 0.8) make output more random.
                         Lower values (e.g., 0.2) make output more deterministic.
                         Range is typically 0.0 to 1.0.
        """
        self._temperature = temperature

    @abstractmethod
    def query(self, prompt: str, conversation: Optional[Conversation] = None) -> str:
        """
        Send a query to the LLM and return the response.

        Args:
            prompt: The prompt to send to the LLM
            conversation: Optional conversation context

        Returns:
            Text response from the LLM
        """
        pass


class OpenAIProvider(LLMProvider):
    """
    Provider for OpenAI's API (GPT models).
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-4",
        temperature: float = 0.5,
    ):
        """
        Initialize the OpenAI provider.

        Args:
            api_key: API key for OpenAI (defaults to environment variable)
            model: Model identifier to use
        """
        super().__init__(temperature)
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.model = model
        self.api_url = "https://api.openai.com/v1/chat/completions"

        if not self.api_key:
            logger.warning(
                "No OpenAI API key provided. Set the OPENAI_API_KEY environment variable."
            )

    def query(self, prompt: str, conversation: Optional[Conversation] = None) -> str:
        """
        Send a query to OpenAI and return the response.

        Args:
            prompt: The prompt to send
            conversation: Optional conversation context

        Returns:
            Text response from the LLM
        """
        try:
            if not self.api_key:
                return self._simulate_response(prompt)

            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            }

            # Use conversation context if provided, otherwise create a new single-turn conversation
            if conversation:
                conversation.add_user_message(prompt)
                messages = conversation.get_messages()
            else:
                messages = [{"role": "user", "content": prompt}]

            data = {
                "model": self.model,
                "messages": messages,
                "temperature": self._temperature,
            }

            response = requests.post(
                self.api_url, headers=headers, json=data, timeout=30
            )

            if response.status_code == 200:
                content = response.json()["choices"][0]["message"]["content"]

                # Add the assistant's response to the conversation if it exists
                if conversation:
                    conversation.add_assistant_message(content)

                return content
            else:
                error_msg = f"OpenAI API call failed with status {response.status_code}: {response.text}"
                logger.error(error_msg)
                raise Exception(error_msg)

        except Exception as e:
            logger.error(f"Error querying OpenAI: {str(e)}")
            return self._error_response(str(e))

    def _simulate_response(self, prompt: str) -> str:
        """Simulate a response when no API key is available (for testing)."""
        logger.info("Using simulated OpenAI response (no API key provided)")
        return json.dumps(
            {
                "task_type": "web application",
                "technologies": ["Python", "JavaScript", "React"],
                "components": ["Frontend UI", "Backend API", "Database"],
                "dependencies": ["React", "Flask", "SQLAlchemy"],
                "complexity_estimate": 6,
                "key_features": [
                    "User authentication",
                    "Data visualization",
                    "API integration",
                ],
                "concerns": ["Scalability", "Security"],
            }
        )

    def _error_response(self, error: str) -> str:
        """Generate an error response."""
        return json.dumps(
            {
                "task_type": "unknown",
                "technologies": [],
                "components": [],
                "dependencies": [],
                "complexity_estimate": 5,
                "key_features": [],
                "concerns": [f"Error connecting to OpenAI service: {error}"],
            }
        )


class GeminiProvider(LLMProvider):
    """
    Provider for Google's Gemini API.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gemini-pro",
        temperature: float = 0.5,
    ):
        """
        Initialize the Gemini provider.

        Args:
            api_key: API key for Google Gemini (defaults to environment variable)
            model: Model identifier to use
        """
        super().__init__(temperature)
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY")
        self.model = model
        self.api_url = f"https://generativelanguage.googleapis.com/v1/models/{model}:generateContent"

        if not self.api_key:
            logger.warning(
                "No Gemini API key provided. Set the GEMINI_API_KEY environment variable."
            )

    def query(self, prompt: str, conversation: Optional[Conversation] = None) -> str:
        """
        Send a query to Gemini and return the response.

        Args:
            prompt: The prompt to send
            conversation: Optional conversation context

        Returns:
            Text response from the LLM
        """
        try:
            if not self.api_key:
                return self._simulate_response(prompt)

            # Format the conversation for Gemini API
            contents = []

            if conversation:
                conversation.add_user_message(prompt)

                for message in conversation.messages:
                    # Gemini uses different role names than OpenAI
                    role = "user" if message.role == "user" else "model"

                    # System messages in Gemini need special handling
                    if message.role == "system":
                        # Convert system message to a user message with a prefix
                        contents.append(
                            {
                                "role": "user",
                                "parts": [
                                    {"text": f"System instruction: {message.content}"}
                                ],
                            }
                        )
                    else:
                        contents.append(
                            {"role": role, "parts": [{"text": message.content}]}
                        )
            else:
                # Single-turn query
                contents = [{"role": "user", "parts": [{"text": prompt}]}]

            data = {
                "contents": contents,
                "generationConfig": {
                    "temperature": self._temperature,
                },
            }

            # Gemini API requires the key as a query parameter
            url = f"{self.api_url}?key={self.api_key}"
            response = requests.post(url, json=data, timeout=30)

            if response.status_code == 200:
                response_data = response.json()

                if "candidates" in response_data and response_data["candidates"]:
                    content = response_data["candidates"][0]["content"]["parts"][0][
                        "text"
                    ]

                    # Add the assistant's response to the conversation if it exists
                    if conversation:
                        conversation.add_assistant_message(content)

                    return content
                else:
                    error_msg = "No response content found in Gemini API response"
                    logger.error(error_msg)
                    raise Exception(error_msg)
            else:
                error_msg = f"Gemini API call failed with status {response.status_code}: {response.text}"
                logger.error(error_msg)
                raise Exception(error_msg)

        except Exception as e:
            logger.error(f"Error querying Gemini: {str(e)}")
            return self._error_response(str(e))

    def _simulate_response(self, prompt: str) -> str:
        """Simulate a response when no API key is available (for testing)."""
        logger.info("Using simulated Gemini response (no API key provided)")
        return json.dumps(
            {
                "task_type": "mobile application",
                "technologies": ["Kotlin", "Swift", "Firebase"],
                "components": ["UI Layer", "Data Layer", "Network Layer"],
                "dependencies": ["Android SDK", "iOS SDK", "Firebase SDK"],
                "complexity_estimate": 7,
                "key_features": [
                    "Push notifications",
                    "Real-time updates",
                    "Offline mode",
                ],
                "concerns": ["Cross-platform compatibility", "Battery usage"],
            }
        )

    def _error_response(self, error: str) -> str:
        """Generate an error response."""
        return json.dumps(
            {
                "task_type": "unknown",
                "technologies": [],
                "components": [],
                "dependencies": [],
                "complexity_estimate": 5,
                "key_features": [],
                "concerns": [f"Error connecting to Gemini service: {error}"],
            }
        )


class ClaudeProvider(LLMProvider):
    """
    Provider for Anthropic's Claude API.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "claude-3-opus-20240229",
        temperature: float = 0.5,
    ):
        """
        Initialize the Claude provider.

        Args:
            api_key: API key for Anthropic (defaults to environment variable)
            model: Model identifier to use
        """
        super().__init__(temperature)
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        self.model = model
        self.api_url = "https://api.anthropic.com/v1/messages"

        if not self.api_key:
            logger.warning(
                "No Anthropic API key provided. Set the ANTHROPIC_API_KEY environment variable."
            )

    def query(self, prompt: str, conversation: Optional[Conversation] = None) -> str:
        """
        Send a query to Claude and return the response.

        Args:
            prompt: The prompt to send
            conversation: Optional conversation context

        Returns:
            Text response from the LLM
        """
        try:
            if not self.api_key:
                return self._simulate_response(prompt)

            headers = {
                "x-api-key": self.api_key,
                "anthropic-version": "2023-06-01",
                "Content-Type": "application/json",
            }

            # Format conversation for Claude API
            system_prompt = None
            messages = []

            if conversation:
                conversation.add_user_message(prompt)

                # Claude handles system messages separately from conversation messages
                for message in conversation.messages:
                    if message.role == "system":
                        system_prompt = message.content
                    else:
                        messages.append(
                            {"role": message.role, "content": message.content}
                        )
            else:
                # Single-turn query
                messages = [{"role": "user", "content": prompt}]

            data = {
                "model": self.model,
                "messages": messages,
                "temperature": self._temperature,
            }

            # Add system prompt if it exists
            if system_prompt:
                data["system"] = system_prompt

            response = requests.post(
                self.api_url, headers=headers, json=data, timeout=30
            )

            if response.status_code == 200:
                content = response.json()["content"][0]["text"]

                # Add the assistant's response to the conversation if it exists
                if conversation:
                    conversation.add_assistant_message(content)

                return content
            else:
                error_msg = f"Claude API call failed with status {response.status_code}: {response.text}"
                logger.error(error_msg)
                raise Exception(error_msg)

        except Exception as e:
            logger.error(f"Error querying Claude: {str(e)}")
            return self._error_response(str(e))

    def _simulate_response(self, prompt: str) -> str:
        """Simulate a response when no API key is available (for testing)."""
        logger.info("Using simulated Claude response (no API key provided)")
        return json.dumps(
            {
                "task_type": "data analysis pipeline",
                "technologies": ["Python", "Pandas", "Airflow"],
                "components": [
                    "Data Ingestion",
                    "Processing Layer",
                    "Analytics Dashboard",
                ],
                "dependencies": ["Pandas", "NumPy", "Scikit-learn", "Apache Airflow"],
                "complexity_estimate": 8,
                "key_features": [
                    "Automated ETL processes",
                    "Machine learning integration",
                    "Real-time analytics",
                ],
                "concerns": ["Data quality", "Processing speed", "Scalability"],
            }
        )

    def _error_response(self, error: str) -> str:
        """Generate an error response."""
        return json.dumps(
            {
                "task_type": "unknown",
                "technologies": [],
                "components": [],
                "dependencies": [],
                "complexity_estimate": 5,
                "key_features": [],
                "concerns": [f"Error connecting to Claude service: {error}"],
            }
        )


class LLMInterface:
    """
    Unified interface for working with multiple LLM providers.
    """

    def __init__(
        self,
        provider: str = "openai",
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        system_message: Optional[str] = None,
        temperature: float = 0.5,
    ):
        """
        Initialize the LLM interface.

        Args:
            provider: The LLM provider to use ('openai', 'gemini', or 'claude')
            api_key: API key for the selected provider
            model: Model identifier to use (provider-specific default if None)
        """
        if not 0 <= temperature <= 1:
            logger.warning(
                f"Temperature {temperature} outside recommended range [0.0, 1.0], clamping."
            )
            temperature = max(0, min(temperature, 1))
        self.temperature = temperature
        self.provider_name = provider.lower()
        self.provider = self._initialize_provider(provider, api_key, model)
        self.conversation = None
        if system_message:
            self._start_conversation(system_message)

    def _initialize_provider(
        self,
        provider: str,
        api_key: Optional[str],
        model: Optional[str],
    ) -> LLMProvider:
        """
        Initialize the appropriate provider based on the selection.

        Args:
            provider: The LLM provider to use
            api_key: API key for the selected provider
            model: Model identifier to use

        Returns:
            An initialized LLM provider
        """
        if provider.lower() == "openai":
            return OpenAIProvider(
                api_key=api_key, model=model or "gpt-4", temperature=self.temperature
            )
        elif provider.lower() == "gemini":
            return GeminiProvider(
                api_key=api_key,
                model=model or "gemini-pro",
                temperature=self.temperature,
            )
        elif provider.lower() == "claude":
            return ClaudeProvider(
                api_key=api_key,
                model=model or "claude-3-opus-20240229",
                temperature=self.temperature,
            )
        else:
            raise ValueError(
                f"Unsupported provider: {provider}. Choose from 'openai', 'gemini', or 'claude'."
            )

    def _start_conversation(self, system_message: Optional[str] = None) -> None:
        """
        Start a new conversation with the LLM.

        Args:
            system_message: Optional system message to set the behavior of the LLM
        """
        self.conversation = Conversation(system_message)
        logger.info(f"Started new conversation with {self.provider_name} provider")

    def update_system_message(self, system_message: str) -> None:
        """
        Update the system message in the current conversation.

        Args:
            system_message: The new system message

        Raises:
            ValueError: If no conversation has been started
        """
        if not self.conversation:
            self._start_conversation(system_message)
            return

        self.conversation.set_system_message(system_message)
        logger.info("Updated system message in the current conversation")

    def query(self, prompt: str, use_conversation: bool = False) -> str:
        """
        Send a query to the LLM and return the response.

        Args:
            prompt: The prompt to send to the LLM
            use_conversation: Whether to use the ongoing conversation context

        Returns:
            Text response from the LLM

        Raises:
            ValueError: If use_conversation is True but no conversation has been started
        """
        if use_conversation and not self.conversation:
            self._start_conversation()

        conversation = self.conversation if use_conversation else None
        return self.provider.query(prompt, conversation)

    def end_conversation(self) -> None:
        """End the current conversation."""
        if self.conversation:
            self.conversation = None
            logger.info("Ended conversation")

    def get_conversation_history(self) -> List[Dict[str, str]]:
        """
        Get the full conversation history.

        Returns:
            List of message dictionaries

        Raises:
            ValueError: If no conversation has been started
        """
        if not self.conversation:
            raise ValueError(
                "No conversation started. Call start_conversation() first."
            )

        return self.conversation.get_messages()

    def switch_provider(
        self,
        provider: str,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
    ) -> None:
        """
        Switch to a different LLM provider.

        Args:
            provider: The LLM provider to use ('openai', 'gemini', or 'claude')
            api_key: API key for the selected provider
            model: Model identifier to use (provider-specific default if None)
            temperature: Optional temperature setting for the new provider
        """
        if temperature:
            self.temperature = temperature
        self.provider_name = provider.lower()
        self.provider = self._initialize_provider(provider, api_key, model)
        logger.info(f"Switched to {provider} provider")
