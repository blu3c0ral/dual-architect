import os
import json
from typing import Dict, Any, Optional
import requests


class LLMInterface:
    """
    Handles communication with the Large Language Model API.
    """

    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-4"):
        """
        Initialize the LLM interface.

        Args:
            api_key: API key for the LLM service (defaults to environment variable)
            model: Model identifier to use
        """
        self.api_key = api_key or os.environ.get("LLM_API_KEY")
        self.model = model

        if not self.api_key:
            print(
                "Warning: No API key provided. Set the LLM_API_KEY environment variable."
            )

    def query(self, prompt: str) -> str:
        """
        Send a query to the LLM and return the response.

        Args:
            prompt: The prompt to send to the LLM

        Returns:
            Text response from the LLM
        """
        try:
            # Example implementation for OpenAI's API
            # This would be adapted to whatever LLM service you're using
            response = self._call_openai_api(prompt)
            return response
        except Exception as e:
            print(f"Error querying LLM: {e}")
            return json.dumps(
                {
                    "task_type": "unknown",
                    "technologies": [],
                    "components": [],
                    "dependencies": [],
                    "complexity_estimate": 5,
                    "key_features": [],
                    "concerns": ["Error connecting to LLM service"],
                }
            )

    def _call_openai_api(self, prompt: str) -> str:
        """
        Make a call to the OpenAI API.

        Args:
            prompt: The prompt to send to the API

        Returns:
            Text response from the API
        """
        # This is a simplified example. In a real implementation, you would use the
        # appropriate SDK or make proper API calls

        if not self.api_key:
            return self._simulate_response(prompt)

        # Example using requests to call API
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        data = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.3,  # Lower temperature for more deterministic responses
        }

        response = requests.post(
            "https://api.openai.com/v1/chat/completions", headers=headers, json=data
        )

        if response.status_code == 200:
            return response.json()["choices"][0]["message"]["content"]
        else:
            raise Exception(
                f"API call failed with status {response.status_code}: {response.text}"
            )

    def _simulate_response(self, prompt: str) -> str:
        """
        Simulated response when no API key is available (for testing purposes).

        Args:
            prompt: The prompt that would be sent to the API

        Returns:
            A simulated response
        """
        print("Using simulated LLM response (no API key provided)")
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
