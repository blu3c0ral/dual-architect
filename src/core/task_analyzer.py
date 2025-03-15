import json
from typing import Dict, Any, List, Optional


class TaskAnalyzer:
    """
    Handles analysis of user requirements and converts them into structured task information.
    Acts as the entry point for the dual-architect system.
    """

    def __init__(self, llm_service=None):
        """
        Initialize the TaskAnalyzer.

        Args:
            llm_service: Service for LLM interactions (defaults to None, will use default service)
        """
        from src.utils.llm_client import LLMInterface

        self.llm_service = llm_service if llm_service else LLMInterface()

    def analyze_requirements(self, user_requirements: str) -> Dict[str, Any]:
        """
        Process user requirements and extract structured task information.

        Args:
            user_requirements: Raw text of user requirements

        Returns:
            Dictionary containing structured task information
        """
        # Generate prompt for LLM to extract task details
        prompt = self._generate_analysis_prompt(user_requirements)

        # Get response from LLM
        response = self.llm_service.query(prompt)

        # Parse and validate the response
        task_structure = self._parse_llm_response(response)

        return task_structure

    def _generate_analysis_prompt(self, user_requirements: str) -> str:
        """
        Create a prompt for the LLM to extract structured information from requirements.

        Args:
            user_requirements: Raw text of user requirements

        Returns:
            Formatted prompt for the LLM
        """
        return f"""
        You are a software architecture assistant. Analyze the following user requirements 
        and extract structured information about the development task.
        
        USER REQUIREMENTS:
        ```
        {user_requirements}
        ```
        
        Please provide a structured JSON response with the following information:
        1. "task_type": The general category of the task (e.g., "web application", "API", "data processing")
        2. "technologies": List of relevant technologies that should be used
        3. "components": Major components or modules needed for the solution
        4. "dependencies": External libraries or services required
        5. "complexity_estimate": A score from 1-10 indicating project complexity
        6. "key_features": List of the most important features to implement
        7. "concerns": Any potential challenges or considerations
        
        Format your response as valid JSON. Do not include any explanations or text outside the JSON structure.
        """

    def _parse_llm_response(self, response: str) -> Dict[str, Any]:
        """
        Parse and validate the LLM response into a structured format.

        Args:
            response: Raw text response from the LLM

        Returns:
            Dictionary containing structured task information
        """
        try:
            # Extract JSON from response (in case there's additional text)
            json_start = response.find("{")
            json_end = response.rfind("}") + 1

            if json_start == -1 or json_end == 0:
                raise ValueError("No valid JSON found in the response")

            json_content = response[json_start:json_end]
            parsed_data = json.loads(json_content)

            # Validate the required fields
            required_fields = [
                "task_type",
                "technologies",
                "components",
                "key_features",
            ]
            for field in required_fields:
                if field not in parsed_data:
                    parsed_data[field] = []

            return parsed_data

        except Exception as e:
            # In case of parsing error, return a basic structure
            print(f"Error parsing LLM response: {e}")
            return {
                "task_type": "unknown",
                "technologies": [],
                "components": [],
                "dependencies": [],
                "complexity_estimate": 5,
                "key_features": [],
                "concerns": ["Unable to parse LLM response correctly"],
            }
