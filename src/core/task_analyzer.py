import json
from typing import Dict, Any, List, Optional, Tuple

from src.utils.prompt_loader import load_prompt
from src.utils.logger import setup_logger


class TaskAnalyzer:
    """
    Handles analysis of user requirements and converts them into structured task information.
    Acts as the entry point for the dual-architect system.
    """

    def __init__(self, llm_service=None, log_level="INFO"):
        """
        Initialize the TaskAnalyzer.

        Args:
            llm_service: Service for LLM interactions (defaults to None, will use default service)
        """
        from src.utils.llm_client import LLMInterface

        self.llm_service = llm_service if llm_service else LLMInterface()
        self.logger = setup_logger("task_analyzer", level=log_level)

    def analyze_requirements(self, user_requirements: str) -> Dict[str, Any]:
        """
        Process user requirements and extract structured task information.

        Args:
            user_requirements: Raw text of user requirements

        Returns:
            Dictionary containing structured task information. Some are optional, see task_analyzer.yaml analysis_prompt for details.:
            {
                "task_type": str
                "technologies": list[str],
                "components": list[str],
                "dependencies": list[str],
                "complexity_estimate": int,
                "key_features": list[str],
                "component_priorities": list[{"name": str, "priority": int}],
                "data_model": {
                    "entities": list[{"names: str, "attributes": list[{"name": str, "type": str, "required": bool}], "relationships": list[{"type": str, "target": str}]}],
                },
                "constraints": list[str],
                "concerns": list[str],
                "alternatives_considered": list[{"technology": str, "pros": list[str], "cons": list[str]}],
                "requirement_gaps": list[str],
                "original_requirements": str
            }
        """
        # Generate prompt for LLM to extract task details
        prompt = self._generate_analysis_prompt(user_requirements)

        # Get response from LLM
        response = self.llm_service.query(prompt)

        # Parse and validate the response
        task_structure = self._parse_llm_response(response)

        if task_structure and isinstance(task_structure, dict):
            task_structure["original_requirements"] = user_requirements

        return task_structure

    def _generate_analysis_prompt(self, user_requirements: str) -> str:
        """
        Create a prompt for the LLM to extract structured information from requirements.

        Args:
            user_requirements: Raw text of user requirements

        Returns:
            Formatted prompt for the LLM
        """
        prompt_template = load_prompt("task_analyzer", "analysis_prompt")
        return prompt_template.format(user_requirements=user_requirements)

    def _parse_llm_response(self, response: str) -> Dict[str, Any]:
        """
        Parse and validate the LLM response into a structured format.

        Args:
            response: Raw text response from the LLM

        Returns:
            Dictionary containing structured task information
            {
                "task_type": str
                "technologies": list[str],
                "components": list[str],
                "dependencies": list[str],
                "complexity_estimate": int,
                "key_features": list[str],
                "component_priorities": list[{"name": str, "priority": int}],
                "data_model": {
                    "entities": list[{"names: str, "attributes": list[{"name": str, "type": str, "required": bool}], "relationships": list[{"type": str, "target": str}]}],
                },
                "constraints": list[str],
                "concerns": list[str],
                "alternatives_considered": list[{"technology": str, "pros": list[str], "cons": list[str]}],
                "requirement_gaps": list[str]
            }
        """
        # Use class logger or create one if needed
        logger = self.logger
        if logger is None:
            logger = setup_logger("task_analyzer", level="INFO")

        try:
            # Extract JSON from response (in case there's additional text)
            json_start = response.find("{")
            json_end = response.rfind("}") + 1

            if json_start == -1 or json_end == 0:
                raise ValueError("No valid JSON found in the response.")

            json_content = response[json_start:json_end]
            parsed_data = json.loads(json_content)

            # Required fields and their default values
            required_fields = {
                "task_type": "unknown",
                "technologies": [],
                "components": [],
                "dependencies": [],
                "complexity_estimate": 5,
                "key_features": [],
                "concerns": [],
            }

            # Ensure all required fields are present with valid types
            for field, default in required_fields.items():
                if field not in parsed_data:
                    parsed_data[field] = default
                elif isinstance(default, list) and not isinstance(
                    parsed_data[field], list
                ):
                    # Convert non-list values to lists for list fields
                    parsed_data[field] = (
                        [parsed_data[field]] if parsed_data[field] else []
                    )
                elif field == "complexity_estimate" and not isinstance(
                    parsed_data[field], int
                ):
                    # Ensure complexity_estimate is an integer
                    parsed_data[field] = int(parsed_data.get(field, default))

            # Optional fields with default values
            optional_fields = {
                "component_priorities": [],
                "data_model": {"entities": []},
                "constraints": [],
                "alternatives_considered": [],
                "requirement_gaps": [],
            }

            # Ensure all optional fields are present with valid types
            for field, default in optional_fields.items():
                if field not in parsed_data:
                    parsed_data[field] = default

            # Add additional validation for nested structures
            if "data_model" in parsed_data and isinstance(
                parsed_data["data_model"], dict
            ):
                if "entities" in parsed_data["data_model"] and not isinstance(
                    parsed_data["data_model"]["entities"], list
                ):
                    parsed_data["data_model"]["entities"] = []

            if "alternatives_considered" in parsed_data and not isinstance(
                parsed_data["alternatives_considered"], list
            ):
                parsed_data["alternatives_considered"] = []

            return parsed_data

        except (ValueError, TypeError, json.JSONDecodeError) as e:
            # Log the error and return a default structure
            logger.error(f"Error parsing LLM response: {e}")
            return {
                "task_type": "unknown",
                "technologies": [],
                "components": [],
                "dependencies": [],
                "complexity_estimate": 5,
                "key_features": [],
                "concerns": ["Unable to parse LLM response correctly"],
                "component_priorities": [],
                "data_model": {"entities": []},
                "constraints": [],
                "alternatives_considered": [],
                "requirement_gaps": [],
            }
