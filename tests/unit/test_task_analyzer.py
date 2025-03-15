import unittest
import json
from src.core.task_analyzer import TaskAnalyzer
from unittest.mock import MagicMock, patch


class TestTaskAnalyzer(unittest.TestCase):

    def setUp(self):
        # Create a mock LLM service
        self.mock_llm = MagicMock()
        self.mock_llm.query.return_value = json.dumps(
            {
                "task_type": "web application",
                "technologies": ["Python", "Flask", "React"],
                "components": ["Frontend", "Backend API", "Database"],
                "dependencies": ["React", "Flask", "PostgreSQL"],
                "complexity_estimate": 7,
                "key_features": ["User authentication", "Dashboard", "Data export"],
                "concerns": ["Security", "Performance"],
            }
        )

        # Create the analyzer with the mock
        self.analyzer = TaskAnalyzer(llm_service=self.mock_llm)

    def test_analyze_requirements(self):
        # Sample user requirements
        requirements = """
        Build a web application for project management with user authentication, 
        dashboard for project overview, and ability to export data as CSV.
        """

        # Analyze the requirements
        result = self.analyzer.analyze_requirements(requirements)

        # Check that the LLM was called with an appropriate prompt
        self.mock_llm.query.assert_called_once()
        prompt_arg = self.mock_llm.query.call_args[0][0]
        self.assertIn(requirements, prompt_arg)

        # Check the response structure
        self.assertEqual(result["task_type"], "web application")
        self.assertIn("Python", result["technologies"])
        self.assertIn("Frontend", result["components"])
        self.assertEqual(result["complexity_estimate"], 7)
        self.assertIn("User authentication", result["key_features"])

    def test_parse_invalid_response(self):
        # Test handling of invalid responses
        self.mock_llm.query.return_value = "This is not a valid JSON response"

        requirements = "Build a simple application"
        result = self.analyzer.analyze_requirements(requirements)

        # Check that we got default values for invalid response
        self.assertEqual(result["task_type"], "unknown")
        self.assertIn("Unable to parse LLM response correctly", result["concerns"])


if __name__ == "__main__":
    unittest.main()
