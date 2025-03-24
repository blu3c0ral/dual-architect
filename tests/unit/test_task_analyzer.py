import unittest
import json
from unittest import mock
from src.core.task_analyzer import TaskAnalyzer
from unittest.mock import MagicMock, patch
from src.utils.logger import setup_logger, cleanup_all_loggers


class TestTaskAnalyzer(unittest.TestCase):

    def setUp(self):
        # Setup a test logger that doesn't output to console
        self.test_logger = setup_logger(
            "test_task_analyzer", level="CRITICAL", console_output=False
        )

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

        # Create the analyzer with the mock and set log level to CRITICAL to minimize output
        self.analyzer = TaskAnalyzer(llm_service=self.mock_llm, log_level="CRITICAL")

    def tearDown(self):
        # Clean up all loggers to prevent resource leaks
        cleanup_all_loggers()

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
        # Check that original requirements are included
        self.assertEqual(result["original_requirements"], requirements)

    def test_parse_invalid_response(self):
        # Test handling of invalid responses
        self.mock_llm.query.return_value = "This is not a valid JSON response"

        requirements = "Build a simple application"
        result = self.analyzer.analyze_requirements(requirements)

        # Check that we got default values for invalid response
        self.assertEqual(result["task_type"], "unknown")
        self.assertIn("Unable to parse LLM response correctly", result["concerns"])
        # Check that original requirements are still included even with parsing failure
        self.assertEqual(result["original_requirements"], requirements)

    def test_parse_non_list_values(self):
        # Test handling of non-list values for list fields
        self.mock_llm.query.return_value = json.dumps(
            {
                "task_type": "web application",
                "technologies": "Python",  # Not a list
                "components": ["Frontend", "Backend"],
                "dependencies": "Flask",  # Not a list
                "complexity_estimate": "6",  # Not an integer
                "key_features": ["Feature 1"],
                "concerns": "Security",  # Not a list
            }
        )

        requirements = "Build a web application with Python and Flask"
        result = self.analyzer.analyze_requirements(requirements)

        # Check that non-list values are converted to lists
        self.assertIsInstance(result["technologies"], list)
        self.assertEqual(result["technologies"], ["Python"])
        self.assertIsInstance(result["dependencies"], list)
        self.assertEqual(result["dependencies"], ["Flask"])
        self.assertIsInstance(result["concerns"], list)
        self.assertEqual(result["concerns"], ["Security"])
        # Check that string complexity is converted to int
        self.assertEqual(result["complexity_estimate"], 6)

    def test_missing_optional_fields(self):
        # Test handling of missing optional fields
        self.mock_llm.query.return_value = json.dumps(
            {
                "task_type": "web application",
                "technologies": ["Python", "Flask"],
                "components": ["Frontend", "Backend"],
                "dependencies": ["Flask"],
                "complexity_estimate": 5,
                "key_features": ["Feature 1"],
                "concerns": ["Security"],
                # Missing optional fields
            }
        )

        requirements = "Build a simple web app"
        result = self.analyzer.analyze_requirements(requirements)

        # Check that optional fields are added with default values
        self.assertIn("component_priorities", result)
        self.assertEqual(result["component_priorities"], [])
        self.assertIn("data_model", result)
        self.assertEqual(result["data_model"], {"entities": []})
        self.assertIn("constraints", result)
        self.assertEqual(result["constraints"], [])
        self.assertIn("alternatives_considered", result)
        self.assertEqual(result["alternatives_considered"], [])
        self.assertIn("requirement_gaps", result)
        self.assertEqual(result["requirement_gaps"], [])

    def test_invalid_nested_data_model(self):
        # Test handling of invalid nested structures
        self.mock_llm.query.return_value = json.dumps(
            {
                "task_type": "web application",
                "data_model": {"entities": "invalid"},  # Should be a list, not a string
            }
        )
        result = self.analyzer.analyze_requirements("Invalid nested structure test")
        self.assertEqual(
            result["data_model"], {"entities": []}
        )  # Should fallback to default

    def test_invalid_alternatives_considered(self):
        # Test handling of invalid 'alternatives_considered' structure
        self.mock_llm.query.return_value = json.dumps(
            {
                "task_type": "web application",
                "alternatives_considered": "invalid",  # Should be a list, not a string
            }
        )
        result = self.analyzer.analyze_requirements("Invalid alternatives test")
        self.assertEqual(
            result["alternatives_considered"], []
        )  # Should fallback to default

    def test_empty_requirements(self):
        # Override the default mock response for this test
        self.mock_llm.query.return_value = "Invalid JSON"
        result = self.analyzer.analyze_requirements("")
        self.assertEqual(result["task_type"], "unknown")
        self.assertEqual(result["original_requirements"], "")

    def test_large_requirements(self):
        # Test handling of very large user requirements
        large_requirements = "Build a system. " * 1000  # Large input
        result = self.analyzer.analyze_requirements(large_requirements)
        self.assertEqual(result["original_requirements"], large_requirements)

    @patch("src.utils.logger.logging.getLogger")
    def test_logging_on_error(self, mock_get_logger):
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger

        # Set the analyzer logger to our mock
        self.analyzer.logger = mock_logger

        # Override the mock response to trigger a JSON parsing error
        self.mock_llm.query.return_value = "Invalid JSON"

        result = self.analyzer.analyze_requirements("Test logging")

        # Verify the logger was called with an error
        mock_logger.error.assert_called_with(
            mock.ANY
        )  # Use mock.ANY to match any error message

    def test_default_logger(self):
        analyzer = TaskAnalyzer(llm_service=self.mock_llm)
        analyzer.logger = None  # Simulate missing logger
        # Override the default mock response for this test
        self.mock_llm.query.return_value = "Invalid JSON"
        result = analyzer.analyze_requirements("Test default logger")
        self.assertEqual(result["task_type"], "unknown")


if __name__ == "__main__":
    unittest.main()
