import unittest
from unittest.mock import MagicMock, patch
import tempfile
import os
import json
from pathlib import Path

from src.core.orchestrator import Pipeline, PipelineConfig, PipelineStage


class TestPipelineOrchestrator(unittest.TestCase):

    def setUp(self):
        # Create a mock task analyzer
        self.mock_task_analyzer = MagicMock()

        # Create a temporary directory for outputs
        self.temp_dir = tempfile.TemporaryDirectory()

        # Basic config for testing
        self.config = PipelineConfig(
            output_dir=self.temp_dir.name, save_intermediate=True
        )

    def tearDown(self):
        """Clean up resources after each test."""
        # Clean up temp directory
        self.temp_dir.cleanup()

        # Clean up all loggers
        from src.utils.logger import cleanup_all_loggers

        # This will close all handlers and return the names of the loggers that were cleaned
        cleaned_loggers = cleanup_all_loggers()

        # Optionally print which loggers were cleaned (helpful for debugging)
        if cleaned_loggers:
            print(f"Cleaned up the following loggers: {', '.join(cleaned_loggers)}")

    def test_pipeline_initialization(self):
        """Test that pipeline initializes correctly with components."""
        pipeline = Pipeline(config=self.config, task_analyzer=self.mock_task_analyzer)

        self.assertEqual(pipeline.config, self.config)
        self.assertEqual(pipeline.task_analyzer, self.mock_task_analyzer)
        self.assertIsNone(pipeline.current_stage)
        self.assertEqual(pipeline.pipeline_results, {})

    def test_run_task_analysis_stage(self):
        """Test that task analysis stage executes correctly."""
        # Setup
        mock_analysis_result = {
            "task_type": "web application",
            "complexity_estimate": 3,
            "components": ["frontend", "backend", "database"],
        }
        self.mock_task_analyzer.analyze_requirements.return_value = mock_analysis_result

        pipeline = Pipeline(config=self.config, task_analyzer=self.mock_task_analyzer)

        # Execute
        pipeline._run_task_analysis("Build a simple web app")

        # Verify
        self.mock_task_analyzer.analyze_requirements.assert_called_once_with(
            "Build a simple web app"
        )
        self.assertEqual(
            pipeline.pipeline_results["task_analysis"], mock_analysis_result
        )
        self.assertEqual(pipeline.current_stage, PipelineStage.TASK_ANALYSIS)

    def test_complexity_check_within_limits(self):
        """Test that complexity check passes when within limits."""
        # Setup
        pipeline = Pipeline(config=self.config, task_analyzer=self.mock_task_analyzer)
        pipeline.pipeline_results = {
            "task_analysis": {"complexity_estimate": 3}  # Below default max of 5
        }

        # Execute
        result = pipeline._check_complexity()

        # Verify
        self.assertTrue(result)
        self.assertNotIn("complexity_warning", pipeline.pipeline_results)

    def test_complexity_check_exceeds_limits(self):
        """Test that complexity check handles requirements that exceed limits."""
        # Setup
        pipeline = Pipeline(config=self.config, task_analyzer=self.mock_task_analyzer)
        pipeline.pipeline_results = {
            "requirements": "Build a complex app",
            "task_analysis": {
                "complexity_estimate": 8,  # Above default max of 5
                "requires_simplification": False,  # Don't force stop
            },
        }

        # Mock LLM service
        pipeline.task_analyzer.llm_service = MagicMock()
        pipeline.task_analyzer.llm_service.query.return_value = json.dumps(
            {
                "stage1": {
                    "name": "Core Functionality",
                    "requirements": "Basic app features",
                    "complexity_estimate": 4,
                    "reasoning": "Simpler scope",
                },
                "stage2": {
                    "name": "Advanced Features",
                    "requirements": "Extra functionality",
                    "complexity_estimate": 5,
                    "reasoning": "Building on core",
                },
                "explanation": "Split into core and advanced features",
            }
        )

        # Execute
        result = pipeline._check_complexity()

        # Verify
        self.assertTrue(
            result
        )  # Should continue since requires_simplification is False
        self.assertIn("complexity_warning", pipeline.pipeline_results)
        self.assertIn("complexity_breakdown", pipeline.pipeline_results)

    def test_full_pipeline_execution(self):
        """Test that full pipeline executes all stages in correct order."""
        # Setup - mock task analyzer only
        self.mock_task_analyzer.analyze_requirements.return_value = {
            "task_type": "cli tool",
            "complexity_estimate": 3,
        }

        # Create pipeline with only the task analyzer
        pipeline = Pipeline(
            config=self.config,
            task_analyzer=self.mock_task_analyzer,
        )

        # Patch the stage execution methods with autospec=True
        with patch.object(
            pipeline,
            "_run_code_generation",
            autospec=True,
        ) as mock_code, patch.object(
            pipeline,
            "_run_test_generation",
            autospec=True,
        ) as mock_test, patch.object(
            pipeline, "_run_validation", autospec=True
        ) as mock_valid, patch.object(
            pipeline, "_run_documentation", autospec=True
        ) as mock_doc:

            # Execute
            result = pipeline.run("Build a simple CLI tool")

            # Verify stages and results
            self.assertEqual(pipeline.current_stage, PipelineStage.COMPLETE)
            self.assertTrue(result["execution_summary"]["completed"])
            self.mock_task_analyzer.analyze_requirements.assert_called_once()
            mock_code.assert_called_once()
            mock_test.assert_called_once()
            mock_valid.assert_called_once()
            mock_doc.assert_called_once()

    def test_stage_retry_logic(self):
        """Test that stage retries work correctly on failure."""
        # Setup
        pipeline = Pipeline(config=self.config, task_analyzer=self.mock_task_analyzer)

        # Create a side effect that fails twice then succeeds
        mock_stage_func = MagicMock()
        mock_stage_func.__name__ = "_run_test_stage"
        mock_stage_func.side_effect = [
            ValueError("First failure"),
            ValueError("Second failure"),
            None,  # Success on third try
        ]

        # Execute with patched time.sleep to avoid waiting
        with patch("time.sleep", return_value=None):
            pipeline._execute_stage(mock_stage_func)

        # Verify
        self.assertEqual(mock_stage_func.call_count, 3)
        self.assertIn("test_stage", pipeline.stage_times)

    def test_save_results(self):
        """Test that results are saved correctly to file."""
        # Setup
        pipeline = Pipeline(config=self.config, task_analyzer=self.mock_task_analyzer)
        pipeline.pipeline_results = {"test_key": "test_value", "nested": {"data": 123}}

        # Execute
        pipeline._save_results("test_output.json")

        # Verify
        output_path = Path(self.temp_dir.name) / "test_output.json"
        self.assertTrue(output_path.exists())

        with open(output_path, "r") as f:
            saved_data = json.load(f)

        self.assertEqual(saved_data, pipeline.pipeline_results)

    def test_error_handling(self):
        """Test that pipeline handles errors gracefully."""
        # Setup - task analyzer that raises an exception
        self.mock_task_analyzer.analyze_requirements.side_effect = Exception(
            "Test error"
        )

        pipeline = Pipeline(config=self.config, task_analyzer=self.mock_task_analyzer)

        # Execute
        result = pipeline.run("Build something")

        # Verify
        self.assertEqual(pipeline.current_stage, PipelineStage.ERROR)
        self.assertIn("error", result)
        self.assertFalse(result["execution_summary"]["completed"])

    def test_skip_specific_stages(self):
        """Test that pipeline properly skips stages not in stages_to_run."""
        # Create config with limited stages
        limited_config = PipelineConfig(
            stages_to_run=[PipelineStage.TASK_ANALYSIS, PipelineStage.DOCUMENTATION],
            output_dir=self.temp_dir.name,
        )

        # Setup pipeline
        pipeline = Pipeline(
            config=limited_config, task_analyzer=self.mock_task_analyzer
        )

        # Setup mocks
        self.mock_task_analyzer.analyze_requirements.return_value = {
            "complexity_estimate": 3
        }

        # Patch all stage methods with autospec=True
        with patch.object(
            pipeline, "_run_task_analysis", autospec=True
        ) as mock_task, patch.object(
            pipeline, "_run_code_generation", autospec=True
        ) as mock_code, patch.object(
            pipeline, "_run_test_generation", autospec=True
        ) as mock_test, patch.object(
            pipeline, "_run_validation", autospec=True
        ) as mock_valid, patch.object(
            pipeline, "_run_documentation", autospec=True
        ) as mock_doc:

            # Execute
            pipeline.run("Test requirement")

            # Verify only task_analysis and documentation were called
            mock_task.assert_called_once()
            mock_doc.assert_called_once()
            mock_code.assert_not_called()
            mock_test.assert_not_called()
            mock_valid.assert_not_called()

    def test_complexity_check_strict_enforcement(self):
        """Test that pipeline stops when complexity exceeds limits and enforcement is strict."""
        # Setup pipeline with high complexity result
        pipeline = Pipeline(config=self.config, task_analyzer=self.mock_task_analyzer)
        pipeline.pipeline_results = {
            "requirements": "Complex task",
            "task_analysis": {
                "complexity_estimate": 8,
                "requires_simplification": True,  # Force stop
            },
        }

        # Setup LLM service mock
        pipeline.task_analyzer.llm_service = MagicMock()
        pipeline.task_analyzer.llm_service.query.return_value = "..."  # JSON response

        # Execute and expect exception
        with self.assertRaises(ValueError):
            pipeline._check_complexity()

        # Verify warning was added
        self.assertIn("complexity_warning", pipeline.pipeline_results)

    def test_complexity_llm_failure_handling(self):
        """Test handling of LLM failures during complexity breakdown."""
        # Setup
        pipeline = Pipeline(config=self.config, task_analyzer=self.mock_task_analyzer)
        pipeline.pipeline_results = {
            "requirements": "Build something complex",
            "task_analysis": {
                "complexity_estimate": 7,
                "requires_simplification": False,
            },
        }

        # Mock LLM failure
        pipeline.task_analyzer.llm_service = MagicMock()
        pipeline.task_analyzer.llm_service.query.side_effect = Exception(
            "LLM service down"
        )

        # Execute
        result = pipeline._check_complexity()

        # Verify graceful handling
        self.assertFalse(result)
        self.assertIn("complexity_warning", pipeline.pipeline_results)
        self.assertNotIn("complexity_breakdown", pipeline.pipeline_results)

    def test_logging_of_stage_execution(self):
        """Test that pipeline properly logs its execution stages."""
        self.mock_task_analyzer.analyze_requirements.return_value = {
            "complexity_estimate": 3
        }

        pipeline = Pipeline(config=self.config, task_analyzer=self.mock_task_analyzer)

        # Mock the stage methods with autospec to avoid __name__ issues
        with patch.object(
            pipeline, "_run_code_generation", autospec=True
        ), patch.object(pipeline, "_run_test_generation", autospec=True), patch.object(
            pipeline, "_run_validation", autospec=True
        ), patch.object(
            pipeline, "_run_documentation", autospec=True
        ):

            # Use assertLogs to capture log output
            with self.assertLogs("pipeline_orchestrator", level="INFO") as log_cm:
                pipeline.run("A simple requirement")

            # Verify expected log messages
            log_output = "\n".join(log_cm.output)
            self.assertIn("Starting pipeline execution", log_output)
            self.assertIn("Starting stage: task_analysis", log_output)
            self.assertIn("Completed stage: task_analysis", log_output)
            self.assertIn("Task complexity", log_output)

    def test_intermediate_files_saved(self):
        """Test that intermediate result files are saved after each stage."""
        self.mock_task_analyzer.analyze_requirements.return_value = {
            "complexity_estimate": 3
        }

        # Create a pipeline with save_intermediate=True
        pipeline = Pipeline(
            config=PipelineConfig(
                output_dir=self.temp_dir.name, save_intermediate=True
            ),
            task_analyzer=self.mock_task_analyzer,
        )

        # Mock the pipeline stages with autospec
        with patch.object(
            pipeline, "_run_code_generation", autospec=True
        ), patch.object(pipeline, "_run_test_generation", autospec=True), patch.object(
            pipeline, "_run_validation", autospec=True
        ), patch.object(
            pipeline, "_run_documentation", autospec=True
        ):

            # Run the pipeline
            pipeline.run("Test requirement")

        # Check for intermediate files
        expected_files = [
            "task_analysis_results.json",
            "code_generation_results.json",
            "test_generation_results.json",
            "validation_results.json",
            "documentation_results.json",
            "final_results.json",
        ]

        for filename in expected_files:
            output_path = Path(self.temp_dir.name) / filename
            self.assertTrue(output_path.exists(), f"{filename} was not created")

            # Verify the file contains valid JSON
            with open(output_path, "r") as f:
                content = json.load(f)
                self.assertIsInstance(content, dict)

    def test_pipeline_with_empty_requirements(self):
        """Test pipeline behavior with empty requirements."""
        pipeline = Pipeline(config=self.config, task_analyzer=self.mock_task_analyzer)

        # Run with empty requirements
        result = pipeline.run("")

        # Verify pipeline handles this gracefully
        self.assertEqual(pipeline.current_stage, PipelineStage.ERROR)
        self.assertIn("error", result)
        self.assertFalse(result["execution_summary"]["completed"])

    def test_malformed_llm_response(self):
        """Test handling of malformed responses from the LLM."""
        # Setup task analyzer to return invalid data
        self.mock_task_analyzer.analyze_requirements.return_value = (
            "This is not a dictionary"
        )

        pipeline = Pipeline(config=self.config, task_analyzer=self.mock_task_analyzer)

        # Run the pipeline and verify error handling
        result = pipeline.run("Build something")

        self.assertEqual(pipeline.current_stage, PipelineStage.ERROR)
        self.assertIn("error", result)


class TestPipelineConfig(unittest.TestCase):

    def test_default_config(self):
        """Test that default configuration is created correctly."""
        config = PipelineConfig()

        self.assertEqual(len(config.stages_to_run), 5)  # All stages
        self.assertEqual(config.retry_attempts, 2)
        self.assertEqual(config.output_dir, "./output")
        self.assertFalse(config.verbose)
        self.assertTrue(config.save_intermediate)

    def test_custom_config(self):
        """Test that custom configuration options are applied."""
        config = PipelineConfig(
            stages_to_run=[PipelineStage.TASK_ANALYSIS, PipelineStage.CODE_GENERATION],
            retry_attempts=3,
            output_dir="./custom_output",
            verbose=True,
            save_intermediate=False,
        )

        self.assertEqual(len(config.stages_to_run), 2)
        self.assertEqual(config.retry_attempts, 3)
        self.assertEqual(config.output_dir, "./custom_output")
        self.assertTrue(config.verbose)
        self.assertFalse(config.save_intermediate)

    def test_config_from_file(self):
        """Test loading configuration from a JSON file."""
        # Create a temporary config file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(
                {
                    "retry_attempts": 5,
                    "verbose": True,
                    "max_complexity": 7,
                    "stages_to_run": ["task_analysis", "code_generation"],
                },
                f,
            )
            config_path = f.name

        try:
            # Load the config
            config = PipelineConfig.from_file(config_path)

            # Verify
            self.assertEqual(config.retry_attempts, 5)
            self.assertTrue(config.verbose)
            self.assertEqual(config.max_complexity, 7)
            self.assertEqual(len(config.stages_to_run), 2)
            self.assertEqual(config.stages_to_run[0], PipelineStage.TASK_ANALYSIS)

        finally:
            # Clean up
            os.unlink(config_path)


if __name__ == "__main__":
    unittest.main()
