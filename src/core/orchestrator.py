import logging
from typing import Dict, Any, List, Optional, Callable, Union
from enum import Enum
import time
import json
import os
from pathlib import Path

from src.core.task_analyzer import TaskAnalyzer
from src.utils.logger import setup_logger
from src.utils.prompt_loader import load_prompt


class PipelineStage(Enum):
    """Enum representing different stages in the pipeline process."""

    TASK_ANALYSIS = "task_analysis"
    CODE_GENERATION = "code_generation"
    TEST_GENERATION = "test_generation"
    VALIDATION = "validation"
    DOCUMENTATION = "documentation"
    COMPLETE = "complete"
    ERROR = "error"


class PipelineConfig:
    """Configuration options for the pipeline execution."""

    def __init__(
        self,
        stages_to_run: List[PipelineStage] = None,
        retry_attempts: int = 2,
        output_dir: str = "./output",
        verbose: bool = False,
        save_intermediate: bool = True,
        max_complexity: int = 5,
        mode: str = "standard",
    ):
        """
        Initialize pipeline configuration.

        Args:
            stages_to_run: List of stages to execute (defaults to all stages)
            retry_attempts: Number of retry attempts for each stage
            output_dir: Directory to save outputs
            verbose: Whether to print detailed logs
            save_intermediate: Whether to save intermediate results
            max_complexity: Maximum allowed complexity score
            mode: Pipeline execution mode (standard, fast, thorough)
        """
        self.stages_to_run = stages_to_run or [
            PipelineStage.TASK_ANALYSIS,
            PipelineStage.CODE_GENERATION,
            PipelineStage.TEST_GENERATION,
            PipelineStage.VALIDATION,
            PipelineStage.DOCUMENTATION,
        ]
        self.retry_attempts = retry_attempts
        self.output_dir = output_dir
        self.verbose = verbose
        self.save_intermediate = save_intermediate
        self.max_complexity = max_complexity
        self.mode = mode

        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

    @classmethod
    def from_file(cls, config_path: str) -> "PipelineConfig":
        """
        Create configuration from a JSON file.

        Args:
            config_path: Path to JSON configuration file

        Returns:
            PipelineConfig instance
        """
        try:
            with open(config_path, "r") as f:
                config_data = json.load(f)

            # Convert stage strings to enum values if present
            if "stages_to_run" in config_data:
                config_data["stages_to_run"] = [
                    PipelineStage(stage) for stage in config_data["stages_to_run"]
                ]

            return cls(**config_data)
        except Exception as e:
            logging.error(f"Failed to load config from {config_path}: {e}")
            return cls()  # Return default config on error


class Pipeline:
    """
    Manages the orchestration of the dual-architect workflow.
    Controls the sequence of operations from requirement analysis to code generation.
    """

    def __init__(
        self,
        config: PipelineConfig = None,
        task_analyzer=None,
        code_generator=None,
        test_generator=None,
        validator=None,
        documenter=None,
    ):
        """
        Initialize the pipeline orchestrator.

        Args:
            config: Pipeline configuration
            task_analyzer: Component for analyzing requirements
            code_generator: Component for generating code
            test_generator: Component for generating tests
            validator: Component for validating results
            documenter: Component for generating documentation
        """
        self.config = config or PipelineConfig()

        # Set up logging
        log_level = logging.DEBUG if self.config.verbose else logging.INFO
        self.logger = setup_logger("pipeline_orchestrator", level=log_level)

        # Initialize components (use defaults if not provided)
        self.task_analyzer = task_analyzer or TaskAnalyzer()

        # Other components will be initialized when needed
        self.code_generator = code_generator
        self.test_generator = test_generator
        self.validator = validator
        self.documenter = documenter

        # Pipeline state
        self.current_stage = None
        self.pipeline_results = {}
        self.start_time = None
        self.stage_times = {}

    def run(self, requirements: str) -> Dict[str, Any]:
        """
        Execute the complete pipeline with the given requirements.

        Args:
            requirements: User requirements text

        Returns:
            Dictionary containing results from all executed stages
        """
        self.start_time = time.time()
        self.current_stage = PipelineStage.TASK_ANALYSIS
        self.pipeline_results = {"requirements": requirements}

        self.logger.info("Starting pipeline execution")

        try:
            # Execute pipeline stages in sequence
            if PipelineStage.TASK_ANALYSIS in self.config.stages_to_run:
                self._execute_stage(self._run_task_analysis, requirements)

                # Check complexity before proceeding
                self._check_complexity()

            # Continue with other stages if configured and no errors
            if (
                PipelineStage.CODE_GENERATION in self.config.stages_to_run
                and self.current_stage != PipelineStage.ERROR
            ):
                self._execute_stage(self._run_code_generation)

            if (
                PipelineStage.TEST_GENERATION in self.config.stages_to_run
                and self.current_stage != PipelineStage.ERROR
            ):
                self._execute_stage(self._run_test_generation)

            if (
                PipelineStage.VALIDATION in self.config.stages_to_run
                and self.current_stage != PipelineStage.ERROR
            ):
                self._execute_stage(self._run_validation)

            if (
                PipelineStage.DOCUMENTATION in self.config.stages_to_run
                and self.current_stage != PipelineStage.ERROR
            ):
                self._execute_stage(self._run_documentation)

            # Mark pipeline as complete if no errors
            if self.current_stage != PipelineStage.ERROR:
                self.current_stage = PipelineStage.COMPLETE
                self.logger.info("Pipeline execution completed successfully")

            # Add execution summary to results
            self._add_execution_summary()

            # Save final results if configured
            if self.config.save_intermediate:
                self._save_results("final_results.json")

            return self.pipeline_results

        except Exception as e:
            self.current_stage = PipelineStage.ERROR
            self.logger.error(f"Pipeline failed with error: {str(e)}", exc_info=True)
            self.pipeline_results["error"] = str(e)
            self._add_execution_summary()

            # Save error results
            if self.config.save_intermediate:
                self._save_results("error_results.json")

            return self.pipeline_results

    def _execute_stage(self, stage_func: Callable, *args, **kwargs) -> None:
        """
        Execute a pipeline stage with retry logic and timing.

        Args:
            stage_func: Function to execute the stage
            *args, **kwargs: Arguments to pass to the stage function
        """
        stage_name = stage_func.__name__.replace("_run_", "")
        self.logger.info(f"Starting stage: {stage_name}")

        start_time = time.time()
        attempts = 0
        last_error = None

        while attempts <= self.config.retry_attempts:
            try:
                if attempts > 0:
                    self.logger.info(f"Retry attempt {attempts} for {stage_name}")

                # Run the stage function
                stage_func(*args, **kwargs)

                # Stage completed successfully
                elapsed = time.time() - start_time
                self.stage_times[stage_name] = elapsed
                self.logger.info(f"Completed stage: {stage_name} in {elapsed:.2f}s")

                # Save intermediate results if configured
                if self.config.save_intermediate:
                    self._save_results(f"{stage_name}_results.json")

                return

            except Exception as e:
                last_error = e
                attempts += 1
                self.logger.warning(f"Stage {stage_name} failed: {str(e)}")

                if attempts <= self.config.retry_attempts:
                    # Wait before retry (exponential backoff)
                    wait_time = 2**attempts
                    self.logger.info(f"Waiting {wait_time}s before retry...")
                    time.sleep(wait_time)

        # If we get here, all attempts failed
        elapsed = time.time() - start_time
        self.stage_times[stage_name] = elapsed
        self.logger.error(f"Stage {stage_name} failed after {attempts} attempts")

        # Update pipeline state
        self.current_stage = PipelineStage.ERROR
        self.pipeline_results[f"{stage_name}_error"] = str(last_error)

        # Re-raise the last error
        raise last_error

    def _run_task_analysis(self, requirements: str) -> None:
        """
        Execute the task analysis stage.

        Args:
            requirements: User requirements text
        """
        analysis_result = self.task_analyzer.analyze_requirements(requirements)
        self.pipeline_results["task_analysis"] = analysis_result
        self.current_stage = PipelineStage.TASK_ANALYSIS

    def _check_complexity(self) -> bool:
        """
        Check if task complexity is within allowed limits.

        If complexity exceeds the maximum, uses LLM to suggest how to break
        down the requirements into simpler stages.

        Returns:
            bool: True if complexity is acceptable, False otherwise
        """
        analysis = self.pipeline_results.get("task_analysis", {})
        complexity = analysis.get("complexity_estimate", 0)

        # Complexity is acceptable
        if complexity <= self.config.max_complexity:
            self.logger.info(
                f"Task complexity ({complexity}/10) is within acceptable range"
            )
            return True

        # Complexity exceeds maximum - get breakdown suggestions
        self.logger.warning(
            f"Task complexity ({complexity}/10) exceeds maximum allowed "
            f"({self.config.max_complexity}). Requesting breakdown suggestions."
        )

        # Original requirements
        requirements = self.pipeline_results.get("requirements", "")

        # Create prompt for LLM to suggest breakdown
        prompt_template = load_prompt("complexity", "breakdown_prompt")
        prompt = prompt_template.format(
            requirements=requirements,
            complexity=complexity,
            max_complexity=self.config.max_complexity,
        )

        try:
            # Get breakdown suggestions from LLM
            llm_response = self.task_analyzer.llm_service.query(prompt)

            # Extract JSON from response
            import json
            import re

            # Find JSON in the response (handling potential text before/after)
            json_match = re.search(r"({[\s\S]*})", llm_response)
            if json_match:
                breakdown_json = json.loads(json_match.group(1))
            else:
                raise ValueError("Could not extract valid JSON from LLM response")

            # Add breakdown suggestions to pipeline results
            self.pipeline_results["complexity_breakdown"] = breakdown_json

            # Add warning to pipeline results
            self.pipeline_results["complexity_warning"] = (
                f"Task complexity ({complexity}/10) exceeds maximum allowed "
                f"({self.config.max_complexity}/10). See 'complexity_breakdown' "
                f"for suggested simplification."
            )

            self.logger.info("Successfully generated task breakdown suggestions")

            # Continue pipeline execution only if override is set
            if not analysis.get("requires_simplification", False):
                self.pipeline_results[
                    "complexity_warning"
                ] += " Proceeding anyway as override is not set."
                return True  # Allow continuation but return False to indicate issue

            return False  # Stop pipeline execution due to high complexity

        except Exception as e:
            error_msg = f"Failed to generate task breakdown: {str(e)}"
            self.logger.error(error_msg)

            # Add generic warning to pipeline results
            self.pipeline_results["complexity_warning"] = (
                f"Task complexity ({complexity}/10) exceeds maximum allowed "
                f"({self.config.max_complexity}/10). Consider breaking the task into "
                f"smaller, more manageable components."
            )

            # Stop pipeline execution if requires_simplification flag is set
            if analysis.get("requires_simplification", False):
                raise ValueError(
                    f"Task complexity too high ({complexity}/10): {error_msg}"
                )

            return False  # Indicate complexity issue but allow continuation

    def _run_code_generation(self) -> None:
        """Execute the code generation stage."""
        # This will be implemented when code_generator is available
        self.logger.info("Code generation stage (placeholder)")
        # For now, just add a placeholder result
        self.pipeline_results["code_generation"] = {"status": "not_implemented"}
        self.current_stage = PipelineStage.CODE_GENERATION

    def _run_test_generation(self) -> None:
        """Execute the test generation stage."""
        # This will be implemented when test_generator is available
        self.logger.info("Test generation stage (placeholder)")
        # For now, just add a placeholder result
        self.pipeline_results["test_generation"] = {"status": "not_implemented"}
        self.current_stage = PipelineStage.TEST_GENERATION

    def _run_validation(self) -> None:
        """Execute the validation stage."""
        # This will be implemented when validator is available
        self.logger.info("Validation stage (placeholder)")
        # For now, just add a placeholder result
        self.pipeline_results["validation"] = {"status": "not_implemented"}
        self.current_stage = PipelineStage.VALIDATION

    def _run_documentation(self) -> None:
        """Execute the documentation generation stage."""
        # This will be implemented when documenter is available
        self.logger.info("Documentation stage (placeholder)")
        # For now, just add a placeholder result
        self.pipeline_results["documentation"] = {"status": "not_implemented"}
        self.current_stage = PipelineStage.DOCUMENTATION

    def _add_execution_summary(self) -> None:
        """Add execution summary to pipeline results."""
        total_time = time.time() - self.start_time

        summary = {
            "total_execution_time": total_time,
            "stage_times": self.stage_times,
            "final_stage": self.current_stage.value,
            "stages_executed": [stage for stage, time in self.stage_times.items()],
            "completed": self.current_stage == PipelineStage.COMPLETE,
        }

        self.pipeline_results["execution_summary"] = summary

    def _save_results(self, filename: str) -> None:
        """
        Save current pipeline results to a file.

        Args:
            filename: Name of the file to save results to
        """
        try:
            output_path = Path(self.config.output_dir) / filename
            with open(output_path, "w") as f:
                json.dump(self.pipeline_results, f, indent=2)

            self.logger.debug(f"Saved results to {output_path}")
        except Exception as e:
            self.logger.error(f"Failed to save results to {filename}: {e}")
