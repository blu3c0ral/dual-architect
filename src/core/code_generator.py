import json
import re
from typing import Dict, List, Any

from src.utils.prompt_loader import load_prompt


class CodeGenerator:
    """
    Component responsible for generating high-quality, structured, documented code based on task requirements.

    Implements a two-step LLM workflow:
    1. Generate detailed function analysis (signatures, docstrings, examples, environment).
    2. Generate implementation strictly adhering to that analysis.

    Ensures code maintainability, readability, documentation quality, and robustness.
    """

    def __init__(self, llm_service):
        """
        Initialize the code generator with an LLM service.

        Args:
            llm_service: The language model service for code generation.
        """
        self.llm_service = llm_service

    def generate_code(self, task_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate implementation code using a structured two-step approach.

        Args:
            task_analysis (Dict[str, Any]): Task analysis including:
                - task_type, technologies, components, dependencies, complexity_estimate,
                  key_features, concerns, original_requirements.

        Returns:
            Dict[str, Any]: Generated implementation details including:
                - code, functions, imports, explanation, file_structure.
        """
        try:
            # Step 1: Generate detailed function analysis
            function_analysis = self._generate_function_analysis(task_analysis)

            # Step 2: Generate actual implementation based on function analysis
            implementation = self._generate_implementation(
                task_analysis, function_analysis
            )

            # Extract detailed function metadata from implementation
            implementation["functions"] = self._extract_function_details(
                implementation["code"]
            )

            return implementation

        except Exception as e:
            return {"error": f"Code generation failed: {str(e)}"}

    def _generate_function_analysis(
        self, task_analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Generate a comprehensive analysis of required functions.

        Returns:
            Dict[str, Any]: Structured function analysis (signatures, docstrings, examples).
        """
        prompt_template = load_prompt("code_generator", "function_analysis_prompt")
        prompt = prompt_template.format(**task_analysis)

        response = self.llm_service.query(prompt)
        analysis = self._parse_json_response(response, "function_analysis")

        if not analysis.get("functions"):
            raise ValueError("Function analysis missing or invalid.")

        return analysis

    def _generate_implementation(
        self, task_analysis: Dict[str, Any], function_analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Generate implementation code based explicitly on the function analysis.

        Returns:
            Dict[str, Any]: Implementation code and documentation.
        """
        prompt_template = load_prompt("code_generator", "implementation_prompt")
        prompt = prompt_template.format(
            original_requirements=task_analysis["original_requirements"],
            task_type=task_analysis["task_type"],
            technologies=", ".join(task_analysis["technologies"]),
            components=", ".join(task_analysis["components"]),
            dependencies=", ".join(task_analysis["dependencies"]),
            key_features=", ".join(task_analysis["key_features"]),
            concerns=", ".join(task_analysis["concerns"]),
            function_analysis=json.dumps(function_analysis, indent=2),
        )

        response = self.llm_service.query(prompt)
        implementation = self._parse_json_response(response, "implementation")

        if not implementation.get("code"):
            raise ValueError("Implementation code missing in LLM response.")

        if not self._validate_code_result(implementation):
            repaired_response = self._repair_code_response(response, implementation)
            implementation = self._parse_json_response(
                repaired_response, "implementation"
            )
            if not self._validate_code_result(implementation):
                raise ValueError("Implementation invalid even after repair attempt.")

        return implementation

    def _parse_json_response(self, response: str, context: str = "") -> Dict[str, Any]:
        """
        Robustly parse JSON from LLM response.

        Args:
            response: Raw LLM response.
            context: Context for clearer error messages.

        Returns:
            Parsed JSON dictionary.
        """
        try:
            json_match = re.search(r"```json\s*([\s\S]*?)\s*```", response)
            if json_match:
                return json.loads(json_match.group(1).strip())

            json_match = re.search(r"(\{[\s\S]*\})", response)
            if json_match:
                return json.loads(json_match.group(1).strip())

            raise ValueError(f"No JSON found in {context} response.")

        except json.JSONDecodeError as e:
            raise ValueError(f"JSON parsing error in {context} response: {str(e)}")

    def _validate_code_result(self, code_result: Dict[str, Any]) -> bool:
        """
        Basic validation check on generated code results.

        Returns:
            bool: True if valid, False otherwise.
        """
        code = code_result.get("code", "")
        return (
            bool(code)
            and code.count("(") == code.count(")")
            and code.count("[") == code.count("]")
            and code.count("{") == code.count("}")
        )

    def _repair_code_response(
        self, original_response: str, initial_result: Dict[str, Any]
    ) -> str:
        """
        Attempt to repair invalid response via follow-up query.

        Returns:
            str: Repaired LLM response.
        """
        issues = []
        if not initial_result.get("code"):
            issues.append("Missing implementation code.")

        repair_prompt = f"""
        Your previous response had issues: {' '.join(issues)}

        Provide a valid JSON response exactly as follows:
        ```json
        {{
          "code": "<implementation code with docstrings and comments>",
          "imports": ["imports"],
          "explanation": "<implementation explanation>",
          "file_structure": {{"optional": "file structure"}}
        }}
        ```

        Original response:
        {original_response}
        """

        return self.llm_service.query(repair_prompt)

    def _extract_function_details(self, code: str) -> List[Dict[str, Any]]:
        """
        Extract detailed function metadata from code.

        Returns:
            List[Dict]: Function metadata.
        """
        pattern = r'def\s+([a-zA-Z_]\w*)\s*\((.*?)\)(?:\s*->\s*([^:]+))?\s*:(?:\s*"""([\s\S]*?)""")?'
        functions = []

        for match in re.finditer(pattern, code):
            name, params, return_type, docstring = match.groups()
            docstring = docstring or ""
            functions.append(
                {
                    "name": name,
                    "signature": f"def {name}({params}) -> {return_type or 'None'}:",
                    "description": docstring.split("\n\n")[0].strip(),
                    "parameters": self._parse_docstring_params(docstring),
                    "returns": self._parse_docstring_returns(docstring),
                    "examples": self._extract_docstring_examples(docstring),
                }
            )

        return functions

    def _parse_docstring_params(self, docstring: str) -> List[Dict[str, str]]:
        match = re.search(r"Args:\s*([\s\S]*?)(Returns:|Examples:|$)", docstring)
        if not match:
            return []
        param_lines = match.group(1).strip().split("\n")
        params = []
        for line in param_lines:
            if ":" in line:
                name, desc = line.split(":", 1)
                params.append({"name": name.strip(), "description": desc.strip()})
        return params

    def _parse_docstring_returns(self, docstring: str) -> Dict[str, str]:
        match = re.search(r"Returns:\s*(.*?)(Args:|Examples:|$)", docstring, re.DOTALL)
        return {"description": match.group(1).strip()} if match else {}

    def _extract_docstring_examples(self, docstring: str) -> List[str]:
        return re.findall(r"```(?:python)?\s*([\s\S]*?)```", docstring)
