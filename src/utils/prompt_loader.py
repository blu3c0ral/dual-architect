import yaml
import os
from pathlib import Path
from typing import Dict, Any

_prompt_cache = {}


def load_prompt(component: str, prompt_name: str) -> str:
    """
    Load a prompt from the prompts directory.

    Args:
        component: Component name (corresponds to YAML file)
        prompt_name: Name of the prompt in the YAML file

    Returns:
        String containing the prompt template
    """
    cache_key = f"{component}:{prompt_name}"

    # Return from cache if available
    if cache_key in _prompt_cache:
        return _prompt_cache[cache_key]

    # Determine prompts directory relative to this file
    base_dir = Path(__file__).parent.parent
    prompt_file = base_dir / "prompts" / f"{component}.yaml"

    try:
        # Load prompts file
        with open(prompt_file, "r") as f:
            prompts = yaml.safe_load(f)

        # Get requested prompt
        if prompt_name not in prompts:
            raise ValueError(f"Prompt '{prompt_name}' not found in {component}.yaml")

        prompt = prompts[prompt_name]

        # Cache for future use
        _prompt_cache[cache_key] = prompt

        return prompt

    except Exception as e:
        raise ValueError(
            f"Failed to load prompt '{prompt_name}' from {component}.yaml: {e}"
        )
