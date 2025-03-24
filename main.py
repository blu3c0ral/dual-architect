from src.core.task_analyzer import TaskAnalyzer
import json


def main():
    """
    Example usage of the Task Analysis Module.
    """
    # Create an instance of TaskAnalyzer
    analyzer = TaskAnalyzer()

    # Sample user requirements
    user_requirements = """
    I need to build a simple inventory item catalog component that will be reused across multiple projects. The component should:
    - Provide CRUD operations for basic inventory items (create, read, update, delete)
    - Each item should have: ID, name, category, quantity, price, and description
    - Support simple search and filtering by name and category
    - Expose a clean REST API for other components to consume
    - Be containerized with Docker for easy deployment
    - Include basic authentication for API access
    - Store data in SQLite for simplicity in this initial version but keep it usable later with different DB types
    - No UI implementation required in this phase
    """

    print("Analyzing user requirements...")

    # Analyze the requirements
    task_info = analyzer.analyze_requirements(user_requirements)

    # Display the results in a cleaner format
    print("\n== Task Analysis Results ==")
    print(f"Task Type: {task_info['task_type']}")

    print("\nSuggested Technologies:")
    for tech in task_info.get("technologies", []):
        print(f"• {tech}")

    print("\nRequired Components:")
    for component in task_info.get("components", []):
        print(f"• {component}")

    print("\nKey Features:")
    for feature in task_info.get("key_features", []):
        print(f"• {feature}")

    print(f"\nComplexity Estimate: {task_info.get('complexity_estimate', 'N/A')}/10")

    if task_info.get("concerns"):
        print("\nPotential Concerns:")
        for concern in task_info["concerns"]:
            print(f"• {concern}")

    # Show dependencies which are important for implementation
    print("\nDependencies:")
    for dependency in task_info.get("dependencies", []):
        print(f"• {dependency}")

    # Print full JSON output for reference
    print("\n== Complete Analysis JSON ==")
    print(json.dumps(task_info, indent=2))


if __name__ == "__main__":
    main()
