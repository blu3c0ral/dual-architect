from src.core.task_analyzer import TaskAnalyzer


def main():
    """
    Example usage of the Task Analysis Module.
    """
    # Create an instance of TaskAnalyzer
    analyzer = TaskAnalyzer()

    # Sample user requirements
    user_requirements = """
    I need to build a web application for managing inventory. It should:
    - Allow users to add, edit, and delete items
    - Track inventory levels and send alerts for low stock
    - Generate reports on inventory movement
    - Support barcode scanning for quick entry
    - Have a responsive design for mobile use
    """

    print("Analyzing user requirements...")

    # Analyze the requirements
    task_info = analyzer.analyze_requirements(user_requirements)

    # Display the results
    print("\nTask Analysis Results:")
    print(f"Task Type: {task_info['task_type']}")
    print("\nSuggested Technologies:")
    for tech in task_info.get("technologies", []):
        print(f"- {tech}")

    print("\nRequired Components:")
    for component in task_info.get("components", []):
        print(f"- {component}")

    print("\nKey Features:")
    for feature in task_info.get("key_features", []):
        print(f"- {feature}")

    print(f"\nComplexity Estimate: {task_info.get('complexity_estimate', 'N/A')}/10")

    if task_info.get("concerns"):
        print("\nPotential Concerns:")
        for concern in task_info["concerns"]:
            print(f"- {concern}")


if __name__ == "__main__":
    main()
