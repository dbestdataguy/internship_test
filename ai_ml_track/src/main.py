import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.grouper import group_transactions


def load_input(filepath: str) -> list[str]:
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Input file not found: {filepath}")

    with open(filepath, "r") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("Input file must contain a JSON array of strings.")

    if not all(isinstance(item, str) for item in data):
        raise ValueError("Every item in the input array must be a string.")

    return data


def main():
    input_path = os.path.join(
        os.path.dirname(__file__), "..", "data", "sample_input.json"
    )

    print("Loading transactions...")
    descriptions = load_input(input_path)
    print(f"Loaded {len(descriptions)} transactions.\n")

    print("Running grouping pipeline...")
    result = group_transactions(descriptions)

    print("\n--- FINAL OUTPUT ---\n")
    print(json.dumps(result, indent=2))

    output_path = os.path.join(
        os.path.dirname(__file__), "..", "data", "output.json"
    )
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"\nOutput saved to data/output.json")


if __name__ == "__main__":
    main()