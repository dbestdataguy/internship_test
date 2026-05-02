"""
TaxStreem AI/ML Track — Starter Template
=========================================
This is optional scaffolding. You may use it, modify it, or ignore it entirely.
What matters is your reasoning, not whether you used this template.

Run: python src/main.py
"""

import json
import os
from typing import TypedDict


# ── Types ─────────────────────────────────────────────────────────────────────

class TransactionGroup(TypedDict):
    label: str
    items: list[str]
    confidence: str          # "high" | "medium" | "low"
    explanation: str


class GroupingSummary(TypedDict):
    total_input: int
    total_groups: int
    ungrouped_count: int


class GroupingResult(TypedDict):
    groups: list[TransactionGroup]
    ungrouped: list[str]
    summary: GroupingSummary


# ── Core Grouping Interface ────────────────────────────────────────────────────

def group_transactions(transactions: list[str]) -> GroupingResult:
    """
    Core grouping function. Replace this with your implementation.

    Args:
        transactions: Raw transaction description strings

    Returns:
        Structured grouping result with labels, explanations, and summary
    """
    raise NotImplementedError("Implement your grouping logic here")


# ── Output Validation ─────────────────────────────────────────────────────────

def validate_output(result: GroupingResult, original_input: list[str]) -> list[str]:
    """
    Sanity check the output before returning.
    Returns a list of validation warnings (empty = valid).
    """
    warnings = []

    # Check all input items are accounted for
    all_grouped = [item for group in result["groups"] for item in group["items"]]
    all_items = all_grouped + result["ungrouped"]

    for txn in original_input:
        if txn not in all_items:
            warnings.append(f"MISSING from output: '{txn}'")

    for item in all_items:
        if item not in original_input:
            warnings.append(f"UNEXPECTED item in output: '{item}'")

    # Check required fields
    for i, group in enumerate(result["groups"]):
        for field in ["label", "items", "confidence", "explanation"]:
            if field not in group or not group[field]:  # type: ignore
                warnings.append(f"Group {i} missing or empty field: '{field}'")

    return warnings



def main():
    # Load sample data
    data_path = os.path.join(os.path.dirname(__file__), "..", "data", "sample_input.json")
    with open(data_path) as f:
        data = json.load(f)

    transactions = data["transactions"]
    print(f"Processing {len(transactions)} transactions...\n")

    # Run grouping
    result = group_transactions(transactions)

    # Validate
    warnings = validate_output(result, transactions)
    if warnings:
        print("⚠️  Validation warnings:")
        for w in warnings:
            print(f"   - {w}")
        print()

    # Print output
    print(json.dumps(result, indent=2, ensure_ascii=False))

    # Summary
    print(f"\n{'─' * 50}")
    print(f"Groups found: {result['summary']['total_groups']}")
    print(f"Ungrouped:    {result['summary']['ungrouped_count']}")


if __name__ == "__main__":
    main()
