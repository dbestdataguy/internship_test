import pytest
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.grouper import parse_llm_response, build_final_output
from src.main import load_input


def test_parse_valid_json():
    raw = '{"groups": [{"label": "Ride-hailing", "items": ["Uber trip 1200"], "confidence": "high", "explanation": "Ride service"}], "ungrouped": []}'
    result = parse_llm_response(raw)
    assert result["groups"][0]["label"] == "Ride-hailing"
    assert result["groups"][0]["confidence"] == "high"


def test_parse_json_with_code_fences():
    raw = '```json\n{"groups": [{"label": "Streaming", "items": ["Netflix subscription 4500"], "confidence": "high", "explanation": "Streaming service"}], "ungrouped": []}\n```'
    result = parse_llm_response(raw)
    assert result["groups"][0]["label"] == "Streaming"


def test_parse_invalid_json_raises_error():
    raw = "this is not json at all"
    with pytest.raises(ValueError) as exc_info:
        parse_llm_response(raw)
    assert "invalid JSON" in str(exc_info.value)


def test_build_final_output_summary():
    parsed = {
        "groups": [
            {"label": "Ride-hailing", "items": ["Uber trip 1200", "Bolt ride 900"], "confidence": "high", "explanation": "Both are ride services"},
            {"label": "Streaming", "items": ["Netflix subscription 4500"], "confidence": "high", "explanation": "Streaming service"}
        ],
        "ungrouped": []
    }
    result = build_final_output(parsed, total_input=3)
    assert result["summary"]["total_input"] == 3
    assert result["summary"]["total_groups"] == 2
    assert result["summary"]["ungrouped_count"] == 0


def test_load_input_file_not_found():
    with pytest.raises(FileNotFoundError):
        load_input("nonexistent_path/fake.json")