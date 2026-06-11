"""Tests for entity_extractor — uses mocked LLM."""

import pytest
from unittest.mock import MagicMock, patch, AsyncMock
import json


def test_parse_extraction_result_valid_json():
    """Should parse valid JSON from LLM response."""
    from backend.services.entity_extractor import EntityExtractor
    extractor = EntityExtractor.__new__(EntityExtractor)

    raw = json.dumps({
        "entities": [
            {"name": "BGE", "type": "技术", "description": "文本嵌入模型"},
        ],
        "relations": [
            {"source": "BGE", "target": "Milvus", "relation": "依赖", "context": "BGE 用于 Milvus"},
        ],
    }, ensure_ascii=False)

    result = extractor._parse_extraction_result(raw)
    assert len(result["entities"]) == 1
    assert result["entities"][0]["name"] == "BGE"
    assert len(result["relations"]) == 1


def test_parse_extraction_result_handles_markdown_json_block():
    """Should handle LLM responses wrapped in ```json ... ``` blocks."""
    from backend.services.entity_extractor import EntityExtractor
    extractor = EntityExtractor.__new__(EntityExtractor)

    raw = '```json\n{"entities": [], "relations": []}\n```'
    result = extractor._parse_extraction_result(raw)
    assert result == {"entities": [], "relations": []}


def test_parse_extraction_result_handles_invalid_json():
    """Should return empty result on invalid JSON."""
    from backend.services.entity_extractor import EntityExtractor
    extractor = EntityExtractor.__new__(EntityExtractor)

    result = extractor._parse_extraction_result("not json at all")
    assert result == {"entities": [], "relations": []}


def test_parse_query_entities_valid():
    """Should parse entity names from query extraction result."""
    from backend.services.entity_extractor import EntityExtractor
    extractor = EntityExtractor.__new__(EntityExtractor)

    raw = '["BGE", "Milvus"]'
    result = extractor._parse_query_entities(raw)
    assert result == ["BGE", "Milvus"]


def test_parse_query_entities_handles_invalid():
    """Should return empty list on invalid input."""
    from backend.services.entity_extractor import EntityExtractor
    extractor = EntityExtractor.__new__(EntityExtractor)

    result = extractor._parse_query_entities("invalid")
    assert result == []
