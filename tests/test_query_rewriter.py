"""Tests for query_rewriter — parse, get_queries, integration."""
import json
import pytest
from unittest.mock import MagicMock, patch
from backend.services.query_rewriter import QueryRewriter, REWRITE_PROMPT


class TestParse:
    """Verify _parse handles various LLM output formats."""

    def test_parse_valid_json(self):
        rewriter = QueryRewriter()
        raw = '{"original": "q", "rewritten": "rw", "sub_queries": [], "changes": []}'
        result = rewriter._parse(raw, "q")
        assert result["rewritten"] == "rw"
        assert result["sub_queries"] == []

    def test_parse_json_with_markdown_fence(self):
        rewriter = QueryRewriter()
        raw = '```json\n{"original": "q", "rewritten": "rw", "sub_queries": [], "changes": []}\n```'
        result = rewriter._parse(raw, "q")
        assert result["rewritten"] == "rw"

    def test_parse_invalid_json_falls_back_to_raw(self):
        rewriter = QueryRewriter()
        raw = "some plain text response"
        result = rewriter._parse(raw, "原始查询")
        assert result["original"] == "原始查询"
        assert result["rewritten"] == raw
        assert result["sub_queries"] == []
        assert result["changes"] == []

    def test_parse_with_sub_queries(self):
        rewriter = QueryRewriter()
        raw = json.dumps({
            "original": "A和B的区别",
            "rewritten": "A和B区别对比",
            "sub_queries": ["A的原理", "B的原理"],
            "changes": ["分解: 对比拆为两个"],
        })
        result = rewriter._parse(raw, "A和B的区别")
        assert len(result["sub_queries"]) == 2


class TestGetQueries:
    """Verify get_queries extracts correct query list."""

    def test_single_rewritten_query(self):
        rewriter = QueryRewriter()
        result = {
            "original": "它是什么",
            "rewritten": "LangChain是什么",
            "sub_queries": [],
            "changes": ["指代消解"],
        }
        queries = rewriter.get_queries(result)
        assert queries == ["LangChain是什么"]

    def test_with_sub_queries(self):
        rewriter = QueryRewriter()
        result = {
            "original": "A和B的区别",
            "rewritten": "A和B区别对比",
            "sub_queries": ["A的原理", "B的原理"],
            "changes": [],
        }
        queries = rewriter.get_queries(result)
        assert len(queries) == 3
        assert queries[0] == "A和B区别对比"
        assert "A的原理" in queries
        assert "B的原理" in queries

    def test_empty_rewritten_still_returns_original(self):
        rewriter = QueryRewriter()
        result = {
            "original": "test",
            "rewritten": "",
            "sub_queries": [],
            "changes": [],
        }
        queries = rewriter.get_queries(result)
        assert queries == ["test"]

    def test_empty_sub_query_filtered_out(self):
        rewriter = QueryRewriter()
        result = {
            "original": "q",
            "rewritten": "rw",
            "sub_queries": ["sub1", "", "sub2"],
            "changes": [],
        }
        queries = rewriter.get_queries(result)
        assert queries == ["rw", "sub1", "sub2"]


class TestRewrite:
    """Verify rewrite method with mocked LLM."""

    def test_rewrite_returns_structured_result(self):
        rewriter = QueryRewriter()
        mock_response = json.dumps({
            "original": "它的作者",
            "rewritten": "《三体》的作者",
            "sub_queries": [],
            "changes": ["指代消解: 它→《三体》"],
        })
        rewriter.llm = MagicMock()
        rewriter.llm.invoke.return_value = MagicMock(content=mock_response)

        result = rewriter.rewrite("它的作者", chat_history="用户: 介绍一下《三体》")
        assert result["rewritten"] == "《三体》的作者"
        assert len(result["changes"]) == 1

    def test_rewrite_llm_failure_returns_original(self):
        rewriter = QueryRewriter()
        rewriter.llm = MagicMock()
        rewriter.llm.invoke.side_effect = RuntimeError("API error")

        result = rewriter.rewrite("测试查询", chat_history="历史")
        assert result["rewritten"] == "测试查询"
        assert result["sub_queries"] == []
        assert result["changes"] == []


class TestPrompt:
    """Verify prompt template contains key requirements."""

    def test_prompt_contains_co_reference(self):
        assert "指代消解" in REWRITE_PROMPT or "Co-reference" in REWRITE_PROMPT

    def test_prompt_contains_decomposition(self):
        assert "分解" in REWRITE_PROMPT or "Decomposition" in REWRITE_PROMPT

    def test_prompt_contains_expansion(self):
        assert "扩展" in REWRITE_PROMPT or "expansion" in REWRITE_PROMPT

    def test_prompt_contains_output_format(self):
        assert "Output" in REWRITE_PROMPT and "JSON" in REWRITE_PROMPT
