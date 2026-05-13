"""Tests for query_router — rule matching and routing logic."""
import pytest
from backend.services.query_router import QueryRouter


class TestQueryRouterRules:
    """Verify rule-based fast path matching."""

    @pytest.fixture
    def router(self):
        return QueryRouter()

    # ── chat rules ──
    @pytest.mark.parametrize("query", [
        "你好",
        "hi",
        "hello",
        "Hello",
        "Hi",
        "早上好",
        "晚上好",
        "下午好",
    ])
    def test_greeting_routes_to_chat(self, router, query):
        assert router.route(query) == "chat"

    @pytest.mark.parametrize("query", [
        "谢谢",
        "感谢",
        "多谢",
        "谢谢你帮我查了这么多资料",
        "thanks",
        "Thanks",
        "thank you",
    ])
    def test_thanks_routes_to_chat(self, router, query):
        assert router.route(query) == "chat"

    @pytest.mark.parametrize("query", [
        "再见",
        "拜拜",
        "bye",
        "Bye",
        "goodbye",
    ])
    def test_farewell_routes_to_chat(self, router, query):
        assert router.route(query) == "chat"

    @pytest.mark.parametrize("query", [
        "你是谁",
        "你能做什么",
        "介绍一下自己",
        "你叫什么",
        "你好，请问你是谁",
    ])
    def test_self_intro_routes_to_chat(self, router, query):
        assert router.route(query) == "chat"

    # ── fast keyword rules ──
    @pytest.mark.parametrize("query", [
        "测试覆盖率最低要求是多少",
        "什么时候开始绩效考核",
        "机房在哪里",
        "项目负责人是谁",
        "合同保管期限不少于几年",
        "供应商分为哪几个等级",  # has "几个"
        "质量目标在哪年制定",
    ])
    def test_fact_lookup_routes_to_fast(self, router, query):
        assert router.route(query) == "fast"

    @pytest.mark.parametrize("query", [
        "预算是否可以随意调整",
        "能不能提前终止合同",
        "这个方案可以吗",
        "是否需要提供资质证明",
        "员工是不是需要参加培训",
        "离职时需不需要卸载软件",
    ])
    def test_yes_no_pattern_routes_to_fast(self, router, query):
        assert router.route(query) == "fast"

    @pytest.mark.parametrize("query", [
        "KPI是什么意思",
        "什么叫OKR",
        "RAG指的是什么",
    ])
    def test_definition_routes_to_fast(self, router, query):
        assert router.route(query) == "fast"

    # ── no rule match, no LLM hint → deep ──
    def test_no_match_without_hint_defaults_to_deep(self, router):
        assert router.route("VIP客户和重要客户在响应时间上有何不同") == "deep"

    # ── no rule match but has LLM hint ──
    @pytest.mark.parametrize("hint,expected", [
        ("fast", "fast"),
        ("precise", "precise"),
        ("deep", "deep"),
    ])
    def test_llm_hint_used_when_rules_dont_match(self, router, hint, expected):
        assert router.route("某个复杂问题描述", llm_hint=hint) == expected

    # ── rule match takes priority over LLM hint ──
    def test_rule_match_overrides_llm_hint(self, router):
        # "是多少" matches fast rule, LLM says precise → fast wins
        assert router.route("预算编制期限是多少", llm_hint="precise") == "fast"

    # ── invalid LLM hint ignored, falls back to deep ──
    def test_invalid_llm_hint_falls_back_to_deep(self, router):
        assert router.route("一个复杂问题", llm_hint="unknown") == "deep"

    # ── empty query ──
    def test_empty_query_defaults_to_deep(self, router):
        assert router.route("") == "deep"

    # ── white-space only ──
    def test_whitespace_only_defaults_to_deep(self, router):
        assert router.route("   ") == "deep"


class TestRouterIsSingletonSafe:
    """Verify router can be instantiated once and reused."""

    def test_multiple_calls_no_side_effects(self):
        router = QueryRouter()
        assert router.route("你好") == "chat"
        assert router.route("价格是多少") == "fast"
        assert router.route("你好") == "chat"  # same call twice
        assert router.route("复杂推理问题", llm_hint="deep") == "deep"
