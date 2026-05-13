import re
from typing import Pattern


class QueryRouter:
    """Two-layer routing decision: regex rules (zero cost) -> LLM hint (from QueryRewriter) -> default deep"""

    CHAT_PATTERNS: list[Pattern] = [
        re.compile(r"^(你好|hi|hello|早上好|晚上好|下午好)\b"),
        re.compile(r"^(谢谢|感谢|多谢|thanks|thank you)\b"),
        re.compile(r"^(再见|拜拜|bye|goodbye)\b"),
        re.compile(r"(你是谁|你能做什么|介绍一下自己|你叫什么)"),
        re.compile(r"^(谢谢你|谢谢您)"),
    ]

    FAST_KEYWORDS: list[tuple[str, Pattern]] = (
        [
            (kw, re.compile(re.escape(kw)))
            for kw in [
                "是多少", "什么时候", "在哪", "是谁", "几年", "几个", "哪年",
                "是否", "能不能", "可以吗", "是不是", "需不需要",
                "是什么", "什么叫", "指的是",
            ]
        ]
    )

    def _match_rules(self, query: str) -> str | None:
        """Try to match query against fast-path rules. Returns route or None."""
        if not query or not query.strip():
            return None

        q = query.strip()

        for pat in self.CHAT_PATTERNS:
            if pat.search(q):
                return "chat"

        for _kw, pat in self.FAST_KEYWORDS:
            if pat.search(q):
                return "fast"

        return None

    def route(self, query: str, llm_hint: str | None = None) -> str:
        """
        Determine the best retrieval strategy for a query.

        Decision order:
        1. Rule match (regex) -> immediate return
        2. LLM hint from QueryRewriter (if available and valid)
        3. Default to "deep"

        Args:
            query: The raw user query
            llm_hint: Route hint from LLM query rewriting step

        Returns:
            One of "chat", "fast", "precise", "deep"
        """
        rule_match = self._match_rules(query)
        if rule_match:
            return rule_match

        if llm_hint and llm_hint in ("fast", "precise", "deep"):
            return llm_hint

        return "deep"


# Global singleton instance
query_router = QueryRouter()
