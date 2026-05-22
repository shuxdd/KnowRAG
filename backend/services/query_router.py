"""
查询路由器模块

使用两层路由决策确定最佳检索策略：
1. 正则规则匹配（零成本）：关键词 + 模式匹配，覆盖 chat/fast/precise/deep
2. LLM 提示（来自 QueryRewriter）：正则未命中时的策略建议
3. 默认策略：deep

路由策略：
- chat: 寒暄/感谢/告别/自我介绍/简单确认（不检索，LLM 直接回复）
- fast: 定义/简单事实/是非判断（仅向量检索）
- precise: 列举/步骤/方法（向量 + BM25 + RRF）
- deep: 对比/原因/推理/多跳（向量 + BM25 + RRF + HyDE + Reranker，默认）
"""

import re
from typing import Pattern


def _compile_keywords(*keywords: str) -> list[Pattern]:
    """Compile keyword list into case-insensitive regex patterns."""
    return [re.compile(re.escape(kw), re.IGNORECASE) for kw in keywords]


class QueryRouter:
    """
    查询路由器

    第一层：正则规则按优先级匹配（chat → deep → precise → fast），命中即返回
    第二层：LLM 提示兜底
    第三层：默认 "deep"
    """

    # ---- chat: 寒暄/感谢/告别/确认 ----
    CHAT: list[Pattern] = [
        # 问候
        re.compile(r"^(你好|hi|hello|嗨|hey|早上好|晚上好|下午好|中午好)", re.IGNORECASE),
        # 感谢
        re.compile(r"(谢谢|感谢|多谢|thanks|thank you|辛苦|费心|好心)"),
        # 告别
        re.compile(r"(再见|拜拜|bye|goodbye|回头见|下次见)"),
        # 自我介绍 / 能力询问
        re.compile(r"(你是谁|你能做什么|介绍一下自己|你叫什么|你是什么|你的能力|你有什么功能)"),
        # 简单确认
        re.compile(r"^(好的|知道了|明白了|懂了|ok|嗯|哦|好嘞|收到|了解)[,.，。!！\s]*$", re.IGNORECASE),
    ]

    # ---- deep: 对比/原因/推理/多跳（优先级最高，命中即走 deep）----
    DEEP: list[Pattern] = _compile_keywords(
        # 对比
        "对比", "比较", "区别", "不同", "异同", "哪个好", "哪个更",
        "优缺点", "优劣", "差异", "哪一个更", "哪个比较", "哪个比较",
        # 原因 / 推理
        "为什么", "原因", "原理", "背景", "根源", "依据",
        # 关系 / 影响
        "关系", "联系", "关联", "影响", "作用", "意义", "后果",
        # 优劣 / 选择
        "优点", "缺点", "好处", "坏处", "更好", "推荐", "建议选择",
        # 多跳 / 综合
        "异同点", "综合评价",
    )

    # ---- precise: 列举/步骤/方法 ----
    PRECISE: list[Pattern] = _compile_keywords(
        # 列举
        "列出", "有哪些", "哪些", "分类", "种类", "类型", "包括哪些",
        "列举", "罗列", "一共", "总共", "分为几", "有几种", "有几类",
        # 步骤 / 方法
        "怎么", "如何", "怎样", "步骤", "方法", "流程", "做法",
        "怎么做", "如何做", "怎样做", "操作步骤",
        # 配置 / 部署 / 安装（通常需要步骤型回答）
        "配置", "部署", "安装", "搭建", "设置",
    )

    # ---- fast: 定义/简单事实/是非判断 ----
    FAST: list[Pattern] = _compile_keywords(
        # 定义
        "是什么", "什么叫", "什么是", "指的是", "定义", "含义", "概念",
        "什么意思", "啥意思",
        # 简单事实
        "是多少", "什么时候", "在哪", "是谁", "几年", "几个",
        "哪年", "哪个", "哪里", "何时", "何人",
        # 数值 / 日期
        "多少", "多大", "多长", "多久", "多远",
        # 是非判断
        "是否", "能不能", "可以吗", "是不是", "需不需要", "有没有",
        "行不行", "对不对", "会不会",
    )

    def _match_rules(self, query: str) -> str | None:
        """First-layer regex match: chat → deep → precise → fast, first hit wins."""
        if not query or not query.strip():
            return None

        q = query.strip()

        # 1. Chat detection: must match at start or be very short
        for pat in self.CHAT:
            if pat.search(q):
                return "chat"
        # Very short queries (< 8 chars) that aren't matched elsewhere → chat
        if len(q) < 8 and not any(
            pat.search(q) for pat in self.DEEP + self.PRECISE + self.FAST
        ):
            return "chat"

        # 2. Deep patterns (comparison / reasoning / multi-hop)
        for pat in self.DEEP:
            if pat.search(q):
                return "deep"

        # 3. Precise patterns (list / steps / methods)
        for pat in self.PRECISE:
            if pat.search(q):
                return "precise"

        # 4. Fast patterns (definition / fact / yes-no)
        for pat in self.FAST:
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
