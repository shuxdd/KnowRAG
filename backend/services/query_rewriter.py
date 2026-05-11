import json
import logging
import re
from typing import List

logger = logging.getLogger(__name__)
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from backend.config import get_settings

settings = get_settings()

REWRITE_PROMPT = """You are a query rewriter for a RAG knowledge base.
Analyze the conversation history and current query, then output a JSON with rewritten queries.

## Tasks (apply all that are needed):

1. **Co-reference resolution**: Replace pronouns ("它", "他", "这个", "那个") with the actual entity from conversation history
2. **Spelling correction**: Fix typos and normalize entity names
3. **Query expansion**: Add key synonyms or related terms (keep concise)
4. **Decomposition**: If the query compares/contrasts or has multiple aspects, split into 2-3 sub-queries

## Output format:
```json
{{
  "original": "<原始查询>",
  "rewritten": "<改写后的单查询，包含扩展词>",
  "sub_queries": ["子查询1", "子查询2"],
  "changes": ["指代消解: X→Y", "扩展: +关键词"]
}}
```

## Examples:

History: [用户: "介绍一下LangChain", 助手: "LangChain是..."]
Query: "它的核心组件有哪些"
Output:
```json
{{
  "original": "它的核心组件有哪些",
  "rewritten": "LangChain的核心组件有哪些",
  "sub_queries": [],
  "changes": ["指代消解: 它→LangChain"]
}}
```

History: (empty)
Query: "RAG和传统搜索有什么区别"
Output:
```json
{{
  "original": "RAG和传统搜索有什么区别",
  "rewritten": "RAG和传统搜索区别对比",
  "sub_queries": ["RAG检索增强生成的原理和流程", "传统搜索技术的原理和特点"],
  "changes": ["分解: 对比问题拆为两个方面"]
}}
```

History: (empty)
Query: "怎么优化RAG"
Output:
```json
{{
  "original": "怎么优化RAG",
  "rewritten": "RAG检索增强生成优化方法 提升召回率 提高准确性",
  "sub_queries": [],
  "changes": ["扩展: +提升召回率 +提高准确性"]
}}
```

Conversation history:
{chat_history}

Current query: {query}

Output ONLY the JSON, no explanation."""


class QueryRewriter:
    """LLM 驱动的查询改写器，处理指代消解、扩展、纠错、分解"""

    def __init__(self):
        self.llm = ChatOpenAI(
            model=settings.qwen_model,
            api_key=settings.qwen_api_key,
            base_url=settings.qwen_base_url,
            temperature=0.0,
            request_timeout=5,
        )
        self.prompt = ChatPromptTemplate.from_template(REWRITE_PROMPT)

    def rewrite(self, query: str, chat_history: str = "(无历史对话)") -> dict:
        """
        改写查询

        Args:
            query: 当前查询
            chat_history: 格式化的对话历史字符串

        Returns:
            {original, rewritten, sub_queries, changes}

        失败时降级返回原始查询
        """
        try:
            messages = self.prompt.format_messages(
                chat_history=chat_history, query=query
            )
            response = self.llm.invoke(messages)
            return self._parse(response.content, query)
        except Exception:
            logger.warning("Query rewriting failed, falling back to original query", exc_info=True)
            return {
                "original": query,
                "rewritten": query,
                "sub_queries": [],
                "changes": [],
            }

    def _parse(self, raw: str, original: str) -> dict:
        """解析 LLM 输出的 JSON，带容错"""
        raw = raw.strip()
        if raw.startswith("```"):
            raw = re.sub(r"```(?:\w*)?\s*([\s\S]*?)\s*```", r"\1", raw).strip()
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return {
                "original": original,
                "rewritten": raw,
                "sub_queries": [],
                "changes": [],
            }

    def get_queries(self, result: dict) -> List[str]:
        """从改写结果中提取需要检索的查询列表"""
        queries = []
        if result.get("rewritten"):
            queries.append(result["rewritten"])
        for sq in result.get("sub_queries", []):
            if sq:
                queries.append(sq)
        return queries if queries else [result.get("original", "")]
