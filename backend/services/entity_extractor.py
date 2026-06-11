"""
实体关系抽取模块

使用 LLM 从文档文本中抽取实体和关系，用于构建知识图谱。
"""

import json
import logging
from typing import Any

from backend.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


EXTRACT_PROMPT = """从以下文本中抽取实体和它们之间的关系。

要求：
1. 实体名称归一化（同义实体合并为一个标准名）
2. 每个实体给出类型和一句话描述
3. 关系要有明确的类型（如：依赖、属于、导致、对比、包含）
4. 标注支撑该关系的原文句子
5. 实体最多 {max_entities} 个，关系最多 {max_relations} 个

输出 JSON 格式：
{{"entities": [{{"name": "...", "type": "...", "description": "..."}}], "relations": [{{"source": "...", "target": "...", "relation": "...", "context": "..."}}]}}

文本内容：
{text}"""


QUERY_ENTITY_PROMPT = """从以下问题中提取关键实体名称，输出 JSON 数组。
要求：只返回与知识相关的实体名词，忽略动词和修饰词。

问题：{query}
输出格式：["实体1", "实体2"]"""


class EntityExtractor:
    """LLM-based entity and relationship extractor."""

    def __init__(self):
        from langchain_openai import ChatOpenAI

        model_name = settings.kg_extract_model or settings.mimo_model
        self._llm = ChatOpenAI(
            model=model_name,
            api_key=settings.mimo_api_key,
            base_url=settings.mimo_base_url,
            max_tokens=4096,
            temperature=0.1,
        )

    def extract_from_chunk(self, text: str) -> dict[str, Any]:
        """从父块文本中抽取实体和关系。"""
        prompt = EXTRACT_PROMPT.format(
            text=text,
            max_entities=settings.kg_max_entities_per_chunk,
            max_relations=settings.kg_max_relations_per_chunk,
        )
        try:
            response = self._llm.invoke(prompt)
            return self._parse_extraction_result(response.content)
        except Exception as e:
            logger.warning(f"Entity extraction failed: {e}")
            return {"entities": [], "relations": []}

    def extract_query_entities(self, query: str) -> list[str]:
        """从用户问题中提取实体名称列表。"""
        prompt = QUERY_ENTITY_PROMPT.format(query=query)
        try:
            response = self._llm.invoke(prompt)
            return self._parse_query_entities(response.content)
        except Exception as e:
            logger.warning(f"Query entity extraction failed: {e}")
            return []

    def _parse_extraction_result(self, raw: str) -> dict[str, Any]:
        """Parse LLM response into entities and relations."""
        raw = raw.strip()
        if raw.startswith("```"):
            lines = raw.split("\n")
            lines = [l for l in lines if not l.startswith("```")]
            raw = "\n".join(lines).strip()

        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            start = raw.find("{")
            end = raw.rfind("}") + 1
            if start >= 0 and end > start:
                try:
                    data = json.loads(raw[start:end])
                except json.JSONDecodeError:
                    logger.warning(f"Failed to parse extraction JSON: {raw[:200]}")
                    return {"entities": [], "relations": []}
            else:
                logger.warning(f"No JSON object found in extraction result: {raw[:200]}")
                return {"entities": [], "relations": []}

        entities = data.get("entities", [])
        relations = data.get("relations", [])

        valid_entities = [e for e in entities if isinstance(e, dict) and "name" in e]
        valid_relations = [
            r for r in relations
            if isinstance(r, dict) and "source" in r and "target" in r and "relation" in r
        ]

        return {"entities": valid_entities, "relations": valid_relations}

    def _parse_query_entities(self, raw: str) -> list[str]:
        """Parse entity names from LLM response."""
        raw = raw.strip()
        if raw.startswith("```"):
            lines = raw.split("\n")
            lines = [l for l in lines if not l.startswith("```")]
            raw = "\n".join(lines).strip()

        try:
            data = json.loads(raw)
            if isinstance(data, list):
                return [str(e) for e in data if isinstance(e, str)]
        except json.JSONDecodeError:
            start = raw.find("[")
            end = raw.rfind("]") + 1
            if start >= 0 and end > start:
                try:
                    data = json.loads(raw[start:end])
                    if isinstance(data, list):
                        return [str(e) for e in data if isinstance(e, str)]
                except json.JSONDecodeError:
                    pass

        logger.warning(f"Failed to parse query entities: {raw[:200]}")
        return []


entity_extractor = EntityExtractor()
