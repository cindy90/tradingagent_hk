"""ScoreCard 双产出工具。

设计：每个分析 Agent 在 SYSTEM prompt 末尾追加一段 schema 提示，
要求 LLM 在 markdown 报告之后 **额外** 输出一段 ```json``` 代码块作为评分卡。

用一次 LLM 调用同时产出 (markdown 报告, 评分卡)，不增加调用次数。
解析失败时降级写空评分卡（不阻塞主流程），但 logger 警告。
"""
from __future__ import annotations

import json
import re
from typing import Any

from loguru import logger
from pydantic import ValidationError

from src.feedback.models import AgentScoreCard


def schema_instruction(score_card_class: type[AgentScoreCard]) -> str:
    """为给定 ScoreCard 类生成给 LLM 的 schema 指令片段。"""
    schema = score_card_class.model_json_schema()
    fields = []
    for name, prop in schema.get("properties", {}).items():
        type_hint = prop.get("type", prop.get("anyOf", [{}])[0].get("type", "any"))
        desc = prop.get("description", "")
        constraint = ""
        if "minimum" in prop and "maximum" in prop:
            constraint = f" [{prop['minimum']}-{prop['maximum']}]"
        elif "enum" in prop:
            constraint = f" [{'/'.join(map(str, prop['enum']))}]"
        fields.append(f"  - {name} ({type_hint}{constraint}): {desc}")

    return (
        "\n\n---\n\n"
        "**强制要求：在上述 markdown 报告之后，再输出一段 ```json``` 代码块作为评分卡，"
        "结构如下（所有数值字段必填，文本字段简洁）：\n\n"
        f"```\n字段说明：\n" + "\n".join(fields) + "\n```\n\n"
        "评分卡示例格式：\n"
        "```json\n"
        + json.dumps(_example_for_class(score_card_class), ensure_ascii=False, indent=2)
        + "\n```\n"
        "**只输出一段 json 代码块，不要重复或追加其它内容。**"
    )


def _example_for_class(cls: type[AgentScoreCard]) -> dict[str, Any]:
    """根据 schema 生成一个最小示例（用于 prompt 中演示格式）。"""
    out: dict[str, Any] = {}
    schema = cls.model_json_schema()
    for name, prop in schema.get("properties", {}).items():
        if "default" in prop:
            out[name] = prop["default"]
            continue
        types = prop.get("type")
        if types is None:
            anyof = prop.get("anyOf", [])
            types = anyof[0].get("type") if anyof else "string"
        if types == "number" or types == "integer":
            out[name] = (prop.get("minimum", 0) + prop.get("maximum", 5)) / 2
        elif types == "array":
            out[name] = []
        elif types == "boolean":
            out[name] = False
        elif types == "object":
            out[name] = {}
        else:
            if "enum" in prop:
                out[name] = prop["enum"][0]
            else:
                out[name] = ""
    return out


def parse_score_card(
    text: str,
    score_card_class: type[AgentScoreCard],
    agent_name: str = "",
) -> AgentScoreCard | None:
    """从 LLM 输出末尾提取最后一个 ```json``` 代码块并校验。

    优先取最后一个（决议 Agent 的输出格式可能多块 json，评分卡应在末尾）。
    """
    matches = list(re.finditer(r"```json\s*(\{.*?\})\s*```", text, re.DOTALL))
    if not matches:
        logger.warning(f"[{agent_name}] 输出未包含 ```json``` 代码块，评分卡缺失")
        return None
    candidate = matches[-1].group(1)
    try:
        parsed = json.loads(candidate)
    except json.JSONDecodeError as e:
        logger.warning(f"[{agent_name}] 评分卡 JSON 解析失败: {e}")
        return None
    try:
        return score_card_class.model_validate(parsed)
    except ValidationError as e:
        logger.warning(f"[{agent_name}] 评分卡 schema 校验失败: {str(e)[:200]}")
        return None


def strip_score_card_block(text: str) -> str:
    """从输出中剥离最后一个 ```json``` 代码块，返回纯 markdown 部分。
    用于把"展示给人看的报告"和"机器消费的评分卡"分开存储。"""
    matches = list(re.finditer(r"```json\s*\{.*?\}\s*```", text, re.DOTALL))
    if not matches:
        return text
    last = matches[-1]
    return (text[:last.start()] + text[last.end():]).strip()
