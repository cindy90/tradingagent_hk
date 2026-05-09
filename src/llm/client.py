from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from anthropic import Anthropic
from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential

from config import get_settings

from .router import ModelTier, resolve_model


@dataclass
class LLMResponse:
    text: str
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    model: str = ""
    raw: Any = None

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


@dataclass
class TokenLedger:
    by_tier: dict[str, dict[str, int]] = field(default_factory=dict)

    def record(self, tier: str, resp: LLMResponse) -> None:
        slot = self.by_tier.setdefault(
            tier,
            {"input": 0, "output": 0, "cache_read": 0, "cache_write": 0, "calls": 0},
        )
        slot["input"] += resp.input_tokens
        slot["output"] += resp.output_tokens
        slot["cache_read"] += resp.cache_read_tokens
        slot["cache_write"] += resp.cache_write_tokens
        slot["calls"] += 1

    def summary(self) -> str:
        lines = ["| Tier | Calls | Input | Output | Cache Read | Cache Write |",
                 "|------|-------|-------|--------|------------|-------------|"]
        for tier, s in self.by_tier.items():
            lines.append(
                f"| {tier} | {s['calls']} | {s['input']} | {s['output']} | "
                f"{s['cache_read']} | {s['cache_write']} |"
            )
        return "\n".join(lines)


class LLMClient:
    def __init__(self, api_key: str | None = None) -> None:
        s = get_settings()
        self._client = Anthropic(api_key=api_key or s.anthropic_api_key)
        self.ledger = TokenLedger()

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=2, max=20))
    def complete(
        self,
        *,
        tier: ModelTier,
        system: str | list[dict] | None = None,
        messages: list[dict],
        max_tokens: int = 4096,
        temperature: float = 0.3,
        cached_system_blocks: list[str] | None = None,
    ) -> LLMResponse:
        """
        cached_system_blocks: 静态文本块列表，自动加 cache_control。
        例如把招股书全文 / 行业基础资料放进来，跨 Agent 复用同一份 cache。
        """
        model = resolve_model(tier)

        sys_param: Any
        if cached_system_blocks:
            blocks: list[dict] = []
            for i, blk in enumerate(cached_system_blocks):
                if not blk:
                    continue
                blocks.append({
                    "type": "text",
                    "text": blk,
                    "cache_control": {"type": "ephemeral"},
                })
            if isinstance(system, str) and system:
                blocks.append({"type": "text", "text": system})
            elif isinstance(system, list):
                blocks.extend(system)
            sys_param = blocks
        else:
            sys_param = system

        kwargs: dict[str, Any] = {
            "model": model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": messages,
        }
        if sys_param:
            kwargs["system"] = sys_param

        logger.debug(f"LLM call tier={tier.value} model={model} msgs={len(messages)}")
        resp = self._client.messages.create(**kwargs)

        text = "".join(b.text for b in resp.content if getattr(b, "type", "") == "text")
        usage = resp.usage
        out = LLMResponse(
            text=text,
            input_tokens=getattr(usage, "input_tokens", 0),
            output_tokens=getattr(usage, "output_tokens", 0),
            cache_read_tokens=getattr(usage, "cache_read_input_tokens", 0) or 0,
            cache_write_tokens=getattr(usage, "cache_creation_input_tokens", 0) or 0,
            model=model,
            raw=resp,
        )
        self.ledger.record(tier.value, out)
        return out
