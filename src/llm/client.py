"""统一 LLM 客户端：支持 Anthropic / Kimi (Moonshot) / DeepSeek。

provider 通过环境变量 LLM_PROVIDER 选择 (anthropic|kimi|deepseek)。
- anthropic: 走原生 SDK，支持 cache_control 显式 prompt caching
- kimi:      走 OpenAI 兼容协议 (https://api.moonshot.cn/v1)，支持 context caching
             （需开 cache=enabled，自动按 prompt 前缀命中）
- deepseek:  走 OpenAI 兼容协议 (https://api.deepseek.com/v1)，自动 prompt caching
             （>1k tokens 自动命中，无需配置）

调用方代码不需要关心 provider 区别：tier → model 路由 + LLMResponse 输出格式
都已抽象。Token 账本统一记录。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

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


# ---------- Provider 实现 ----------

class _AnthropicProvider:
    def __init__(self, api_key: str):
        try:
            from anthropic import Anthropic
        except ImportError as e:
            raise RuntimeError(
                "未安装 anthropic SDK。pip install anthropic，或切换 LLM_PROVIDER=kimi/deepseek"
            ) from e
        self.client = Anthropic(api_key=api_key)

    def complete(
        self,
        *,
        model: str,
        system: str | list[dict] | None,
        messages: list[dict],
        max_tokens: int,
        temperature: float,
        cached_system_blocks: list[str] | None,
    ) -> LLMResponse:
        sys_param: Any
        if cached_system_blocks:
            blocks: list[dict] = []
            for blk in cached_system_blocks:
                if not blk:
                    continue
                blocks.append({
                    "type": "text", "text": blk,
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

        resp = self.client.messages.create(**kwargs)
        text = "".join(b.text for b in resp.content if getattr(b, "type", "") == "text")
        usage = resp.usage
        return LLMResponse(
            text=text,
            input_tokens=getattr(usage, "input_tokens", 0),
            output_tokens=getattr(usage, "output_tokens", 0),
            cache_read_tokens=getattr(usage, "cache_read_input_tokens", 0) or 0,
            cache_write_tokens=getattr(usage, "cache_creation_input_tokens", 0) or 0,
            model=model,
            raw=resp,
        )


class _OpenAICompatProvider:
    """统一 Kimi / DeepSeek / 任何 OpenAI 兼容服务。"""

    def __init__(self, api_key: str, base_url: str, provider_name: str):
        try:
            from openai import OpenAI
        except ImportError as e:
            raise RuntimeError("未安装 openai SDK。pip install openai") from e
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.provider_name = provider_name

    def complete(
        self,
        *,
        model: str,
        system: str | list[dict] | None,
        messages: list[dict],
        max_tokens: int,
        temperature: float,
        cached_system_blocks: list[str] | None,
    ) -> LLMResponse:
        # OpenAI 协议把 system 合并进 messages
        full_messages: list[dict] = []
        # 把 cached_system_blocks 拼接到 system 前部
        # Kimi / DeepSeek 的 prompt caching 都基于"prompt 前缀"自动命中，
        # 把静态内容放最前面是关键。
        sys_text_parts: list[str] = []
        if cached_system_blocks:
            sys_text_parts.extend(b for b in cached_system_blocks if b)
        if isinstance(system, str) and system:
            sys_text_parts.append(system)
        elif isinstance(system, list):
            for b in system:
                if isinstance(b, dict) and b.get("type") == "text":
                    sys_text_parts.append(b.get("text", ""))

        if sys_text_parts:
            full_messages.append({"role": "system", "content": "\n\n".join(sys_text_parts)})
        full_messages.extend(messages)

        resp = self.client.chat.completions.create(
            model=model,
            messages=full_messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        choice = resp.choices[0]
        text = choice.message.content or ""
        usage = resp.usage
        # DeepSeek 在 usage 里有 prompt_cache_hit_tokens / prompt_cache_miss_tokens
        cache_read = 0
        cache_write = 0
        if usage is not None:
            cache_read = getattr(usage, "prompt_cache_hit_tokens", 0) or 0
            # cache_write 概念在 OpenAI 协议里不直接存在；DeepSeek 是"自动写"，无显式计费
        return LLMResponse(
            text=text,
            input_tokens=getattr(usage, "prompt_tokens", 0) or 0,
            output_tokens=getattr(usage, "completion_tokens", 0) or 0,
            cache_read_tokens=cache_read,
            cache_write_tokens=cache_write,
            model=model,
            raw=resp,
        )


def _build_provider():
    s = get_settings()
    name = (s.llm_provider or "anthropic").lower()
    if name == "anthropic":
        if not s.anthropic_api_key:
            raise RuntimeError("LLM_PROVIDER=anthropic 但 ANTHROPIC_API_KEY 未配置")
        return _AnthropicProvider(s.anthropic_api_key), name
    if name == "kimi":
        if not s.kimi_api_key:
            raise RuntimeError("LLM_PROVIDER=kimi 但 KIMI_API_KEY 未配置")
        return _OpenAICompatProvider(
            api_key=s.kimi_api_key,
            base_url=s.kimi_base_url or "https://api.moonshot.cn/v1",
            provider_name="kimi",
        ), name
    if name == "deepseek":
        if not s.deepseek_api_key:
            raise RuntimeError("LLM_PROVIDER=deepseek 但 DEEPSEEK_API_KEY 未配置")
        return _OpenAICompatProvider(
            api_key=s.deepseek_api_key,
            base_url=s.deepseek_base_url or "https://api.deepseek.com/v1",
            provider_name="deepseek",
        ), name
    raise RuntimeError(f"未知 LLM_PROVIDER: {name}")


# ---------- 公共 LLMClient ----------

class LLMClient:
    def __init__(self, provider=None, provider_name: str | None = None):
        if provider is None:
            self._provider, self._provider_name = _build_provider()
        else:
            self._provider = provider
            self._provider_name = provider_name or "custom"
        self.ledger = TokenLedger()

    @property
    def provider_name(self) -> str:
        return self._provider_name

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
        model = resolve_model(tier)
        logger.debug(
            f"LLM call provider={self._provider_name} tier={tier.value} model={model} "
            f"msgs={len(messages)}"
        )
        out = self._provider.complete(
            model=model,
            system=system,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            cached_system_blocks=cached_system_blocks,
        )
        self.ledger.record(tier.value, out)
        return out
