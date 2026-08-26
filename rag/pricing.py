from __future__ import annotations

from rag.models import TokenUsage

"""Per-model $/token pricing table for cost estimation.

English: There is no billing API to query at runtime, so prices are kept as
hardcoded constants here and used only to *estimate* cost per query for the
evaluation dashboard (architecturev2.txt section 7, "System Metrics" ->
"Cost per Query"). Ollama is always $0 since it runs on local hardware.
Prices are USD per 1,000 tokens, matching common provider rate cards, and
were last checked August 2026 — re-verify against the provider's pricing
page before trusting cost figures for a real budget decision.
中文: 目前沒有可在執行期查詢的計費 API，因此價格先以固定常數維護於此，僅用於
評估儀表板估算每次查詢的成本（對應 architecturev2.txt 第 7 節「System
Metrics」的「Cost per Query」）。Ollama 因為在本機硬體上執行，成本永遠是 $0。
價格單位為每 1,000 tokens 的美元，對應常見 provider 的費率表，最後確認時間為
2026 年 8 月——實際用於預算決策前，請自行對照 provider 官方價格頁面確認。
"""

# (prompt $/1K tokens, completion $/1K tokens)
_CHAT_PRICE_PER_1K: dict[str, tuple[float, float]] = {
    "gpt-4o-mini": (0.00015, 0.0006),
    "gpt-4o": (0.0025, 0.01),
    "anthropic.claude-3-haiku-20240307-v1:0": (0.00025, 0.00125),
}

# $/1K tokens, embeddings have no separate completion cost.
_EMBED_PRICE_PER_1K: dict[str, float] = {
    "text-embedding-3-small": 0.00002,
    "amazon.titan-embed-text-v2:0": 0.00002,
}


def estimate_chat_cost_usd(model: str, usage: TokenUsage) -> float | None:
    """Estimate the $ cost of one generate() call's token usage.

    Returns None when the model isn't in the pricing table (e.g. an Ollama
    model, or an OpenAI/Bedrock model not yet added here) rather than
    silently returning 0, so callers can distinguish "known to be free" from
    "cost unknown".
    """

    price = _CHAT_PRICE_PER_1K.get(model)
    if price is None:
        return None
    prompt_price, completion_price = price
    return round(
        (usage.prompt_tokens / 1000) * prompt_price
        + (usage.completion_tokens / 1000) * completion_price,
        6,
    )


def estimate_embed_cost_usd(model: str, token_count: int) -> float | None:
    """Estimate the $ cost of embedding `token_count` tokens with `model`."""

    price = _EMBED_PRICE_PER_1K.get(model)
    if price is None:
        return None
    return round((token_count / 1000) * price, 6)


def is_free_provider(provider: str) -> bool:
    """Ollama always runs on local hardware, so its marginal $ cost is 0."""

    return provider == "ollama"
