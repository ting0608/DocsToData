from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv

VALID_PROVIDERS = {"openai", "ollama", "bedrock"}
VALID_AI_MODES = {"local", "aws"}
VALID_VECTOR_STORES = {"faiss"}

# Which providers make sense to expose by default for each deployment mode.
# architecturev2.txt section 10: local -> Ollama/FAISS, production -> Bedrock.
# OpenAI is left available in both modes since the current app already
# depends on it for the default/legacy provider and it works from anywhere.
_DEFAULT_ENABLED_PROVIDERS: dict[str, set[str]] = {
    "local": {"ollama", "openai"},
    "aws": {"openai", "bedrock"},
}


@dataclass(frozen=True)
class AppConfig:
    """Deployment-mode configuration from architecturev2.txt section 10.

    English: `enabled_providers` governs which `provider` values the API
    will accept — this generalizes the old `DISABLE_OLLAMA` flag into a
    proper allow-list driven by `AI_MODE`/`LLM_PROVIDER`/`ENABLED_PROVIDERS`.
    `default_llm_provider` is what CLI/tools fall back to when no explicit
    provider is given.
    中文: `enabled_providers` 決定 API 會接受哪些 `provider` 值——這是把舊有的
    `DISABLE_OLLAMA` flag 一般化為由 `AI_MODE`/`LLM_PROVIDER`/
    `ENABLED_PROVIDERS` 驅動的白名單機制。`default_llm_provider` 則是在沒有
    明確指定 provider 時，CLI/工具會使用的預設值。
    """

    ai_mode: str
    default_llm_provider: str
    vector_store: str
    enabled_providers: frozenset[str]


def load_app_config() -> AppConfig:
    """Load and validate `AI_MODE`/`LLM_PROVIDER`/`VECTOR_STORE` from `.env`.

    English: Fails fast with a clear `ValueError` on invalid values rather
    than silently falling back, so misconfiguration is caught at startup.
    中文: 遇到不合法的值會直接拋出明確的 `ValueError`，而不是悄悄使用預設值，
    確保設定錯誤能在啟動時就被發現。
    """

    load_dotenv()

    ai_mode = os.getenv("AI_MODE", "local").strip().lower()
    if ai_mode not in VALID_AI_MODES:
        raise ValueError(f"AI_MODE must be one of {sorted(VALID_AI_MODES)}, got '{ai_mode}'")

    default_provider = os.getenv("LLM_PROVIDER", "ollama" if ai_mode == "local" else "bedrock").strip().lower()
    if default_provider not in VALID_PROVIDERS:
        raise ValueError(f"LLM_PROVIDER must be one of {sorted(VALID_PROVIDERS)}, got '{default_provider}'")

    vector_store = os.getenv("VECTOR_STORE", "faiss").strip().lower()
    if vector_store not in VALID_VECTOR_STORES:
        raise ValueError(f"VECTOR_STORE must be one of {sorted(VALID_VECTOR_STORES)}, got '{vector_store}'")

    enabled = set(_DEFAULT_ENABLED_PROVIDERS[ai_mode])

    # Backward-compatible: the pre-v2 `DISABLE_OLLAMA=true` flag (used on
    # Cloud Run) still works and removes ollama even in local mode.
    if os.getenv("DISABLE_OLLAMA", "").strip().lower() in ("1", "true", "yes"):
        enabled.discard("ollama")

    # Explicit override for advanced setups (comma-separated provider list).
    explicit = os.getenv("ENABLED_PROVIDERS", "").strip()
    if explicit:
        enabled = {p.strip().lower() for p in explicit.split(",") if p.strip()}
        unknown = enabled - VALID_PROVIDERS
        if unknown:
            raise ValueError(f"ENABLED_PROVIDERS contains unknown provider(s): {sorted(unknown)}")

    return AppConfig(
        ai_mode=ai_mode,
        default_llm_provider=default_provider,
        vector_store=vector_store,
        enabled_providers=frozenset(enabled),
    )
