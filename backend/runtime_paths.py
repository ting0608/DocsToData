from __future__ import annotations

import os
from pathlib import Path

"""Runtime-aware data paths for local dev, Docker/Cloud Run, and AWS Lambda.

English: The rest of the codebase (rag/rag_local/rag_aws pipelines,
backend/app.py) refers to data locations with simple relative strings like
"data/index" or "data/uploads". That works fine when the process can write
next to the repo (local dev, Docker container with a bind-mounted volume,
Cloud Run). On AWS Lambda, the deployment package/image filesystem is
read-only except for `/tmp` (ephemeral per execution environment, wiped
between cold starts), so those same relative paths must be redirected under
`/tmp` instead. This module centralizes that decision in one place so
`backend/app.py` doesn't need provider-specific branching scattered through
every route.
中文: 專案其餘程式碼（rag/rag_local/rag_aws pipeline、backend/app.py）都用簡單
的相對路徑字串（例如 "data/index"、"data/uploads"）指定資料位置。在本機開發、
掛載 volume 的 Docker 容器、或 Cloud Run 上都沒問題，因為程式可以直接寫在
repo 旁邊。但在 AWS Lambda 上，部署套件/映像檔的檔案系統是唯讀的，只有
`/tmp` 可寫（且每個執行環境的 `/tmp` 是暫時性的，冷啟動之間會被清空），因此
同樣的相對路徑必須改為導向 `/tmp`。這個模組把這個判斷邏輯集中管理，避免
`backend/app.py` 的每個路由都要各自判斷 provider。
"""


def is_lambda() -> bool:
    """True when running inside an AWS Lambda execution environment."""

    return bool(os.getenv("AWS_LAMBDA_FUNCTION_NAME"))


def writable_base_dir(local_base_dir: Path) -> Path:
    """Return the directory that data paths should be resolved against.

    English: `/tmp` on Lambda (the only writable path), otherwise the
    caller-supplied local base (typically the repo root).
    中文: 在 Lambda 上回傳 `/tmp`（唯一可寫的路徑），否則回傳呼叫端提供的本機
    base 目錄（通常是 repo 根目錄）。
    """

    if is_lambda():
        return Path("/tmp")
    return local_base_dir


def app_data_dir(local_base_dir: Path, *relative: str) -> str:
    """Resolve a data-relative path (e.g. "data/index") against the correct
    writable base for the current runtime.

    English: If `relative` is already an absolute path, it is returned as-is
    (mirrors `pathlib.Path.joinpath` semantics) so callers that already pass
    a fully-qualified path (e.g. a custom `out_dir` from an API request)
    keep working unchanged.
    中文: 若 `relative` 本身已是絕對路徑，則原樣回傳（沿用
    `pathlib.Path.joinpath` 的行為），讓已經傳入完整路徑（例如 API 請求中自訂
    的 `out_dir`）的呼叫端行為維持不變。
    """

    return str(writable_base_dir(local_base_dir).joinpath(*relative))
