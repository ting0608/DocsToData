from __future__ import annotations

from mangum import Mangum

from backend.app import app

"""AWS Lambda entry point for the FastAPI app.

English: Mangum adapts ASGI apps (FastAPI/Starlette) to the Lambda/API
Gateway request-response shape, so the exact same `backend/app.py` used for
local dev, Docker, and Cloud Run also runs on Lambda unchanged. This module
is the `Handler` referenced by `template.yaml`
(`backend.lambda_handler.handler`).
中文: Mangum 把 ASGI 應用（FastAPI/Starlette）轉接成 Lambda/API Gateway 的
請求/回應格式，因此本機開發、Docker、Cloud Run 用的同一份 `backend/app.py`
可以原封不動在 Lambda 上執行。這個模組就是 `template.yaml` 中
`Handler: backend.lambda_handler.handler` 所指向的進入點。
"""

handler = Mangum(app, lifespan="off")
