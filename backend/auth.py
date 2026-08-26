from __future__ import annotations

import os
import threading
from typing import Any

import jwt
from fastapi import Header, HTTPException
from jwt import PyJWKClient

"""AWS Cognito JWT verification for the Go Invoice API.

English: Verifies Cognito access tokens (RS256, JWKS) on protected routes.
When Cognito env vars are not configured, auth falls back to an open "dev mode"
so local development keeps working without an AWS account.
中文: 驗證 Cognito 存取權杖 (RS256 + JWKS)。若未設定 Cognito 環境變數，會退回到
開發模式（不驗證），方便本機開發時不需要 AWS 帳號即可運作。
"""

_jwk_client_lock = threading.Lock()
_jwk_client: PyJWKClient | None = None
_jwk_client_url: str | None = None


def _cognito_region() -> str:
    return os.getenv("COGNITO_REGION", "").strip()


def _user_pool_id() -> str:
    return os.getenv("COGNITO_USER_POOL_ID", "").strip()


def _client_id() -> str:
    return os.getenv("COGNITO_CLIENT_ID", "").strip()


def _domain() -> str:
    return os.getenv("COGNITO_DOMAIN", "").strip()


def _sso_provider() -> str:
    """Optional Cognito identity provider name (e.g. an enterprise SAML/OIDC IdP)
    for a direct 'Continue with SSO' button that skips the Hosted UI IdP picker."""

    return os.getenv("COGNITO_SSO_PROVIDER", "").strip()


def auth_enabled() -> bool:
    """True once the three required Cognito settings are present."""

    return bool(_cognito_region() and _user_pool_id() and _client_id())


def _issuer() -> str:
    return f"https://cognito-idp.{_cognito_region()}.amazonaws.com/{_user_pool_id()}"


def _jwks_url() -> str:
    return f"{_issuer()}/.well-known/jwks.json"


def _get_jwk_client() -> PyJWKClient:
    """Lazily build (and cache) the JWKS client for the configured user pool."""

    global _jwk_client, _jwk_client_url
    url = _jwks_url()
    with _jwk_client_lock:
        if _jwk_client is None or _jwk_client_url != url:
            _jwk_client = PyJWKClient(url)
            _jwk_client_url = url
        return _jwk_client


def _decode_access_token(token: str) -> dict[str, Any]:
    jwk_client = _get_jwk_client()
    signing_key = jwk_client.get_signing_key_from_jwt(token)
    claims = jwt.decode(
        token,
        signing_key.key,
        algorithms=["RS256"],
        issuer=_issuer(),
        options={"verify_aud": False},
    )
    if claims.get("token_use") != "access":
        raise jwt.InvalidTokenError("Expected a Cognito access token")
    if claims.get("client_id") != _client_id():
        raise jwt.InvalidTokenError("Token was not issued for this app client")
    return claims


def get_current_user(authorization: str | None = Header(default=None)) -> dict[str, Any]:
    """FastAPI dependency: resolve the caller's identity from a bearer token.

    English: Returns a dev-mode identity when Cognito is not configured.
    中文: 若未設定 Cognito，回傳開發模式的預設身分。
    """

    if not auth_enabled():
        return {"sub": "dev-user", "username": "dev", "email": None, "dev_mode": True}

    if not authorization or not authorization.lower().startswith("bearer "):
        raise HTTPException(status_code=401, detail="Missing bearer token")

    token = authorization.split(" ", 1)[1].strip()
    try:
        claims = _decode_access_token(token)
    except Exception as exc:  # noqa: BLE001 - surface as 401 regardless of jwt error subtype
        raise HTTPException(status_code=401, detail=f"Invalid or expired token: {exc}") from exc

    return {
        "sub": claims.get("sub"),
        "username": claims.get("username") or claims.get("sub"),
        "email": claims.get("email"),
        "scope": claims.get("scope"),
        "dev_mode": False,
    }


def auth_public_config() -> dict[str, Any]:
    """Non-secret config the frontend needs to drive the Cognito Hosted UI flow."""

    return {
        "enabled": auth_enabled(),
        "region": _cognito_region(),
        "user_pool_id": _user_pool_id(),
        "client_id": _client_id(),
        "domain": _domain(),
        "sso_provider": _sso_provider() or None,
    }
