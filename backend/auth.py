from __future__ import annotations

import os
import threading
from typing import Any

import boto3
import jwt
from botocore.exceptions import ClientError
from dotenv import load_dotenv
from fastapi import Header, HTTPException
from jwt import PyJWKClient

# English: Other modules (rag/config.py, rag_local/config.py, ...) call
# load_dotenv() themselves, but only once a RagPipeline is instantiated --
# too late for auth_enabled()/auth_public_config(), which read env vars as
# soon as the app starts (e.g. GET /auth/config before any /query call).
# Calling it here too (idempotent, no-op if already loaded) ensures
# COGNITO_* is picked up from .env immediately in local/Docker dev.
# 中文: 其他模組會自行呼叫 load_dotenv()，但要等到 RagPipeline 被建立時才會
# 呼叫——對 auth_enabled()/auth_public_config() 來說太晚了，因為這兩個函式在
# App 啟動後就會讀取環境變數（例如還沒呼叫過 /query 就先呼叫了
# GET /auth/config）。在這裡也呼叫一次（重複呼叫是安全的，已載入則不做任何
#事），確保本機/Docker 開發時能立即讀到 .env 裡的 COGNITO_*。
load_dotenv()

"""AWS Cognito authentication for the Go Invoice API.

English: Two halves live here. (1) Token verification: verifies Cognito
access tokens (RS256, JWKS) on protected routes -- used by every
`Depends(get_current_user)` route. When Cognito env vars are not
configured, auth falls back to an open "dev mode" so local development
keeps working without an AWS account. (2) Classic username/password auth:
thin boto3 wrappers around Cognito's InitiateAuth/SignUp/etc, used by the
`/auth/*` routes in backend/app.py for the in-app email/password login
form (as opposed to the Hosted UI redirect flow in frontend/js/auth.js's
`signIn()`, which stays available for SSO).
中文: 這裡有兩個部分。(1) 權杖驗證：驗證 Cognito 存取權杖 (RS256 + JWKS)，用於
所有 `Depends(get_current_user)` 路由。若未設定 Cognito 環境變數，會退回到
開發模式（不驗證），方便本機開發時不需要 AWS 帳號即可運作。(2) 傳統帳號密碼
驗證：對 Cognito 的 InitiateAuth/SignUp 等 API 做簡單的 boto3 包裝，供
backend/app.py 的 `/auth/*` 路由使用，實作應用內的 email/密碼登入表單
（相對於 frontend/js/auth.js 的 `signIn()` 導向 Hosted UI 流程，該流程仍保留
給 SSO 使用）。
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


# ---------------------------------------------------------------------------
# Classic username/password auth (InitiateAuth / SignUp / etc)
#
# English: `cognito-idp`'s InitiateAuth/SignUp/ConfirmSignUp/ForgotPassword/
# GlobalSignOut family are unauthenticated API operations (no SigV4/IAM
# credentials required -- verified against a live user pool while building
# this) as long as boto3 has *some* region configured, so a default
# (no-credentials) boto3 client works fine both locally and on Lambda.
# 中文: `cognito-idp` 的 InitiateAuth/SignUp/ConfirmSignUp/ForgotPassword/
# GlobalSignOut 這幾個都是不需驗證的 API（不需要 SigV4/IAM 憑證，此結論已對
# 實際的 user pool 測試驗證過），只要 boto3 有設定好 region 即可，因此本機與
# Lambda 上用預設（無憑證）的 boto3 client 呼叫都沒問題。
# ---------------------------------------------------------------------------

_cognito_client_lock = threading.Lock()
_cognito_client: Any = None


def _cognito_idp():
    """Lazily build (and cache) the boto3 cognito-idp client for the configured region."""

    global _cognito_client
    with _cognito_client_lock:
        if _cognito_client is None:
            _cognito_client = boto3.client("cognito-idp", region_name=_cognito_region())
        return _cognito_client


def _require_auth_configured() -> None:
    if not auth_enabled():
        raise HTTPException(status_code=503, detail="Cognito is not configured on this server.")


# Map Cognito's boto3 exception codes to HTTP statuses + a friendly message
# prefix. Anything not listed here falls back to 400 with the raw message.
_COGNITO_ERROR_STATUS: dict[str, int] = {
    "NotAuthorizedException": 401,
    "UserNotFoundException": 401,
    "UserNotConfirmedException": 403,
    "UsernameExistsException": 409,
    "CodeMismatchException": 400,
    "ExpiredCodeException": 400,
    "InvalidPasswordException": 400,
    "InvalidParameterException": 400,
    "TooManyRequestsException": 429,
    "TooManyFailedAttemptsException": 429,
    "LimitExceededException": 429,
    "PasswordResetRequiredException": 403,
    "AliasExistsException": 409,
}


def _raise_from_cognito_error(exc: ClientError) -> None:
    code = exc.response.get("Error", {}).get("Code", "")
    message = exc.response.get("Error", {}).get("Message") or str(exc)
    status = _COGNITO_ERROR_STATUS.get(code, 400)
    raise HTTPException(status_code=status, detail=message) from exc


def _auth_result_to_tokens(auth_result: dict[str, Any]) -> dict[str, Any]:
    """Normalize a Cognito AuthenticationResult into the shape the frontend expects."""

    return {
        "access_token": auth_result.get("AccessToken"),
        "id_token": auth_result.get("IdToken"),
        # Only present on the *initial* sign-in response, not on a
        # REFRESH_TOKEN_AUTH response (Cognito doesn't rotate/return it then) --
        # callers must keep reusing the refresh token obtained at sign_in().
        "refresh_token": auth_result.get("RefreshToken"),
        "expires_in": auth_result.get("ExpiresIn"),
        "token_type": auth_result.get("TokenType", "Bearer"),
    }


def sign_in(username: str, password: str) -> dict[str, Any]:
    """Classic username/password sign-in via Cognito's USER_PASSWORD_AUTH flow.

    English: Returns access/id/refresh tokens on success. Raises HTTPException
    (401/403/etc, mapped from Cognito's error code) on failure -- including
    the case where Cognito demands a further challenge (e.g. NEW_PASSWORD_REQUIRED),
    which this simple flow does not support.
    """

    _require_auth_configured()
    try:
        resp = _cognito_idp().initiate_auth(
            AuthFlow="USER_PASSWORD_AUTH",
            AuthParameters={"USERNAME": username, "PASSWORD": password},
            ClientId=_client_id(),
        )
    except ClientError as exc:
        _raise_from_cognito_error(exc)

    if "ChallengeName" in resp:
        raise HTTPException(
            status_code=403,
            detail=f"Additional sign-in step required: {resp['ChallengeName']}",
        )

    return _auth_result_to_tokens(resp.get("AuthenticationResult", {}))


def refresh_tokens(refresh_token: str) -> dict[str, Any]:
    """Exchange a refresh token for a new access token (+ id token).

    English: Cognito does not return a new refresh token here, so the
    frontend must keep reusing the one obtained at sign_in() until it
    itself expires (default 30 days) or the user signs out.
    """

    _require_auth_configured()
    try:
        resp = _cognito_idp().initiate_auth(
            AuthFlow="REFRESH_TOKEN_AUTH",
            AuthParameters={"REFRESH_TOKEN": refresh_token},
            ClientId=_client_id(),
        )
    except ClientError as exc:
        _raise_from_cognito_error(exc)

    tokens = _auth_result_to_tokens(resp.get("AuthenticationResult", {}))
    tokens["refresh_token"] = refresh_token
    return tokens


def sign_up(email: str, password: str) -> dict[str, Any]:
    """Register a new user. Self-service sign-up on the Hosted UI stays enabled,
    so this mirrors that same behavior for the in-app form: the account is
    created but unconfirmed until confirm_sign_up() is called with the emailed code."""

    _require_auth_configured()
    try:
        resp = _cognito_idp().sign_up(
            ClientId=_client_id(),
            Username=email,
            Password=password,
            UserAttributes=[{"Name": "email", "Value": email}],
        )
    except ClientError as exc:
        _raise_from_cognito_error(exc)

    return {
        "user_confirmed": resp.get("UserConfirmed", False),
        "user_sub": resp.get("UserSub"),
    }


def confirm_sign_up(email: str, code: str) -> None:
    """Confirm a new account using the verification code emailed by Cognito."""

    _require_auth_configured()
    try:
        _cognito_idp().confirm_sign_up(ClientId=_client_id(), Username=email, ConfirmationCode=code)
    except ClientError as exc:
        _raise_from_cognito_error(exc)


def resend_confirmation_code(email: str) -> None:
    """Re-send the sign-up confirmation code (e.g. the first one expired or was lost)."""

    _require_auth_configured()
    try:
        _cognito_idp().resend_confirmation_code(ClientId=_client_id(), Username=email)
    except ClientError as exc:
        _raise_from_cognito_error(exc)


def sign_out(access_token: str) -> None:
    """Invalidate every refresh token issued for this user (all devices/sessions)."""

    _require_auth_configured()
    try:
        _cognito_idp().global_sign_out(AccessToken=access_token)
    except ClientError as exc:
        _raise_from_cognito_error(exc)
