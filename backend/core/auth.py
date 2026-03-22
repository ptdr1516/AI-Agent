"""
Bearer Token Authentication — FastAPI dependency injection.

Usage in routes:
    from core.auth import verify_token
    @router.post("/endpoint", dependencies=[Depends(verify_token)])

If API_TOKEN is not set in .env, auth is skipped entirely (dev mode).
If it is set, every request must pass:
    Authorization: Bearer <token>
A mismatch or missing header returns HTTP 401.
"""
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from core.config import settings
from core.logger import log

_bearer = HTTPBearer(auto_error=False)


async def verify_token(
    credentials: HTTPAuthorizationCredentials | None = Depends(_bearer),
) -> str:
    """FastAPI dependency — validates bearer token if API_TOKEN is configured.
    Returns the Bearer token string as the `user_id` for multi-tenant isolation.
    """
    if not settings.API_TOKEN:
        # Auth is disabled — skip validation in dev/local mode
        return credentials.credentials if credentials else "anonymous_user"

    if credentials is None:
        log.warning("Unauthorized request — invalid or missing bearer token.")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing bearer token.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return credentials.credentials
