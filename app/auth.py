from __future__ import annotations

import hmac

from fastapi import Header, HTTPException, Request


async def authorize(
    request: Request, authorization: str | None = Header(default=None)
) -> None:
    expected = request.app.state.settings.api_key
    if not expected:
        return
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing bearer token")
    supplied = authorization.removeprefix("Bearer ").strip()
    if not hmac.compare_digest(supplied, expected):
        raise HTTPException(status_code=403, detail="Invalid bearer token")
