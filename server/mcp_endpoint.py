"""Central stateless MCP endpoint for the self-hosted Mem0 server.

Serves the occ-memory tools over MCP Streamable HTTP (stateless mode,
JSON responses). Identity always comes from the X-API-Key header resolved
per request; no tool accepts user_id. Mounted into the FastAPI app by
server/main.py; requires the session manager lifespan to be running.
"""

from __future__ import annotations

from contextvars import ContextVar

from fastapi import HTTPException
from mcp.server import MCPServer
from mcp.server.transport_security import TransportSecuritySettings
from starlette.responses import JSONResponse

from auth import _resolve_user_from_api_key
from db import SessionLocal
from models import User

mcp = MCPServer(name="mem0")

current_user: ContextVar[User] = ContextVar("mcp_current_user")


def _user() -> User:
    try:
        return current_user.get()
    except LookupError:  # pragma: no cover - auth wrapper always sets it
        raise RuntimeError("MCP tool called without an authenticated user.")


def resolve_mcp_user(api_key: str) -> User:
    """Resolve a personal API key to its user. Admin/legacy keys are not
    accepted on the MCP path: every memory operation needs a real identity."""
    with SessionLocal() as db:
        return _resolve_user_from_api_key(api_key, db)


class ApiKeyAuth:
    """ASGI wrapper enforcing X-API-Key before the MCP app sees the request.

    ponytail: sync DB lookup inside the event loop, same as the sync REST
    handlers; move to a threadpool if MCP traffic ever matters.
    """

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        headers = {k.decode().lower(): v.decode() for k, v in scope.get("headers", [])}
        api_key = headers.get("x-api-key")
        if not api_key:
            response = JSONResponse(
                {"detail": "Authentication required. Provide an X-API-Key header."},
                status_code=401,
            )
            await response(scope, receive, send)
            return
        try:
            user = resolve_mcp_user(api_key)
        except HTTPException as exc:
            response = JSONResponse({"detail": exc.detail}, status_code=exc.status_code)
            await response(scope, receive, send)
            return
        token = current_user.set(user)
        try:
            await self.app(scope, receive, send)
        finally:
            current_user.reset(token)


# API-key auth is the gate; Host-header pinning would break VM deployments.
_inner_app = mcp.streamable_http_app(
    streamable_http_path="/mcp",
    json_response=True,
    stateless_http=True,
    transport_security=TransportSecuritySettings(enable_dns_rebinding_protection=False),
)
asgi_app = ApiKeyAuth(_inner_app)


def lifespan_context():
    """Session-manager lifespan; Starlette does not run mounted sub-app
    lifespans, so server/main.py enters this from the FastAPI lifespan."""
    return mcp.session_manager.run()
