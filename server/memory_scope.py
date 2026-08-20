from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

ScopeName = Literal["global", "project"]
RESERVED_SCOPE_METADATA = frozenset(
    {
        "user_id",
        "scope",
        "project_id",
        "scope_key",
        "import_id",
        "source_ref",
        "source_sha256",
        "import_content_sha256",
        "content_sha256",
    }
)


class ScopeError(ValueError):
    pass


@dataclass(frozen=True)
class MemoryScope:
    scope: ScopeName
    project_id: str | None

    @property
    def scope_key(self) -> str:
        if self.scope == "global":
            return "global"
        return f"project:{self.project_id}"


def resolve_scope(scope: str, project_id: str | None) -> MemoryScope:
    normalized_scope = scope.strip().lower()
    normalized_project = project_id.strip() if project_id else None

    if normalized_scope not in {"global", "project"}:
        raise ScopeError("scope must be 'global' or 'project'")
    if normalized_scope == "project" and not normalized_project:
        raise ScopeError("project_id is required for project scope")
    if normalized_scope == "global" and normalized_project:
        raise ScopeError("project_id must be omitted for global scope")

    return MemoryScope(
        scope=normalized_scope,
        project_id=normalized_project,
    )


def bind_memory_metadata(
    *,
    user_id: str,
    target: MemoryScope,
    metadata: dict[str, Any] | None,
) -> dict[str, Any]:
    del user_id  # The core Memory.add call receives user_id separately.
    supplied = dict(metadata or {})
    reserved = RESERVED_SCOPE_METADATA.intersection(supplied)
    if reserved:
        names = ", ".join(sorted(reserved))
        raise ScopeError(f"reserved metadata cannot be supplied by the client: {names}")

    return {
        **supplied,
        "scope": target.scope,
        "project_id": target.project_id,
        "scope_key": target.scope_key,
    }


def scope_run_id(bound_metadata: dict[str, Any]) -> str | None:
    """Session identifier that keeps inference inside one scope.

    mem0 builds both its recent-message context and its deduplication search
    from user_id/agent_id/run_id only — scope_key is dropped. Without a
    run_id, an inferred write therefore sees the user's messages and memories
    from every other project and can extract their content into this scope.
    Passing the scope key as run_id restores that boundary; reads are
    unaffected because they filter on scope_key.
    """
    scope_key = bound_metadata.get("scope_key")
    return str(scope_key) if scope_key else None


def build_read_filter(user_id: str, target: MemoryScope) -> dict[str, Any]:
    if target.scope == "global":
        return {"user_id": user_id, "scope_key": "global"}

    return {
        "user_id": user_id,
        "OR": [
            {"scope_key": "global"},
            {"scope_key": target.scope_key},
        ],
    }


def memory_owner_id(memory: Any) -> str | None:
    if isinstance(memory, dict):
        owner = memory.get("user_id")
        if owner is None and isinstance(memory.get("metadata"), dict):
            owner = memory["metadata"].get("user_id")
        return str(owner) if owner is not None else None

    payload = getattr(memory, "payload", None) or {}
    owner = payload.get("user_id")
    return str(owner) if owner is not None else None
