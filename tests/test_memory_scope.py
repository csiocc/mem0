import pytest

from memory_scope import (
    RESERVED_SCOPE_METADATA,
    ScopeError,
    bind_memory_metadata,
    build_read_filter,
    resolve_scope,
)


def test_project_scope_requires_project_id():
    with pytest.raises(ScopeError, match="project_id is required"):
        resolve_scope("project", None)


def test_global_scope_rejects_project_id():
    with pytest.raises(ScopeError, match="must be omitted"):
        resolve_scope("global", "LernendeLab")


def test_project_scope_has_stable_key():
    target = resolve_scope("project", " LernendeLab ")

    assert target.scope == "project"
    assert target.project_id == "LernendeLab"
    assert target.scope_key == "project:LernendeLab"


def test_metadata_is_bound_to_authenticated_user_and_scope():
    target = resolve_scope("project", "LernendeLab")

    metadata = bind_memory_metadata(
        user_id="11111111-1111-1111-1111-111111111111",
        target=target,
        metadata={"type": "decision"},
    )

    assert metadata == {
        "type": "decision",
        "scope": "project",
        "project_id": "LernendeLab",
        "scope_key": "project:LernendeLab",
    }


@pytest.mark.parametrize("key", sorted(RESERVED_SCOPE_METADATA))
def test_client_cannot_override_scope_metadata(key):
    target = resolve_scope("global", None)

    with pytest.raises(ScopeError, match="reserved metadata"):
        bind_memory_metadata(
            user_id="11111111-1111-1111-1111-111111111111",
            target=target,
            metadata={key: "attacker-value"},
        )


def test_project_reads_include_global_and_current_project():
    target = resolve_scope("project", "LernendeLab")

    assert build_read_filter("user-1", target) == {
        "user_id": "user-1",
        "OR": [
            {"scope_key": "global"},
            {"scope_key": "project:LernendeLab"},
        ],
    }


def test_global_reads_exclude_project_memories():
    target = resolve_scope("global", None)

    assert build_read_filter("user-1", target) == {
        "user_id": "user-1",
        "scope_key": "global",
    }
