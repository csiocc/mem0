"""Write-time guards shared by the verbatim memory write paths.

Fork-only module. The two verbatim writers — the MCP ``add_memory`` tool and
the file import appender — store caller text unchanged, so this is where an
exact duplicate or an instruction aimed at a later reader has to be caught.

Inferred writes are deliberately not guarded here. Their stored text is
produced by the extraction model rather than supplied by the caller, and a
conversation that merely discusses prompt injection must stay storable.
"""

from __future__ import annotations

import re
from typing import Any

# Server-set metadata key holding the canonical content digest. Listed in
# RESERVED_SCOPE_METADATA so a client cannot forge it to dodge the check.
CONTENT_HASH_KEY = "content_sha256"

# Deliberately narrow: phrasings whose only purpose is to redirect whoever
# reads the memory back out of the untrusted-data frame. Broader rules such
# as "run this command" would reject ordinary operational notes.
_OVERRIDE_PATTERNS = (
    re.compile(
        r"\b(?:ignore|disregard|forget)\s+(?:all\s+|any\s+|the\s+)*"
        r"(?:previous|prior|preceding|earlier|above)\s+\w*\s*instruction",
        re.IGNORECASE,
    ),
    re.compile(r"\bignore\s+everything\s+(?:above|before)\b", re.IGNORECASE),
)


class InjectedInstructionError(ValueError):
    """Verbatim text carried instructions directed at the reading agent."""


def reject_override_instructions(text: str) -> None:
    """Refuse text that exists to override a reading agent's instructions.

    Subclasses ValueError so the existing handlers map it to a 400 or a tool
    error without introducing a new error category.
    """
    for pattern in _OVERRIDE_PATTERNS:
        if pattern.search(text):
            raise InjectedInstructionError(
                "text contains instructions directed at the reading agent and was not stored"
            )


def duplicate_memory_id(memory: Any, *, user_id: str, scope_key: str, digest: str) -> str | None:
    """Id of an identical memory already stored in this scope, else None.

    Exact-content match only: the digest comes from the canonicalised text,
    so whitespace and Unicode form differences still count as duplicates
    while paraphrases deliberately do not.
    """
    found = memory.get_all(
        filters={"user_id": user_id, "scope_key": scope_key, CONTENT_HASH_KEY: digest},
        top_k=1,
    )
    entries = found.get("results", []) if isinstance(found, dict) else found
    if not entries:
        return None
    first = entries[0]
    return str(first.get("id", "")) if isinstance(first, dict) else ""
