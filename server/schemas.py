import uuid
from typing import Literal

from pydantic import BaseModel, Field


class MessageResponse(BaseModel):
    message: str


class MemoryImportBeginRequest(BaseModel):
    source_ref: str = Field(min_length=1)
    source_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    scope: Literal["global", "project"] = "project"
    project_id: str | None = None


class MemoryImportItem(BaseModel):
    text: str = Field(min_length=1)
    source_section: str | None = None


class MemoryImportAppendRequest(BaseModel):
    memories: list[MemoryImportItem] = Field(min_length=1)


class MemoryImportStatusResponse(BaseModel):
    state: str
    import_id: uuid.UUID
    stored_count: int
    duplicate_count: int
    failed_count: int
    errors: list[dict] = Field(default_factory=list)
