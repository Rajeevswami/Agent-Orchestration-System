"""Pydantic request/response schemas for the REST API."""

from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field


class ConversationCreate(BaseModel):
    title: str = "New conversation"


class ConversationOut(BaseModel):
    id: str
    title: str
    created_at: str

    class Config:
        from_attributes = True


class RunCreate(BaseModel):
    task: str = Field(..., min_length=1, description="The user's task for the agent team")
    conversation_id: Optional[str] = None
    mode: str = Field("sync", description="'sync' executes before returning; 'async' queues/starts in background")
    metadata: dict[str, Any] = Field(default_factory=dict)


class EscalationDecision(BaseModel):
    decision: str = Field(..., description="approve | reject | modify")
    comment: str = ""
    modified_answer: Optional[str] = Field(
        None, description="Replacement/adjusted answer when decision=modify"
    )

    def as_resume_value(self) -> dict[str, Any]:
        value: dict[str, Any] = {"decision": self.decision, "comment": self.comment}
        if self.modified_answer:
            value["modified_answer"] = self.modified_answer
        return value


class MemorySearch(BaseModel):
    q: str = Field(..., min_length=1)
    k: int = Field(5, ge=1, le=50)


class RunOut(BaseModel):
    id: str
    conversation_id: str
    task: str
    status: str
    provider: str
    final_answer: Optional[str] = None
    error: Optional[str] = None
    created_at: str
    started_at: Optional[str] = None
    finished_at: Optional[str] = None
    output_data: dict[str, Any] = Field(default_factory=dict)


class MessageOut(BaseModel):
    id: int
    role: str
    name: Optional[str] = None
    content: str
    tool_calls: Optional[list] = None
    seq: int


class EventOut(BaseModel):
    id: int
    type: str
    actor: str
    payload: dict[str, Any]
    created_at: str


class EscalationOut(BaseModel):
    id: str
    run_id: str
    kind: str
    status: str
    reason: str
    question: str
    context: dict[str, Any]
    decision: Optional[dict[str, Any]] = None
    created_at: str
    resolved_at: Optional[str] = None
