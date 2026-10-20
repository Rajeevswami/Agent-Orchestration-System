"""Deterministic scripted chat models for tests and offline demos.

A :class:`ScriptedChatModel` replays a script of assistant turns — plain text
or tool calls — so the whole orchestration graph (supervisor routing, tool
loops, escalation interrupts) can be exercised without any API key.
"""

from __future__ import annotations

import itertools
from typing import Any, Callable, List, Optional, Sequence, Union

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import ConfigDict

ScriptItem = Union[str, dict, AIMessage]

_id_counter = itertools.count(1)


def _coerce(item: ScriptItem) -> AIMessage:
    if isinstance(item, AIMessage):
        return item
    if isinstance(item, str):
        return AIMessage(content=item)
    if isinstance(item, dict):
        tool_calls = item.get("tool_calls")
        if tool_calls:
            calls = []
            for tc in tool_calls:
                calls.append(
                    {
                        "name": tc["name"],
                        "args": tc.get("args", {}),
                        "id": tc.get("id", f"call_{next(_id_counter)}"),
                        "type": "tool_call",
                    }
                )
            return AIMessage(content=item.get("content", ""), tool_calls=calls)
        return AIMessage(content=item.get("content", ""))
    raise TypeError(f"Unsupported script item: {item!r}")


class ScriptedChatModel(BaseChatModel):
    """Replays canned responses; records every prompt it is shown.

    Parameters
    ----------
    script:
        Sequence of items (``str`` for plain replies, ``dict`` with
        ``tool_calls`` for tool use).
    fallback:
        Optional callable ``(messages) -> str`` used when the script runs out.
    name:
        Label used in ``_llm_type``.
    """

    # Raw script items are kept unvalidated (plain dicts are coerced at call
    # time by ``_coerce``), otherwise pydantic eagerly builds AIMessages.
    script: Sequence[Any]
    fallback: Optional[Callable[[Sequence[BaseMessage]], str]] = None
    label: str = "scripted"
    calls: List[List[BaseMessage]] = []

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @property
    def _llm_type(self) -> str:
        return self.label

    @property
    def supports_tool_calling(self) -> bool:
        return True

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        self.calls.append(list(messages))
        index = len(self.calls) - 1
        if index < len(self.script):
            message = _coerce(self.script[index])
        else:
            text = (
                self.fallback(messages)
                if self.fallback
                else "Scripted model exhausted — final response."
            )
            message = AIMessage(content=text)
        return ChatResult(generations=[ChatGeneration(message=message)])


class ScriptedRegistry:
    """Role -> model registry consulted by the provider router."""

    def __init__(self) -> None:
        self._models: dict[str, ScriptedChatModel] = {}

    def register(self, role: str, model: ScriptedChatModel) -> None:
        self._models[role] = model

    def get(self, role: str) -> ScriptedChatModel:
        if role not in self._models:
            # Unregistered roles get a role-aware fallback so scripted mode can
            # still orchestrate arbitrary tasks end-to-end (used by the console).
            self._models[role] = ScriptedChatModel(
                script=[], fallback=smart_fallback(role), label=f"scripted-{role}"
            )
        return self._models[role]

    def clear(self) -> None:
        self._models.clear()


def smart_fallback(role: str) -> Callable[[Sequence[BaseMessage]], str]:
    """Role-aware deterministic replies for unscripted turns.

    The supervisor fallback implements real routing: it looks at which workers
    have already reported (``[name]`` markers in the outputs section) and
    delegates to the next one, so scripted mode orchestrates any task.
    """

    def supervisor(messages: Sequence[BaseMessage]) -> str:
        import json as _json
        import re as _re

        text = str(messages[-1].content if messages else "")
        done = set(_re.findall(r"^\[([a-z]+)\]", text, _re.M))
        if "researcher" not in done:
            action, instructions = "researcher", "Gather the key facts and context for this task."
        elif "analyst" not in done:
            action, instructions = "analyst", "Analyze the gathered facts; quantify the important points."
        elif "writer" not in done:
            action, instructions = "writer", "Compose the final answer from the gathered material."
        else:
            action, instructions = "FINISH", ""
        return _json.dumps(
            {
                "action": action,
                "reasoning": f"Scripted routing: next step is {action}.",
                "instructions": instructions,
                "confidence": 0.85,
            }
        )

    def worker(messages: Sequence[BaseMessage]) -> str:
        text = str(messages[-1].content if messages else "")
        instruction = ""
        if "Supervisor instructions for this turn" in text:
            instruction = text.split("Supervisor instructions for this turn", 1)[1]
            instruction = instruction.split("\n", 2)[1].strip() if "\n" in instruction else instruction[:120]
        return (
            f"({role}) Completed the assigned step in scripted mode. "
            f"Focus was: {instruction or 'the supervisor instructions'}.\n\n"
            f"SUMMARY: {role} finished the requested step; findings recorded for the supervisor."
        )

    return supervisor if role == "supervisor" else worker


REGISTRY = ScriptedRegistry()


def register_script(role: str, script: Sequence[ScriptItem], **kwargs: Any) -> ScriptedChatModel:
    model = ScriptedChatModel(script=list(script), **kwargs)
    REGISTRY.register(role, model)
    return model
