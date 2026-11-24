"""Deep Agents engine: real graph and real local computer, scripted model."""

from __future__ import annotations

import asyncio
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic_ai import ConversationTypeAdapter
from pydantic_ai.durable_exec.temporal import PydanticAIPlugin
from pydantic_ai.messages import ModelRequest, ModelResponse, TextPart, ToolCallPart, ToolReturnPart
from pydantic_ai.workspaces import LocalWorkspaceBackend, Workspace
from temporalio.converter import DataConverter
from temporalio.testing import WorkflowEnvironment

from moyai import deepagents_engine
from moyai.contracts import PreparedTurn, RunScope
from moyai.deepagents_engine import engine_for, run_turn, to_graph_messages
from moyai.models import Turn
from moyai.settings import Settings
from moyai.storage import EncryptedPayloadCodec
from moyai.worker import make_worker
from moyai.workflow import TurnWorkflow


class ScriptedModel(BaseChatModel):
    script: list[AIMessage]
    seen: list[list[Any]]

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def bind_tools(self, tools: Any, **kwargs: Any) -> ScriptedModel:
        return self

    def _generate(self, messages: list[Any], stop: Any = None, run_manager: Any = None, **kwargs: Any) -> ChatResult:
        self.seen.append(list(messages))
        reply = self.script[len(self.seen) - 1]
        return ChatResult(generations=[ChatGeneration(message=reply)])


def call(name: str, args: dict[str, Any], call_id: str) -> AIMessage:
    return AIMessage(
        "", tool_calls=[{"name": name, "args": args, "id": call_id, "type": "tool_call"}],
        usage_metadata={"input_tokens": 10, "output_tokens": 2, "total_tokens": 12},
    )


def coding_script(directory: Any) -> list[AIMessage]:
    return [
        call("write_file", {"file_path": f"{directory}/note.txt", "content": "from deep agents"}, "call-1"),
        call("execute", {"command": "cat note.txt && echo ' ran'"}, "call-2"),
        AIMessage("The file is written.", usage_metadata={"input_tokens": 30, "output_tokens": 5, "total_tokens": 35}),
    ]


def prepared_turn(**changes: Any) -> PreparedTurn:
    scope = RunScope(session_id="session", turn_id="turn", actor_id="actor", model="test")
    return PreparedTurn(scope=scope, prompt="Write the note.", **changes)


async def test_graph_tools_run_in_the_turn_computer_and_history_stays_native(tmp_path):
    model = ScriptedModel(script=coding_script(tmp_path), seen=[])
    completed = await run_turn(
        prepared_turn(), model=model, workspace=Workspace(LocalWorkspaceBackend(tmp_path)),
    )
    assert completed.output == "The file is written."
    assert "from deep agents" in (tmp_path / "note.txt").read_text()
    results = [message for message in model.seen[-1] if isinstance(message, ToolMessage)]
    assert "from deep agents" in results[-1].content and "ran" in results[-1].content
    assert completed.usage["requests"] == 3 and completed.usage["input_tokens"] == 50
    saved = ConversationTypeAdapter.validate_json(completed.conversation).messages
    assert [type(message) for message in saved] == [
        ModelRequest, ModelResponse, ModelRequest, ModelResponse, ModelRequest, ModelResponse,
    ]
    assert isinstance(saved[1].parts[0], ToolCallPart) and saved[1].parts[0].tool_name == "write_file"
    assert isinstance(saved[2].parts[0], ToolReturnPart)
    assert saved[-1].parts == [TextPart("The file is written.")] or saved[-1].parts[0].content == "The file is written."

    follow_up = ScriptedModel(script=[AIMessage("It says: from deep agents")], seen=[])
    again = await run_turn(
        PreparedTurn(scope=completed.scope, prompt="What did it say?", conversation=completed.conversation),
        model=follow_up, workspace=Workspace(LocalWorkspaceBackend(tmp_path)),
    )
    assert again.output == "It says: from deep agents"
    sent = follow_up.seen[0]
    assert [message.type for message in sent[1:]] == ["human", "ai", "tool", "ai", "tool", "ai", "human"]


def test_unanswered_tool_calls_are_not_sent_back_to_the_graph():
    history = [
        ModelResponse(parts=[TextPart("Working."), ToolCallPart("execute", {"command": "ls"}, "lost")]),
    ]
    (message,) = to_graph_messages(history)
    assert message.content == "Working." and message.tool_calls == []


def test_only_ordinary_turns_change_engine():
    settings = Settings(_env_file=None, agent_engine="deepagents")
    assert engine_for(settings, prepared_turn()) == "deepagents"
    assert engine_for(Settings(_env_file=None), prepared_turn()) == "pydantic"
    delegated = prepared_turn()
    delegated.scope.delegation_depth = 1
    assert engine_for(settings, delegated) == "pydantic"
    attention = prepared_turn()
    attention.scope.attention = True
    assert engine_for(settings, attention) == "pydantic"


async def test_deepagents_turn_completes_in_real_temporal_server(runtime, active, monkeypatch, tmp_path):
    _session, turn, _prepared = active
    runtime.settings.agent_engine = "deepagents"
    model = ScriptedModel(script=coding_script(tmp_path / "computer"), seen=[])
    monkeypatch.setattr(deepagents_engine, "chat_model", lambda _runtime, _model: model)
    environment = await WorkflowEnvironment.start_local(
        plugins=[PydanticAIPlugin()],
        data_converter=DataConverter(payload_codec=EncryptedPayloadCodec(runtime.security, runtime.objects)),
    )
    async with environment:
        runtime.client = environment.client
        async with make_worker(runtime):
            result = await asyncio.wait_for(runtime.client.execute_workflow(
                TurnWorkflow.run, turn.id, id=turn.workflow_id, task_queue=runtime.settings.temporal_queue,
            ), timeout=60)
    assert result["status"] == "completed"
    async with runtime.database.sessions() as db:
        saved = await db.get(Turn, turn.id)
        assert saved.status == "completed"
        assert runtime.security.unseal(saved.result).decode() == "The file is written."
    assert (tmp_path / "computer" / "note.txt").read_text().strip() == "from deep agents"
    assert len(model.seen) == 3
