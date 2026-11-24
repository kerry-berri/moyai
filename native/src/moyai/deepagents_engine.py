"""Deep Agents as an alternative agent loop for ordinary turns.

Selected with MOYAI_AGENT_ENGINE=deepagents. The graph runs on a trusted worker
inside one Temporal activity: model calls go to the LiteLLM gateway (Responses
API) with the worker's key, and file/shell tools run in the turn's computer.
The stored conversation stays in the native SDK format, so the web and Slack
surfaces, history and a later switch back to the default engine keep working.
"""

from __future__ import annotations

import asyncio
from contextlib import suppress
from dataclasses import asdict
from typing import TYPE_CHECKING, Any

from deepagents import create_deep_agent
from deepagents.backends.protocol import ExecuteResponse, FileDownloadResponse, FileUploadResponse
from deepagents.backends.sandbox import BaseSandbox
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_openai import ChatOpenAI
from pydantic import TypeAdapter
from pydantic_ai import Conversation, ConversationTypeAdapter
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.usage import RunUsage
from pydantic_ai.workspaces import Workspace, WorkspaceRef, WorkspaceTimeoutError
from temporalio import activity
from temporalio.exceptions import ApplicationError

from .contracts import CompletedTurn, PreparedTurn
from .inference_usage import INFERENCE_ACTIVITY, InferenceActivity

if TYPE_CHECKING:
    from .runtime import Runtime
    from .settings import Settings

ENGINE = "deepagents"
DEFAULT_COMMAND_TIMEOUT = 120
INSTRUCTIONS = (
    "You are Moyai, a coding teammate. Take the user's task through implementation, verification "
    "and a clear handoff. Your file and shell tools act on the session's isolated computer; "
    "repository work belongs under /workspace. Inspect repository instructions before editing, "
    "preserve unrelated work, and say what you verified and what remains uncertain."
)


def engine_for(settings: Settings, prepared: PreparedTurn) -> str:
    """Only ordinary turns change loop; reviews, swarms and saved answers keep the default one."""
    scope = prepared.scope
    if (
        settings.agent_engine != ENGINE or scope.attention or scope.responsibility_id is not None
        or scope.swarm_id or scope.delegation_depth or prepared.deferred_results is not None
    ):
        return "pydantic"
    return ENGINE


class WorkspaceSandbox(BaseSandbox):
    """The turn's computer as a Deep Agents sandbox backend.

    Deep Agents calls the synchronous methods from worker threads, so each one
    is forwarded to the event loop that owns the workspace connection.
    """

    def __init__(self, workspace: Workspace, loop: asyncio.AbstractEventLoop):
        self.workspace, self.loop = workspace, loop

    def _wait(self, coroutine: Any) -> Any:
        return asyncio.run_coroutine_threadsafe(coroutine, self.loop).result()

    @property
    def id(self) -> str:
        ref = self.workspace.ref
        return f"{ref.provider}:{ref.id}" if ref else "pending"

    async def _execute(self, command: str, timeout: int | None) -> ExecuteResponse:
        try:
            result = await self.workspace.run(command, shell=True, timeout=timeout or DEFAULT_COMMAND_TIMEOUT)
        except WorkspaceTimeoutError as error:
            return ExecuteResponse(
                output=f"{error.stdout}{error.stderr}\nCommand timed out.".strip(), exit_code=124,
            )
        return ExecuteResponse(output=result.stdout + result.stderr, exit_code=result.exit_code)

    def execute(self, command: str, *, timeout: int | None = None) -> ExecuteResponse:
        return self._wait(self._execute(command, timeout))

    async def _upload(self, files: list[tuple[str, bytes]]) -> list[FileUploadResponse]:
        responses = []
        for path, content in files:
            try:
                await self.workspace.write_bytes(path, content)
                responses.append(FileUploadResponse(path=path, error=None))
            except IsADirectoryError:
                responses.append(FileUploadResponse(path=path, error="is_directory"))
            except PermissionError:
                responses.append(FileUploadResponse(path=path, error="permission_denied"))
        return responses

    def upload_files(self, files: list[tuple[str, bytes]]) -> list[FileUploadResponse]:
        return self._wait(self._upload(files))

    async def _download(self, paths: list[str]) -> list[FileDownloadResponse]:
        responses = []
        for path in paths:
            try:
                responses.append(FileDownloadResponse(path=path, content=await self.workspace.read_bytes(path)))
            except FileNotFoundError:
                responses.append(FileDownloadResponse(path=path, error="file_not_found"))
            except IsADirectoryError:
                responses.append(FileDownloadResponse(path=path, error="is_directory"))
            except PermissionError:
                responses.append(FileDownloadResponse(path=path, error="permission_denied"))
        return responses

    def download_files(self, paths: list[str]) -> list[FileDownloadResponse]:
        return self._wait(self._download(paths))


def _text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            block if isinstance(block, str) else block.get("text", "")
            for block in content
            if isinstance(block, str) or (isinstance(block, dict) and block.get("type") in {"text", "output_text"})
        )
    return "" if content is None else str(content)


def to_graph_messages(messages: list[ModelMessage]) -> list[BaseMessage]:
    """Saved native history as graph input. Calls without a saved result are dropped."""
    returned = {
        part.tool_call_id for message in messages if isinstance(message, ModelRequest)
        for part in message.parts if isinstance(part, ToolReturnPart)
    }
    result: list[BaseMessage] = []
    for message in messages:
        if isinstance(message, ModelRequest):
            for part in message.parts:
                if isinstance(part, UserPromptPart):
                    result.append(HumanMessage(_text(part.content)))
                elif isinstance(part, ToolReturnPart):
                    result.append(ToolMessage(
                        part.model_response_str(), tool_call_id=part.tool_call_id, name=part.tool_name,
                    ))
        elif isinstance(message, ModelResponse):
            text = "".join(part.content for part in message.parts if isinstance(part, TextPart))
            calls = [
                {"name": part.tool_name, "args": part.args_as_dict(), "id": part.tool_call_id, "type": "tool_call"}
                for part in message.parts
                if isinstance(part, ToolCallPart) and part.tool_call_id in returned
            ]
            if text or calls:
                result.append(AIMessage(text, tool_calls=calls))
    return result


def to_native_messages(messages: list[BaseMessage], model: str) -> list[ModelMessage]:
    result: list[ModelMessage] = []
    for message in messages:
        if isinstance(message, HumanMessage):
            result.append(ModelRequest(parts=[UserPromptPart(content=_text(message.content))]))
        elif isinstance(message, AIMessage):
            parts: list[Any] = []
            if text := _text(message.content):
                parts.append(TextPart(content=text))
            parts.extend(
                ToolCallPart(tool_name=call["name"], args=call["args"], tool_call_id=call["id"] or "")
                for call in message.tool_calls
            )
            if parts:
                result.append(ModelResponse(parts=parts, model_name=model))
        elif isinstance(message, ToolMessage):
            part = ToolReturnPart(
                tool_name=message.name or "tool", content=_text(message.content), tool_call_id=message.tool_call_id,
            )
            previous = result[-1] if result else None
            if isinstance(previous, ModelRequest) and all(
                isinstance(existing, ToolReturnPart) for existing in previous.parts
            ):
                result[-1] = ModelRequest(parts=[*previous.parts, part])
            else:
                result.append(ModelRequest(parts=[part]))
    return result


def graph_usage(messages: list[BaseMessage]) -> RunUsage:
    usage = RunUsage()
    for message in messages:
        if isinstance(message, AIMessage):
            usage.requests += 1
            metadata = message.usage_metadata or {}
            usage.input_tokens += metadata.get("input_tokens", 0)
            usage.output_tokens += metadata.get("output_tokens", 0)
            usage.tool_calls += len(message.tool_calls)
    return usage


def chat_model(runtime: Runtime, model: str) -> ChatOpenAI:
    """The worker's gateway client, so spend is recorded the same way as the default engine."""
    client: Any = getattr(runtime.models[model], "client")  # noqa: B009
    return ChatOpenAI(
        model=model, api_key=client.api_key, base_url=str(client.base_url),
        root_async_client=client, async_client=client.chat.completions,
        use_responses_api=True, store=False, use_previous_response_id=False, max_retries=0,
    )


def workspace_for(runtime: Runtime, reference: dict[str, Any] | None) -> Workspace:
    providers = runtime.workspace_capability
    if providers is None:
        from pydantic_ai_harness import ModalSandbox

        from .agent import workspace_image
        from .lambda_workspace import WorkspaceProviders

        settings = runtime.settings
        providers = WorkspaceProviders(settings, ModalSandbox(
            image=workspace_image(), app_name=settings.modal_app, working_dir="/workspace",
            sandbox_timeout=settings.computer_lifetime, idle_timeout=settings.computer_idle_timeout,
        ))
    ref = TypeAdapter(WorkspaceRef).validate_python(reference) if reference else None
    backend = providers.get_workspace(None, ref=ref)  # type: ignore[arg-type]
    if backend is None:
        raise ApplicationError("The saved computer has an unsupported provider.", non_retryable=True)
    return Workspace(backend)


async def run_turn(
    prepared: PreparedTurn, *, model: Any, workspace: Workspace, recursion_limit: int = 500,
) -> CompletedTurn:
    """Run one prompt to a final answer. No Temporal or database state is touched here."""
    scope = prepared.scope
    conversation = (
        ConversationTypeAdapter.validate_json(prepared.conversation)
        if prepared.conversation else Conversation(conversation_id=scope.conversation_id)
    )
    history = to_graph_messages(conversation.messages)
    graph = create_deep_agent(
        model=model, system_prompt=INSTRUCTIONS,
        backend=WorkspaceSandbox(workspace, asyncio.get_running_loop()),
    )
    graph_input: Any = {"messages": [*history, HumanMessage(prepared.prompt)]}
    state = await graph.ainvoke(graph_input, config={"recursion_limit": recursion_limit})
    produced = state["messages"][len(history):]
    answers = [message for message in produced if isinstance(message, AIMessage) and not message.tool_calls]
    if not answers:
        raise ApplicationError("Deep Agents ended without a final answer.", non_retryable=True)
    usage = graph_usage(produced)
    conversation.messages.extend(to_native_messages(produced, scope.model))
    conversation.usage = conversation.usage + usage
    ref = workspace.ref
    return CompletedTurn(
        scope=scope,
        conversation=ConversationTypeAdapter.dump_json(conversation).decode(),
        output=_text(answers[-1].content),
        workspace=TypeAdapter(WorkspaceRef).dump_python(ref, mode="json") if ref else None,
        usage=asdict(usage),
    )


@activity.defn(name="moyai_deepagents_turn_v1")
async def deepagents_turn(prepared: PreparedTurn) -> CompletedTurn:
    from .runtime import current

    async def heartbeat() -> None:
        while True:
            activity.heartbeat("Deep Agents is working.")
            await asyncio.sleep(10)

    runtime, scope, info = current(), prepared.scope, activity.info()
    if prepared.images:
        raise ApplicationError(
            "Image attachments are not supported by the Deep Agents engine.", non_retryable=True,
        )
    token = INFERENCE_ACTIVITY.set(InferenceActivity(
        runtime.database, scope,
        f"{info.workflow_namespace}/{info.workflow_id}/{info.workflow_run_id}/{info.activity_id}/{info.attempt}",
    ))
    heartbeats = asyncio.create_task(heartbeat())
    try:
        return await run_turn(
            prepared, model=chat_model(runtime, scope.model),
            workspace=workspace_for(runtime, prepared.workspace),
        )
    finally:
        INFERENCE_ACTIVITY.reset(token)
        heartbeats.cancel()
        with suppress(asyncio.CancelledError):
            await heartbeats


__all__ = ["ENGINE", "WorkspaceSandbox", "deepagents_turn", "engine_for", "run_turn"]
