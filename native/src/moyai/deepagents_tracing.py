"""Model and tool spans for Deep Agents turns, on the same provider as the default loop."""

from __future__ import annotations

import json
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any
from uuid import UUID

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import AIMessage, BaseMessage, ToolMessage
from opentelemetry import context as otel_context
from opentelemetry.trace import Span, SpanKind, Status, StatusCode, set_span_in_context

from .lifecycle_tracing import scope_attributes
from .presentation import redact

if TYPE_CHECKING:
    from .contracts import RunScope
    from .tracing import LensTracing

ROLES = {"human": "user", "ai": "assistant", "system": "system", "tool": "tool"}


def _text(content: Any) -> str:
    if isinstance(content, str):
        return content
    return "".join(
        block if isinstance(block, str) else str(block.get("text", ""))
        for block in content if isinstance(block, (str, dict))
    )


class TurnTrace:
    """One `invoke_agent` span per turn, with a `chat` span per model call and an
    `execute_tool` span per tool call beneath it.

    Prompt and answer text follow LITELLM_TRACE_INCLUDE_CONTENT. Tool arguments
    and results are never exported, matching the default loop.
    """

    def __init__(self, tracing: LensTracing, scope: RunScope):
        self.tracer = tracing.instrumentation.tracer
        self.include_content, self.secrets = tracing.include_content, tracing.secrets
        self.scope, self.base = scope, {**scope_attributes(scope), "moyai.engine": "deepagents"}
        self.parent: otel_context.Context | None = None
        self.callback = _ToolSpans(self)

    @contextmanager
    def agent(self) -> Iterator[Span]:
        with self.tracer.start_as_current_span("invoke_agent moyai", attributes={
            **self.base, "gen_ai.operation.name": "invoke_agent", "gen_ai.request.model": self.scope.model,
        }) as span:
            self.parent = set_span_in_context(span)
            yield span

    def messages(self, messages: list[BaseMessage]) -> str:
        """OpenTelemetry GenAI message parts; text only when content tracing is on."""
        rendered = []
        for message in messages:
            parts: list[dict[str, Any]] = []
            if isinstance(message, ToolMessage):
                parts.append({"type": "tool_call_response", "id": message.tool_call_id, "name": message.name})
            else:
                text = _text(message.content)
                if text:
                    part = {"type": "text"}
                    if self.include_content:
                        part["content"] = str(redact(text, self.secrets))
                    parts.append(part)
                if isinstance(message, AIMessage):
                    parts.extend(
                        {"type": "tool_call", "id": call["id"], "name": call["name"]} for call in message.tool_calls
                    )
            rendered.append({"role": ROLES.get(message.type, message.type), "parts": parts})
        return json.dumps(rendered)

    @contextmanager
    def chat(self, messages: list[BaseMessage]) -> Iterator[Callable[[AIMessage], None]]:
        """Current while the request is in flight, so the gateway's spans join this trace."""
        model = self.scope.model
        with self.tracer.start_as_current_span(f"chat {model}", context=self.parent, kind=SpanKind.CLIENT, attributes={
            **self.base, "gen_ai.operation.name": "chat", "gen_ai.provider.name": "litellm",
            "gen_ai.request.model": model, "gen_ai.input.messages": self.messages(messages),
        }) as span:
            def record(reply: AIMessage) -> None:
                usage: Any = reply.usage_metadata or {}
                details = usage.get("input_token_details") or {}
                span.set_attributes({
                    "gen_ai.response.model": str(reply.response_metadata.get("model_name") or model),
                    "gen_ai.usage.input_tokens": int(usage.get("input_tokens", 0)),
                    "gen_ai.usage.output_tokens": int(usage.get("output_tokens", 0)),
                    "gen_ai.usage.cache_read.input_tokens": int(details.get("cache_read") or 0),
                    "gen_ai.response.finish_reasons": ["tool_call" if reply.tool_calls else "stop"],
                    "gen_ai.output.messages": self.messages([reply]),
                })
                # Lens prices a call from the gateway's spend log, which is keyed by the
                # gateway call ID; the Responses ID it returns is not in that log.
                headers: Any = reply.response_metadata.get("headers") or {}
                if call_id := headers.get("x-litellm-call-id"):
                    span.set_attribute("litellm.call_id", str(call_id))
                elif reply.id:
                    span.set_attribute("gen_ai.response.id", reply.id)
            yield record


class _ToolSpans(BaseCallbackHandler):
    run_inline = True

    def __init__(self, trace: TurnTrace):
        self.trace, self.spans = trace, dict[UUID, Span]()

    def on_tool_start(self, serialized: dict[str, Any] | None, input_str: str, *, run_id: UUID, **kwargs: Any) -> None:
        name = str((serialized or {}).get("name") or kwargs.get("name") or "tool")
        attributes = {**self.trace.base, "gen_ai.operation.name": "execute_tool", "gen_ai.tool.name": name}
        if call_id := kwargs.get("tool_call_id"):
            attributes["gen_ai.tool.call.id"] = str(call_id)
        self.spans[run_id] = self.trace.tracer.start_span(
            f"execute_tool {name}", context=self.trace.parent, attributes=attributes,
        )

    def on_tool_end(self, output: Any, *, run_id: UUID, **kwargs: Any) -> None:
        if span := self.spans.pop(run_id, None):
            if getattr(output, "status", "") == "error":
                span.set_status(Status(StatusCode.ERROR))
            span.end()

    def on_tool_error(self, error: BaseException, *, run_id: UUID, **kwargs: Any) -> None:
        if span := self.spans.pop(run_id, None):
            span.set_attribute("error.type", type(error).__name__)
            span.set_status(Status(StatusCode.ERROR))
            span.end()
