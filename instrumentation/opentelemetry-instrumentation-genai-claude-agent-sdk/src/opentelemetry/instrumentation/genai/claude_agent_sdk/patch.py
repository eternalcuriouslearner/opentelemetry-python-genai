# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""Wrappers turning Claude Agent SDK runs into ``invoke_agent`` telemetry.

The Claude Agent SDK drives an agent loop through the bundled Claude Code
CLI, so there is no per-API-call surface to patch. Telemetry is emitted at
the agent level via ``opentelemetry-util-genai``: ``query()`` and each
``ClaudeSDKClient.receive_response()`` turn become ``invoke_agent`` spans
carrying the prompt, the assistant output messages, token usage, model,
and session id extracted from the streamed messages.

Messages that belong to a subagent run (``parent_tool_use_id`` set) are not
folded into the root span; dedicated subagent ``invoke_agent`` and
``execute_tool`` spans are a follow-up.
"""

from __future__ import annotations

from collections.abc import Mapping as MappingABC
from typing import Any, AsyncIterator, Awaitable, Callable, Mapping

from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAIAttributes,
)
from opentelemetry.util.genai.handler import TelemetryHandler
from opentelemetry.util.genai.invocation import AgentInvocation, Error
from opentelemetry.util.genai.types import (
    GenericPart,
    InputMessage,
    MessagePart,
    OutputMessage,
    Reasoning,
    Text,
    ToolCallRequest,
)

_PROVIDER = GenAIAttributes.GenAiProviderNameValues.ANTHROPIC.value

# Attribute set on the ClaudeSDKClient instance to link the prompt passed to
# connect()/query() with the span of the next receive_response() turn.
_LAST_PROMPT_ATTR = "_otel_genai_last_prompt"

# semconv error.type fallback value for failures that carry no exception.
_OTHER_ERROR = "_OTHER"

_error_types: dict[str, type[Exception]] = {}


def _error_type(name: str) -> type[Exception]:
    """Return an exception type whose qualname is the given error code.

    util-genai's ``Error`` only accepts an exception type, but result
    failures reported by the CLI are plain messages with a domain-specific
    error code (e.g. ``error_max_turns``). Synthesizing a type per code
    (bounded set) surfaces that code as ``error.type``.
    """
    etype = _error_types.get(name)
    if etype is None:
        etype = type(name, (Exception,), {})
        _error_types[name] = etype
    return etype


def _get_field(obj: Any, key: str, default: Any = None) -> Any:
    if isinstance(obj, MappingABC):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _safe_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _coerce_usage(usage: Any) -> Mapping[str, Any]:
    if isinstance(usage, MappingABC):
        return usage
    return {
        "input_tokens": getattr(usage, "input_tokens", None),
        "output_tokens": getattr(usage, "output_tokens", None),
        "cache_read_input_tokens": getattr(
            usage, "cache_read_input_tokens", None
        ),
        "cache_creation_input_tokens": getattr(
            usage, "cache_creation_input_tokens", None
        ),
        "cache_write_input_tokens": getattr(
            usage, "cache_write_input_tokens", None
        ),
    }


def _extract_model_name_from_usage(model_usage: Any) -> str | None:
    if isinstance(model_usage, MappingABC) and model_usage:
        # The CLI emits per-model usage as `{model_name: {outputTokens, ...}}`.
        # Multi-model runs (main model + a router / fast model) must surface
        # the model that did the bulk of the generation, not whichever key
        # dict iteration happens to yield first.
        def _output_tokens(name: Any) -> int:
            entry = model_usage[name]
            if isinstance(entry, MappingABC):
                value = (
                    entry.get("outputTokens") or entry.get("output_tokens") or 0
                )
                try:
                    return int(value)
                except (TypeError, ValueError):
                    return 0
            return 0

        best = max(model_usage.keys(), key=_output_tokens, default=None)
        if best is not None:
            return str(best)
    if isinstance(model_usage, (list, tuple)):
        for entry in model_usage:
            name = (
                _get_field(entry, "model")
                or _get_field(entry, "name")
                or _get_field(entry, "model_name")
            )
            if name:
                return str(name)
    if model_usage is not None:
        name = (
            _get_field(model_usage, "model")
            or _get_field(model_usage, "name")
            or _get_field(model_usage, "model_name")
        )
        if name:
            return str(name)
    return None


def _extract_model_name(msg: Any) -> str | None:
    raw_name = _get_field(msg, "model") or _get_field(msg, "model_name")
    if raw_name:
        return str(raw_name)
    model_usage = _get_field(msg, "modelUsage") or _get_field(
        msg, "model_usage"
    )
    usage_name = _extract_model_name_from_usage(model_usage)
    if usage_name:
        return usage_name
    usage = _get_field(msg, "usage")
    model_usage = _get_field(usage, "modelUsage") or _get_field(
        usage, "model_usage"
    )
    usage_name = _extract_model_name_from_usage(model_usage)
    if usage_name:
        return usage_name
    raw_name = _get_field(usage, "model") or _get_field(usage, "model_name")
    if raw_name:
        return str(raw_name)
    data = _get_field(msg, "data", {})
    raw_name = _get_field(data, "model") or _get_field(data, "model_name")
    if raw_name:
        return str(raw_name)
    inner = _get_field(msg, "message")
    if inner is not None:
        raw_name = _get_field(inner, "model")
        if raw_name:
            return str(raw_name)
    return None


def _extract_session_id(msg: Any) -> str | None:
    session_id = _get_field(msg, "session_id")
    if session_id is None:
        session_id = _get_field(_get_field(msg, "data", {}), "session_id")
    return str(session_id) if session_id else None


def _is_system_init_message(msg: Any) -> bool:
    if _get_field(msg, "subtype") != "init":
        return False
    if _get_field(msg, "type") == "system":
        return True
    return bool(_extract_session_id(msg) or _extract_model_name(msg))


def _is_result_success_message(msg: Any) -> bool:
    msg_type = _get_field(msg, "type")
    subtype = _get_field(msg, "subtype")
    is_error = bool(_get_field(msg, "is_error"))
    if msg_type == "result" and subtype == "success" and not is_error:
        return True
    return (
        subtype == "success"
        and _get_field(msg, "usage") is not None
        and not is_error
    )


def _is_result_error_message(msg: Any) -> bool:
    if _get_field(msg, "is_error") is True:
        return True
    msg_type = _get_field(msg, "type")
    subtype = _get_field(msg, "subtype")
    if (
        msg_type == "result"
        and isinstance(subtype, str)
        and subtype.startswith("error")
    ):
        return True
    return (
        isinstance(subtype, str)
        and subtype.startswith("error")
        and _get_field(msg, "usage") is not None
    )


def _normalize_finish_reason(stop_reason: Any) -> str | None:
    if not stop_reason:
        return None
    stop_reason = str(stop_reason)
    normalized = {
        "end_turn": "stop",
        "stop_sequence": "stop",
        "max_tokens": "length",
        "tool_use": "tool_calls",
    }.get(stop_reason)
    return normalized or stop_reason


def _extract_message_content(message: Any) -> list[Any] | None:
    content = getattr(message, "content", None)
    if isinstance(content, list):
        return content
    inner = _get_field(message, "message")
    content = _get_field(inner, "content")
    return content if isinstance(content, list) else None


def _is_tool_use_block(block: Any) -> bool:
    block_type = _get_field(block, "type")
    if block_type is not None:
        return str(block_type) == "tool_use"
    return (
        _get_field(block, "id") is not None
        and _get_field(block, "name") is not None
        and _get_field(block, "input") is not None
        and _get_field(block, "tool_use_id") is None
    )


def _is_tool_result_block(block: Any) -> bool:
    block_type = _get_field(block, "type")
    if block_type is not None:
        return str(block_type) == "tool_result"
    return _get_field(block, "tool_use_id") is not None


def _is_text_block(block: Any) -> bool:
    block_type = _get_field(block, "type")
    if block_type is not None:
        return str(block_type) == "text"
    return (
        _get_field(block, "text") is not None
        and _get_field(block, "id") is None
        and _get_field(block, "tool_use_id") is None
    )


def _is_thinking_block(block: Any) -> bool:
    block_type = _get_field(block, "type")
    if block_type is not None:
        return str(block_type) == "thinking"
    return _get_field(block, "thinking") is not None


def _is_assistant_message(msg: Any) -> bool:
    if _get_field(msg, "type") == "assistant":
        return True
    inner = _get_field(msg, "message")
    if inner is not None and _get_field(inner, "role") == "assistant":
        return True
    # SDK AssistantMessage dataclasses have content blocks and a model name
    # but no `type` discriminator.
    return (
        _extract_message_content(msg) is not None
        and _get_field(msg, "model") is not None
    )


def _output_message_from_assistant(msg: Any) -> OutputMessage | None:
    content = _extract_message_content(msg)
    if content is None:
        return None
    parts: list[MessagePart] = []
    for block in content:
        if _is_tool_use_block(block):
            tool_id = _get_field(block, "id")
            parts.append(
                ToolCallRequest(
                    arguments=_get_field(block, "input"),
                    name=str(_get_field(block, "name", "")),
                    id=str(tool_id) if tool_id is not None else None,
                )
            )
        elif _is_text_block(block):
            parts.append(Text(content=str(_get_field(block, "text", ""))))
        elif _is_thinking_block(block):
            parts.append(
                Reasoning(content=str(_get_field(block, "thinking", "")))
            )
        elif not _is_tool_result_block(block):
            parts.append(GenericPart(value=block))
    if not parts:
        return None
    finish_reason = _normalize_finish_reason(_get_field(msg, "stop_reason"))
    return OutputMessage(
        role="assistant", parts=parts, finish_reason=finish_reason or "stop"
    )


class _AgentRunState:
    """Accumulates message-derived state for one ``invoke_agent`` span."""

    def __init__(
        self, invocation: AgentInvocation, capture_content: bool
    ) -> None:
        self.invocation = invocation
        self._capture_content = capture_content
        self._output_messages: list[OutputMessage] = []
        self._finish_reasons: list[str] = []
        self._pending_error: Error | None = None
        self._result_text: str | None = None
        self._finished = False

    def process_message(self, msg: Any) -> None:
        # Messages that belong to a subagent run must not be folded into
        # the root span.
        parent_tool_use_id = _get_field(msg, "parent_tool_use_id")
        if parent_tool_use_id is not None and str(parent_tool_use_id):
            return

        invocation = self.invocation
        model = _extract_model_name(msg)
        if model and invocation.request_model is None:
            invocation.request_model = model
        session_id = _extract_session_id(msg)
        if session_id and invocation.conversation_id is None:
            invocation.conversation_id = session_id

        if _is_system_init_message(msg):
            return
        if _is_result_success_message(msg):
            self._apply_usage(msg)
            result = _get_field(msg, "result")
            if result is not None:
                self._result_text = str(result)
            return
        if _is_result_error_message(msg):
            self._apply_usage(msg)
            subtype = _get_field(msg, "subtype")
            errors = _get_field(msg, "errors")
            message = f"Result error: {subtype}"
            if errors:
                message = f"{message}: {errors}"
            error_code = (
                str(subtype)
                if isinstance(subtype, str) and subtype.startswith("error")
                else _OTHER_ERROR
            )
            self._pending_error = Error(
                message=message, type=_error_type(error_code)
            )
            return
        if _is_assistant_message(msg):
            finish_reason = _normalize_finish_reason(
                _get_field(msg, "stop_reason")
            )
            if finish_reason and finish_reason not in self._finish_reasons:
                self._finish_reasons.append(finish_reason)
            if self._capture_content:
                output_message = _output_message_from_assistant(msg)
                if output_message is not None:
                    self._output_messages.append(output_message)

    def _apply_usage(self, msg: Any) -> None:
        invocation = self.invocation
        usage = _coerce_usage(_get_field(msg, "usage"))
        input_tokens = _safe_int(usage.get("input_tokens"))
        output_tokens = _safe_int(usage.get("output_tokens"))
        cache_read = _safe_int(usage.get("cache_read_input_tokens"))
        cache_creation = _safe_int(
            usage.get("cache_creation_input_tokens")
            if usage.get("cache_creation_input_tokens") is not None
            else usage.get("cache_write_input_tokens")
        )
        if input_tokens is not None:
            invocation.input_tokens = input_tokens
        if output_tokens is not None:
            invocation.output_tokens = output_tokens
        if cache_read is not None:
            invocation.cache_read_input_tokens = cache_read
        if cache_creation is not None:
            invocation.cache_creation_input_tokens = cache_creation

    def finish(self, error: BaseException | None = None) -> None:
        if self._finished:
            return
        self._finished = True
        invocation = self.invocation
        if self._capture_content:
            if not self._output_messages and self._result_text is not None:
                self._output_messages.append(
                    OutputMessage(
                        role="assistant",
                        parts=[Text(content=self._result_text)],
                        finish_reason="stop",
                    )
                )
            invocation.output_messages = self._output_messages
        if self._finish_reasons:
            invocation.finish_reasons = self._finish_reasons
        if error is not None:
            invocation.fail(error)
        elif self._pending_error is not None:
            invocation.fail(self._pending_error)
        else:
            invocation.stop()


def _extract_prompt(
    args: tuple[Any, ...],
    kwargs: Mapping[str, Any],
) -> Any:
    prompt = kwargs.get("prompt") if kwargs else None
    if prompt is None and args:
        prompt = args[0]
    return prompt


def _prompt_input_messages(prompt: Any) -> list[InputMessage]:
    # Streaming-mode prompts are async iterables that cannot be consumed
    # here; only plain string prompts are captured.
    if isinstance(prompt, str):
        return [InputMessage(role="user", parts=[Text(content=prompt)])]
    return []


def _start_agent_invocation(
    handler: TelemetryHandler, capture_content: bool, prompt: Any
) -> AgentInvocation:
    invocation = handler.invoke_local_agent()
    invocation.provider = _PROVIDER
    if capture_content:
        invocation.input_messages = _prompt_input_messages(prompt)
    return invocation


def query_wrapper(
    handler: TelemetryHandler,
) -> Callable[..., AsyncIterator[Any]]:
    """Wrap ``claude_agent_sdk.query`` in an ``invoke_agent`` span."""
    capture_content = handler.should_capture_content()

    async def traced_query(
        wrapped: Callable[..., AsyncIterator[Any]],
        instance: Any,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> AsyncIterator[Any]:
        del instance
        prompt = _extract_prompt(args, kwargs)
        invocation = _start_agent_invocation(handler, capture_content, prompt)
        state = _AgentRunState(invocation, capture_content)
        error: BaseException | None = None
        try:
            async for message in wrapped(*args, **kwargs):
                state.process_message(message)
                yield message
        except Exception as exc:
            error = exc
            raise
        finally:
            state.finish(error)

    return traced_query


def client_connect_wrapper(
    handler: TelemetryHandler,
) -> Callable[..., Awaitable[Any]]:
    """Record the prompt passed to connect() for the first response turn."""
    del handler

    async def traced_connect(
        wrapped: Callable[..., Awaitable[Any]],
        instance: Any,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> Any:
        prompt = _extract_prompt(args, kwargs)
        if prompt is not None:
            setattr(instance, _LAST_PROMPT_ATTR, prompt)
        return await wrapped(*args, **kwargs)

    return traced_connect


def client_query_wrapper(
    handler: TelemetryHandler,
) -> Callable[..., Awaitable[Any]]:
    """Record the prompt passed to query() for the next response turn."""
    del handler

    async def traced_client_query(
        wrapped: Callable[..., Awaitable[Any]],
        instance: Any,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> Any:
        setattr(instance, _LAST_PROMPT_ATTR, _extract_prompt(args, kwargs))
        return await wrapped(*args, **kwargs)

    return traced_client_query


def client_receive_response_wrapper(
    handler: TelemetryHandler,
) -> Callable[..., AsyncIterator[Any]]:
    """Wrap each ``receive_response()`` turn in an ``invoke_agent`` span."""
    capture_content = handler.should_capture_content()

    async def traced_receive_response(
        wrapped: Callable[..., AsyncIterator[Any]],
        instance: Any,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> AsyncIterator[Any]:
        prompt = getattr(instance, _LAST_PROMPT_ATTR, None)
        invocation = _start_agent_invocation(handler, capture_content, prompt)
        state = _AgentRunState(invocation, capture_content)
        error: BaseException | None = None
        try:
            async for message in wrapped(*args, **kwargs):
                state.process_message(message)
                yield message
        except Exception as exc:
            error = exc
            raise
        finally:
            state.finish(error)

    return traced_receive_response
