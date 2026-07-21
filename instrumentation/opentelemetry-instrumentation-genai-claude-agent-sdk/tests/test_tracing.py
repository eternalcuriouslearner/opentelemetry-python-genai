# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any

import pytest

from opentelemetry.instrumentation.genai.claude_agent_sdk.patch import (
    client_query,
    client_receive_response,
    query,
)
from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAI,
)
from opentelemetry.trace import SpanKind, StatusCode, get_current_span
from opentelemetry.util.genai.environment_variables import (
    OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT,
)
from opentelemetry.util.genai.handler import TelemetryHandler


def _handler(tracer_provider, logger_provider, meter_provider):
    return TelemetryHandler(
        tracer_provider=tracer_provider,
        logger_provider=logger_provider,
        meter_provider=meter_provider,
    )


def _root_messages(result: str = "ok") -> list[dict[str, Any]]:
    return [
        {
            "type": "system",
            "subtype": "init",
            "session_id": "session-root",
            "model": "claude-test",
        },
        {
            "type": "assistant",
            "message": {
                "content": [{"type": "text", "text": result}],
                "stop_reason": "end_turn",
            },
            "model": "claude-test",
        },
        {
            "type": "result",
            "subtype": "success",
            "is_error": False,
            "session_id": "session-root",
            "result": result,
            "usage": {
                "input_tokens": 3,
                "output_tokens": 1,
                "cache_creation_input_tokens": 2,
                "cache_read_input_tokens": 4,
            },
        },
    ]


def _subagent_tool_messages() -> list[dict[str, Any]]:
    return [
        {
            "type": "system",
            "subtype": "init",
            "session_id": "session-root",
            "model": "claude-test",
        },
        {
            "type": "assistant",
            "message": {
                "content": [
                    {
                        "type": "tool_use",
                        "id": "agent-1",
                        "name": "Agent",
                        "input": {
                            "subagent_type": "general-purpose",
                            "description": "Run a command",
                            "prompt": "Count the file",
                        },
                    }
                ]
            },
            "model": "claude-test",
        },
        {
            "type": "system",
            "subtype": "task_started",
            "tool_use_id": "agent-1",
            "description": "Run a command",
        },
        {
            "type": "assistant",
            "parent_tool_use_id": "agent-1",
            "message": {
                "content": [
                    {
                        "type": "tool_use",
                        "id": "bash-1",
                        "name": "Bash",
                        "input": {"command": "wc -c pyproject.toml"},
                    }
                ]
            },
            "model": "claude-test",
        },
        {
            "type": "user",
            "parent_tool_use_id": "agent-1",
            "message": {
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "bash-1",
                        "content": "42 pyproject.toml",
                        "is_error": False,
                    }
                ]
            },
        },
        {
            "type": "user",
            "message": {
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "agent-1",
                        "content": "42 pyproject.toml",
                        "is_error": False,
                    }
                ]
            },
            "tool_use_result": {
                "usage": {"input_tokens": 2, "output_tokens": 3}
            },
        },
        {
            "type": "result",
            "subtype": "success",
            "is_error": False,
            "session_id": "session-root",
            "result": "42 pyproject.toml",
            "usage": {"input_tokens": 5, "output_tokens": 6},
        },
    ]


async def _messages(values):
    for value in values:
        yield value


async def _collect(stream):
    return [message async for message in stream]


def test_one_shot_query_emits_invoke_agent_span(
    monkeypatch,
    tracer_provider,
    logger_provider,
    meter_provider,
    span_exporter,
):
    monkeypatch.setenv(
        OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT, "SPAN_ONLY"
    )
    handler = _handler(tracer_provider, logger_provider, meter_provider)

    def fake_query(*args, **kwargs):
        return _messages(_root_messages())

    async def exercise():
        stream = query(handler)(
            fake_query,
            None,
            (),
            {"prompt": "Reply with ok", "options": None},
        )
        assert await _collect(stream) == _root_messages()

    asyncio.run(exercise())

    (span,) = span_exporter.get_finished_spans()
    assert span.name == "invoke_agent Claude"
    assert span.kind is SpanKind.CLIENT
    assert span.status.status_code is StatusCode.UNSET
    attributes = dict(span.attributes or {})
    assert attributes[GenAI.GEN_AI_OPERATION_NAME] == "invoke_agent"
    assert attributes[GenAI.GEN_AI_PROVIDER_NAME] == "anthropic"
    assert attributes[GenAI.GEN_AI_AGENT_NAME] == "Claude"
    assert attributes[GenAI.GEN_AI_REQUEST_MODEL] == "claude-test"
    assert attributes[GenAI.GEN_AI_CONVERSATION_ID] == "session-root"
    assert attributes[GenAI.GEN_AI_USAGE_INPUT_TOKENS] == 3
    assert attributes[GenAI.GEN_AI_USAGE_OUTPUT_TOKENS] == 1
    assert "Reply with ok" in attributes[GenAI.GEN_AI_INPUT_MESSAGES]
    assert "ok" in attributes[GenAI.GEN_AI_OUTPUT_MESSAGES]


def test_query_reconstructs_subagent_and_tool_lineage(
    monkeypatch,
    tracer_provider,
    logger_provider,
    meter_provider,
    span_exporter,
):
    monkeypatch.setenv(
        OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT, "SPAN_ONLY"
    )
    handler = _handler(tracer_provider, logger_provider, meter_provider)

    def fake_query(*args, **kwargs):
        return _messages(_subagent_tool_messages())

    async def exercise():
        stream = query(handler)(
            fake_query,
            None,
            (),
            {"prompt": "Delegate this", "options": None},
        )
        await _collect(stream)

    asyncio.run(exercise())

    spans = {span.name: span for span in span_exporter.get_finished_spans()}
    assert set(spans) == {
        "invoke_agent Claude",
        "invoke_agent general-purpose",
        "execute_tool Bash",
    }
    root = spans["invoke_agent Claude"]
    subagent = spans["invoke_agent general-purpose"]
    tool = spans["execute_tool Bash"]
    assert subagent.kind is SpanKind.INTERNAL
    assert subagent.parent is not None
    assert subagent.parent.span_id == root.context.span_id
    assert tool.parent is not None
    assert tool.parent.span_id == subagent.context.span_id
    assert not any(
        span.name == "execute_tool Agent" for span in spans.values()
    )

    tool_attributes = dict(tool.attributes or {})
    assert tool_attributes[GenAI.GEN_AI_TOOL_NAME] == "Bash"
    assert tool_attributes[GenAI.GEN_AI_TOOL_CALL_ID] == "bash-1"
    assert "wc -c" in tool_attributes[GenAI.GEN_AI_TOOL_CALL_ARGUMENTS]
    assert (
        "42 pyproject.toml" in tool_attributes[GenAI.GEN_AI_TOOL_CALL_RESULT]
    )


def test_persistent_client_creates_one_span_per_serialized_turn(
    tracer_provider,
    logger_provider,
    meter_provider,
    span_exporter,
):
    handler = _handler(tracer_provider, logger_provider, meter_provider)
    client = SimpleNamespace(options=SimpleNamespace(model="claude-test"))
    send_prompts = []

    async def send(prompt):
        send_prompts.append(prompt)

    def receive():
        return _messages(_root_messages(result=send_prompts[-1]))

    async def exercise():
        traced_send = client_query(handler)
        traced_receive = client_receive_response()
        for prompt in ("first", "second"):
            await traced_send(send, client, (prompt,), {})
            stream = traced_receive(receive, client, (), {})
            await _collect(stream)

    asyncio.run(exercise())

    spans = span_exporter.get_finished_spans()
    assert [span.name for span in spans] == [
        "invoke_agent Claude",
        "invoke_agent Claude",
    ]
    assert spans[0].context.trace_id != spans[1].context.trace_id
    assert getattr(client, "_otel_genai_turn_state") is None


def test_query_stream_exception_is_reraised_unchanged_and_fails_span(
    tracer_provider,
    logger_provider,
    meter_provider,
    span_exporter,
):
    handler = _handler(tracer_provider, logger_provider, meter_provider)
    expected = RuntimeError("sdk failed")

    async def failing_messages():
        raise expected
        yield  # pragma: no cover

    def fake_query(*args, **kwargs):
        return failing_messages()

    async def exercise():
        stream = query(handler)(
            fake_query, None, (), {"prompt": "hello", "options": None}
        )
        with pytest.raises(RuntimeError) as caught:
            await anext(stream)
        assert caught.value is expected

    asyncio.run(exercise())

    (span,) = span_exporter.get_finished_spans()
    assert span.status.status_code is StatusCode.ERROR


def test_invoke_agent_is_current_when_one_shot_sdk_stream_starts(
    tracer_provider,
    logger_provider,
    meter_provider,
    span_exporter,
):
    handler = _handler(tracer_provider, logger_provider, meter_provider)
    current_span_ids = []

    async def sdk_messages():
        current_span_ids.append(get_current_span().get_span_context().span_id)
        for message in _root_messages():
            yield message

    def fake_query(*args, **kwargs):
        return sdk_messages()

    async def exercise():
        stream = query(handler)(
            fake_query, None, (), {"prompt": "hello", "options": None}
        )
        await _collect(stream)

    asyncio.run(exercise())

    (span,) = span_exporter.get_finished_spans()
    assert current_span_ids == [span.context.span_id]
