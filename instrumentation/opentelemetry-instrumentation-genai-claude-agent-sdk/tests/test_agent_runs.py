# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""Span tests for query() and ClaudeSDKClient runs.

Integration tests replay recorded CLI sessions through the SDK's public
``Transport`` interface (see ``tests/conformance/cli_replay.py``); unit
tests drive the wrappers with an in-test fake ``query`` async generator.
"""

from __future__ import annotations

import importlib
import json
from typing import Any

import pytest

from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAIAttributes,
)
from opentelemetry.semconv.attributes import error_attributes
from opentelemetry.trace import StatusCode

from .conformance.cli_replay import replay_transport

TOOL_PROMPT = (
    "Use the Bash tool to run: wc -c 'pyproject.toml' and respond with "
    "exactly the output. Do not answer unless you executed the tool."
)


def _spans_by_operation(spans: Any) -> dict[str, list[Any]]:
    grouped: dict[str, list[Any]] = {}
    for span in spans:
        operation = (span.attributes or {}).get(
            GenAIAttributes.GEN_AI_OPERATION_NAME
        )
        grouped.setdefault(str(operation), []).append(span)
    return grouped


def _output_part_types(span: Any) -> list[str]:
    payload = (span.attributes or {}).get(GenAIAttributes.GEN_AI_OUTPUT_MESSAGES)
    if not payload:
        return []
    return [
        part["type"]
        for message in json.loads(payload)
        for part in message["parts"]
    ]


# ---- Integration tests: replayed CLI sessions ----


@pytest.mark.asyncio
async def test_query_agent_span_attributes(
    instrument_with_content, span_exporter
):
    from claude_agent_sdk import query  # noqa: PLC0415

    prompt = "Reply with exactly the word: ok"
    async for _ in query(
        prompt=prompt, transport=replay_transport("invoke_agent.yaml")
    ):
        pass

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]
    assert span.name == "invoke_agent"
    attrs = dict(span.attributes or {})
    assert (
        attrs[GenAIAttributes.GEN_AI_OPERATION_NAME]
        == GenAIAttributes.GenAiOperationNameValues.INVOKE_AGENT.value
    )
    assert (
        attrs[GenAIAttributes.GEN_AI_PROVIDER_NAME]
        == GenAIAttributes.GenAiProviderNameValues.ANTHROPIC.value
    )
    assert isinstance(attrs[GenAIAttributes.GEN_AI_CONVERSATION_ID], str)
    assert isinstance(attrs[GenAIAttributes.GEN_AI_REQUEST_MODEL], str)
    assert isinstance(attrs[GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS], int)
    assert isinstance(attrs[GenAIAttributes.GEN_AI_USAGE_OUTPUT_TOKENS], int)
    assert isinstance(
        attrs[GenAIAttributes.GEN_AI_USAGE_CACHE_READ_INPUT_TOKENS], int
    )
    assert isinstance(
        attrs[GenAIAttributes.GEN_AI_USAGE_CACHE_CREATION_INPUT_TOKENS], int
    )
    input_messages = json.loads(
        attrs[GenAIAttributes.GEN_AI_INPUT_MESSAGES]
    )
    assert input_messages == [
        {"role": "user", "parts": [{"content": prompt, "type": "text"}]}
    ]
    assert _output_part_types(span) == ["text"]


@pytest.mark.asyncio
async def test_query_tool_spans(instrument_with_content, span_exporter):
    from claude_agent_sdk import ClaudeAgentOptions, query  # noqa: PLC0415

    options = ClaudeAgentOptions(
        allowed_tools=["Bash"], permission_mode="bypassPermissions"
    )
    async for _ in query(
        prompt=TOOL_PROMPT,
        options=options,
        transport=replay_transport("tool_calling.yaml"),
    ):
        pass

    spans = span_exporter.get_finished_spans()
    grouped = _spans_by_operation(spans)
    (agent_span,) = grouped["invoke_agent"]
    (tool_span,) = grouped["execute_tool"]

    assert tool_span.name == "execute_tool Bash"
    assert tool_span.parent is not None
    assert tool_span.parent.span_id == agent_span.context.span_id
    attrs = dict(tool_span.attributes or {})
    assert attrs[GenAIAttributes.GEN_AI_TOOL_NAME] == "Bash"
    assert attrs[GenAIAttributes.GEN_AI_TOOL_TYPE] == "extension"
    assert isinstance(attrs[GenAIAttributes.GEN_AI_TOOL_CALL_ID], str)
    arguments = json.loads(attrs[GenAIAttributes.GEN_AI_TOOL_CALL_ARGUMENTS])
    assert "wc -c" in arguments["command"]
    assert isinstance(attrs[GenAIAttributes.GEN_AI_TOOL_CALL_RESULT], str)
    assert tool_span.status.status_code == StatusCode.UNSET

    assert "tool_call" in _output_part_types(agent_span)


@pytest.mark.asyncio
async def test_query_subagent_spans(instrument_with_content, span_exporter):
    from claude_agent_sdk import ClaudeAgentOptions, query  # noqa: PLC0415

    options = ClaudeAgentOptions(
        allowed_tools=["Bash", "Agent"], permission_mode="bypassPermissions"
    )
    async for _ in query(
        prompt=TOOL_PROMPT,
        options=options,
        transport=replay_transport("subagent.yaml"),
    ):
        pass

    spans = span_exporter.get_finished_spans()
    by_id = {span.context.span_id: span for span in spans}
    grouped = _spans_by_operation(spans)
    assert len(grouped["invoke_agent"]) == 2
    assert len(grouped["execute_tool"]) == 2

    root_span = next(
        span for span in grouped["invoke_agent"] if span.parent is None
    )
    subagent_span = next(
        span for span in grouped["invoke_agent"] if span.parent is not None
    )
    subagent_attrs = dict(subagent_span.attributes or {})
    assert (
        subagent_attrs[GenAIAttributes.GEN_AI_AGENT_NAME]
        == "general-purpose"
    )
    assert subagent_span.name == "invoke_agent general-purpose"

    # invoke_agent -> execute_tool Agent -> invoke_agent subagent -> Bash
    agent_tool_span = by_id[subagent_span.parent.span_id]
    assert (
        dict(agent_tool_span.attributes or {})[
            GenAIAttributes.GEN_AI_TOOL_NAME
        ]
        == "Agent"
    )
    assert agent_tool_span.parent.span_id == root_span.context.span_id

    bash_span = next(
        span
        for span in grouped["execute_tool"]
        if span is not agent_tool_span
    )
    assert bash_span.parent.span_id == subagent_span.context.span_id

    # The subagent span must close before its spawning tool span.
    assert subagent_span.end_time <= agent_tool_span.end_time


@pytest.mark.asyncio
async def test_client_receive_response_spans(
    instrument_with_content, span_exporter
):
    from claude_agent_sdk import (  # noqa: PLC0415
        ClaudeAgentOptions,
        ClaudeSDKClient,
    )

    options = ClaudeAgentOptions(
        allowed_tools=["Bash"], permission_mode="bypassPermissions"
    )
    async with ClaudeSDKClient(
        options=options, transport=replay_transport("reasoning.yaml")
    ) as client:
        await client.query(TOOL_PROMPT)
        async for _ in client.receive_response():
            pass

    spans = span_exporter.get_finished_spans()
    grouped = _spans_by_operation(spans)
    (agent_span,) = grouped["invoke_agent"]
    (tool_span,) = grouped["execute_tool"]

    assert tool_span.parent.span_id == agent_span.context.span_id
    attrs = dict(agent_span.attributes or {})
    input_messages = json.loads(attrs[GenAIAttributes.GEN_AI_INPUT_MESSAGES])
    assert input_messages == [
        {"role": "user", "parts": [{"content": TOOL_PROMPT, "type": "text"}]}
    ]
    assert _output_part_types(agent_span) == [
        "reasoning",
        "tool_call",
        "text",
    ]


@pytest.mark.asyncio
async def test_query_no_content_capture(instrument_no_content, span_exporter):
    from claude_agent_sdk import ClaudeAgentOptions, query  # noqa: PLC0415

    options = ClaudeAgentOptions(
        allowed_tools=["Bash"], permission_mode="bypassPermissions"
    )
    async for _ in query(
        prompt=TOOL_PROMPT,
        options=options,
        transport=replay_transport("tool_calling.yaml"),
    ):
        pass

    spans = span_exporter.get_finished_spans()
    grouped = _spans_by_operation(spans)
    (agent_span,) = grouped["invoke_agent"]
    (tool_span,) = grouped["execute_tool"]

    agent_attrs = dict(agent_span.attributes or {})
    assert GenAIAttributes.GEN_AI_INPUT_MESSAGES not in agent_attrs
    assert GenAIAttributes.GEN_AI_OUTPUT_MESSAGES not in agent_attrs
    # Non-content attributes are still recorded.
    assert isinstance(
        agent_attrs[GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS], int
    )

    tool_attrs = dict(tool_span.attributes or {})
    assert GenAIAttributes.GEN_AI_TOOL_CALL_ARGUMENTS not in tool_attrs
    assert GenAIAttributes.GEN_AI_TOOL_CALL_RESULT not in tool_attrs
    assert tool_attrs[GenAIAttributes.GEN_AI_TOOL_NAME] == "Bash"


@pytest.mark.asyncio
async def test_metrics_recorded(instrument_with_content, metric_reader):
    from claude_agent_sdk import ClaudeAgentOptions, query  # noqa: PLC0415

    options = ClaudeAgentOptions(
        allowed_tools=["Bash"], permission_mode="bypassPermissions"
    )
    async for _ in query(
        prompt=TOOL_PROMPT,
        options=options,
        transport=replay_transport("tool_calling.yaml"),
    ):
        pass

    metrics_data = metric_reader.get_metrics_data()
    metrics = {
        metric.name: metric
        for resource_metrics in metrics_data.resource_metrics
        for scope_metrics in resource_metrics.scope_metrics
        for metric in scope_metrics.metrics
    }
    duration_ops = {
        point.attributes[GenAIAttributes.GEN_AI_OPERATION_NAME]
        for point in metrics[
            "gen_ai.client.operation.duration"
        ].data.data_points
    }
    assert {"invoke_agent", "execute_tool"} <= duration_ops
    for point in metrics["gen_ai.client.operation.duration"].data.data_points:
        assert (
            point.attributes[GenAIAttributes.GEN_AI_PROVIDER_NAME]
            == "anthropic"
        )
    token_types = {
        point.attributes[GenAIAttributes.GEN_AI_TOKEN_TYPE]
        for point in metrics["gen_ai.client.token.usage"].data.data_points
    }
    assert {"input", "output"} <= token_types


# ---- Unit tests: fake message streams ----


class _FakeQuery:
    """Temporarily replaces claude_agent_sdk.query.query with a fake."""

    def __init__(self, fake: Any) -> None:
        self._fake = fake
        self._module = importlib.import_module("claude_agent_sdk.query")
        self._original = self._module.query

    def __enter__(self) -> None:
        setattr(self._module, "query", self._fake)

    def __exit__(self, *exc_info: Any) -> None:
        setattr(self._module, "query", self._original)


async def _drain(agen: Any) -> None:
    async for _ in agen:
        pass


@pytest.mark.asyncio
async def test_result_error_sets_error_status(
    tracer_provider, logger_provider, meter_provider, span_exporter
):
    from opentelemetry.instrumentation.genai.claude_agent_sdk import (  # noqa: PLC0415
        ClaudeAgentSDKInstrumentor,
    )

    async def fake_query(**kwargs: Any):
        yield {
            "type": "result",
            "subtype": "error_max_turns",
            "is_error": True,
            "usage": {"input_tokens": 3, "output_tokens": 1},
            "session_id": "sess-err",
        }

    with _FakeQuery(fake_query):
        instrumentor = ClaudeAgentSDKInstrumentor()
        instrumentor.instrument(
            tracer_provider=tracer_provider,
            logger_provider=logger_provider,
            meter_provider=meter_provider,
        )
        try:
            query_module = importlib.import_module("claude_agent_sdk.query")
            await _drain(query_module.query(prompt="hello"))
        finally:
            instrumentor.uninstrument()

    (span,) = span_exporter.get_finished_spans()
    assert span.status.status_code == StatusCode.ERROR
    attrs = dict(span.attributes or {})
    assert attrs[error_attributes.ERROR_TYPE] == "error_max_turns"
    assert attrs[GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS] == 3
    assert attrs[GenAIAttributes.GEN_AI_CONVERSATION_ID] == "sess-err"


@pytest.mark.asyncio
async def test_query_exception_reraised_and_recorded(
    tracer_provider, logger_provider, meter_provider, span_exporter
):
    from opentelemetry.instrumentation.genai.claude_agent_sdk import (  # noqa: PLC0415
        ClaudeAgentSDKInstrumentor,
    )

    async def failing_query(**kwargs: Any):
        raise RuntimeError("Simulated failure")
        yield  # pragma: no cover - makes this an async generator

    with _FakeQuery(failing_query):
        instrumentor = ClaudeAgentSDKInstrumentor()
        instrumentor.instrument(
            tracer_provider=tracer_provider,
            logger_provider=logger_provider,
            meter_provider=meter_provider,
        )
        try:
            query_module = importlib.import_module("claude_agent_sdk.query")
            with pytest.raises(RuntimeError, match="Simulated failure"):
                await _drain(query_module.query(prompt="hello"))
        finally:
            instrumentor.uninstrument()

    (span,) = span_exporter.get_finished_spans()
    assert span.status.status_code == StatusCode.ERROR
    attrs = dict(span.attributes or {})
    assert attrs[error_attributes.ERROR_TYPE] == "RuntimeError"


@pytest.mark.asyncio
async def test_abandoned_and_failed_tools(
    tracer_provider, logger_provider, meter_provider, span_exporter
):
    from opentelemetry.instrumentation.genai.claude_agent_sdk import (  # noqa: PLC0415
        ClaudeAgentSDKInstrumentor,
    )

    async def fake_query(**kwargs: Any):
        yield {
            "type": "assistant",
            "message": {
                "content": [
                    {
                        "type": "tool_use",
                        "id": "toolu_fail",
                        "name": "Bash",
                        "input": {"command": "false"},
                    },
                    {
                        "type": "tool_use",
                        "id": "toolu_open",
                        "name": "Read",
                        "input": {"path": "x"},
                    },
                ]
            },
        }
        yield {
            "type": "user",
            "message": {
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "toolu_fail",
                        "content": "boom",
                        "is_error": True,
                    }
                ]
            },
        }
        # toolu_open never gets a result: abandoned.
        yield {
            "type": "result",
            "subtype": "success",
            "result": "done",
            "usage": {"input_tokens": 1, "output_tokens": 1},
            "session_id": "sess",
        }

    with _FakeQuery(fake_query):
        instrumentor = ClaudeAgentSDKInstrumentor()
        instrumentor.instrument(
            tracer_provider=tracer_provider,
            logger_provider=logger_provider,
            meter_provider=meter_provider,
        )
        try:
            query_module = importlib.import_module("claude_agent_sdk.query")
            await _drain(query_module.query(prompt="hello"))
        finally:
            instrumentor.uninstrument()

    spans = span_exporter.get_finished_spans()
    grouped = _spans_by_operation(spans)
    (agent_span,) = grouped["invoke_agent"]
    assert agent_span.status.status_code == StatusCode.UNSET

    tool_spans = {
        dict(span.attributes or {})[
            GenAIAttributes.GEN_AI_TOOL_CALL_ID
        ]: span
        for span in grouped["execute_tool"]
    }
    assert tool_spans["toolu_fail"].status.status_code == StatusCode.ERROR
    assert tool_spans["toolu_open"].status.status_code == StatusCode.ERROR
    for span in tool_spans.values():
        assert (
            dict(span.attributes or {})[error_attributes.ERROR_TYPE]
            == "_OTHER"
        )


# ---- Unit tests: model-name extraction from per-model usage ----
# (Regression coverage carried over from the OpenInference package for
# multi-model runs, where the span must surface the model that did the
# bulk of the generation rather than an arbitrary dict key.)


def _extract(usage: Any) -> str | None:
    from opentelemetry.instrumentation.genai.claude_agent_sdk.patch import (  # noqa: PLC0415
        _extract_model_name_from_usage,
    )

    return _extract_model_name_from_usage(usage)


def test_single_model_dict_returns_that_model():
    usage = {"claude-sonnet-4-6": {"outputTokens": 4, "inputTokens": 3}}
    assert _extract(usage) == "claude-sonnet-4-6"


def test_multi_model_dict_picks_max_output_tokens():
    usage = {
        "claude-haiku-4-5": {"outputTokens": 5, "inputTokens": 200},
        "claude-sonnet-4-6": {"outputTokens": 350, "inputTokens": 8},
    }
    assert _extract(usage) == "claude-sonnet-4-6"


def test_multi_model_dict_order_does_not_matter():
    usage = {
        "claude-sonnet-4-6": {"outputTokens": 350, "inputTokens": 8},
        "claude-haiku-4-5": {"outputTokens": 5, "inputTokens": 200},
    }
    assert _extract(usage) == "claude-sonnet-4-6"


def test_snake_case_output_tokens_also_accepted():
    usage = {
        "claude-haiku-4-5": {"output_tokens": 5},
        "claude-sonnet-4-6": {"output_tokens": 400},
    }
    assert _extract(usage) == "claude-sonnet-4-6"


def test_missing_output_tokens_does_not_crash():
    usage = {"model-a": {"inputTokens": 10}, "model-b": {"inputTokens": 20}}
    assert _extract(usage) in ("model-a", "model-b")


def test_non_mapping_entry_value_does_not_crash():
    usage = {"model-a": "unexpected", "model-b": {"outputTokens": 99}}
    assert _extract(usage) == "model-b"


def test_non_int_output_tokens_does_not_crash():
    usage = {
        "model-a": {"outputTokens": "not-a-number"},
        "model-b": {"outputTokens": 50},
    }
    assert _extract(usage) == "model-b"


def test_empty_dict_returns_none():
    assert _extract({}) is None


def test_list_of_entries_returns_first_named():
    usage = [
        {"model": "claude-sonnet-4-6", "outputTokens": 10},
        {"model": "claude-haiku-4-5", "outputTokens": 200},
    ]
    assert _extract(usage) == "claude-sonnet-4-6"


def test_object_with_model_attribute():
    class FakeUsage:
        model = "claude-sonnet-4-6"

    assert _extract(FakeUsage()) == "claude-sonnet-4-6"


def test_none_returns_none():
    assert _extract(None) is None


@pytest.mark.parametrize("value", ["", 0, [], {}])
def test_falsy_inputs_return_none(value: Any):
    assert _extract(value) is None
