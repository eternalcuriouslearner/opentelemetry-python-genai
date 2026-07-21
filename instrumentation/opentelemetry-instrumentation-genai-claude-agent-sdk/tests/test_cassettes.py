# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import hashlib
import importlib
from pathlib import Path

import pytest

from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAI,
)

QUERY_REAL_AGENT = "test_query_real_agent_span.yaml"
CLIENT_REAL_AGENT = "test_client_real_agent_span.yaml"
QUERY_TOOL_MESSAGES = "test_query_tool_spans_from_messages.yaml"
QUERY_TOOL_FALLBACK = "test_query_tool_fallback_when_hooks_unavailable.yaml"
QUERY_TASK_SUBAGENT = "test_query_task_subagent_spans.yaml"
CLIENT_TOOL_HOOKS = "test_client_tool_hooks_create_tool_spans.yaml"

OPENINFERENCE_CASSETTE_BLOBS = {
    CLIENT_REAL_AGENT: "5a9e850e9de66952314b2e429871ba85a2f76a6e",
    CLIENT_TOOL_HOOKS: "bac30796c18d91d11549c086b4c43fd90f9494d7",
    QUERY_REAL_AGENT: "98382a35ef62546d32fde7d7d13eed9617cdf3ab",
    QUERY_TASK_SUBAGENT: "6e02e02759e131a0f9047d3944d72d3e248b8221",
    QUERY_TOOL_FALLBACK: "3e33d0c44d129468af316bf1dbc6edd6411e511f",
    QUERY_TOOL_MESSAGES: "b1d9fecbfc9973580cfabd28640d5f5127e4e693",
}


def _span_named(spans, name):
    matches = [span for span in spans if span.name == name]
    assert matches, f"No {name!r} span in {[span.name for span in spans]}"
    return matches[-1]


def _git_blob_hash(path: Path) -> str:
    content = path.read_bytes()
    header = f"blob {len(content)}\0".encode()
    return hashlib.sha1(header + content, usedforsecurity=False).hexdigest()


def _run_one_shot(transport):
    async def exercise():
        query_function = importlib.import_module("claude_agent_sdk").query
        async for _ in query_function(
            prompt="Replay the recorded turn",
            transport=transport,
        ):
            pass

    asyncio.run(exercise())


def _run_persistent(transport, prompts):
    async def exercise():
        client_type = importlib.import_module(
            "claude_agent_sdk"
        ).ClaudeSDKClient

        async with client_type(transport=transport) as client:
            for prompt in prompts:
                await client.query(prompt)
                async for _ in client.receive_response():
                    pass

    asyncio.run(exercise())


def test_openinference_cassettes_are_complete_and_unmodified():
    cassette_dir = Path(__file__).parent / "cassettes" / "openinference"
    cassettes = {path.name: path for path in cassette_dir.glob("*.yaml")}

    assert cassettes.keys() == OPENINFERENCE_CASSETTE_BLOBS.keys()
    assert {
        name: _git_blob_hash(path) for name, path in cassettes.items()
    } == OPENINFERENCE_CASSETTE_BLOBS


def test_one_shot_query_uses_complete_openinference_cassette(
    instrument_claude_agent_sdk,
    cassette_transport,
    span_exporter,
):
    _run_one_shot(cassette_transport(QUERY_REAL_AGENT))

    (span,) = span_exporter.get_finished_spans()
    assert span.name == "invoke_agent Claude"
    attributes = dict(span.attributes or {})
    assert attributes[GenAI.GEN_AI_OPERATION_NAME] == "invoke_agent"
    assert attributes[GenAI.GEN_AI_PROVIDER_NAME] == "anthropic"
    assert attributes[GenAI.GEN_AI_REQUEST_MODEL] == "claude-sonnet-4-6"
    assert (
        attributes[GenAI.GEN_AI_CONVERSATION_ID]
        == "446de397-5452-4a8a-a368-67ef07fafecb"
    )


def test_persistent_client_uses_complete_openinference_cassette(
    instrument_claude_agent_sdk,
    cassette_transport,
    span_exporter,
):
    _run_persistent(cassette_transport(CLIENT_REAL_AGENT), ("first",))

    (span,) = span_exporter.get_finished_spans()
    assert span.name == "invoke_agent Claude"
    assert dict(span.attributes or {})[GenAI.GEN_AI_OPERATION_NAME] == (
        "invoke_agent"
    )


@pytest.mark.parametrize(
    "cassette_name",
    (QUERY_TOOL_MESSAGES, QUERY_TOOL_FALLBACK),
)
def test_one_shot_tool_lineage_uses_each_complete_openinference_cassette(
    instrument_claude_agent_sdk,
    cassette_transport,
    span_exporter,
    cassette_name,
):
    _run_one_shot(cassette_transport(cassette_name))

    spans = span_exporter.get_finished_spans()
    root = _span_named(spans, "invoke_agent Claude")
    tool = _span_named(spans, "execute_tool Bash")
    assert tool.parent is not None
    assert tool.parent.span_id == root.context.span_id
    assert dict(tool.attributes or {})[GenAI.GEN_AI_TOOL_CALL_ID].startswith(
        "toolu_"
    )


def test_one_shot_subagent_lineage_uses_complete_openinference_cassette(
    instrument_claude_agent_sdk,
    cassette_transport,
    span_exporter,
):
    _run_one_shot(cassette_transport(QUERY_TASK_SUBAGENT))

    spans = span_exporter.get_finished_spans()
    root = _span_named(spans, "invoke_agent Claude")
    subagent = _span_named(spans, "invoke_agent general-purpose")
    tool = _span_named(spans, "execute_tool Bash")
    assert subagent.parent is not None
    assert subagent.parent.span_id == root.context.span_id
    assert tool.parent is not None
    assert tool.parent.span_id == subagent.context.span_id
    assert not any(span.name == "execute_tool Agent" for span in spans)


def test_persistent_tool_lineage_ignores_recorded_hook_callbacks(
    instrument_claude_agent_sdk,
    cassette_transport,
    span_exporter,
):
    _run_persistent(cassette_transport(CLIENT_TOOL_HOOKS), ("run tool",))

    spans = span_exporter.get_finished_spans()
    root = _span_named(spans, "invoke_agent Claude")
    tool = _span_named(spans, "execute_tool Bash")
    assert tool.parent is not None
    assert tool.parent.span_id == root.context.span_id
    assert not any(span.name == "claude_code.hook" for span in spans)


def test_persistent_multi_turn_composes_complete_cassettes(
    instrument_claude_agent_sdk,
    cassette_transport,
    span_exporter,
):
    _run_persistent(
        cassette_transport(CLIENT_REAL_AGENT, CLIENT_REAL_AGENT),
        ("first", "second"),
    )

    roots = [
        span
        for span in span_exporter.get_finished_spans()
        if span.name == "invoke_agent Claude"
    ]
    assert len(roots) == 2
    assert roots[0].context.trace_id != roots[1].context.trace_id


def test_persistent_multi_turn_subagent_and_tool_lineage_composes_cassettes(
    instrument_claude_agent_sdk,
    cassette_transport,
    span_exporter,
):
    _run_persistent(
        cassette_transport(CLIENT_REAL_AGENT, QUERY_TASK_SUBAGENT),
        ("ready", "delegate and run a tool"),
    )

    spans = span_exporter.get_finished_spans()
    roots = [span for span in spans if span.name == "invoke_agent Claude"]
    subagent = _span_named(spans, "invoke_agent general-purpose")
    tool = _span_named(spans, "execute_tool Bash")
    assert len(roots) == 2
    assert subagent.parent is not None
    assert subagent.parent.span_id == roots[-1].context.span_id
    assert tool.parent is not None
    assert tool.parent.span_id == subagent.context.span_id
