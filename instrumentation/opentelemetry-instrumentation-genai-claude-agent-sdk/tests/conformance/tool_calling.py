# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""Conformance scenario: a ``query()`` run where the agent uses a tool.

The recorded run has the model call the Bash tool before answering, so the
``invoke_agent`` span's output messages must carry a ``tool_call`` part
alongside the final text.

The instrumentation currently emits agent-level spans only, so the tool
execution shows up as message content rather than a nested ``execute_tool``
span (a planned follow-up).
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

from claude_agent_sdk import ClaudeAgentOptions

from opentelemetry.instrumentation.genai.claude_agent_sdk import (
    ClaudeAgentSDKInstrumentor,
)
from opentelemetry.sdk._logs import LoggerProvider
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.test.weaver_live_check import LiveCheckReport
from opentelemetry.test_util_genai.conformance import Scenario
from opentelemetry.test_util_genai.instrumentor import instrument

from .cli_replay import replay_transport

PROMPT = (
    "Use the Bash tool to run: wc -c 'pyproject.toml' and respond with "
    "exactly the output. Do not answer unless you executed the tool."
)


class ToolCallingScenario(Scenario):
    expected_spans = ("invoke_agent",)
    expected_metrics = (
        "gen_ai.client.operation.duration",
        "gen_ai.client.token.usage",
    )

    def run(
        self,
        *,
        tracer_provider: TracerProvider,
        meter_provider: MeterProvider,
        logger_provider: LoggerProvider,
        vcr: Any,
    ) -> None:
        del vcr  # CLI-protocol replay, not HTTP — see tests/conformance/cli_replay.py

        async def _run() -> None:
            # Import after instrument() so the patched module attribute is
            # picked up (a module-top import would bind the original).
            from claude_agent_sdk import query  # noqa: PLC0415

            options = ClaudeAgentOptions(
                allowed_tools=["Bash"], permission_mode="bypassPermissions"
            )
            async for _ in query(
                prompt=PROMPT,
                options=options,
                transport=replay_transport("tool_calling.yaml"),
            ):
                pass

        with instrument(
            ClaudeAgentSDKInstrumentor(),
            tracer_provider=tracer_provider,
            logger_provider=logger_provider,
            meter_provider=meter_provider,
            content_capture="SPAN_ONLY",
        ):
            asyncio.run(_run())

    def validate(self, report: LiveCheckReport) -> None:
        super().validate(report)
        spans = _invoke_agent_spans(report)
        assert len(spans) == 1, f"expected one invoke_agent span, saw {spans}"
        span = spans[0]

        output_messages = json.loads(_attr(span, "gen_ai.output.messages"))
        part_types = {
            part["type"]
            for message in output_messages
            for part in message["parts"]
        }
        assert "tool_call" in part_types, (
            f"expected a tool_call part on an output message, saw {part_types}"
        )
        assert "text" in part_types, part_types

        tool_calls = [
            part
            for message in output_messages
            for part in message["parts"]
            if part["type"] == "tool_call"
        ]
        assert tool_calls[0]["name"] == "Bash", tool_calls
        assert isinstance(tool_calls[0]["id"], str)
        assert "wc -c" in tool_calls[0]["arguments"]["command"], tool_calls


def _invoke_agent_spans(report: LiveCheckReport) -> list[dict[str, Any]]:
    return [
        entry["span"]
        for entry in report["samples"]
        if "span" in entry
        and _attr(entry["span"], "gen_ai.operation.name") == "invoke_agent"
    ]


def _attr(span: dict[str, Any], name: str) -> Any:
    for attr in span["attributes"]:
        if attr["name"] == name:
            return attr["value"]
    return None
