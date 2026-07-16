# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""Conformance scenario: a ``ClaudeSDKClient`` turn with thinking content.

Drives the interactive-client surface (``connect`` / ``query`` /
``receive_response``): the recorded turn starts with a thinking block, then
a tool call, then the final text, so the turn's ``invoke_agent`` span must
carry ``reasoning``, ``tool_call``, and ``text`` output parts.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

from claude_agent_sdk import ClaudeAgentOptions, ClaudeSDKClient

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


class ReasoningScenario(Scenario):
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
            options = ClaudeAgentOptions(
                allowed_tools=["Bash"], permission_mode="bypassPermissions"
            )
            async with ClaudeSDKClient(
                options=options,
                transport=replay_transport("reasoning.yaml"),
            ) as client:
                await client.query(PROMPT)
                async for _ in client.receive_response():
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

        input_messages = json.loads(_attr(span, "gen_ai.input.messages"))
        assert input_messages == [
            {"role": "user", "parts": [{"content": PROMPT, "type": "text"}]}
        ], input_messages

        part_types = {
            part["type"]
            for message in json.loads(_attr(span, "gen_ai.output.messages"))
            for part in message["parts"]
        }
        assert "reasoning" in part_types, (
            f"expected a reasoning part on an output message, saw {part_types}"
        )
        assert {"tool_call", "text"} <= part_types, part_types


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
