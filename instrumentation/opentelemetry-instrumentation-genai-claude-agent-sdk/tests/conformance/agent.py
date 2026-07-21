# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""Conformance scenario for Claude agent, subagent, and tool spans."""

from __future__ import annotations

import asyncio
import importlib
from pathlib import Path
from typing import Any

from opentelemetry.instrumentation.genai.claude_agent_sdk import (
    ClaudeAgentSDKInstrumentor,
)
from opentelemetry.sdk._logs import LoggerProvider
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.test.weaver_live_check import LiveCheckReport
from opentelemetry.test_util_genai.conformance import (
    ExpectedViolation,
    Scenario,
)
from opentelemetry.test_util_genai.instrumentor import instrument

from .._replay import load_cassette  # noqa: TID252


class AgentScenario(Scenario):
    expected_spans = {"invoke_agent": 2, "execute_tool": 1}
    expected_metrics = (
        "gen_ai.client.operation.duration",
        "gen_ai.client.token.usage",
    )
    expected_violations = (
        # The SDK invokes a local Claude Code subprocess and does not expose
        # the upstream Anthropic service endpoint to this instrumentation.
        ExpectedViolation(
            advice_id="genai_expected_attribute_missing",
            message_substring="server.address",
        ),
    )

    def run(
        self,
        *,
        tracer_provider: TracerProvider,
        meter_provider: MeterProvider,
        logger_provider: LoggerProvider,
        vcr: Any,
    ) -> None:
        del vcr
        cassette = (
            Path(__file__).parents[1]
            / "cassettes"
            / "openinference"
            / "test_query_task_subagent_spans.yaml"
        )

        async def exercise() -> None:
            query_function = importlib.import_module("claude_agent_sdk").query
            async for _ in query_function(
                prompt="Delegate this task and run the Bash tool",
                transport=load_cassette(cassette),
            ):
                pass

        with instrument(
            ClaudeAgentSDKInstrumentor(),
            tracer_provider=tracer_provider,
            meter_provider=meter_provider,
            logger_provider=logger_provider,
            content_capture="SPAN_ONLY",
        ):
            asyncio.run(exercise())

    def validate(self, report: LiveCheckReport) -> None:
        super().validate(report)
        agent_names = {
            attribute["value"]
            for entry in report["samples"]
            if "span" in entry
            for attribute in entry["span"]["attributes"]
            if attribute["name"] == "gen_ai.agent.name"
        }
        assert agent_names == {"Claude", "general-purpose"}
