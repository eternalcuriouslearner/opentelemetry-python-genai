# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""Conformance scenario: a ``query()`` run that delegates to a subagent.

The recorded run has the root agent invoke the Agent tool to spawn a
``general-purpose`` subagent, which runs the Bash tool. Emits nested
``invoke_agent`` spans (root and subagent) plus ``execute_tool`` spans for
the spawning Agent tool and the subagent's Bash tool.
"""

from __future__ import annotations

import asyncio
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
    "You must use the Agent tool to delegate to a subagent. The subagent "
    "must use the Bash tool to run: wc -c 'pyproject.toml' and respond "
    "with exactly the output. Do not answer unless you executed the tool."
)


class MultiAgentScenario(Scenario):
    expected_spans = ("invoke_agent", "execute_tool")
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
                allowed_tools=["Bash", "Agent"],
                permission_mode="bypassPermissions",
            )
            async for _ in query(
                prompt=PROMPT,
                options=options,
                transport=replay_transport("subagent.yaml"),
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
        agent_spans = [
            entry["span"]
            for entry in report["samples"]
            if "span" in entry
            and _attr(entry["span"], "gen_ai.operation.name")
            == "invoke_agent"
        ]
        assert len(agent_spans) == 2, (
            f"expected a root and a subagent invoke_agent span, saw "
            f"{len(agent_spans)}"
        )
        agent_names = {
            _attr(span, "gen_ai.agent.name") for span in agent_spans
        }
        assert "general-purpose" in agent_names, agent_names

        tool_names = {
            _attr(entry["span"], "gen_ai.tool.name")
            for entry in report["samples"]
            if "span" in entry
            and _attr(entry["span"], "gen_ai.operation.name")
            == "execute_tool"
        }
        assert tool_names == {"Agent", "Bash"}, tool_names


def _attr(span: dict[str, Any], name: str) -> Any:
    for attr in span["attributes"]:
        if attr["name"] == name:
            return attr["value"]
    return None
