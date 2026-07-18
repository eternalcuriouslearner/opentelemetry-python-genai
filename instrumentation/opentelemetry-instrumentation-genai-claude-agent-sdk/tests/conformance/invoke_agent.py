# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""Conformance scenario: a plain ``query()`` agent run.

A one-shot ``query()`` with a text-only response, replayed from a recorded
CLI session. Emits a single ``invoke_agent`` span carrying the prompt, the
assistant's text output, token usage, model, and session id.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

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

PROMPT = "Reply with exactly the word: ok"


class InvokeAgentScenario(Scenario):
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

            async for _ in query(
                prompt=PROMPT,
                transport=replay_transport("invoke_agent.yaml"),
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

        input_messages = json.loads(_attr(span, "gen_ai.input.messages"))
        assert input_messages == [
            {"role": "user", "parts": [{"content": PROMPT, "type": "text"}]}
        ], input_messages

        output_parts = _part_types(_attr(span, "gen_ai.output.messages"))
        assert output_parts == ["text"], output_parts

        assert isinstance(_attr(span, "gen_ai.conversation.id"), str)
        assert isinstance(_attr(span, "gen_ai.request.model"), str)
        assert isinstance(_attr(span, "gen_ai.usage.input_tokens"), int)
        assert isinstance(_attr(span, "gen_ai.usage.output_tokens"), int)


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


def _part_types(messages_json: str | None) -> list[str]:
    messages = json.loads(messages_json) if messages_json else []
    return [part["type"] for message in messages for part in message["parts"]]
