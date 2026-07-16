# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""Conformance scenario for a basic Claude Agent SDK query invocation."""

from __future__ import annotations

import asyncio
import json
import os
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any
from unittest import mock

import yaml
from claude_agent_sdk import ClaudeAgentOptions, query

from opentelemetry.instrumentation.genai.claude_agent_sdk import (
    ClaudeAgentSDKInstrumentor,
)
from opentelemetry.sdk._logs import LoggerProvider
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.test.weaver_live_check import LiveCheckReport
from opentelemetry.test_util_genai.conformance import Scenario
from opentelemetry.test_util_genai.instrumentor import instrument

try:
    from claude_agent_sdk._internal.transport import Transport
except ImportError:  # pragma: no cover - only for SDK compatibility
    Transport = object  # type: ignore[assignment,misc]

_CASSETTE = (
    Path(__file__).parent.parent
    / "cassettes"
    / "test_query_real_agent_span.yaml"
)
_FAKE_KEY = "test-key-no-real-call"


class _ReplayTransport(Transport):  # type: ignore[misc,valid-type]
    """Replay Claude Agent SDK subprocess messages from an OpenInference cassette."""

    def __init__(self, messages: list[dict[str, Any]], prompt: str) -> None:
        self._messages = messages
        self._prompt = prompt
        self._sdk_control_request_ids: list[str] = []

    async def connect(self) -> None:
        pass

    async def write(self, data: str) -> None:
        try:
            message = json.loads(data.strip())
        except json.JSONDecodeError:
            return
        if message.get("type") == "control_request":
            self._sdk_control_request_ids.append(str(message["request_id"]))

    def read_messages(self) -> AsyncIterator[dict[str, Any]]:
        return self._gen()

    async def _gen(self) -> AsyncIterator[dict[str, Any]]:
        response_index = 0
        started = False
        stopped = False
        for message in self._messages:
            if message.get("type") == "control_response":
                message = dict(message)
                response = dict(message["response"])
                if response_index < len(self._sdk_control_request_ids):
                    response["request_id"] = self._sdk_control_request_ids[
                        response_index
                    ]
                    response_index += 1
                message["response"] = response
                yield message
                continue
            if not started:
                started = True
                yield _hook_control_request(
                    request_id="hook_user_prompt",
                    callback_id="hook_0",
                    input_data={
                        "hook_event_name": "UserPromptSubmit",
                        "prompt": self._prompt,
                    },
                )
            if not stopped and message.get("type") == "result":
                stopped = True
                yield _hook_control_request(
                    request_id="hook_stop",
                    callback_id="hook_1",
                    input_data={
                        "hook_event_name": "Stop",
                        "stop_hook_active": False,
                    },
                )
            yield message

    async def close(self) -> None:
        pass

    async def end_input(self) -> None:
        pass

    def is_ready(self) -> bool:
        return True


def _hook_control_request(
    *, request_id: str, callback_id: str, input_data: dict[str, Any]
) -> dict[str, Any]:
    return {
        "type": "control_request",
        "request_id": request_id,
        "request": {
            "subtype": "hook_callback",
            "callback_id": callback_id,
            "tool_use_id": None,
            "input": input_data,
        },
    }


def _load_transport(prompt: str) -> _ReplayTransport:
    data = yaml.safe_load(_CASSETTE.read_text())
    return _ReplayTransport(messages=data["messages"], prompt=prompt)


class QueryScenario(Scenario):
    expected_spans = ("invoke_agent",)
    expected_metrics = ("gen_ai.client.operation.duration",)

    def run(
        self,
        *,
        tracer_provider: TracerProvider,
        meter_provider: MeterProvider,
        logger_provider: LoggerProvider,
        vcr: Any,
    ) -> None:
        async def run_query() -> None:
            prompt = "Reply with exactly: ok"
            options = ClaudeAgentOptions(model="claude-sonnet-4-6")
            async for _message in query(
                prompt=prompt,
                options=options,
                transport=_load_transport(prompt),
            ):
                pass

        key_override = (
            {}
            if os.getenv("ANTHROPIC_API_KEY")
            else {"ANTHROPIC_API_KEY": _FAKE_KEY}
        )
        with mock.patch.dict(os.environ, key_override):
            with instrument(
                ClaudeAgentSDKInstrumentor(),
                tracer_provider=tracer_provider,
                logger_provider=logger_provider,
                meter_provider=meter_provider,
                content_capture="SPAN_ONLY",
            ):
                asyncio.run(run_query())

    def validate(self, report: LiveCheckReport) -> None:
        super().validate(report)
        agent_names = {
            attr["value"]
            for entry in report["samples"]
            if "span" in entry
            for attr in entry["span"]["attributes"]
            if attr["name"] == "gen_ai.agent.name"
        }
        provider_names = {
            attr["value"]
            for entry in report["samples"]
            if "span" in entry
            for attr in entry["span"]["attributes"]
            if attr["name"] == "gen_ai.provider.name"
        }
        assert "ClaudeAgentSDK.query" in agent_names
        assert "anthropic" in provider_names
