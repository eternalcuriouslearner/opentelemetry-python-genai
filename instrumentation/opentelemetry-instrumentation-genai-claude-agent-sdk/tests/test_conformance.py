# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""Per-scenario conformance tests for the Claude Agent SDK instrumentation."""

from __future__ import annotations

import pytest

# Skip collection when weaver_live_check or OTLP exporters aren't installed
# (non-conformance envs).
pytest.importorskip("opentelemetry.test.weaver_live_check")
pytest.importorskip("opentelemetry.exporter.otlp.proto.grpc")

from opentelemetry.test.weaver_live_check import WeaverLiveCheck  # noqa: E402
from opentelemetry.test_util_genai.conformance import (  # noqa: E402
    Scenario,
    run_conformance,
)

from .conformance.invoke_agent import InvokeAgentScenario
from .conformance.multi_agent import MultiAgentScenario
from .conformance.reasoning import ReasoningScenario
from .conformance.tool_calling import ToolCallingScenario


@pytest.mark.parametrize(
    "scenario",
    [
        pytest.param(InvokeAgentScenario()),
        pytest.param(ToolCallingScenario()),
        pytest.param(ReasoningScenario()),
        pytest.param(MultiAgentScenario()),
    ],
    ids=lambda s: type(s).__name__,
)
def test_conformance(
    scenario: Scenario, weaver_live_check: WeaverLiveCheck
) -> None:
    # The Claude Agent SDK replays recorded CLI sessions through a custom
    # Transport (tests/conformance/cli_replay.py) instead of HTTP cassettes, so no VCR
    # object is needed.
    run_conformance(scenario, vcr=None, weaver=weaver_live_check)
