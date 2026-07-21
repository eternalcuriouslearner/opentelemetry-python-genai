# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""Test configuration and fixtures for Claude Agent SDK instrumentation tests."""
# pylint: disable=redefined-outer-name

from pathlib import Path

import pytest

from ._replay import load_cassettes

pytest_plugins = ["opentelemetry.test_util_genai.fixtures"]


@pytest.fixture
def cassette_transport():
    """Replay one or more complete donated OpenInference cassettes."""

    cassette_dir = Path(__file__).parent / "cassettes" / "openinference"

    def load(*names: str):
        return load_cassettes(*(cassette_dir / name for name in names))

    return load


@pytest.fixture
def instrument_claude_agent_sdk(
    tracer_provider, logger_provider, meter_provider
):
    """Fixture to instrument Claude Agent SDK with test providers."""
    # pylint: disable=import-outside-toplevel
    from opentelemetry.instrumentation.genai.claude_agent_sdk import (  # noqa: PLC0415
        ClaudeAgentSDKInstrumentor,
    )

    instrumentor = ClaudeAgentSDKInstrumentor()
    instrumentor.instrument(
        tracer_provider=tracer_provider,
        logger_provider=logger_provider,
        meter_provider=meter_provider,
    )
    yield instrumentor
    instrumentor.uninstrument()


@pytest.fixture
def uninstrument_claude_agent_sdk():
    """Fixture to ensure Claude Agent SDK is uninstrumented after test."""
    yield
    # pylint: disable=import-outside-toplevel
    from opentelemetry.instrumentation.genai.claude_agent_sdk import (  # noqa: PLC0415
        ClaudeAgentSDKInstrumentor,
    )

    ClaudeAgentSDKInstrumentor().uninstrument()
