# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for the ClaudeAgentSDKInstrumentor class."""

import sys
import types

import anyio

from opentelemetry.instrumentation.genai.claude_agent_sdk import (
    ClaudeAgentSDKInstrumentor,
)


def test_instrumentor_instantiation():
    """Test that the instrumentor can be instantiated."""
    instrumentor = ClaudeAgentSDKInstrumentor()
    assert instrumentor is not None
    assert isinstance(instrumentor, ClaudeAgentSDKInstrumentor)


def test_instrumentation_dependencies():
    """Test that instrumentation dependencies are correctly reported."""
    instrumentor = ClaudeAgentSDKInstrumentor()
    dependencies = instrumentor.instrumentation_dependencies()

    assert dependencies is not None
    assert len(dependencies) > 0
    assert "claude-agent-sdk >= 0.1.14" in dependencies


def test_instrument_uninstrument_cycle(
    tracer_provider, logger_provider, meter_provider
):
    """Test that instrument() and uninstrument() can be called multiple times."""
    instrumentor = ClaudeAgentSDKInstrumentor()

    # First instrumentation
    instrumentor.instrument(
        tracer_provider=tracer_provider,
        logger_provider=logger_provider,
        meter_provider=meter_provider,
    )

    # First uninstrumentation
    instrumentor.uninstrument()

    # Second instrumentation (should work)
    instrumentor.instrument(
        tracer_provider=tracer_provider,
        logger_provider=logger_provider,
        meter_provider=meter_provider,
    )

    # Second uninstrumentation
    instrumentor.uninstrument()


def test_multiple_instrumentation_calls(
    tracer_provider, logger_provider, meter_provider
):
    """Test that multiple instrument() calls don't cause issues."""
    instrumentor = ClaudeAgentSDKInstrumentor()

    # First call
    instrumentor.instrument(
        tracer_provider=tracer_provider,
        logger_provider=logger_provider,
        meter_provider=meter_provider,
    )

    # Second call (should be idempotent or handle gracefully)
    instrumentor.instrument(
        tracer_provider=tracer_provider,
        logger_provider=logger_provider,
        meter_provider=meter_provider,
    )

    # Clean up
    instrumentor.uninstrument()


def test_uninstrument_without_instrument():
    """Test that uninstrument() can be called without prior instrument()."""
    instrumentor = ClaudeAgentSDKInstrumentor()

    # This should not raise an error
    instrumentor.uninstrument()


def test_instrument_with_no_providers():
    """Test that instrument() works without explicit providers."""
    instrumentor = ClaudeAgentSDKInstrumentor()

    # Should use global providers
    instrumentor.instrument()

    # Clean up
    instrumentor.uninstrument()


def test_instrumentor_has_required_attributes():
    """Test that the instrumentor has the required methods."""
    instrumentor = ClaudeAgentSDKInstrumentor()

    assert hasattr(instrumentor, "instrument")
    assert hasattr(instrumentor, "uninstrument")
    assert hasattr(instrumentor, "instrumentation_dependencies")
    assert callable(instrumentor.instrument)
    assert callable(instrumentor.uninstrument)
    assert callable(instrumentor.instrumentation_dependencies)


def test_query_injects_hooks_and_creates_agent_span(
    monkeypatch,
    tracer_provider,
    span_exporter,
    logger_provider,
    meter_provider,
):
    """Test query() uses Claude Agent SDK hooks for invoke_agent spans."""

    class HookMatcher:
        def __init__(self, matcher=None, hooks=None):
            self.matcher = matcher
            self.hooks = hooks or []

    class ClaudeAgentOptions:
        def __init__(self, model=None, hooks=None):
            self.model = model
            self.hooks = hooks

    async def _run_hooks(options, event_name, input_data):
        for matcher in options.hooks[event_name]:
            for hook in matcher.hooks:
                await hook(input_data, None, {"signal": None})

    async def query(prompt, options=None):
        await _run_hooks(
            options,
            "UserPromptSubmit",
            {"hook_event_name": "UserPromptSubmit", "prompt": prompt},
        )
        yield {"type": "assistant"}
        await _run_hooks(
            options,
            "Stop",
            {"hook_event_name": "Stop", "stop_hook_active": False},
        )

    module = types.ModuleType("claude_agent_sdk")
    module.ClaudeAgentOptions = ClaudeAgentOptions
    module.HookMatcher = HookMatcher
    module.query = query
    monkeypatch.setitem(sys.modules, "claude_agent_sdk", module)

    instrumentor = ClaudeAgentSDKInstrumentor()
    instrumentor.instrument(
        tracer_provider=tracer_provider,
        logger_provider=logger_provider,
        meter_provider=meter_provider,
    )

    options = ClaudeAgentOptions(model="claude-sonnet-4-5")

    async def run_query():
        messages = []
        async for message in module.query(prompt="Hello", options=options):
            messages.append(message)
        return messages

    messages = anyio.run(run_query)

    instrumentor.uninstrument()

    assert len(messages) == 1
    assert "UserPromptSubmit" in options.hooks
    assert "Stop" in options.hooks
    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]
    assert span.name == "invoke_agent ClaudeAgentSDK.query"
    assert span.attributes["gen_ai.operation.name"] == "invoke_agent"
    assert span.attributes["gen_ai.provider.name"] == "anthropic"
    assert span.attributes["gen_ai.agent.name"] == "ClaudeAgentSDK.query"
    assert span.attributes["gen_ai.request.model"] == "claude-sonnet-4-5"


def test_query_preserves_existing_hooks(
    monkeypatch,
    tracer_provider,
    logger_provider,
    meter_provider,
):
    """Test instrumentation composes with user-provided SDK hooks."""

    class HookMatcher:
        def __init__(self, matcher=None, hooks=None):
            self.matcher = matcher
            self.hooks = hooks or []

    class ClaudeAgentOptions:
        def __init__(self, hooks=None):
            self.hooks = hooks

    async def user_hook(input_data, tool_use_id, context):
        return {}

    async def query(prompt, options=None):
        yield {"type": "assistant"}

    module = types.ModuleType("claude_agent_sdk")
    module.ClaudeAgentOptions = ClaudeAgentOptions
    module.HookMatcher = HookMatcher
    module.query = query
    monkeypatch.setitem(sys.modules, "claude_agent_sdk", module)

    instrumentor = ClaudeAgentSDKInstrumentor()
    instrumentor.instrument(
        tracer_provider=tracer_provider,
        logger_provider=logger_provider,
        meter_provider=meter_provider,
    )

    user_matcher = HookMatcher(matcher=None, hooks=[user_hook])
    options = ClaudeAgentOptions(hooks={"UserPromptSubmit": [user_matcher]})

    async def run_query():
        async for _ in module.query(prompt="Hello", options=options):
            pass

    try:
        anyio.run(run_query)
    finally:
        instrumentor.uninstrument()

    assert options.hooks["UserPromptSubmit"][1] is user_matcher


def test_query_stream_error_records_exception(
    monkeypatch,
    tracer_provider,
    span_exporter,
    logger_provider,
    meter_provider,
):
    """Test query() records and re-raises stream errors after hook start."""

    class HookMatcher:
        def __init__(self, matcher=None, hooks=None):
            self.matcher = matcher
            self.hooks = hooks or []

    class ClaudeAgentOptions:
        def __init__(self, hooks=None):
            self.hooks = hooks

    expected_error = ConnectionError("stream failed")

    async def _run_hooks(options, event_name, input_data):
        for matcher in options.hooks[event_name]:
            for hook in matcher.hooks:
                await hook(input_data, None, {"signal": None})

    async def query(prompt, options=None):
        await _run_hooks(
            options,
            "UserPromptSubmit",
            {"hook_event_name": "UserPromptSubmit", "prompt": prompt},
        )
        raise expected_error
        yield  # pragma: no cover

    module = types.ModuleType("claude_agent_sdk")
    module.ClaudeAgentOptions = ClaudeAgentOptions
    module.HookMatcher = HookMatcher
    module.query = query
    monkeypatch.setitem(sys.modules, "claude_agent_sdk", module)

    instrumentor = ClaudeAgentSDKInstrumentor()
    instrumentor.instrument(
        tracer_provider=tracer_provider,
        logger_provider=logger_provider,
        meter_provider=meter_provider,
    )

    async def run_query():
        async for _ in module.query(prompt="Hello"):
            pass

    try:
        try:
            anyio.run(run_query)
        except ConnectionError as error:
            assert error is expected_error
        else:  # pragma: no cover
            raise AssertionError("expected ConnectionError")
    finally:
        instrumentor.uninstrument()

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    assert spans[0].attributes["error.type"] == "ConnectionError"
