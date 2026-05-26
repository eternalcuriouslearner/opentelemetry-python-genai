# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for async Messages.create instrumentation."""

import pytest
from anthropic import NotFoundError

from opentelemetry.instrumentation.genai.anthropic.wrappers import (
    AsyncMessagesStreamWrapper,
)
from opentelemetry.semconv._incubating.attributes import (
    error_attributes as ErrorAttributes,
)
from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAIAttributes,
)
from opentelemetry.semconv._incubating.attributes import (
    server_attributes as ServerAttributes,
)


def normalize_stop_reason(stop_reason):
    """Map Anthropic stop reasons to GenAI semconv values."""
    return {
        "end_turn": "stop",
        "stop_sequence": "stop",
        "max_tokens": "length",
        "tool_use": "tool_calls",
    }.get(stop_reason, stop_reason)


def expected_input_tokens(usage):
    """Compute semconv input tokens from Anthropic usage."""
    base = getattr(usage, "input_tokens", 0) or 0
    cache_creation = getattr(usage, "cache_creation_input_tokens", 0) or 0
    cache_read = getattr(usage, "cache_read_input_tokens", 0) or 0
    return base + cache_creation + cache_read


@pytest.mark.asyncio
@pytest.mark.vcr()
async def test_async_messages_create_basic(
    span_exporter, async_anthropic_client, instrument_no_content
):
    """Test basic async message creation produces correct span."""
    model = "claude-sonnet-4-20250514"
    messages = [{"role": "user", "content": "Say hello in one word."}]

    response = await async_anthropic_client.messages.create(
        model=model,
        max_tokens=100,
        messages=messages,
    )

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1

    span = spans[0]
    assert span.name == f"chat {model}"
    assert span.attributes[GenAIAttributes.GEN_AI_OPERATION_NAME] == "chat"
    assert (
        span.attributes[GenAIAttributes.GEN_AI_SYSTEM] == "anthropic"
    )
    assert span.attributes[GenAIAttributes.GEN_AI_REQUEST_MODEL] == model
    assert (
        span.attributes[GenAIAttributes.GEN_AI_RESPONSE_ID] == response.id
    )
    assert (
        span.attributes[GenAIAttributes.GEN_AI_RESPONSE_MODEL]
        == response.model
    )
    assert (
        span.attributes[GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS]
        == expected_input_tokens(response.usage)
    )
    assert (
        span.attributes[GenAIAttributes.GEN_AI_USAGE_OUTPUT_TOKENS]
        == response.usage.output_tokens
    )
    assert (
        span.attributes[GenAIAttributes.GEN_AI_RESPONSE_FINISH_REASONS]
        == (normalize_stop_reason(response.stop_reason),)
    )
    assert (
        span.attributes[ServerAttributes.SERVER_ADDRESS]
        == "api.anthropic.com"
    )


@pytest.mark.asyncio
@pytest.mark.vcr()
async def test_async_messages_create_streaming(
    span_exporter, async_anthropic_client, instrument_no_content
):
    """Test async create(stream=True) returns a wrapped stream and records a span."""
    model = "claude-sonnet-4-20250514"
    messages = [{"role": "user", "content": "Say hello in one word."}]

    response_id = None
    response_model = None
    stop_reason = None
    input_tokens = None
    output_tokens = None

    stream = await async_anthropic_client.messages.create(
        model=model,
        max_tokens=100,
        messages=messages,
        stream=True,
    )
    assert isinstance(stream, AsyncMessagesStreamWrapper)

    async with stream:
        async for chunk in stream:
            if chunk.type == "message_start":
                response_id = chunk.message.id
                response_model = chunk.message.model
                input_tokens = chunk.message.usage.input_tokens
            elif chunk.type == "message_delta":
                stop_reason = chunk.delta.stop_reason
                output_tokens = chunk.usage.output_tokens

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1

    span = spans[0]
    assert span.attributes[GenAIAttributes.GEN_AI_REQUEST_MODEL] == model
    assert span.attributes[GenAIAttributes.GEN_AI_RESPONSE_ID] == response_id
    assert (
        span.attributes[GenAIAttributes.GEN_AI_RESPONSE_MODEL]
        == response_model
    )
    assert (
        span.attributes[GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS]
        == input_tokens
    )
    assert (
        span.attributes[GenAIAttributes.GEN_AI_USAGE_OUTPUT_TOKENS]
        == output_tokens
    )
    assert (
        span.attributes[GenAIAttributes.GEN_AI_RESPONSE_FINISH_REASONS]
        == (normalize_stop_reason(stop_reason),)
    )


@pytest.mark.asyncio
@pytest.mark.vcr()
async def test_async_messages_create_api_error(
    span_exporter, async_anthropic_client, instrument_no_content
):
    """Test async API errors are recorded and re-raised unchanged."""
    model = "invalid-model-name"
    messages = [{"role": "user", "content": "Hello"}]

    with pytest.raises(NotFoundError):
        await async_anthropic_client.messages.create(
            model=model,
            max_tokens=100,
            messages=messages,
        )

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]
    assert span.attributes[GenAIAttributes.GEN_AI_REQUEST_MODEL] == model
    assert ErrorAttributes.ERROR_TYPE in span.attributes
    assert "NotFoundError" in span.attributes[ErrorAttributes.ERROR_TYPE]
