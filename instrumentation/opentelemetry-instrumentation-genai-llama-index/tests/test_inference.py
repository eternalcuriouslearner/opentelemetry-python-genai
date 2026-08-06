# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from importlib.metadata import version

import pytest
from llama_index.core.llms import ChatMessage, MockLLM
from openai import AuthenticationError

from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAIAttributes,
)
from opentelemetry.semconv.attributes import (
    error_attributes,
    server_attributes,
)
from opentelemetry.trace import SpanKind, StatusCode


def _assert_inference_span(span: ReadableSpan) -> None:
    attributes = span.attributes
    assert attributes is not None
    assert span.name == "chat gpt-4o-mini"
    assert span.kind is SpanKind.CLIENT
    assert span.status.status_code is StatusCode.UNSET
    assert attributes[GenAIAttributes.GEN_AI_OPERATION_NAME] == "chat"
    assert isinstance(attributes[GenAIAttributes.GEN_AI_OPERATION_NAME], str)
    assert attributes[GenAIAttributes.GEN_AI_PROVIDER_NAME] == (
        GenAIAttributes.GenAiProviderNameValues.OPENAI.value
    )
    assert attributes[GenAIAttributes.GEN_AI_REQUEST_MODEL] == "gpt-4o-mini"
    assert attributes[server_attributes.SERVER_ADDRESS] == "api.openai.com"
    assert isinstance(attributes[server_attributes.SERVER_ADDRESS], str)
    assert attributes[GenAIAttributes.GEN_AI_REQUEST_TEMPERATURE] == 0.1
    assert isinstance(
        attributes[GenAIAttributes.GEN_AI_REQUEST_TEMPERATURE], float
    )
    assert attributes[GenAIAttributes.GEN_AI_RESPONSE_ID] == (
        "chatcmpl-BIRHLZwGBZn076WDqD1eV6AMp1sQc"
    )
    assert attributes[GenAIAttributes.GEN_AI_RESPONSE_MODEL] == (
        "gpt-4o-mini-2024-07-18"
    )
    assert attributes[GenAIAttributes.GEN_AI_RESPONSE_FINISH_REASONS] == (
        "stop",
    )
    assert attributes[GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS] == 9
    assert isinstance(
        attributes[GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS], int
    )
    # util-genai 1.0b0 added reasoning tokens to output_tokens; current
    # util-genai follows semconv and reports the provider's output total as-is.
    expected_output_tokens = (
        13 if version("opentelemetry-util-genai") == "1.0b0" else 10
    )
    assert (
        attributes[GenAIAttributes.GEN_AI_USAGE_OUTPUT_TOKENS]
        == expected_output_tokens
    )
    assert isinstance(
        attributes[GenAIAttributes.GEN_AI_USAGE_OUTPUT_TOKENS], int
    )
    assert (
        attributes[GenAIAttributes.GEN_AI_USAGE_CACHE_READ_INPUT_TOKENS] == 1
    )
    assert isinstance(
        attributes[GenAIAttributes.GEN_AI_USAGE_CACHE_READ_INPUT_TOKENS], int
    )
    assert (
        attributes[GenAIAttributes.GEN_AI_USAGE_REASONING_OUTPUT_TOKENS] == 3
    )
    assert isinstance(
        attributes[GenAIAttributes.GEN_AI_USAGE_REASONING_OUTPUT_TOKENS], int
    )

    input_messages = json.loads(
        attributes[GenAIAttributes.GEN_AI_INPUT_MESSAGES]
    )
    assert input_messages == [
        {
            "role": "user",
            "parts": [{"content": "Hello!", "type": "text"}],
        }
    ]
    output_messages = json.loads(
        attributes[GenAIAttributes.GEN_AI_OUTPUT_MESSAGES]
    )
    assert output_messages == [
        {
            "role": "assistant",
            "parts": [
                {
                    "content": "Hello! How can I assist you today?",
                    "type": "text",
                }
            ],
            "finish_reason": "stop",
        }
    ]


def test_chat_inference(
    span_exporter,
    metric_reader,
    instrument_llama_index_with_content,
    openai_llm,
    vcr,
) -> None:
    with vcr.use_cassette("inference.yaml"):
        response = openai_llm.chat(
            [ChatMessage(role="user", content="Hello!")]
        )

    assert response.message.content == "Hello! How can I assist you today?"
    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    _assert_inference_span(spans[0])
    metrics_data = metric_reader.get_metrics_data()
    assert metrics_data is not None
    metrics = metrics_data.resource_metrics[0].scope_metrics[0].metrics
    assert {metric.name for metric in metrics} == {
        "gen_ai.client.operation.duration",
        "gen_ai.client.token.usage",
    }


@pytest.mark.asyncio
async def test_async_chat_inference(
    span_exporter,
    instrument_llama_index_with_content,
    openai_llm,
    vcr,
) -> None:
    with vcr.use_cassette("inference.yaml"):
        response = await openai_llm.achat(
            [ChatMessage(role="user", content="Hello!")]
        )

    assert response.message.content == "Hello! How can I assist you today?"
    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    _assert_inference_span(spans[0])


def test_completion_inference(
    span_exporter,
    instrument_llama_index_with_content,
) -> None:
    response = MockLLM(max_tokens=3).complete("Hello!")

    assert response.text == "text text text"
    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]
    assert span.name == "text_completion unknown"
    assert span.attributes is not None
    assert span.attributes[GenAIAttributes.GEN_AI_OPERATION_NAME] == (
        GenAIAttributes.GenAiOperationNameValues.TEXT_COMPLETION.value
    )
    assert isinstance(
        span.attributes[GenAIAttributes.GEN_AI_OPERATION_NAME], str
    )
    assert json.loads(
        span.attributes[GenAIAttributes.GEN_AI_INPUT_MESSAGES]
    ) == [
        {
            "role": "user",
            "parts": [{"content": "Hello!", "type": "text"}],
        }
    ]
    assert json.loads(
        span.attributes[GenAIAttributes.GEN_AI_OUTPUT_MESSAGES]
    ) == [
        {
            "role": "assistant",
            "parts": [{"content": "text text text", "type": "text"}],
            "finish_reason": "",
        }
    ]


@pytest.mark.asyncio
async def test_async_completion_inference(
    span_exporter,
    instrument_llama_index_with_content,
) -> None:
    response = await MockLLM(max_tokens=3).acomplete("Hello!")

    assert response.text == "text text text"
    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    assert spans[0].attributes is not None
    assert spans[0].attributes[GenAIAttributes.GEN_AI_OPERATION_NAME] == (
        GenAIAttributes.GenAiOperationNameValues.TEXT_COMPLETION.value
    )
    assert json.loads(
        spans[0].attributes[GenAIAttributes.GEN_AI_OUTPUT_MESSAGES]
    ) == [
        {
            "role": "assistant",
            "parts": [{"content": "text text text", "type": "text"}],
            "finish_reason": "",
        }
    ]


def test_chat_inference_error(
    span_exporter,
    instrument_llama_index,
    openai_llm,
    vcr,
) -> None:
    with vcr.use_cassette("inference_error.yaml"):
        with pytest.raises(AuthenticationError) as exc_info:
            openai_llm.chat([ChatMessage(role="user", content="Hello!")])

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]
    assert span.status.status_code is StatusCode.ERROR
    assert span.attributes is not None
    error_type = span.attributes[error_attributes.ERROR_TYPE]
    assert error_type in {
        type(exc_info.value).__qualname__,
        f"{type(exc_info.value).__module__}.{type(exc_info.value).__qualname__}",
    }
    assert isinstance(error_type, str)
