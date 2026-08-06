# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""Conformance scenario: LlamaIndex OpenAI chat inference."""

from __future__ import annotations

from typing import Any

from llama_index.core.llms import ChatMessage
from llama_index.llms.openai import OpenAI

from opentelemetry.instrumentation.genai.llama_index import (
    LlamaIndexInstrumentor,
)
from opentelemetry.sdk._logs import LoggerProvider
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.test_util_genai.conformance import Scenario
from opentelemetry.test_util_genai.instrumentor import instrument


class InferenceScenario(Scenario):
    expected_spans = {"chat": 1}
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
        with instrument(
            LlamaIndexInstrumentor(),
            tracer_provider=tracer_provider,
            logger_provider=logger_provider,
            meter_provider=meter_provider,
            content_capture="SPAN_ONLY",
        ):
            llm = OpenAI(model="gpt-4o-mini", temperature=0.1)
            with vcr.use_cassette("inference.yaml"):
                llm.chat([ChatMessage(role="user", content="Hello!")])
