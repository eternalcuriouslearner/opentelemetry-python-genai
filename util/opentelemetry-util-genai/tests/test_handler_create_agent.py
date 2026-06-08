# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import os
from unittest import TestCase
from unittest.mock import patch

import pytest

from opentelemetry.instrumentation._semconv import (
    OTEL_SEMCONV_STABILITY_OPT_IN,
    _OpenTelemetrySemanticConventionStability,
)
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)
from opentelemetry.sdk.trace.sampling import Decision, SamplingResult
from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAI,
)
from opentelemetry.semconv.attributes import server_attributes
from opentelemetry.trace import INVALID_SPAN, SpanKind
from opentelemetry.trace.status import StatusCode
from opentelemetry.util.genai.environment_variables import (
    OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT,
)
from opentelemetry.util.genai.handler import TelemetryHandler
from opentelemetry.util.genai.invocation import CreateAgentInvocation
from opentelemetry.util.genai.types import Error, Text


class _CreateAgentTestBase(TestCase):
    def setUp(self) -> None:
        self.span_exporter = InMemorySpanExporter()
        self.tracer_provider = TracerProvider()
        self.tracer_provider.add_span_processor(
            SimpleSpanProcessor(self.span_exporter)
        )
        self.handler = TelemetryHandler(
            tracer_provider=self.tracer_provider,
        )

    def tearDown(self) -> None:
        _OpenTelemetrySemanticConventionStability._initialized = False

    def _get_finished_spans(self):
        return self.span_exporter.get_finished_spans()

    def _get_single_finished_span(self):
        spans = self._get_finished_spans()
        self.assertEqual(len(spans), 1)
        return spans[0]


class TelemetryHandlerCreateAgentTest(_CreateAgentTestBase):
    # ------------------------------------------------------------------
    # create_agent
    # ------------------------------------------------------------------

    def test_create_agent_creates_span(self) -> None:
        invocation = self.handler.create_agent("openai")
        self.assertIsNot(invocation.span, INVALID_SPAN)
        invocation.stop()

    def test_create_agent_span_name_with_agent_name(self) -> None:
        invocation = self.handler.create_agent(
            "openai", agent_name="Math Tutor"
        )
        invocation.stop()

        span = self._get_single_finished_span()
        self.assertEqual(span.name, "create_agent Math Tutor")

    def test_create_agent_span_name_without_agent_name(self) -> None:
        invocation = self.handler.create_agent("openai")
        invocation.stop()

        span = self._get_single_finished_span()
        self.assertEqual(span.name, "create_agent")

    def test_create_agent_span_kind_is_client(self) -> None:
        invocation = self.handler.create_agent("openai")
        invocation.stop()

        span = self._get_single_finished_span()
        self.assertEqual(span.kind, SpanKind.CLIENT)

    def test_create_agent_records_monotonic_start(self) -> None:
        with patch("timeit.default_timer", return_value=42.0):
            invocation = self.handler.create_agent("openai")
        self.assertEqual(invocation._monotonic_start_s, 42.0)
        invocation.stop()

    # ------------------------------------------------------------------
    # stop (required + conditionally required attributes)
    # ------------------------------------------------------------------

    def test_stop_sets_operation_name(self) -> None:
        invocation = self.handler.create_agent("openai")
        invocation.stop()

        span = self._get_single_finished_span()
        self.assertEqual(
            span.attributes[GenAI.GEN_AI_OPERATION_NAME], "create_agent"
        )

    def test_stop_sets_provider_name(self) -> None:
        invocation = self.handler.create_agent("openai")
        invocation.stop()

        span = self._get_single_finished_span()
        self.assertEqual(
            span.attributes[GenAI.GEN_AI_PROVIDER_NAME], "openai"
        )

    def test_stop_sets_request_model(self) -> None:
        invocation = self.handler.create_agent("openai", request_model="gpt-4")
        invocation.stop()

        span = self._get_single_finished_span()
        self.assertEqual(span.attributes[GenAI.GEN_AI_REQUEST_MODEL], "gpt-4")

    def test_stop_sets_server_address_and_port(self) -> None:
        invocation = self.handler.create_agent(
            "openai",
            server_address="api.openai.com",
            server_port=443,
        )
        invocation.stop()

        span = self._get_single_finished_span()
        attrs = span.attributes
        self.assertEqual(
            attrs[server_attributes.SERVER_ADDRESS], "api.openai.com"
        )
        self.assertEqual(attrs[server_attributes.SERVER_PORT], 443)

    def test_stop_sets_agent_attributes(self) -> None:
        invocation = self.handler.create_agent(
            "openai", agent_name="Math Tutor"
        )
        invocation.agent_id = "agent-123"
        invocation.agent_description = "A test agent"
        invocation.agent_version = "1.0.0"
        invocation.stop()

        span = self._get_single_finished_span()
        attrs = span.attributes
        self.assertEqual(attrs[GenAI.GEN_AI_AGENT_ID], "agent-123")
        self.assertEqual(attrs[GenAI.GEN_AI_AGENT_NAME], "Math Tutor")
        self.assertEqual(attrs[GenAI.GEN_AI_AGENT_DESCRIPTION], "A test agent")
        self.assertEqual(attrs[GenAI.GEN_AI_AGENT_VERSION], "1.0.0")

    # ------------------------------------------------------------------
    # stop (opt-in attributes set after construction)
    # ------------------------------------------------------------------

    @patch.dict(
        os.environ,
        {
            OTEL_SEMCONV_STABILITY_OPT_IN: "gen_ai_latest_experimental",
            OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT: "SPAN_ONLY",
        },
    )
    def test_stop_sets_system_instructions_when_content_capture_enabled(
        self,
    ) -> None:
        _OpenTelemetrySemanticConventionStability._initialized = False
        _OpenTelemetrySemanticConventionStability._initialize()

        invocation = self.handler.create_agent("openai")
        invocation.system_instruction = [Text(content="teach math")]
        invocation.stop()

        span = self._get_single_finished_span()
        raw = span.attributes[GenAI.GEN_AI_SYSTEM_INSTRUCTIONS]
        self.assertIsInstance(raw, str)
        self.assertEqual(
            json.loads(raw),
            [{"content": "teach math", "type": "text"}],
        )

    def test_stop_suppresses_system_instructions_when_content_capture_disabled(
        self,
    ) -> None:
        invocation = self.handler.create_agent("openai")
        invocation.system_instruction = [Text(content="teach math")]
        invocation.stop()

        span = self._get_single_finished_span()
        self.assertNotIn(GenAI.GEN_AI_SYSTEM_INSTRUCTIONS, span.attributes)

    def test_stop_sets_custom_attributes(self) -> None:
        invocation = self.handler.create_agent("openai")
        invocation.attributes["custom.key"] = "value"
        invocation.stop()

        span = self._get_single_finished_span()
        self.assertEqual(span.attributes["custom.key"], "value")

    def test_stop_omits_none_attributes(self) -> None:
        invocation = self.handler.create_agent("openai")
        invocation.stop()

        span = self._get_single_finished_span()
        attrs = span.attributes
        self.assertNotIn(GenAI.GEN_AI_REQUEST_MODEL, attrs)
        self.assertNotIn(server_attributes.SERVER_ADDRESS, attrs)
        self.assertNotIn(server_attributes.SERVER_PORT, attrs)
        self.assertNotIn(GenAI.GEN_AI_AGENT_ID, attrs)
        self.assertNotIn(GenAI.GEN_AI_AGENT_NAME, attrs)
        self.assertNotIn(GenAI.GEN_AI_AGENT_DESCRIPTION, attrs)
        self.assertNotIn(GenAI.GEN_AI_AGENT_VERSION, attrs)

    # ------------------------------------------------------------------
    # fail
    # ------------------------------------------------------------------

    def test_fail_sets_error_status(self) -> None:
        invocation = self.handler.create_agent("openai")
        invocation.fail(Error(message="timeout", type=TimeoutError))

        span = self._get_single_finished_span()
        self.assertEqual(span.status.status_code, StatusCode.ERROR)
        self.assertEqual(span.status.description, "timeout")

    def test_fail_sets_error_type_attribute(self) -> None:
        invocation = self.handler.create_agent("openai")
        invocation.fail(Error(message="bad", type=ConnectionError))

        span = self._get_single_finished_span()
        self.assertEqual(span.attributes["error.type"], "ConnectionError")

    def test_fail_sets_operation_name(self) -> None:
        invocation = self.handler.create_agent("openai")
        invocation.fail(Error(message="err", type=RuntimeError))

        span = self._get_single_finished_span()
        self.assertEqual(
            span.attributes[GenAI.GEN_AI_OPERATION_NAME], "create_agent"
        )

    def test_fail_with_exception_instance(self) -> None:
        invocation = self.handler.create_agent("openai")
        invocation.fail(ValueError("oops"))

        span = self._get_single_finished_span()
        self.assertEqual(span.status.status_code, StatusCode.ERROR)
        self.assertEqual(span.attributes["error.type"], "ValueError")


class TelemetryHandlerCreateAgentContextManagerTest(_CreateAgentTestBase):
    # ------------------------------------------------------------------
    # create_agent context manager
    # ------------------------------------------------------------------

    def test_context_manager_creates_and_ends_span(self) -> None:
        with self.handler.create_agent(
            "openai", agent_name="Math Tutor"
        ) as inv:
            self.assertIsNot(inv.span, INVALID_SPAN)

        span = self._get_single_finished_span()
        self.assertEqual(span.name, "create_agent Math Tutor")

    def test_context_manager_default_invocation(self) -> None:
        with self.handler.create_agent("openai") as inv:
            self.assertIsInstance(inv, CreateAgentInvocation)
            self.assertIsNone(inv.agent_name)
            self.assertEqual(inv._operation_name, "create_agent")

    def test_context_manager_success_has_unset_status(self) -> None:
        with self.handler.create_agent("openai"):
            pass

        span = self._get_single_finished_span()
        self.assertEqual(span.status.status_code, StatusCode.UNSET)

    def test_context_manager_reraises_exception(self) -> None:
        with pytest.raises(ValueError, match="create failed"):
            with self.handler.create_agent("openai"):
                raise ValueError("create failed")

    def test_context_manager_marks_error_on_exception(self) -> None:
        with pytest.raises(RuntimeError):
            with self.handler.create_agent("openai"):
                raise RuntimeError("agent service down")

        span = self._get_single_finished_span()
        self.assertEqual(span.status.status_code, StatusCode.ERROR)
        self.assertEqual(span.attributes["error.type"], "RuntimeError")

    def test_context_manager_sets_attributes_on_span(self) -> None:
        with self.handler.create_agent("openai") as inv:
            inv.agent_id = "agent-123"

        span = self._get_single_finished_span()
        attrs = span.attributes
        self.assertEqual(attrs[GenAI.GEN_AI_PROVIDER_NAME], "openai")
        self.assertEqual(attrs[GenAI.GEN_AI_AGENT_ID], "agent-123")


class TelemetryHandlerCreateAgentSamplingTest(_CreateAgentTestBase):
    def test_sampling_attributes_available_at_span_creation(self) -> None:
        """Sampling-relevant attributes must be present at start_span() time."""
        captured_attributes: dict = {}

        class AttributeCapturingSampler:  # pylint: disable=no-self-use
            def should_sample(
                self,
                parent_context,
                trace_id,
                name,
                kind=None,
                attributes=None,
                links=None,
            ):
                captured_attributes.update(attributes or {})
                return SamplingResult(Decision.RECORD_AND_SAMPLE, attributes)

            def get_description(self):
                return "AttributeCapturingSampler"

        sampler_provider = TracerProvider(sampler=AttributeCapturingSampler())
        sampler_provider.add_span_processor(
            SimpleSpanProcessor(self.span_exporter)
        )
        handler = TelemetryHandler(tracer_provider=sampler_provider)

        invocation = handler.create_agent(
            "openai",
            request_model="gpt-4",
            server_address="api.openai.com",
            server_port=443,
            agent_name="Math Tutor",
        )
        invocation.stop()

        self.assertEqual(
            captured_attributes[GenAI.GEN_AI_OPERATION_NAME], "create_agent"
        )
        self.assertEqual(
            captured_attributes[GenAI.GEN_AI_PROVIDER_NAME], "openai"
        )
        self.assertEqual(
            captured_attributes[GenAI.GEN_AI_REQUEST_MODEL], "gpt-4"
        )
        self.assertEqual(
            captured_attributes[server_attributes.SERVER_ADDRESS],
            "api.openai.com",
        )
        self.assertEqual(
            captured_attributes[server_attributes.SERVER_PORT], 443
        )
        self.assertEqual(
            captured_attributes[GenAI.GEN_AI_AGENT_NAME], "Math Tutor"
        )
