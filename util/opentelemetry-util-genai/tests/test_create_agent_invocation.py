# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import unittest

from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)
from opentelemetry.trace import INVALID_SPAN
from opentelemetry.util.genai.handler import TelemetryHandler
from opentelemetry.util.genai.types import Text


class TestCreateAgentInvocation(unittest.TestCase):
    def setUp(self):
        self.span_exporter = InMemorySpanExporter()
        tracer_provider = TracerProvider()
        tracer_provider.add_span_processor(
            SimpleSpanProcessor(self.span_exporter)
        )
        self.handler = TelemetryHandler(tracer_provider=tracer_provider)

    def test_default_values(self):
        invocation = self.handler.create_agent("openai")
        invocation.stop()

        assert invocation._operation_name == "create_agent"
        assert invocation.provider == "openai"
        assert invocation.request_model is None
        assert invocation.server_address is None
        assert invocation.server_port is None
        assert invocation.agent_name is None
        assert invocation.agent_id is None
        assert invocation.agent_description is None
        assert invocation.agent_version is None
        assert not invocation.system_instruction
        assert invocation.span is not INVALID_SPAN
        assert not invocation.attributes

    def test_custom_agent_name(self):
        invocation = self.handler.create_agent(
            "openai", agent_name="Math Tutor"
        )
        invocation.stop()

        assert invocation.agent_name == "Math Tutor"

    def test_with_system_instruction(self):
        instruction = Text(content="Explain math step by step")
        invocation = self.handler.create_agent("openai")
        invocation.system_instruction = [instruction]
        invocation.stop()

        assert len(invocation.system_instruction) == 1
        assert invocation.system_instruction[0].content == (
            "Explain math step by step"
        )

    def test_default_system_instruction_lists_are_independent(self):
        inv1 = self.handler.create_agent("openai")
        inv2 = self.handler.create_agent("openai")
        inv1.system_instruction.append(Text(content="one"))

        assert len(inv2.system_instruction) == 0
        inv2.stop()
        inv1.stop()

    def test_default_attributes_are_independent(self):
        inv1 = self.handler.create_agent("openai")
        inv2 = self.handler.create_agent("openai")
        inv1.attributes["foo"] = "bar"

        assert "foo" not in inv2.attributes
        inv2.stop()
        inv1.stop()

    def test_full_construction(self):
        invocation = self.handler.create_agent(
            "openai",
            request_model="gpt-4",
            server_address="api.openai.com",
            server_port=443,
            agent_name="Math Tutor",
        )
        invocation.agent_id = "agent-123"
        invocation.agent_description = "A test agent"
        invocation.agent_version = "1.0.0"
        invocation.system_instruction = [Text(content="teach")]
        invocation.stop()

        assert invocation.provider == "openai"
        assert invocation.request_model == "gpt-4"
        assert invocation.server_address == "api.openai.com"
        assert invocation.server_port == 443
        assert invocation.agent_name == "Math Tutor"
        assert invocation.agent_id == "agent-123"
        assert invocation.agent_description == "A test agent"
        assert invocation.agent_version == "1.0.0"
        assert invocation.system_instruction[0].content == "teach"
