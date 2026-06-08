# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any

from opentelemetry._logs import Logger
from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAI,
)
from opentelemetry.trace import SpanKind, Tracer
from opentelemetry.util.genai._agent_invocation import (
    _get_agent_base_attributes,
    _get_agent_common_attributes,
    _get_agent_metric_attributes,
)
from opentelemetry.util.genai._invocation import (
    Error,
    GenAIInvocation,
    get_content_attributes,
)
from opentelemetry.util.genai.completion_hook import CompletionHook
from opentelemetry.util.genai.metrics import InvocationMetricsRecorder
from opentelemetry.util.genai.types import MessagePart


class CreateAgentInvocation(GenAIInvocation):
    """Represents a single create_agent client span.

    Use handler.create_agent() rather than constructing this directly.

    Reference: https://github.com/open-telemetry/semantic-conventions/blob/main/docs/gen-ai/gen-ai-agent-spans.md#create-agent-client-span
    """

    def __init__(
        self,
        tracer: Tracer,
        metrics_recorder: InvocationMetricsRecorder,
        logger: Logger,
        completion_hook: CompletionHook,
        provider: str,
        *,
        request_model: str | None = None,
        server_address: str | None = None,
        server_port: int | None = None,
        agent_name: str | None = None,
    ) -> None:
        """Use handler.create_agent() instead of calling this directly."""
        _operation_name = GenAI.GenAiOperationNameValues.CREATE_AGENT.value
        super().__init__(
            tracer,
            metrics_recorder,
            logger,
            completion_hook,
            operation_name=_operation_name,
            span_name=f"{_operation_name} {agent_name}"
            if agent_name
            else _operation_name,
            span_kind=SpanKind.CLIENT,
        )
        self.provider = provider
        self.request_model = request_model
        self.server_address = server_address
        self.server_port = server_port

        self.agent_name: str | None = agent_name
        self.agent_id: str | None = None
        self.agent_description: str | None = None
        self.agent_version: str | None = None
        self.system_instruction: list[MessagePart] = []

        self._start(self._get_base_attributes())

    def _get_base_attributes(self) -> dict[str, Any]:
        """Return sampling-relevant attributes available at span creation time."""
        return _get_agent_base_attributes(
            operation_name=self._operation_name,
            provider=self.provider,
            request_model=self.request_model,
            agent_name=self.agent_name,
            server_address=self.server_address,
            server_port=self.server_port,
        )

    def _get_common_attributes(self) -> dict[str, Any]:
        return _get_agent_common_attributes(
            operation_name=self._operation_name,
            provider=self.provider,
            request_model=self.request_model,
            server_address=self.server_address,
            server_port=self.server_port,
            agent_name=self.agent_name,
            agent_id=self.agent_id,
            agent_description=self.agent_description,
            agent_version=self.agent_version,
        )

    def _get_content_attributes_for_span(self) -> dict[str, Any]:
        return get_content_attributes(
            input_messages=[],
            output_messages=[],
            system_instruction=self.system_instruction,
            tool_definitions=None,
            for_span=True,
        )

    def _get_metric_attributes(self) -> dict[str, Any]:
        return _get_agent_metric_attributes(
            operation_name=self._operation_name,
            provider=self.provider,
            request_model=self.request_model,
            server_address=self.server_address,
            server_port=self.server_port,
            metric_attributes=self.metric_attributes,
        )

    def _apply_finish(self, error: Error | None = None) -> None:
        if error is not None:
            self._apply_error_attributes(error)

        attributes: dict[str, Any] = {}
        attributes.update(self._get_common_attributes())
        attributes.update(self._get_content_attributes_for_span())
        attributes.update(self.attributes)
        self.span.set_attributes(attributes)
        self._call_completion_hook(system_instruction=self.system_instruction)
        self._metrics_recorder.record(self)
