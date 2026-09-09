# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests finishing an invocation from a different execution context.

Frameworks that run each step of a workflow as its own ``asyncio`` task copy
the context when the task is created, so an invocation started in one step and
finished in another cannot reset the token it attached. Resetting it anyway
raises ``ValueError`` inside ``opentelemetry.context.detach``, which logs the
failure with a traceback on every such finish.
"""

from __future__ import annotations

import asyncio
import contextvars
import logging
import threading
from unittest import TestCase

from opentelemetry.context import get_current
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)
from opentelemetry.trace.status import StatusCode
from opentelemetry.util.genai.handler import TelemetryHandler
from opentelemetry.util.genai.invocation import AgentInvocation

_TIMEOUT = 5


class _DetachFailureHandler(logging.Handler):
    """Collect the records ``opentelemetry.context.detach`` logs on failure."""

    def __init__(self) -> None:
        super().__init__()
        self.records: list[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:
        if "Failed to detach context" in record.getMessage():
            self.records.append(record)


class CrossTaskFinishTest(TestCase):
    def setUp(self) -> None:
        self.span_exporter = InMemorySpanExporter()
        self.tracer_provider = TracerProvider()
        self.tracer_provider.add_span_processor(
            SimpleSpanProcessor(self.span_exporter)
        )
        self.handler = TelemetryHandler(tracer_provider=self.tracer_provider)
        self.detach_failures = _DetachFailureHandler()
        self.context_logger = logging.getLogger("opentelemetry.context")
        self.context_logger.addHandler(self.detach_failures)
        self.addCleanup(
            self.context_logger.removeHandler, self.detach_failures
        )

    async def _start_in_own_task(self) -> AgentInvocation:
        async def start() -> AgentInvocation:
            return self.handler.invoke_local_agent(
                request_model="test-model", agent_name="agent"
            )

        return await asyncio.create_task(start())

    def test_stop_from_another_task_does_not_log_detach_failure(self) -> None:
        async def scenario() -> None:
            invocation = await self._start_in_own_task()

            async def finish() -> None:
                invocation.stop()

            await asyncio.create_task(finish())

        asyncio.run(asyncio.wait_for(scenario(), _TIMEOUT))

        self.assertEqual(self.detach_failures.records, [])
        spans = self.span_exporter.get_finished_spans()
        self.assertEqual(len(spans), 1)
        self.assertEqual(spans[0].status.status_code, StatusCode.UNSET)

    def test_fail_from_another_task_does_not_log_detach_failure(self) -> None:
        async def scenario() -> None:
            invocation = await self._start_in_own_task()

            async def finish() -> None:
                invocation.fail(RuntimeError("boom"))

            await asyncio.create_task(finish())

        asyncio.run(asyncio.wait_for(scenario(), _TIMEOUT))

        self.assertEqual(self.detach_failures.records, [])
        spans = self.span_exporter.get_finished_spans()
        self.assertEqual(len(spans), 1)
        self.assertEqual(spans[0].status.status_code, StatusCode.ERROR)

    def test_stop_from_a_child_task_does_not_log_detach_failure(self) -> None:
        """A task inherits its parent's Context object, not the right to reset it.

        ``asyncio`` copies the context when a task is created, so the child sees
        the very same ``Context`` the parent attached; only the token's owner may
        reset it.
        """

        async def scenario() -> None:
            invocation = self.handler.invoke_local_agent(
                request_model="test-model", agent_name="agent"
            )
            self.assertIs(get_current(), invocation._span_context)

            async def finish() -> None:
                # The child sees the attached context, yet cannot reset it.
                self.assertIs(get_current(), invocation._span_context)
                invocation.stop()

            await asyncio.create_task(finish())

        asyncio.run(asyncio.wait_for(scenario(), _TIMEOUT))

        self.assertEqual(self.detach_failures.records, [])
        self.assertEqual(len(self.span_exporter.get_finished_spans()), 1)

    def test_stop_from_another_thread_does_not_log_detach_failure(
        self,
    ) -> None:
        def scenario() -> None:
            invocation = self.handler.invoke_local_agent(
                request_model="test-model", agent_name="agent"
            )
            worker = threading.Thread(target=invocation.stop)
            worker.start()
            worker.join(_TIMEOUT)
            self.assertFalse(worker.is_alive())

        # The finishing thread cannot reset the token this thread attached, so
        # the span stays on this context. Run in a throwaway copy to keep that
        # out of the contexts the rest of the suite runs in.
        contextvars.copy_context().run(scenario)

        self.assertEqual(self.detach_failures.records, [])
        self.assertEqual(len(self.span_exporter.get_finished_spans()), 1)

    def test_same_task_finish_restores_the_ambient_context(self) -> None:
        before = get_current()
        invocation = self.handler.invoke_local_agent(
            request_model="test-model", agent_name="agent"
        )
        self.assertIsNot(get_current(), before)

        invocation.stop()

        self.assertIs(get_current(), before)
        self.assertEqual(self.detach_failures.records, [])
