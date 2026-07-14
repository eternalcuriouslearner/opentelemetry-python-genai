# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""
OpenTelemetry Claude Agent SDK Instrumentation
===============================================

Instrumentation for the `Claude Agent SDK
<https://github.com/anthropics/claude-agent-sdk-python>`_.

The Claude Agent SDK runs an agent loop through the bundled Claude Code
CLI, so telemetry is emitted at the agent level: ``query()`` and each
``ClaudeSDKClient.receive_response()`` turn produce ``invoke_agent`` spans
carrying the prompt, assistant output messages, token usage, model, and
session id.

Usage
-----

.. code-block:: python

    from opentelemetry.instrumentation.genai.claude_agent_sdk import ClaudeAgentSDKInstrumentor
    from claude_agent_sdk import AssistantMessage, TextBlock, query

    # Enable instrumentation
    ClaudeAgentSDKInstrumentor().instrument()

    # Use Claude Agent SDK normally
    import anyio

    async def main():
        async for message in query(prompt="Hello!"):
            if isinstance(message, AssistantMessage):
                for block in message.content:
                    if isinstance(block, TextBlock):
                        print(block.text)

    anyio.run(main)

Configuration
-------------

Message content capture can be enabled by setting the environment variable:
``OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT=true``

API
---
"""

from __future__ import annotations

import importlib
from typing import Any, Collection

from wrapt import wrap_function_wrapper

from opentelemetry.instrumentation.genai.claude_agent_sdk.package import (
    _instruments,
)
from opentelemetry.instrumentation.genai.claude_agent_sdk.patch import (
    client_connect_wrapper,
    client_query_wrapper,
    client_receive_response_wrapper,
    query_wrapper,
)
from opentelemetry.instrumentation.instrumentor import BaseInstrumentor
from opentelemetry.instrumentation.utils import unwrap
from opentelemetry.util.genai.handler import TelemetryHandler


class ClaudeAgentSDKInstrumentor(BaseInstrumentor):
    """An instrumentor for the Claude Agent SDK.

    This instrumentor traces agent runs (``query()`` and
    ``ClaudeSDKClient`` response turns) as ``invoke_agent`` spans and
    optionally captures message content.
    """

    # pylint: disable=no-self-use
    def instrumentation_dependencies(self) -> Collection[str]:
        return _instruments

    def _instrument(self, **kwargs: Any) -> None:
        """Enable Claude Agent SDK instrumentation.

        Args:
            **kwargs: Optional arguments
                - tracer_provider: TracerProvider instance
                - meter_provider: MeterProvider instance
                - logger_provider: LoggerProvider instance
                - completion_hook: CompletionHook instance
        """
        handler = TelemetryHandler(
            tracer_provider=kwargs.get("tracer_provider"),
            meter_provider=kwargs.get("meter_provider"),
            logger_provider=kwargs.get("logger_provider"),
            completion_hook=kwargs.get("completion_hook"),
        )

        wrap_function_wrapper(
            "claude_agent_sdk.query",
            "query",
            query_wrapper(handler),
        )
        wrap_function_wrapper(
            "claude_agent_sdk.client",
            "ClaudeSDKClient.connect",
            client_connect_wrapper(handler),
        )
        wrap_function_wrapper(
            "claude_agent_sdk.client",
            "ClaudeSDKClient.query",
            client_query_wrapper(handler),
        )
        wrap_function_wrapper(
            "claude_agent_sdk.client",
            "ClaudeSDKClient.receive_response",
            client_receive_response_wrapper(handler),
        )
        self._sync_package_query_export()

    def _uninstrument(self, **kwargs: Any) -> None:
        """Disable Claude Agent SDK instrumentation."""
        query_module = importlib.import_module("claude_agent_sdk.query")
        unwrap(query_module, "query")
        client_module = importlib.import_module("claude_agent_sdk.client")
        client_class = client_module.ClaudeSDKClient
        unwrap(client_class, "connect")
        unwrap(client_class, "query")
        unwrap(client_class, "receive_response")
        self._sync_package_query_export()

    @staticmethod
    def _sync_package_query_export() -> None:
        """Point the package-level ``query`` re-export at the module attribute.

        The ``claude_agent_sdk`` package re-exports ``query`` from its
        ``claude_agent_sdk.query`` submodule at import time, so patching the
        submodule attribute alone would leave
        ``from claude_agent_sdk import query`` resolving to the unpatched
        function (and vice versa on uninstrument).
        """
        package = importlib.import_module("claude_agent_sdk")
        query_module = importlib.import_module("claude_agent_sdk.query")
        setattr(package, "query", query_module.query)
