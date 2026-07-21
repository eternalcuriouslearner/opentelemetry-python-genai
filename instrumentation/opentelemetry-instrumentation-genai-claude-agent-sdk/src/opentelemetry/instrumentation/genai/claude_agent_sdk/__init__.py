# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""
OpenTelemetry Claude Agent SDK Instrumentation
===============================================

Instrumentation for the `Claude Agent SDK
<https://github.com/anthropics/claude-agent-sdk-python>`_.

Usage
-----

.. code-block:: python

    from opentelemetry.instrumentation.genai.claude_agent_sdk import ClaudeAgentSDKInstrumentor
    from claude_agent_sdk import ClaudeAgentOptions, AgentDefinition, AssistantMessage, TextBlock, query

    # Enable instrumentation
    ClaudeAgentSDKInstrumentor().instrument()

    # Use Claude Agent SDK normally
    import anyio

    async def main():
        options = ClaudeAgentOptions(
            agents={
                "assistant": AgentDefinition(
                    description="A helpful assistant",
                    prompt="You are a helpful assistant.",
                    tools=["Read"],
                    model="sonnet",
                ),
            },
        )

        async for message in query(prompt="Hello!", options=options):
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

import importlib
from typing import Any, Collection

from wrapt import wrap_function_wrapper

from opentelemetry.instrumentation.genai.claude_agent_sdk.package import (
    _instruments,
)
from opentelemetry.instrumentation.instrumentor import BaseInstrumentor
from opentelemetry.instrumentation.utils import unwrap
from opentelemetry.util.genai.handler import TelemetryHandler

from .patch import (
    client_connect,
    client_disconnect,
    client_query,
    client_receive_response,
    query,
)


class ClaudeAgentSDKInstrumentor(BaseInstrumentor):
    """An instrumentor for the Claude Agent SDK.

    This instrumentor traces agent turns, subagents, and tools reconstructed
    from Claude Agent SDK messages.
    """

    def __init__(self) -> None:
        """Initialize the instrumentor without an active telemetry handler."""

        super().__init__()
        self._handler: TelemetryHandler | None = None

    # pylint: disable=no-self-use
    def instrumentation_dependencies(self) -> Collection[str]:
        """Return the Claude Agent SDK versions supported by this package.

        Returns:
            The package dependency constraints checked before instrumentation.
        """

        return _instruments

    def _instrument(self, **kwargs: Any) -> None:
        """Enable Claude Agent SDK instrumentation.

        Args:
            **kwargs: Instrumentation configuration. Supported values include
                ``tracer_provider``, ``meter_provider``, ``logger_provider``,
                and ``completion_hook``.
        """

        handler = TelemetryHandler(
            tracer_provider=kwargs.get("tracer_provider"),
            meter_provider=kwargs.get("meter_provider"),
            logger_provider=kwargs.get("logger_provider"),
            completion_hook=kwargs.get("completion_hook"),
        )
        self._handler = handler

        wrap_function_wrapper(
            "claude_agent_sdk.query",
            "query",
            query(handler),
        )
        wrap_function_wrapper(
            "claude_agent_sdk.client",
            "ClaudeSDKClient.connect",
            client_connect(handler),
        )
        wrap_function_wrapper(
            "claude_agent_sdk.client",
            "ClaudeSDKClient.query",
            client_query(handler),
        )
        wrap_function_wrapper(
            "claude_agent_sdk.client",
            "ClaudeSDKClient.receive_response",
            client_receive_response(),
        )
        wrap_function_wrapper(
            "claude_agent_sdk.client",
            "ClaudeSDKClient.disconnect",
            client_disconnect(),
        )

        # The SDK re-exports query from its package root. Point that export at
        # the wrapped function so both supported import styles are traced.
        claude_agent_sdk = importlib.import_module("claude_agent_sdk")
        query_module = importlib.import_module("claude_agent_sdk.query")
        setattr(claude_agent_sdk, "query", query_module.query)

    def _uninstrument(self, **kwargs: Any) -> None:
        """Disable Claude Agent SDK instrumentation.

        This removes all patches applied during instrumentation.

        Args:
            **kwargs: Reserved by ``BaseInstrumentor`` for uninstrumentation
                options. This instrumentor does not currently use them.
        """
        claude_agent_sdk = importlib.import_module("claude_agent_sdk")
        query_module = importlib.import_module("claude_agent_sdk.query")
        client_module = importlib.import_module("claude_agent_sdk.client")

        unwrap(query_module, "query")
        unwrap(client_module.ClaudeSDKClient, "connect")
        unwrap(client_module.ClaudeSDKClient, "query")
        unwrap(client_module.ClaudeSDKClient, "receive_response")
        unwrap(client_module.ClaudeSDKClient, "disconnect")
        setattr(claude_agent_sdk, "query", query_module.query)
        self._handler = None
