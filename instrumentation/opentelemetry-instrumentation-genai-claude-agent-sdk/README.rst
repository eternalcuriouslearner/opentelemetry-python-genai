OpenTelemetry Claude Agent SDK Instrumentation
==============================================

|pypi|

.. |pypi| image:: https://badge.fury.io/py/opentelemetry-instrumentation-genai-claude-agent-sdk.svg
   :target: https://pypi.org/project/opentelemetry-instrumentation-genai-claude-agent-sdk/

This library traces agent operations performed through the
`Claude Agent SDK <https://github.com/anthropics/claude-agent-sdk-python>`_.

It emits OpenTelemetry GenAI semantic-convention spans for one-shot
``query()`` calls and persistent ``ClaudeSDKClient`` turns. Claude Code's own
telemetry is neither disabled nor modified; both telemetry sources may be
exported side by side.

Installation
------------

::

    pip install opentelemetry-instrumentation-genai-claude-agent-sdk

If you don't have a Claude Agent SDK application yet, try our `examples <examples>`_
which only need a valid Anthropic API key.

Check out the `zero-code example <examples/zero-code>`_ for a quick start.

Usage
-----

This section describes how to set up Claude Agent SDK instrumentation if you're setting OpenTelemetry up manually.
Check out the `manual example <examples/manual>`_ for more details.

.. code-block:: python

    from opentelemetry.instrumentation.genai.claude_agent_sdk import ClaudeAgentSDKInstrumentor
    from claude_agent_sdk import ClaudeAgentOptions, AgentDefinition, AssistantMessage, TextBlock, query

    # Instrument Claude Agent SDK
    ClaudeAgentSDKInstrumentor().instrument()

    # Use Claude Agent SDK as normal
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

        async for message in query(
            prompt="Hello, Claude!",
            options=options,
        ):
            if isinstance(message, AssistantMessage):
                for block in message.content:
                    if isinstance(block, TextBlock):
                        print(block.text)

    anyio.run(main)


Telemetry model
---------------

The instrumentation reconstructs agent activity from the SDK message stream
without installing SDK hooks:

* A one-shot query or persistent client turn emits a CLIENT
  ``invoke_agent Claude`` span.
* An ``Agent`` or legacy ``Task`` launcher emits one INTERNAL
  ``invoke_agent <subagent>`` span. It does not also emit an
  ``execute_tool Agent`` launcher span.
* Ordinary tools such as ``Read``, ``Bash``, and MCP tools emit INTERNAL
  ``execute_tool <name>`` spans beneath the active agent.

For persistent clients, the initial implementation supports serialized turns:
in the same async task, call ``query()``, fully consume
``receive_response()``, and then send the next query. Direct draining through
``receive_messages()`` and overlapping queries are not yet correlated as
separate turns. Parallel sibling tool calls are also outside the initial
serialized message-stack model.

Claude Code native telemetry
****************************

This instrumentation does not register hooks, suppress ``claude_code.*``
spans, or alter Claude Code's telemetry configuration. For one-shot queries,
the ``invoke_agent`` span is current before the SDK starts its transport, so
SDK-supported W3C subprocess propagation can connect Claude Code's native
trace to the application trace. A persistent Claude subprocess is connected
once, so its native spans remain associated with the context active at
``connect()`` rather than a fresh context for every later turn.


Configuration
-------------

Capture Message Content
***********************

By default, prompts and completions are not captured. To capture message content, set the
environment variable ``OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT`` to one of
``NO_CONTENT``, ``SPAN_ONLY``, ``EVENT_ONLY``, or ``SPAN_AND_EVENT``:

::

    export OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT=SPAN_AND_EVENT


References
----------

* `OpenTelemetry Project <https://opentelemetry.io/>`_
* `OpenTelemetry GenAI semantic conventions <https://opentelemetry.io/docs/specs/semconv/gen-ai/>`_
* `Claude Agent SDK (Python) <https://github.com/anthropics/claude-agent-sdk-python>`_
* `Claude Agent SDK observability <https://code.claude.com/docs/en/agent-sdk/observability>`_
* `Donated OpenInference Claude Agent SDK instrumentation <https://github.com/open-telemetry/donation-openinference/tree/main/python/instrumentation/openinference-instrumentation-claude-agent-sdk>`_
* `Anthropic Documentation <https://docs.anthropic.com/>`_
