OpenTelemetry Claude Agent SDK Instrumentation
==============================================

|pypi|

.. |pypi| image:: https://badge.fury.io/py/opentelemetry-instrumentation-genai-claude-agent-sdk.svg
   :target: https://pypi.org/project/opentelemetry-instrumentation-genai-claude-agent-sdk/

This library traces agent runs made with the
`Claude Agent SDK <https://github.com/anthropics/claude-agent-sdk-python>`_,
following the OpenTelemetry GenAI semantic conventions:

* ``query()`` and each ``ClaudeSDKClient.receive_response()`` turn become
  ``invoke_agent`` spans carrying the prompt, assistant output messages,
  token usage, model, and session id (``gen_ai.conversation.id``).
* Tool executions become nested ``execute_tool`` spans.
* Subagent runs (e.g. via the Agent/Task tool) become nested
  ``invoke_agent`` spans under the spawning tool's span, named from the
  subagent type.

Spans are derived from the SDK's streamed messages. When the model issues
parallel tool calls in a single turn, sibling tool spans may be parented
under each other rather than side by side; attributes remain correct.

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


Configuration
-------------

Capture Message Content
***********************

By default, prompts and completions are not captured. To capture message content, set the
environment variable ``OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT`` to one of
``NO_CONTENT``, ``SPAN_ONLY``, ``EVENT_ONLY``, or ``SPAN_AND_EVENT``:

::

    export OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT=SPAN_AND_EVENT


Correlate with Claude Code's native traces
******************************************

The Claude Code CLI that the SDK drives has its own built-in (beta)
OpenTelemetry tracing (``claude_code.*`` spans for model requests, tool
executions, and hooks), exported directly from the CLI process. This
instrumentation neither requires nor conflicts with it: the SDK propagates
W3C trace context into the CLI subprocess, so when native tracing is
enabled the CLI's spans join the same trace, alongside the semantic
convention spans emitted here (for ``query()`` runs, nested under the
``invoke_agent`` span). See the `Agent SDK observability guide
<https://code.claude.com/docs/en/agent-sdk/observability>`_ for the CLI
telemetry configuration.

References
----------

* `OpenTelemetry Project <https://opentelemetry.io/>`_
* `OpenTelemetry GenAI semantic conventions <https://opentelemetry.io/docs/specs/semconv/gen-ai/>`_
* `Claude Agent SDK (Python) <https://github.com/anthropics/claude-agent-sdk-python>`_
* `Anthropic Documentation <https://docs.anthropic.com/>`_
