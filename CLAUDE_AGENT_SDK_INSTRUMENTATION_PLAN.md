# Claude Agent SDK Instrumentation Plan

## Objective

Add functional instrumentation to `opentelemetry-instrumentation-genai-claude-agent-sdk` using the SDK patching and message-stream observation pattern from the donated OpenInference implementation.

The instrumentation will:

- Emit OpenTelemetry GenAI telemetry through `opentelemetry-util-genai`.
- Operate independently of Claude Code's native telemetry.
- Coexist with native `claude_code.*` spans without modifying or suppressing them.
- Avoid injecting Claude Agent SDK hooks. Instrumentation-created hooks would produce additional `claude_code.hook` spans that represent observability machinery rather than application behavior.
- Initially support serialized turns: `query()` followed by a fully drained `receive_response()` before the next query.

## Implementation steps

### 1. Port the instrumentation structure

- Use `open-telemetry/donation-openinference` as the behavioral reference.
- Implement in the existing `opentelemetry-instrumentation-genai-claude-agent-sdk` package.
- Use only the public `opentelemetry-util-genai` surface for spans, metrics, events, content capture, and completion hooks.
- Preserve instrumentation suppression and restore every patched method during `uninstrument()`.

### 2. Patch the Claude Agent SDK lifecycle

Patch:

- Module-level `claude_agent_sdk.query()` for one-shot calls.
- The package-level `claude_agent_sdk.query` re-export.
- `ClaudeSDKClient.connect()` to capture an initial prompt when supplied.
- `ClaudeSDKClient.query()` to begin and store a persistent-client turn.
- `ClaudeSDKClient.receive_response()` to observe and complete the turn.
- `ClaudeSDKClient.disconnect()` to close unfinished invocations and clear state.

Do not install or merge `PreToolUse`, `PostToolUse`, or other SDK hooks.

### 3. Add per-turn state

Maintain bounded state for the active serialized turn:

```python
TurnState(
    root_invocation,
    tools_by_id={},
    subagents_by_parent_tool_use_id={},
)
```

Use `tool_use_id` to match tool starts with results and `parent_tool_use_id` to associate subagent messages and nested tools with the correct subagent.

### 4. Process streamed messages

Observe each message before yielding it unchanged to the application:

- `SystemMessage(init)`: enrich the applicable agent invocation with session/conversation ID and model information.
- Ordinary `ToolUseBlock`: start an `execute_tool` invocation and index it by tool-use ID.
- `Agent` or `Task` `ToolUseBlock`: start an `invoke_agent` subagent invocation and index it by launcher tool-use ID.
- `ToolResultBlock`: stop or fail the matching tool or subagent invocation.
- Message with `parent_tool_use_id`: process it within the corresponding subagent lifecycle.
- `ResultMessage`: record output, usage, model, conversation, and error information, then complete the applicable agent invocation.

Under the preferred semantic model, do not emit a separate `execute_tool Agent/Task` span in addition to the subagent `invoke_agent` span.

### 5. Emit OTel GenAI semantic conventions

Produce this logical hierarchy:

```text
invoke_agent Claude
├── execute_tool Read
└── invoke_agent subagent
    └── execute_tool Bash
```

Map available SDK data to standard attributes, including:

- `gen_ai.operation.name`
- `gen_ai.provider.name`
- `gen_ai.agent.name`
- `gen_ai.conversation.id`
- Request/response model attributes
- Input, output, and cache token usage
- `gen_ai.tool.name`
- `gen_ai.tool.call.id`
- Tool arguments and results when content capture is enabled
- `error.type`

Do not add unconventional attributes when no applicable semantic convention exists. In particular, omit total cost unless a standard attribute supports it.

### 6. Finalize safely

On normal completion, failure, generator closure, or disconnect:

1. Close unfinished tool invocations.
2. Close unfinished subagent invocations.
3. Close the root agent invocation.
4. Clear all per-turn state.

Instrumentation failures must not alter SDK behavior, and exceptions from the SDK must be re-raised unchanged.

### 7. Preserve native Claude telemetry

- Do not enable, disable, or reconfigure Claude Code telemetry.
- Do not suppress or transform native spans.
- Do not attempt collector-side conversion.
- Allow OTel GenAI spans and native `claude_code.*` spans to be exported side by side.
- Document that one-shot calls can inherit the active application trace context when the CLI subprocess starts, while a persistent client generally retains the context active at `connect()`.

## Testing steps

Use two primary test tracks: cassette-backed behavioral tests and Weaver conformance tests. Add a separate opt-in integration test for alignment with native Claude telemetry.

### 1. Reuse donated OpenInference cassettes

- Donated repository: <https://github.com/open-telemetry/donation-openinference>
- Claude Agent SDK package: <https://github.com/open-telemetry/donation-openinference/tree/main/python/instrumentation/openinference-instrumentation-claude-agent-sdk>
- Behavioral tests: <https://github.com/open-telemetry/donation-openinference/blob/main/python/instrumentation/openinference-instrumentation-claude-agent-sdk/tests/test_instrumentor.py>
- Cassette transport fixture: <https://github.com/open-telemetry/donation-openinference/blob/main/python/instrumentation/openinference-instrumentation-claude-agent-sdk/tests/conftest.py>
- Cassette directory: <https://github.com/open-telemetry/donation-openinference/tree/main/python/instrumentation/openinference-instrumentation-claude-agent-sdk/tests/cassettes/test_instrumentor>

The donated cassette directory currently contains:

- `test_query_real_agent_span.yaml`
- `test_client_real_agent_span.yaml`
- `test_query_tool_spans_from_messages.yaml`
- `test_query_tool_fallback_when_hooks_unavailable.yaml`
- `test_query_task_subagent_spans.yaml`
- `test_client_tool_hooks_create_tool_spans.yaml`

Reuse procedure:

- First try to reuse the recorded cassettes and `cassette_transport` from `open-telemetry/donation-openinference`.
- Adapt the cassette transport to this repository's shared test fixtures where possible.
- Since Claude Agent SDK communicates with the CLI over a local pipe rather than HTTP, treat these as VCR-style cassette tests rather than assuming ordinary HTTP VCR interception will work.
- Record only scenarios missing from the donated repository.
- Scrub credentials and account-identifying data from every new cassette.
- Mark any AI-synthesized cassette with the repository-required re-recording TODO.

### 2. Cassette-backed behavioral scenarios

#### One-shot query

Expected hierarchy:

```text
application
└── invoke_agent Claude
```

Verify output, model, conversation ID, usage, status, error handling, and cleanup.

#### Persistent multi-turn conversation

Expected hierarchy:

```text
application
├── invoke_agent turn 1
└── invoke_agent turn 2
```

Verify:

- One completed `invoke_agent` span per turn.
- Both turns share the expected `gen_ai.conversation.id`.
- Turn spans are siblings rather than accidentally nested.
- The first turn ends before the second begins.

#### Multi-turn conversation with a subagent

Expected hierarchy:

```text
application
├── invoke_agent turn 1
└── invoke_agent turn 2
    └── invoke_agent subagent
```

Verify that the subagent belongs to the correct turn and is correlated using `parent_tool_use_id`.

#### Multi-turn conversation with a subagent and tool call

Expected hierarchy:

```text
application
└── invoke_agent turn
    └── invoke_agent subagent
        └── execute_tool Bash
```

Verify:

- Root, subagent, and tool parent/child lineage.
- Tool-use/result correlation by `tool_use_id`.
- Tool success and failure status.
- Arguments and results follow content-capture configuration.
- No additional `execute_tool Agent/Task` span under the preferred semantic model.

### 3. Weaver conformance scenarios

Add scenarios such as:

- `OneShotAgentScenario`
- `MultiTurnAgentScenario`
- `SubagentScenario`
- `SubagentToolCallingScenario`

Use the shared conformance runner and Weaver live check to validate:

- Semantic-convention attribute names and value types.
- Required and conditionally required attributes.
- Span names and span kinds.
- Exact `gen_ai.operation.name` counts with no undeclared operations.
- Metrics emitted through `opentelemetry-util-genai`.
- No unexpected semantic-convention violations.

Add scenario-specific assertions for trace IDs, parent span IDs, conversation IDs, tool-call IDs, and the expected Claude-specific hierarchy.

### 4. Native Claude telemetry alignment

Response cassettes cannot fully validate native spans because the Claude Code CLI exports them directly. Add an opt-in integration or documented manual test using a real CLI and a local OTLP receiver.

For one-shot calls, inspect the expected alignment:

```text
application
└── invoke_agent Claude
    └── claude_code.interaction
        ├── claude_code.llm_request
        └── claude_code.tool
```

For persistent clients, verify and document the connect-time propagation limitation: native turns may remain associated with the context active at `connect()`, while the GenAI instrumentation emits a separate `invoke_agent` span for each consumed turn.

Also verify:

- OTel GenAI and native Claude spans coexist in the same backend.
- Native spans are not suppressed or replaced.
- No instrumentation-induced `claude_code.hook` spans are emitted.
- Conversation/session identifiers provide correlation where direct parenting is unavailable.

### 5. Package lifecycle and failure tests

- Instrument/uninstrument restores every patched function and re-export.
- Instrumentation suppression bypasses all added telemetry.
- SDK exceptions are re-raised unchanged and mark the active invocation as failed.
- Stream closure and `disconnect()` close unfinished child and root invocations.
- Existing user options and hooks remain unchanged.
- Run oldest, latest, conformance, type-check, and pre-commit environments required by the repository.

## Deferred hardening

The initial implementation does not promise support for:

- Overlapping or concurrent `ClaudeSDKClient.query()` calls.
- Draining turns through `receive_messages()` instead of `receive_response()`.
- Parallel sibling tool calls.
- Abandoned turns that are never drained or disconnected.
- Fresh native CLI trace parenting for every turn of an already-connected persistent client.

These behaviors should be addressed in focused follow-up changes after the serialized-turn implementation is working and conformant.
