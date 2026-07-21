# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAI,
)
from opentelemetry.util.genai.handler import TelemetryHandler
from opentelemetry.util.genai.invocation import (
    AgentInvocation,
    ToolInvocation,
)
from opentelemetry.util.genai.types import (
    InputMessage,
    MessagePart,
    OutputMessage,
    Reasoning,
    Text,
    ToolCallRequest,
    ToolCallResponse,
)

ANTHROPIC = GenAI.GenAiProviderNameValues.ANTHROPIC.value
AGENT_LAUNCHER_NAMES = frozenset(("agent", "task"))


class ClaudeAgentResultError(RuntimeError):
    """A Claude Agent SDK result reported an unsuccessful turn."""


class ClaudeToolResultError(RuntimeError):
    """A Claude Agent SDK tool result reported an unsuccessful execution."""


@dataclass
class _ToolState:
    """Associate an open tool invocation with its parent launcher ID."""

    invocation: ToolInvocation
    parent_tool_use_id: str | None


@dataclass
class _SubagentState:
    """Associate an open subagent invocation with its parent launcher ID."""

    invocation: AgentInvocation
    parent_tool_use_id: str | None


class TurnState:
    """Reconstruct one serialized Claude agent turn from streamed messages.

    A turn owns one root agent invocation and tracks open tool and subagent
    invocations by Claude ``tool_use_id`` until matching result messages close
    them.
    """

    def __init__(
        self,
        handler: TelemetryHandler,
        *,
        prompt: Any = None,
        options: Any = None,
    ) -> None:
        """Start telemetry for one Claude agent turn.

        Args:
            handler: Telemetry handler used to create agent and tool
                invocations.
            prompt: User prompt for the turn. String prompts are captured when
                content capture is enabled.
            options: Claude agent options used to obtain the requested model
                and optional system prompt.
        """

        self._handler = handler
        self._capture_content = handler.should_capture_content()
        self._finished = False
        self._tools: dict[str, _ToolState] = {}
        self._subagents: dict[str, _SubagentState] = {}

        self.root = handler.invoke_remote_agent(
            ANTHROPIC,
            request_model=_string_field(options, "model"),
            agent_name="Claude",
        )
        if self._capture_content and isinstance(prompt, str):
            self.root.input_messages = [
                InputMessage(role="user", parts=[Text(content=prompt)])
            ]
        system_prompt = _field(options, "system_prompt")
        if self._capture_content and isinstance(system_prompt, str):
            self.root.system_instruction = [Text(content=system_prompt)]

    @property
    def finished(self) -> bool:
        """Report whether this turn has already been finalized.

        Returns:
            ``True`` after the root invocation has been stopped or failed.
        """

        return self._finished

    def on_message(self, message: Any) -> None:
        """Process one message from the Claude Agent SDK stream.

        System, assistant, user, and result messages update or finalize the
        reconstructed invocation hierarchy. Messages received after the turn
        finishes are ignored.

        Args:
            message: Parsed Claude SDK message or equivalent mapping.
        """

        if self._finished:
            return

        message_type = _message_type(message)
        if message_type == "system":
            self._on_system(message)
        elif message_type == "assistant":
            self._on_assistant(message)
        elif message_type == "user":
            self._on_user(message)
        elif message_type == "result":
            self._on_result(message)

    def stop(self) -> None:
        """Finalize the turn and all open descendants successfully."""

        self._finish(None)

    def fail(self, error: BaseException) -> None:
        """Finalize the turn and all open descendants as failed.

        Args:
            error: Failure recorded on each still-open invocation.
        """

        self._finish(error)

    def _on_system(self, message: Any) -> None:
        """Apply identity and lifecycle metadata from a system message.

        Initialization messages set model and conversation identity on the
        applicable agent. ``task_started`` messages enrich a known subagent.

        Args:
            message: Claude system message or equivalent mapping.
        """

        data = _message_data(message)
        subtype = _string_field(message, "subtype") or _string_field(
            data, "subtype"
        )
        parent_tool_use_id = _string_field(data, "parent_tool_use_id")

        if subtype == "init":
            target = self._agent_for_parent(parent_tool_use_id)
            _set_agent_identity(target, data)
            return

        if subtype == "task_started":
            tool_use_id = _string_field(data, "tool_use_id")
            subagent = self._subagents.get(tool_use_id or "")
            if subagent is not None:
                _set_agent_identity(subagent.invocation, data)
                description = _string_field(data, "description")
                if description:
                    subagent.invocation.agent_description = description

    def _on_assistant(self, message: Any) -> None:
        """Process assistant content and start requested invocations.

        Text and reasoning blocks become output message parts. Tool-use blocks
        start either an ordinary tool invocation or an agent invocation for
        Claude's ``Agent`` and ``Task`` launchers.

        Args:
            message: Claude assistant message or equivalent mapping.
        """

        parent_tool_use_id = _string_field(message, "parent_tool_use_id")
        target = self._agent_for_parent(parent_tool_use_id)
        _set_agent_identity(target, message)

        parts: list[MessagePart] = []
        for block in _content_blocks(message):
            block_type = _block_type(block)
            if block_type == "text":
                text = _string_field(block, "text")
                if text is not None:
                    parts.append(Text(content=text))
            elif block_type == "thinking":
                thinking = _string_field(block, "thinking")
                if thinking is not None:
                    parts.append(Reasoning(content=thinking))
            elif block_type == "tool_use":
                tool_use_id = _string_field(block, "id")
                name = _string_field(block, "name") or ""
                arguments = _field(block, "input")
                parts.append(
                    ToolCallRequest(
                        arguments=arguments,
                        name=name,
                        id=tool_use_id,
                    )
                )
                if tool_use_id:
                    if name.lower() in AGENT_LAUNCHER_NAMES:
                        self._start_subagent(
                            name,
                            tool_use_id,
                            arguments,
                            parent_tool_use_id,
                            target,
                        )
                    else:
                        self._start_tool(
                            name,
                            tool_use_id,
                            arguments,
                            parent_tool_use_id,
                        )

        if self._capture_content and parts:
            target.output_messages.append(
                OutputMessage(
                    role="assistant",
                    parts=parts,
                    finish_reason=_string_field(message, "stop_reason") or "",
                )
            )

    def _on_user(self, message: Any) -> None:
        """Process user content and finish matching tool calls.

        Claude represents tool outputs as user-message ``tool_result`` blocks.
        Their ``tool_use_id`` closes the corresponding tool or subagent.

        Args:
            message: Claude user message or equivalent mapping.
        """

        parent_tool_use_id = _string_field(message, "parent_tool_use_id")
        target = self._agent_for_parent(parent_tool_use_id)
        response_parts: list[MessagePart] = []

        for block in _content_blocks(message):
            if _block_type(block) != "tool_result":
                text = _string_field(block, "text")
                if text is not None:
                    response_parts.append(Text(content=text))
                continue

            tool_use_id = _string_field(block, "tool_use_id")
            result = _field(block, "content")
            is_error = bool(_field(block, "is_error"))
            response_parts.append(
                ToolCallResponse(response=result, id=tool_use_id)
            )
            if not tool_use_id:
                continue
            if tool_use_id in self._subagents:
                self._finish_subagent(
                    tool_use_id,
                    result=result,
                    usage=_field(message, "tool_use_result"),
                    is_error=is_error,
                )
            else:
                self._finish_tool(
                    tool_use_id, result=result, is_error=is_error
                )

        if self._capture_content and response_parts:
            target.input_messages.append(
                InputMessage(role="user", parts=response_parts)
            )

    def _on_result(self, message: Any) -> None:
        """Apply final usage and result data, then finish an agent invocation.

        A result with ``parent_tool_use_id`` finishes that subagent. A root
        result finishes the complete turn.

        Args:
            message: Claude result message or equivalent mapping.
        """

        parent_tool_use_id = _string_field(message, "parent_tool_use_id")
        target = self._agent_for_parent(parent_tool_use_id)
        _set_agent_identity(target, message)
        _set_usage(target, _field(message, "usage"))
        _set_model_usage(target, message)

        result = _field(message, "result")
        if (
            self._capture_content
            and isinstance(result, str)
            and not _contains_text_output(target, result)
        ):
            target.output_messages.append(
                OutputMessage(
                    role="assistant",
                    parts=[Text(content=result)],
                    finish_reason="error"
                    if _result_is_error(message)
                    else "stop",
                )
            )

        stop_reason = _string_field(message, "stop_reason")
        target.finish_reasons = [
            stop_reason or ("error" if _result_is_error(message) else "stop")
        ]

        if parent_tool_use_id and parent_tool_use_id in self._subagents:
            self._finish_subagent(
                parent_tool_use_id,
                result=result,
                usage=_field(message, "usage"),
                is_error=_result_is_error(message),
            )
            return

        if _result_is_error(message):
            self.fail(ClaudeAgentResultError(_result_error_message(message)))
        else:
            self.stop()

    def _start_tool(
        self,
        name: str,
        tool_use_id: str,
        arguments: Any,
        parent_tool_use_id: str | None,
    ) -> None:
        """Start and index an ordinary tool invocation.

        Duplicate tool-use IDs are ignored because each recorded tool call
        must own exactly one invocation.

        Args:
            name: Tool name reported by Claude.
            tool_use_id: Identifier used to match the eventual tool result.
            arguments: Arguments supplied to the tool.
            parent_tool_use_id: Launcher ID of the containing subagent, if
                this tool belongs to one.
        """

        if tool_use_id in self._tools:
            return
        invocation = self._handler.tool(
            name,
            tool_call_id=tool_use_id,
            tool_type="function",
        )
        if invocation.should_capture_content_on_span:
            invocation.arguments = arguments
        self._tools[tool_use_id] = _ToolState(
            invocation=invocation,
            parent_tool_use_id=parent_tool_use_id,
        )

    def _start_subagent(
        self,
        launcher_name: str,
        tool_use_id: str,
        arguments: Any,
        parent_tool_use_id: str | None,
        parent: AgentInvocation,
    ) -> None:
        """Start and index an agent launched by ``Agent`` or ``Task``.

        Args:
            launcher_name: Claude launcher tool name used as the fallback
                agent name.
            tool_use_id: Launcher call ID used to correlate subagent messages
                and results.
            arguments: Launcher arguments containing agent metadata and its
                prompt.
            parent_tool_use_id: Launcher ID of an enclosing subagent, if any.
            parent: Agent invocation whose model is inherited by the new
                subagent.
        """

        if tool_use_id in self._subagents:
            return
        arguments_dict = (
            cast("dict[str, Any]", arguments)
            if isinstance(arguments, dict)
            else {}
        )
        agent_name = (
            _string_field(arguments_dict, "subagent_type")
            or _string_field(arguments_dict, "name")
            or launcher_name
        )
        invocation = self._handler.invoke_local_agent(
            request_model=parent.request_model,
            agent_name=agent_name,
        )
        invocation.provider = ANTHROPIC
        invocation.agent_description = _string_field(
            arguments_dict, "description"
        )
        prompt = _string_field(arguments_dict, "prompt")
        if self._capture_content and prompt:
            invocation.input_messages = [
                InputMessage(role="user", parts=[Text(content=prompt)])
            ]
        self._subagents[tool_use_id] = _SubagentState(
            invocation=invocation,
            parent_tool_use_id=parent_tool_use_id,
        )

    def _finish_tool(
        self, tool_use_id: str, *, result: Any, is_error: bool
    ) -> None:
        """Finish the ordinary tool matched by a tool result.

        Args:
            tool_use_id: Identifier of the tool invocation to finish.
            result: Value returned by the tool.
            is_error: Whether Claude marked the tool result as unsuccessful.
        """

        state = self._tools.pop(tool_use_id, None)
        if state is None:
            return
        if state.invocation.should_capture_content_on_span:
            state.invocation.tool_result = result
        if is_error:
            state.invocation.fail(
                ClaudeToolResultError(_result_text(result) or "Tool failed")
            )
        else:
            state.invocation.stop()

    def _finish_subagent(
        self,
        tool_use_id: str,
        *,
        result: Any,
        usage: Any,
        is_error: bool,
    ) -> None:
        """Finish the subagent matched by a launcher result.

        Args:
            tool_use_id: Launcher call ID identifying the subagent.
            result: Final output produced by the subagent.
            usage: Token usage reported directly or inside a launcher result.
            is_error: Whether Claude marked the subagent result as
                unsuccessful.
        """

        state = self._subagents.pop(tool_use_id, None)
        if state is None:
            return
        invocation = state.invocation
        usage_value: Any = usage
        if isinstance(usage, dict):
            usage_dict = cast("dict[str, Any]", usage)
            if isinstance(usage_dict.get("usage"), dict):
                usage_value = usage_dict["usage"]
        _set_usage(invocation, usage_value)
        if (
            self._capture_content
            and result is not None
            and not _contains_text_output(invocation, _result_text(result))
        ):
            invocation.output_messages.append(
                OutputMessage(
                    role="assistant",
                    parts=[Text(content=_result_text(result))],
                    finish_reason="error" if is_error else "stop",
                )
            )
        invocation.finish_reasons = ["error" if is_error else "stop"]
        if is_error:
            invocation.fail(
                ClaudeAgentResultError(
                    _result_text(result) or "Subagent failed"
                )
            )
        else:
            invocation.stop()

    def _agent_for_parent(
        self, parent_tool_use_id: str | None
    ) -> AgentInvocation:
        """Resolve the agent that owns a streamed message.

        Args:
            parent_tool_use_id: Launcher call ID attached to the message, or
                ``None`` for a root-agent message.

        Returns:
            The matching subagent invocation, falling back to the root agent.
        """

        if parent_tool_use_id:
            subagent = self._subagents.get(parent_tool_use_id)
            if subagent is not None:
                return subagent.invocation
        return self.root

    def _finish(self, error: BaseException | None) -> None:
        """Finalize the full invocation hierarchy exactly once.

        Open tools and subagents are closed in reverse insertion order before
        the root invocation, preserving the reconstructed stack lifecycle.

        Args:
            error: Failure applied to all open invocations, or ``None`` for
                successful completion.
        """

        if self._finished:
            return
        self._finished = True

        for tool_use_id in reversed(tuple(self._tools)):
            tool = self._tools.pop(tool_use_id)
            if error is None:
                tool.invocation.stop()
            else:
                tool.invocation.fail(error)
        for tool_use_id in reversed(tuple(self._subagents)):
            subagent = self._subagents.pop(tool_use_id)
            if error is None:
                subagent.invocation.stop()
            else:
                subagent.invocation.fail(error)
        if error is None:
            self.root.stop()
        else:
            self.root.fail(error)


def _message_type(message: Any) -> str | None:
    """Determine the normalized Claude message type.

    Args:
        message: Parsed SDK message or equivalent mapping.

    Returns:
        Lowercase message type, or ``None`` when it cannot be inferred.
    """

    message_type = _string_field(message, "type")
    if message_type:
        return message_type.lower()
    name = type(message).__name__
    if _field(message, "subtype") is not None and isinstance(
        _field(message, "data"), dict
    ):
        return "system"
    if name.endswith("Message"):
        return name[: -len("Message")].lower()
    return None


def _message_data(message: Any) -> Any:
    """Extract a system message's nested data payload when present.

    Args:
        message: Parsed SDK message or equivalent mapping.

    Returns:
        The nested data mapping, otherwise the original message.
    """

    data = _field(message, "data")
    return cast("dict[str, Any]", data) if isinstance(data, dict) else message


def _content_blocks(message: Any) -> list[Any]:
    """Normalize message content into a list of content blocks.

    Args:
        message: Parsed SDK message or equivalent mapping.

    Returns:
        Content blocks from the message. Plain string content becomes one
        synthetic text block, and missing content becomes an empty list.
    """

    content = _field(message, "content")
    if content is None:
        inner = _field(message, "message")
        content = _field(inner, "content")
    if isinstance(content, str):
        return [{"type": "text", "text": content}]
    return cast("list[Any]", content) if isinstance(content, list) else []


def _block_type(block: Any) -> str | None:
    """Determine the normalized type of a Claude content block.

    Args:
        block: Parsed SDK content block or equivalent mapping.

    Returns:
        One of the supported lowercase block types, or ``None`` when the type
        cannot be inferred.
    """

    block_type = _string_field(block, "type")
    if block_type:
        return block_type.lower()
    name = type(block).__name__
    if name.endswith("Block"):
        name = name[: -len("Block")]
    mapping = {
        "Text": "text",
        "Thinking": "thinking",
        "ToolUse": "tool_use",
        "ToolResult": "tool_result",
    }
    return mapping.get(name)


def _set_agent_identity(invocation: AgentInvocation, source: Any) -> None:
    """Copy conversation and model identity onto an agent invocation.

    Args:
        invocation: Agent invocation to update.
        source: SDK message or mapping that may contain ``session_id`` and
            ``model`` fields.
    """

    session_id = _string_field(source, "session_id")
    model = _string_field(source, "model")
    if session_id:
        invocation.conversation_id = session_id
    if model:
        invocation.request_model = model


def _set_usage(invocation: AgentInvocation, usage: Any) -> None:
    """Copy standard token counters onto an agent invocation.

    Args:
        invocation: Agent invocation to update.
        usage: Claude usage mapping. Non-mapping values are ignored.
    """

    if not isinstance(usage, dict):
        return
    usage_dict = cast("dict[str, Any]", usage)
    invocation.input_tokens = _integer_field(usage_dict, "input_tokens")
    invocation.output_tokens = _integer_field(usage_dict, "output_tokens")
    invocation.cache_creation_input_tokens = _integer_field(
        usage_dict, "cache_creation_input_tokens"
    )
    invocation.cache_read_input_tokens = _integer_field(
        usage_dict, "cache_read_input_tokens"
    )


def _set_model_usage(invocation: AgentInvocation, message: Any) -> None:
    """Apply aggregate model usage from a Claude result message.

    When multiple models are present, the model with the largest output token
    count is used as the invocation's request model.

    Args:
        invocation: Agent invocation to update.
        message: Result message containing ``model_usage`` or ``modelUsage``.
    """

    model_usage = _field(message, "model_usage") or _field(
        message, "modelUsage"
    )
    if not isinstance(model_usage, dict) or not model_usage:
        return
    model_usage_dict = cast("dict[str, Any]", model_usage)
    model, usage = max(
        model_usage_dict.items(),
        key=lambda item: _integer_field(item[1], "outputTokens") or 0,
    )
    invocation.request_model = model
    if not isinstance(usage, dict):
        return
    invocation.input_tokens = _integer_field(usage, "inputTokens")
    invocation.output_tokens = _integer_field(usage, "outputTokens")
    invocation.cache_creation_input_tokens = _integer_field(
        usage, "cacheCreationInputTokens"
    )
    invocation.cache_read_input_tokens = _integer_field(
        usage, "cacheReadInputTokens"
    )


def _result_is_error(message: Any) -> bool:
    """Determine whether a Claude result represents failure.

    Args:
        message: Result or tool-result message to inspect.

    Returns:
        ``True`` when ``is_error`` is set or the subtype is not ``success``.
    """

    if bool(_field(message, "is_error")):
        return True
    subtype = _string_field(message, "subtype")
    return bool(subtype and subtype != "success")


def _contains_text_output(invocation: AgentInvocation, value: str) -> bool:
    """Check whether an invocation already contains an exact text output.

    Args:
        invocation: Agent invocation whose output messages are searched.
        value: Exact text value to find.

    Returns:
        ``True`` when a matching text part is already present.
    """

    return any(
        isinstance(part, Text) and part.content == value
        for message in invocation.output_messages
        for part in message.parts
    )


def _result_error_message(message: Any) -> str:
    """Build a readable exception message from a failed result.

    Args:
        message: Failed Claude result message.

    Returns:
        Text from ``errors`` or ``result``, with a generic fallback.
    """

    errors = _field(message, "errors")
    if errors:
        return _result_text(errors)
    result = _field(message, "result")
    return _result_text(result) or "Claude agent invocation failed"


def _result_text(value: Any) -> str:
    """Normalize a result value into text.

    Args:
        value: Scalar result, content-block list, or ``None``.

    Returns:
        String result. List entries are joined with newline characters.
    """

    if isinstance(value, str):
        return value
    if isinstance(value, list):
        values: list[str] = []
        for item in cast("list[Any]", value):
            text = _string_field(item, "text")
            values.append(text if text is not None else str(item))
        return "\n".join(values)
    return "" if value is None else str(value)


def _field(value: Any, name: str) -> Any:
    """Read a field from either a mapping or an SDK object.

    Args:
        value: Mapping or object containing the field.
        name: Field or attribute name to read.

    Returns:
        Field value, or ``None`` when it is absent.
    """

    if isinstance(value, dict):
        return cast("dict[str, Any]", value).get(name)
    return getattr(value, name, None)


def _string_field(value: Any, name: str) -> str | None:
    """Read a field only when its value is a string.

    Args:
        value: Mapping or object containing the field.
        name: Field or attribute name to read.

    Returns:
        String field value, or ``None`` for absent and non-string values.
    """

    result = _field(value, name)
    return result if isinstance(result, str) else None


def _integer_field(value: Any, name: str) -> int | None:
    """Read a field only when its value is a non-boolean integer.

    Args:
        value: Mapping or object containing the field.
        name: Field or attribute name to read.

    Returns:
        Integer field value, or ``None`` for absent, boolean, and other values.
    """

    result = _field(value, name)
    return (
        result
        if isinstance(result, int) and not isinstance(result, bool)
        else None
    )


__all__ = ["TurnState"]
