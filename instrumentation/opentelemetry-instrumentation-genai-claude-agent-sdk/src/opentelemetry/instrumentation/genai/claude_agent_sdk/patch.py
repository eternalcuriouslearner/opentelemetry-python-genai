# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from wrapt import wrap_function_wrapper

from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAI,
)
from opentelemetry.util.genai.handler import TelemetryHandler
from opentelemetry.util.genai.types import InputMessage, Text

_CLAUDE_AGENT_SDK_MODULE = "claude_agent_sdk"
_PATCHED = False
_ORIGINAL_QUERY: Any | None = None

HookCallback = Callable[[Any, str | None, Any], Awaitable[dict[str, Any]]]


@dataclass
class _QuerySession:
    telemetry_handler: TelemetryHandler
    invocation: Any | None = None
    closed: bool = False

    def start(self, input_data: Any, options: Any) -> None:
        if self.invocation is not None:
            return
        prompt = _get_attr(input_data, "prompt")
        self.invocation = self.telemetry_handler.invoke_local_agent(
            request_model=_get_attr(options, "model"),
            agent_name="ClaudeAgentSDK.query",
        )
        self.invocation.provider = (
            GenAI.GenAiProviderNameValues.ANTHROPIC.value
        )
        if prompt is not None:
            self.invocation.input_messages = [
                InputMessage(role="user", parts=[Text(str(prompt))])
            ]

    def stop(self) -> None:
        if self.closed:
            return
        self.closed = True
        if self.invocation is not None:
            self.invocation.stop()

    def fail(self, error: Exception) -> None:
        if self.closed:
            return
        self.closed = True
        if self.invocation is not None:
            self.invocation.fail(error)


def patch(telemetry_handler: TelemetryHandler) -> None:
    global _PATCHED, _ORIGINAL_QUERY  # pylint: disable=global-statement
    if _PATCHED:
        return
    module = importlib.import_module(_CLAUDE_AGENT_SDK_MODULE)
    _ORIGINAL_QUERY = getattr(module, "query")
    wrap_function_wrapper(
        _CLAUDE_AGENT_SDK_MODULE,
        "query",
        _wrap_query(telemetry_handler),
    )
    _PATCHED = True


def unpatch() -> None:
    global _PATCHED, _ORIGINAL_QUERY  # pylint: disable=global-statement
    if not _PATCHED:
        return
    module = importlib.import_module(_CLAUDE_AGENT_SDK_MODULE)
    if _ORIGINAL_QUERY is not None:
        setattr(module, "query", _ORIGINAL_QUERY)
    _ORIGINAL_QUERY = None
    _PATCHED = False


def _wrap_query(telemetry_handler: TelemetryHandler) -> Any:
    def wrapper(
        wrapped: Any,
        instance: Any,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> Any:
        module = importlib.import_module(_CLAUDE_AGENT_SDK_MODULE)
        options, args, kwargs = _get_or_create_options(module, args, kwargs)
        session = _QuerySession(telemetry_handler=telemetry_handler)
        _inject_invocation_hooks(module, options, session)
        result = wrapped(*args, **kwargs)
        return _QueryStream(result, session)

    return wrapper


def _get_or_create_options(
    module: Any, args: tuple[Any, ...], kwargs: dict[str, Any]
) -> tuple[Any, tuple[Any, ...], dict[str, Any]]:
    options = kwargs.get("options")
    if options is not None:
        return options, args, kwargs
    if len(args) > 1:
        if args[1] is not None:
            return args[1], args, kwargs
        options = module.ClaudeAgentOptions()
        updated_args = (args[0], options, *args[2:])
        return options, updated_args, kwargs
    options = module.ClaudeAgentOptions()
    kwargs["options"] = options
    return options, args, kwargs


def _inject_invocation_hooks(
    module: Any, options: Any, session: _QuerySession
) -> None:
    if not hasattr(options, "hooks"):
        return
    if options.hooks is None:
        options.hooks = {}
    for event_name, hook in (
        ("UserPromptSubmit", _user_prompt_submit_hook(session, options)),
        ("Stop", _stop_hook(session)),
    ):
        options.hooks.setdefault(event_name, [])
        options.hooks[event_name].insert(
            0,
            module.HookMatcher(matcher=None, hooks=[hook]),
        )


def _user_prompt_submit_hook(
    session: _QuerySession, options: Any
) -> HookCallback:
    async def hook(
        input_data: Any, tool_use_id: str | None, context: Any
    ) -> dict[str, Any]:
        session.start(input_data, options)
        return {}

    return hook


def _stop_hook(session: _QuerySession) -> HookCallback:
    async def hook(
        input_data: Any, tool_use_id: str | None, context: Any
    ) -> dict[str, Any]:
        session.stop()
        return {}

    return hook


class _QueryStream:
    def __init__(
        self, stream: AsyncIterator[Any], session: _QuerySession
    ) -> None:
        self._stream = stream
        self._session = session

    def __aiter__(self) -> _QueryStream:
        return self

    async def __anext__(self) -> Any:
        try:
            return await self._stream.__anext__()
        except StopAsyncIteration:
            self._session.stop()
            raise
        except Exception as error:
            self._session.fail(error)
            raise

    async def aclose(self) -> None:
        try:
            aclose = getattr(self._stream, "aclose", None)
            if aclose is not None:
                await aclose()
        except Exception as error:
            self._session.fail(error)
            raise
        self._session.stop()


def _get_attr(value: Any, name: str) -> Any:
    if value is None:
        return None
    if isinstance(value, dict):
        return value.get(name)
    return getattr(value, name, None)
