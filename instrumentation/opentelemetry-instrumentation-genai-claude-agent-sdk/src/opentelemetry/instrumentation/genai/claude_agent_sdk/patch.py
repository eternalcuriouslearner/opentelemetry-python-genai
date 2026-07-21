# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from typing import Any

from opentelemetry.instrumentation.utils import is_instrumentation_enabled
from opentelemetry.util.genai.handler import TelemetryHandler

from ._stream import ClaudeMessageStream
from .message_processor import TurnState

_STATE_ATTRIBUTE = "_otel_genai_turn_state"
_CONNECTING_ATTRIBUTE = "_otel_genai_connecting"
_logger = logging.getLogger(__name__)


def query(handler: TelemetryHandler) -> Callable[..., Any]:
    """Build a wrapper for the one-shot SDK query generator.

    Args:
        handler: Telemetry handler used to create the agent invocation.

    Returns:
        A wrapt-compatible callback that instruments the returned message
        stream.
    """

    def traced_method(
        wrapped: Callable[..., Any],
        instance: Any,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> Any:
        """Wrap one invocation of ``claude_agent_sdk.query``.

        Args:
            wrapped: Original SDK query function.
            instance: Bound instance supplied by wrapt. This is normally
                ``None`` for the module-level query function.
            args: Positional arguments passed to the SDK query function.
            kwargs: Keyword arguments passed to the SDK query function.

        Returns:
            The original stream when instrumentation is disabled, otherwise
            a stream that reconstructs and records the agent turn.
        """

        stream = wrapped(*args, **kwargs)
        if not is_instrumentation_enabled():
            return stream
        prompt = _argument(args, kwargs, 0, "prompt")
        options = _argument(args, kwargs, 1, "options")
        return ClaudeMessageStream(
            stream,
            state_factory=lambda: TurnState(
                handler, prompt=prompt, options=options
            ),
        )

    return traced_method


def client_connect(handler: TelemetryHandler) -> Callable[..., Any]:
    """Build a wrapper for persistent-client connection setup.

    The wrapper starts a turn only when ``connect`` includes an initial
    prompt. A client connected without a prompt has no invocation to trace.

    Args:
        handler: Telemetry handler used to create the initial turn.

    Returns:
        A wrapt-compatible asynchronous callback for
        ``ClaudeSDKClient.connect``.
    """

    async def traced_method(
        wrapped: Callable[..., Awaitable[Any]],
        instance: Any,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> Any:
        """Trace a client connection that sends an initial prompt.

        Args:
            wrapped: Original SDK ``connect`` method.
            instance: Persistent ``ClaudeSDKClient`` being connected.
            args: Positional arguments passed to ``connect``.
            kwargs: Keyword arguments passed to ``connect``.

        Returns:
            The value returned by the original ``connect`` method.

        Raises:
            BaseException: Re-raises any failure from the original method
                after finalizing the active turn as failed.
        """

        prompt = _argument(args, kwargs, 0, "prompt")
        if prompt is None or not is_instrumentation_enabled():
            return await wrapped(*args, **kwargs)

        try:
            state = _start_client_turn(handler, instance, prompt)
            setattr(instance, _CONNECTING_ATTRIBUTE, True)
        except Exception:  # pylint: disable=broad-exception-caught
            _logger.debug(
                "Failed to start Claude Agent SDK connect telemetry",
                exc_info=True,
            )
            return await wrapped(*args, **kwargs)
        try:
            return await wrapped(*args, **kwargs)
        except BaseException as error:
            _finalize_client_state(instance, state, error)
            raise
        finally:
            try:
                setattr(instance, _CONNECTING_ATTRIBUTE, False)
            except Exception:  # pylint: disable=broad-exception-caught
                _logger.debug(
                    "Failed to clear Claude Agent SDK connect state",
                    exc_info=True,
                )

    return traced_method


def client_query(handler: TelemetryHandler) -> Callable[..., Any]:
    """Build a wrapper that starts telemetry for each client query.

    Args:
        handler: Telemetry handler used to create each turn.

    Returns:
        A wrapt-compatible asynchronous callback for
        ``ClaudeSDKClient.query``.
    """

    async def traced_method(
        wrapped: Callable[..., Awaitable[Any]],
        instance: Any,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> Any:
        """Start a turn before forwarding a prompt to the SDK.

        Args:
            wrapped: Original SDK ``query`` method.
            instance: Persistent ``ClaudeSDKClient`` receiving the query.
            args: Positional arguments passed to ``query``.
            kwargs: Keyword arguments passed to ``query``.

        Returns:
            The value returned by the original ``query`` method.

        Raises:
            BaseException: Re-raises any SDK failure after finalizing the
                active turn as failed.
        """

        if not is_instrumentation_enabled():
            return await wrapped(*args, **kwargs)

        prompt = _argument(args, kwargs, 0, "prompt")
        try:
            state = _start_client_turn(handler, instance, prompt)
        except Exception:  # pylint: disable=broad-exception-caught
            _logger.debug(
                "Failed to start Claude Agent SDK turn telemetry",
                exc_info=True,
            )
            return await wrapped(*args, **kwargs)
        try:
            return await wrapped(*args, **kwargs)
        except BaseException as error:
            _finalize_client_state(instance, state, error)
            raise

    return traced_method


def client_receive_response() -> Callable[..., Any]:
    """Build a wrapper that observes one persistent-client response.

    Returns:
        A wrapt-compatible callback for
        ``ClaudeSDKClient.receive_response``.
    """

    def traced_method(
        wrapped: Callable[..., Any],
        instance: Any,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> Any:
        """Attach the pending turn to its response message stream.

        Args:
            wrapped: Original SDK ``receive_response`` method.
            instance: Persistent ``ClaudeSDKClient`` receiving the response.
            args: Positional arguments passed to ``receive_response``.
            kwargs: Keyword arguments passed to ``receive_response``.

        Returns:
            The original response stream when no turn is pending, otherwise
            a stream that records messages and finalizes the pending turn.
        """

        stream = wrapped(*args, **kwargs)
        state = _client_state(instance)
        if state is None:
            return stream
        try:
            return ClaudeMessageStream(
                stream,
                state=state,
                on_finalize=lambda finalized: _clear_client_state(
                    instance, finalized
                ),
            )
        except Exception:  # pylint: disable=broad-exception-caught
            _logger.debug(
                "Failed to wrap a Claude Agent SDK response stream",
                exc_info=True,
            )
            return stream

    return traced_method


def client_disconnect() -> Callable[..., Any]:
    """Build a wrapper that closes unfinished telemetry on disconnect.

    Returns:
        A wrapt-compatible asynchronous callback for
        ``ClaudeSDKClient.disconnect``.
    """

    async def traced_method(
        wrapped: Callable[..., Awaitable[Any]],
        instance: Any,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> Any:
        """Disconnect the SDK client and finalize any pending turn.

        Args:
            wrapped: Original SDK ``disconnect`` method.
            instance: Persistent ``ClaudeSDKClient`` being disconnected.
            args: Positional arguments passed to ``disconnect``.
            kwargs: Keyword arguments passed to ``disconnect``.

        Returns:
            The value returned by the original ``disconnect`` method.

        Raises:
            BaseException: Re-raises any SDK failure after marking the
                pending turn as failed.
        """

        state = _client_state(instance)
        try:
            result = await wrapped(*args, **kwargs)
        except BaseException as error:
            if state is not None and not getattr(
                instance, _CONNECTING_ATTRIBUTE, False
            ):
                _finalize_client_state(instance, state, error)
            raise
        if state is not None and not getattr(
            instance, _CONNECTING_ATTRIBUTE, False
        ):
            _finalize_client_state(instance, state)
        return result

    return traced_method


def _start_client_turn(
    handler: TelemetryHandler, instance: Any, prompt: Any
) -> TurnState:
    """Create and store the single pending turn for an SDK client.

    Any previously undrained turn is stopped before the new state is stored.

    Args:
        handler: Telemetry handler used to create the invocation.
        instance: Persistent SDK client that owns the turn.
        prompt: Prompt sent for the new turn.

    Returns:
        The newly created turn state.
    """

    previous = _client_state(instance)
    if previous is not None:
        # The initial implementation supports one outstanding turn. Ending an
        # undrained turn prevents state from accumulating without changing SDK
        # control flow.
        try:
            previous.stop()
        except Exception:  # pylint: disable=broad-exception-caught
            _logger.debug(
                "Failed to close the previous Claude Agent SDK turn",
                exc_info=True,
            )
    state = TurnState(
        handler,
        prompt=prompt,
        options=getattr(instance, "options", None),
    )
    setattr(instance, _STATE_ATTRIBUTE, state)
    return state


def _client_state(instance: Any) -> TurnState | None:
    """Read the pending telemetry state from an SDK client.

    Args:
        instance: SDK client that may own a pending turn.

    Returns:
        The pending turn state, or ``None`` when no valid state is stored.
    """

    state = getattr(instance, _STATE_ATTRIBUTE, None)
    return state if isinstance(state, TurnState) else None


def _clear_client_state(instance: Any, expected: TurnState) -> None:
    """Clear a pending turn only when it is the expected state.

    Args:
        instance: SDK client that owns the state.
        expected: State that must still be current before it is cleared.
    """

    if _client_state(instance) is expected:
        setattr(instance, _STATE_ATTRIBUTE, None)


def _finalize_client_state(
    instance: Any,
    state: TurnState,
    error: BaseException | None = None,
) -> None:
    """Finalize and clear a persistent client's pending turn.

    Telemetry failures are logged and never alter SDK behavior.

    Args:
        instance: SDK client that owns the state.
        state: Pending turn to stop or fail.
        error: SDK failure to record, or ``None`` for successful completion.
    """

    try:
        if error is None:
            state.stop()
        else:
            state.fail(error)
    except Exception:  # pylint: disable=broad-exception-caught
        _logger.debug(
            "Failed to finalize Claude Agent SDK client telemetry",
            exc_info=True,
        )
    finally:
        try:
            _clear_client_state(instance, state)
        except Exception:  # pylint: disable=broad-exception-caught
            _logger.debug(
                "Failed to clear Claude Agent SDK client telemetry",
                exc_info=True,
            )


def _argument(
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    index: int,
    name: str,
) -> Any:
    """Resolve an SDK argument from positional or keyword inputs.

    Args:
        args: Positional arguments supplied to the wrapped method.
        kwargs: Keyword arguments supplied to the wrapped method.
        index: Expected positional index of the argument.
        name: Keyword name of the argument.

    Returns:
        The supplied argument value, or ``None`` when it was omitted.
    """

    if name in kwargs:
        return kwargs[name]
    return args[index] if len(args) > index else None


__all__ = [
    "client_connect",
    "client_disconnect",
    "client_query",
    "client_receive_response",
    "query",
]
