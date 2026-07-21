# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
from collections.abc import AsyncIterable, AsyncIterator, Awaitable, Callable
from typing import Generic, TypeVar

from .message_processor import TurnState

MessageT = TypeVar("MessageT")

_logger = logging.getLogger(__name__)


class ClaudeMessageStream(AsyncIterator[MessageT], Generic[MessageT]):
    """Instrument a Claude SDK async generator without changing its API.

    The wrapper observes each SDK message, drives the associated turn state,
    and finalizes telemetry exactly once when the turn, stream, or caller
    closes.
    """

    def __init__(
        self,
        stream: AsyncIterable[MessageT],
        *,
        state: TurnState | None = None,
        state_factory: Callable[[], TurnState] | None = None,
        on_finalize: Callable[[TurnState], None] | None = None,
    ) -> None:
        """Initialize an instrumented Claude message stream.

        Args:
            stream: SDK message stream to proxy.
            state: Existing turn state for a persistent-client response.
            state_factory: Lazy factory used to start a one-shot query turn
                when iteration begins.
            on_finalize: Optional callback invoked after the turn state is
                finalized.
        """

        self._stream = stream
        self._iterator = aiter(stream)
        self._state = state
        self._state_factory = state_factory
        self._on_finalize = on_finalize
        self._finalized = False

    def __aiter__(self) -> ClaudeMessageStream[MessageT]:
        """Return this wrapper as its own asynchronous iterator.

        Returns:
            This message stream wrapper.
        """

        return self

    async def __anext__(self) -> MessageT:
        """Return and process the next SDK message.

        Returns:
            The next unmodified message from the wrapped SDK stream.

        Raises:
            StopAsyncIteration: When the wrapped stream is exhausted.
            BaseException: Re-raises any failure from the wrapped stream after
                finalizing the turn as failed.
        """

        self._ensure_state()
        try:
            message = await anext(self._iterator)
        except StopAsyncIteration:
            self._finalize_success()
            raise
        except BaseException as error:
            self._finalize_failure(error)
            raise

        state = self._state
        if state is not None:
            try:
                state.on_message(message)
            except Exception:  # pylint: disable=broad-exception-caught
                # Instrumentation must never break the instrumented SDK.
                _logger.debug(
                    "Failed to process a Claude Agent SDK message",
                    exc_info=True,
                )
                self._finalize_success()
            else:
                if state.finished:
                    self._finalize_success()
        return message

    async def aclose(self) -> None:
        """Close the wrapped stream and finalize telemetry.

        Raises:
            BaseException: Re-raises any failure from the wrapped stream's
                close method after finalizing the turn as failed.
        """

        try:
            close = getattr(self._stream, "aclose", None)
            if close is None:
                close = getattr(self._stream, "close", None)
            if close is not None:
                result = close()
                if isinstance(result, Awaitable):
                    await result
        except BaseException as error:
            self._finalize_failure(error)
            raise
        self._finalize_success()

    def __getattr__(self, name: str):
        """Forward unknown attributes to the wrapped SDK stream.

        Args:
            name: Attribute name requested by the caller.

        Returns:
            The matching attribute from the wrapped stream.

        Raises:
            AttributeError: If the wrapped stream does not expose ``name``.
        """

        return getattr(self._stream, name)

    def _ensure_state(self) -> None:
        """Create the lazy turn state once when iteration starts.

        Telemetry construction failures are logged and do not interrupt the
        SDK stream.
        """

        if self._state is not None or self._state_factory is None:
            return
        try:
            self._state = self._state_factory()
        except Exception:  # pylint: disable=broad-exception-caught
            _logger.debug(
                "Failed to start Claude Agent SDK telemetry",
                exc_info=True,
            )
        finally:
            self._state_factory = None

    def _finalize_success(self) -> None:
        """Stop the turn state successfully at most once."""

        if self._finalized:
            return
        self._finalized = True
        state = self._state
        if state is None:
            return
        try:
            state.stop()
        except Exception:  # pylint: disable=broad-exception-caught
            _logger.debug(
                "Failed to finalize Claude Agent SDK telemetry",
                exc_info=True,
            )
        finally:
            self._notify_finalized(state)

    def _finalize_failure(self, error: BaseException) -> None:
        """Fail the turn state at most once.

        Args:
            error: Stream failure to record on the active invocation.
        """

        if self._finalized:
            return
        self._finalized = True
        state = self._state
        if state is None:
            return
        try:
            state.fail(error)
        except Exception:  # pylint: disable=broad-exception-caught
            _logger.debug(
                "Failed to record a Claude Agent SDK stream error",
                exc_info=True,
            )
        finally:
            self._notify_finalized(state)

    def _notify_finalized(self, state: TurnState) -> None:
        """Notify the owner after a turn state has been finalized.

        Args:
            state: Turn state that was stopped or failed.
        """

        if self._on_finalize is None:
            return
        try:
            self._on_finalize(state)
        except Exception:  # pylint: disable=broad-exception-caught
            _logger.debug(
                "Failed to clear Claude Agent SDK telemetry state",
                exc_info=True,
            )


__all__ = ["ClaudeMessageStream"]
