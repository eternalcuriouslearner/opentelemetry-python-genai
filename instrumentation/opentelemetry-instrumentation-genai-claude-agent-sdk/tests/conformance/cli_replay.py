# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""Replay transport for pre-recorded Claude Code CLI sessions.

The Claude Agent SDK talks to the bundled Claude Code CLI over a subprocess
transport, not HTTP, so tests replay recorded CLI message streams through
the SDK's public ``Transport`` interface instead of VCR cassettes. The
cassettes under ``tests/cassettes/`` are YAML files with a single
``messages`` list, recorded from real CLI sessions (originally in the
donation-openinference repository).

Re-record by running the equivalent prompt against a real CLI with
``ANTHROPIC_API_KEY`` set and capturing ``Transport.read_messages()``
output; see the recording fixture in donation-openinference's
claude-agent-sdk package for a reference implementation.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, AsyncIterator

import yaml
from claude_agent_sdk import Transport

CASSETTE_DIR = Path(__file__).parent.parent / "cassettes"


class ReplayTransport(Transport):
    """Replays pre-recorded CLI messages from a cassette.

    Handles request-ID substitution: the SDK generates a new random
    request_id for each control_request it writes. Outgoing writes are
    intercepted to capture the actual IDs, which are substituted into the
    cassette's control_response messages so the SDK's pending control
    responses match.
    """

    def __init__(self, messages: list[dict[str, Any]]) -> None:
        self._messages = messages
        # SDK-originated control_request IDs, captured in write() order.
        self._sdk_control_request_ids: list[str] = []

    async def connect(self) -> None:
        pass

    async def write(self, data: str) -> None:
        try:
            msg = json.loads(data.strip())
        except json.JSONDecodeError:
            return
        if msg.get("type") == "control_request":
            self._sdk_control_request_ids.append(msg["request_id"])

    def read_messages(self) -> AsyncIterator[dict[str, Any]]:
        return self._gen()

    async def _gen(self) -> AsyncIterator[dict[str, Any]]:
        response_idx = 0
        for msg in self._messages:
            if msg.get("type") == "control_response":
                if response_idx < len(self._sdk_control_request_ids):
                    actual_id = self._sdk_control_request_ids[response_idx]
                    response_idx += 1
                    msg = {
                        **msg,
                        "response": {
                            **msg["response"],
                            "request_id": actual_id,
                        },
                    }
            yield msg

    async def close(self) -> None:
        pass

    async def end_input(self) -> None:
        pass

    def is_ready(self) -> bool:
        return True


def replay_transport(cassette_name: str) -> ReplayTransport:
    """Build a ReplayTransport from ``tests/cassettes/<cassette_name>``."""
    path = CASSETTE_DIR / cassette_name
    data = yaml.safe_load(path.read_text())
    messages = data.get("messages") or []
    if not messages:
        raise AssertionError(f"Cassette at {path} is missing or empty")
    return ReplayTransport(messages)
