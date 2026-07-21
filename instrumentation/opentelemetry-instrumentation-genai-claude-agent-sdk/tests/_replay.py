# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import anyio
import yaml

try:
    from claude_agent_sdk._internal.transport import Transport
except ImportError:
    Transport = object  # type: ignore[misc,assignment]


class ReplayTransport(Transport):
    """Replay recorded Claude subprocess messages through the SDK parser."""

    def __init__(self, messages: list[dict[str, Any]]) -> None:
        self._messages = messages
        self._control_request_ids: list[str] = []

    async def connect(self) -> None:
        return None

    async def write(self, data: str) -> None:
        try:
            message = json.loads(data.strip())
        except json.JSONDecodeError:
            return
        if message.get("type") == "control_request":
            self._control_request_ids.append(message["request_id"])

    def read_messages(self):
        return self._replay()

    async def _replay(self):
        response_index = 0
        for original in self._messages:
            message = original
            if message.get("type") == "control_response":
                if response_index >= len(self._control_request_ids):
                    await anyio.sleep(0)
                # Claude Agent SDK 0.1.x did not initialize the control
                # protocol for one-shot string prompts. Such replays have no
                # request ID to substitute, so omit the recorded handshake.
                if response_index >= len(self._control_request_ids):
                    continue
                request_id = self._control_request_ids[response_index]
                response_index += 1
                message = {
                    **message,
                    "response": {
                        **message["response"],
                        "request_id": request_id,
                    },
                }
            yield message

    async def close(self) -> None:
        return None

    async def end_input(self) -> None:
        return None

    def is_ready(self) -> bool:
        return True


def load_cassette_messages(path: Path) -> list[dict[str, Any]]:
    """Load every recorded message from a donated cassette."""

    data = yaml.safe_load(path.read_text())
    return cast(list[dict[str, Any]], data["messages"])


def load_cassettes(*paths: Path) -> ReplayTransport:
    """Replay complete cassettes in order without rewriting their contents."""

    messages = [
        message for path in paths for message in load_cassette_messages(path)
    ]
    return ReplayTransport(messages)


def load_cassette(path: Path) -> ReplayTransport:
    return load_cassettes(path)
