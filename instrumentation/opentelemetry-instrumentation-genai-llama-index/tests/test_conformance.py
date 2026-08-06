# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any

import pytest

pytest.importorskip("opentelemetry.test.weaver_live_check")
pytest.importorskip("opentelemetry.exporter.otlp.proto.grpc")

from opentelemetry.test.weaver_live_check import WeaverLiveCheck  # noqa: E402
from opentelemetry.test_util_genai.conformance import (  # noqa: E402
    run_conformance,
)

from .conformance.inference import InferenceScenario  # noqa: E402


def test_conformance(vcr: Any, weaver_live_check: WeaverLiveCheck) -> None:
    run_conformance(InferenceScenario(), vcr=vcr, weaver=weaver_live_check)
