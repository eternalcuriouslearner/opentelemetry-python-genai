# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import inspect
import logging
from collections.abc import Mapping, Sequence
from typing import Any, cast
from urllib.parse import urlparse

from llama_index.core.base.llms.base import BaseLLM
from llama_index.core.base.llms.types import (
    ChatMessage,
    ChatResponse,
    CompletionResponse,
    TextBlock,
)
from llama_index.core.instrumentation.event_handlers import BaseEventHandler
from llama_index.core.instrumentation.events import BaseEvent
from llama_index.core.instrumentation.events.llm import (
    LLMChatEndEvent,
    LLMChatStartEvent,
    LLMCompletionEndEvent,
    LLMCompletionStartEvent,
)
from llama_index.core.instrumentation.span import BaseSpan
from llama_index.core.instrumentation.span_handlers import BaseSpanHandler
from pydantic import PrivateAttr

from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAIAttributes,
)
from opentelemetry.util.genai.handler import TelemetryHandler
from opentelemetry.util.genai.invocation import InferenceInvocation
from opentelemetry.util.genai.types import (
    InputMessage,
    MessagePart,
    OutputMessage,
    Text,
)

logger = logging.getLogger(__name__)

_CHAT_METHODS = {"chat", "achat"}
_COMPLETION_METHODS = {"complete", "acomplete"}

_PROVIDERS_BY_INTEGRATION = {
    "anthropic": GenAIAttributes.GenAiProviderNameValues.ANTHROPIC.value,
    "azure_openai": GenAIAttributes.GenAiProviderNameValues.AZURE_AI_OPENAI.value,
    "bedrock": GenAIAttributes.GenAiProviderNameValues.AWS_BEDROCK.value,
    "cohere": GenAIAttributes.GenAiProviderNameValues.COHERE.value,
    "deepseek": GenAIAttributes.GenAiProviderNameValues.DEEPSEEK.value,
    "gemini": GenAIAttributes.GenAiProviderNameValues.GCP_GEMINI.value,
    "groq": GenAIAttributes.GenAiProviderNameValues.GROQ.value,
    "mistralai": GenAIAttributes.GenAiProviderNameValues.MISTRAL_AI.value,
    "openai": GenAIAttributes.GenAiProviderNameValues.OPENAI.value,
    "perplexity": GenAIAttributes.GenAiProviderNameValues.PERPLEXITY.value,
    "vertex": GenAIAttributes.GenAiProviderNameValues.GCP_VERTEX_AI.value,
    "xai": GenAIAttributes.GenAiProviderNameValues.X_AI.value,
}


def _value(value: Any, name: str) -> Any:
    if isinstance(value, Mapping):
        return cast(Mapping[str, Any], value).get(name)
    return getattr(value, name, None)


def _method_name(span_id: str) -> str:
    return span_id.partition("-")[0].rsplit(".", 1)[-1]


def _operation_name(method_name: str) -> str | None:
    if method_name in _CHAT_METHODS:
        return GenAIAttributes.GenAiOperationNameValues.CHAT.value
    if method_name in _COMPLETION_METHODS:
        return GenAIAttributes.GenAiOperationNameValues.TEXT_COMPLETION.value
    return None


def _is_native_operation(instance: BaseLLM, operation_name: str) -> bool:
    try:
        is_chat_model = instance.metadata.is_chat_model
    except Exception:  # LLM integrations can compute metadata dynamically.
        return True
    if operation_name == GenAIAttributes.GenAiOperationNameValues.CHAT.value:
        return is_chat_model
    return not is_chat_model


def _provider_name(instance: BaseLLM) -> str:
    module = type(instance).__module__
    marker = "llama_index.llms."
    if marker in module:
        integration = module.partition(marker)[2].partition(".")[0]
        if provider := _PROVIDERS_BY_INTEGRATION.get(integration):
            return provider
        return integration.replace("_", ".")
    return type(instance).__name__.lower()


def _request_model(instance: BaseLLM) -> str | None:
    try:
        model_name = instance.metadata.model_name
    except Exception:  # LLM integrations can compute metadata dynamically.
        model_name = getattr(instance, "model", None)
    return model_name if isinstance(model_name, str) and model_name else None


def _server(instance: BaseLLM) -> tuple[str | None, int | None]:
    base_url = getattr(instance, "api_base", None)
    if base_url is None:
        client = getattr(instance, "_client", None)
        base_url = getattr(client, "base_url", None)
    if base_url is None:
        return None, None
    try:
        parsed = urlparse(str(base_url))
        return parsed.hostname, parsed.port
    except ValueError:
        return None, None


def _set_request_parameters(
    invocation: InferenceInvocation, instance: BaseLLM
) -> None:
    parameters = {
        "temperature": "temperature",
        "top_p": "top_p",
        "frequency_penalty": "frequency_penalty",
        "presence_penalty": "presence_penalty",
        "max_tokens": "max_tokens",
        "seed": "seed",
    }
    for invocation_name, instance_name in parameters.items():
        value = getattr(instance, instance_name, None)
        if value is not None:
            setattr(invocation, invocation_name, value)

    stop = getattr(instance, "stop", None)
    if isinstance(stop, str):
        invocation.stop_sequences = [stop]
    elif isinstance(stop, Sequence):
        invocation.stop_sequences = [
            item for item in cast(Sequence[Any], stop) if isinstance(item, str)
        ]


def _message_parts(message: ChatMessage) -> list[MessagePart]:
    return [
        Text(content=block.text)
        for block in message.blocks
        if isinstance(block, TextBlock) and block.text
    ]


def _input_messages(messages: Sequence[ChatMessage]) -> list[InputMessage]:
    return [
        InputMessage(role=message.role.value, parts=_message_parts(message))
        for message in messages
    ]


def _finish_reasons(response: ChatResponse | CompletionResponse) -> list[str]:
    raw = response.raw
    choices = _value(raw, "choices")
    if not isinstance(choices, Sequence):
        reason = _value(_value(response, "additional_kwargs"), "finish_reason")
        return [reason] if isinstance(reason, str) and reason else []
    return [
        reason
        for choice in cast(Sequence[Any], choices)
        if isinstance((reason := _value(choice, "finish_reason")), str)
        and reason
    ]


def _set_response_metadata(
    invocation: InferenceInvocation,
    response: ChatResponse | CompletionResponse,
) -> None:
    raw = response.raw
    response_id = _value(raw, "id")
    if isinstance(response_id, str):
        invocation.response_id = response_id
    response_model = _value(raw, "model")
    if isinstance(response_model, str):
        invocation.response_model_name = response_model

    usage = _value(raw, "usage") or _value(
        _value(response, "additional_kwargs"), "usage"
    )
    input_tokens = _value(usage, "prompt_tokens")
    if input_tokens is None:
        input_tokens = _value(usage, "input_tokens")
    output_tokens = _value(usage, "completion_tokens")
    if output_tokens is None:
        output_tokens = _value(usage, "output_tokens")
    if isinstance(input_tokens, int) and not isinstance(input_tokens, bool):
        invocation.input_tokens = input_tokens
    if isinstance(output_tokens, int) and not isinstance(output_tokens, bool):
        invocation.output_tokens = output_tokens

    prompt_details = _value(usage, "prompt_tokens_details")
    cache_read_tokens = _value(prompt_details, "cached_tokens")
    if cache_read_tokens is None:
        cache_read_tokens = _value(usage, "cache_read_input_tokens")
    completion_details = _value(usage, "completion_tokens_details")
    reasoning_tokens = _value(completion_details, "reasoning_tokens")
    if reasoning_tokens is None:
        reasoning_tokens = _value(usage, "reasoning_tokens")
    cache_creation_tokens = _value(usage, "cache_creation_input_tokens")
    if isinstance(cache_read_tokens, int) and not isinstance(
        cache_read_tokens, bool
    ):
        invocation.cache_read_input_tokens = cache_read_tokens
    if isinstance(cache_creation_tokens, int) and not isinstance(
        cache_creation_tokens, bool
    ):
        invocation.cache_creation_input_tokens = cache_creation_tokens
    if isinstance(reasoning_tokens, int) and not isinstance(
        reasoning_tokens, bool
    ):
        invocation.thinking_tokens = reasoning_tokens

    finish_reasons = _finish_reasons(response)
    if finish_reasons:
        invocation.finish_reasons = finish_reasons


class _InferenceSpan(BaseSpan):
    _invocation: InferenceInvocation = PrivateAttr()

    def __init__(
        self,
        *,
        id_: str,
        parent_id: str | None,
        invocation: InferenceInvocation,
    ) -> None:
        super().__init__(id_=id_, parent_id=parent_id)
        self._invocation = invocation

    def process_event(self, event: BaseEvent) -> None:
        if isinstance(event, LLMChatStartEvent):
            self._invocation.input_messages = _input_messages(event.messages)
        elif isinstance(event, LLMCompletionStartEvent):
            self._invocation.input_messages = [
                InputMessage(role="user", parts=[Text(content=event.prompt)])
            ]
        elif isinstance(event, LLMChatEndEvent):
            if event.response is None:
                return
            response = event.response
            _set_response_metadata(self._invocation, response)
            reasons = _finish_reasons(response)
            self._invocation.output_messages = [
                OutputMessage(
                    role=response.message.role.value,
                    parts=_message_parts(response.message),
                    finish_reason=reasons[0] if reasons else "",
                )
            ]
        elif isinstance(event, LLMCompletionEndEvent):
            response = event.response
            _set_response_metadata(self._invocation, response)
            reasons = _finish_reasons(response)
            self._invocation.output_messages = [
                OutputMessage(
                    role="assistant",
                    parts=[Text(content=response.text)],
                    finish_reason=reasons[0] if reasons else "",
                )
            ]


class LlamaIndexSpanHandler(BaseSpanHandler[_InferenceSpan]):
    _handler: TelemetryHandler = PrivateAttr()

    def __init__(self, handler: TelemetryHandler) -> None:
        super().__init__()
        self._handler = handler

    def new_span(
        self,
        id_: str,
        bound_args: inspect.BoundArguments,
        instance: Any | None = None,
        parent_span_id: str | None = None,
        tags: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> _InferenceSpan | None:
        if not isinstance(instance, BaseLLM):
            return None
        operation_name = _operation_name(_method_name(id_))
        if operation_name is None or not _is_native_operation(
            instance, operation_name
        ):
            return None
        if parent_span_id in self.open_spans and operation_name == (
            _operation_name(_method_name(parent_span_id))
        ):
            return None
        server_address, server_port = _server(instance)
        invocation = self._handler.inference(
            _provider_name(instance),
            request_model=_request_model(instance),
            server_address=server_address,
            server_port=server_port,
            operation_name=operation_name,
        )
        _set_request_parameters(invocation, instance)
        return _InferenceSpan(
            id_=id_, parent_id=parent_span_id, invocation=invocation
        )

    def prepare_to_exit_span(
        self,
        id_: str,
        bound_args: inspect.BoundArguments,
        instance: Any | None = None,
        result: Any | None = None,
        **kwargs: Any,
    ) -> _InferenceSpan | None:
        span = self.open_spans.get(id_)
        if span is not None:
            span._invocation.stop()
        return span

    def prepare_to_drop_span(
        self,
        id_: str,
        bound_args: inspect.BoundArguments,
        instance: Any | None = None,
        err: BaseException | None = None,
        **kwargs: Any,
    ) -> _InferenceSpan | None:
        span = self.open_spans.get(id_)
        if span is not None:
            if err is None:
                span._invocation.stop()
            else:
                span._invocation.fail(err)
        return span


class LlamaIndexEventHandler(BaseEventHandler):
    _span_handler: LlamaIndexSpanHandler = PrivateAttr()

    def __init__(self, span_handler: LlamaIndexSpanHandler) -> None:
        super().__init__()
        self._span_handler = span_handler

    def handle(self, event: BaseEvent, **kwargs: Any) -> Any:
        if event.span_id is None:
            return event
        span = self._span_handler.open_spans.get(event.span_id)
        if span is not None:
            try:
                span.process_event(event)
            except Exception:
                logger.exception(
                    "Failed to process LlamaIndex event %s",
                    type(event).__qualname__,
                )
        return event
