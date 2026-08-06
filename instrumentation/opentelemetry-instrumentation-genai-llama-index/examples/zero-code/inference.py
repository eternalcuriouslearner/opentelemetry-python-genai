# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

from llama_index.core.llms import ChatMessage
from llama_index.llms.openai import OpenAI

response = OpenAI(model="gpt-4o-mini").chat(
    [ChatMessage(role="user", content="What is OpenTelemetry?")]
)
print(response.message.content)
