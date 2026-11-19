# Conformance run after semconv pin bump (investigation only)

- SEMCONV_GENAI_REF: e6658f14c9d9ea0760825f029cb82e7673f48fab (was 94f432d7)
- weaver: 0.28.0 (aarch64-apple-darwin, sha256 b334b37372325f77374b461fd6a11d38bc050122832eb83a9732fcd6831a5ab1 verified against the release .sha256)
- Registry cache (`~/.cache/otel-conformance/semconv/genai-e6658f14...`) cleared before the run.
- Per-env logs in `conformance-logs/`, per-scenario JSON (collect-all harness) in `conformance-logs/results/`.

## Per-env results

| env | result | summary | notes |
|---|---|---|---|
| py314-test-instrumentation-genai-openai-conformance | FAIL | 12 failed, 1 passed (112s) | gen_ai.client.token.usage / gen_ai.token.type gone from registry; time_to_first_chunk / time_per_output_chunk renamed; only ResponsesFetchScenario passes |
| py314-test-instrumentation-genai-openai_agents-conformance | PASS | 1 passed (10s) | OrchestrationScenario; #1103's declared gen_ai.workflow.name violations are still SEEN (they come from repo policies/genai_span_validation.rego, not the registry) |
| py314-test-instrumentation-google-genai-conformance | FAIL | 6 failed, 1 passed (56s) | token.usage removal only; FetchResponseScenario passes |
| py314-test-instrumentation-genai-agno-conformance | FAIL | 1 failed, 5 passed (48s) | only EmbeddingScenario fails (token.usage); 3 workflow scenarios pass, their #1103 workflow.name expectations still met |
| py314-test-instrumentation-genai-smolagents-conformance | FAIL | 4 failed, 1 passed (38s) | token.usage + streaming chunk metric rename; AgentScenario passes |
| py314-test-instrumentation-genai-anthropic-conformance | FAIL | 13 failed (102s) | token.usage + streaming chunk metric rename; nothing else |
| py314-test-instrumentation-genai-bedrock-conformance | FAIL | 7 failed, 4 passed (83s) | token.usage + chunk metric rename; agent/retrieve scenarios pass |
| py314-test-instrumentation-genai-dspy-conformance | FAIL | 2 failed, 7 passed (69s) | HostedEmbedder, LM fail on token.usage only |
| py314-test-instrumentation-genai-langchain-conformance | FAIL | 11 failed, 1 passed (94s) | token.usage + chunk rename only; WorkflowScenario's #1103 workflow.name expectations still met |
| py314-test-instrumentation-genai-llama_index-conformance | PASS | 3 passed (23s) | WorkflowScenario's #1103 workflow.name expectations still met |
| py314-test-instrumentation-genai-qwen-agent-conformance | PASS | 1 passed (11s) | InvokeAgentScenario |
| py314-test-instrumentation-genai-portkey-conformance | FAIL | 3 failed (25s) | token.usage + chunk rename only |
| py314-test-instrumentation-genai-ollama-conformance | FAIL | 4 failed (30s) | token.usage only |
| py313-test-instrumentation-genai-qwenpaw-conformance | PASS | 1 passed (16s) | InvokeAgentScenario |

Rego re-check (policies/genai_span_validation.rego: invoke_workflow keyed/expected attr gen_ai.workflow.name -> gen_ai.agent.name, committed as experiment): with this change all 6 #1103 workflow scenarios (agno x3, langchain, llama_index, openai_agents) report both declared gen_ai.workflow.name ExpectedViolations as UNMET (declared but no longer seen), and no new violation appears. Logs: conformance-logs/regopatch-*.log


Old-registry control (versions.env temporarily SEMCONV_GENAI_REF=94f432d7, weaver 0.28.0, new opentelemetry-python pin; not committed): google-genai 7 passed, openai 13 passed. Logs: conformance-logs/oldpin-weaver028-*.log

