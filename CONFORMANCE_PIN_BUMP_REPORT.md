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
