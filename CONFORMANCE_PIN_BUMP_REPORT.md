# Conformance run after semconv pin bump (investigation only)

- SEMCONV_GENAI_REF: e6658f14c9d9ea0760825f029cb82e7673f48fab (was 94f432d7)
- weaver: 0.28.0 (aarch64-apple-darwin, sha256 b334b37372325f77374b461fd6a11d38bc050122832eb83a9732fcd6831a5ab1 verified against the release .sha256)
- Registry cache (`~/.cache/otel-conformance/semconv/genai-e6658f14...`) cleared before the run.
- Per-env logs in `conformance-logs/`, per-scenario JSON (collect-all harness) in `conformance-logs/results/`.

## Per-env results

| env | result | summary | notes |
|---|---|---|---|
| py314-test-instrumentation-genai-openai-conformance | FAIL | 12 failed, 1 passed (112s) | gen_ai.client.token.usage / gen_ai.token.type gone from registry; time_to_first_chunk / time_per_output_chunk renamed; only ResponsesFetchScenario passes |
