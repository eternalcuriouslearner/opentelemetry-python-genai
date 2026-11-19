# Conformance run after semconv pin bump (investigation only)

- SEMCONV_GENAI_REF: e6658f14c9d9ea0760825f029cb82e7673f48fab (was 94f432d7, 89 commits)
- weaver: 0.28.0 (aarch64-apple-darwin, sha256 b334b37372325f77374b461fd6a11d38bc050122832eb83a9732fcd6831a5ab1 verified against the release .sha256)
- Registry cache (`~/.cache/otel-conformance/semconv/genai-e6658f14...`) cleared before the run.
- Branch head before results: b48a861f5. Runs used `tox` from a py3.13 uv env (`uv sync --python 3.13`; the default 3.15 interpreter fails to build pydantic-core 2.41.5).
- Per-env logs in `conformance-logs/`, per-scenario JSON (collect-all harness) in `conformance-logs/results/`, raw weaver reports in `conformance-logs/weaver_reports/<env>/`.
- No env was skipped or hung. No "weaver binary not on PATH" skip anywhere. The `Transient error ... exporting metrics to localhost:<port>` lines in every log are post-run noise from the OTLP exporter after weaver stops; they do not affect results.

## 1. Per-env results (new pin e6658f14, weaver 0.28.0)

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

Totals: 10 FAIL, 4 PASS, 0 skip, 0 hung. 63 failing scenarios.

Extra runs:

- **Rego re-check** (uncommitted local edit to `policies/genai_span_validation.rego`: invoke_workflow keyed/expected attr `gen_ai.workflow.name` -> `gen_ai.agent.name`): all 6 #1103 workflow scenarios (agno x3, langchain, llama_index, openai_agents) report both declared `gen_ai.workflow.name` ExpectedViolations as UNMET ("declared but no longer seen"), with no new violation. Logs: `conformance-logs/regopatch-*.log`.
- **Old-registry control** (uncommitted local edit `SEMCONV_GENAI_REF=94f432d7`, weaver 0.28.0, new opentelemetry-python pin): google-genai 7 passed, openai 13 passed. Weaver 0.28 + the new test-util pin are green on the current registry. Logs: `conformance-logs/oldpin-weaver028-*.log`.

## 2. Harness prerequisites

| change | commit | needed for real? |
|---|---|---|
| WEAVER_VERSION v0.26.1 -> v0.28.0 | 37e4c262d | **Yes.** v0.26.1 cannot parse `unref` in the new registry. Proven green on the old registry (control run above), so it can land first. |
| opentelemetry-python pin 14971cef -> 2fa016b2 (open-telemetry/opentelemetry-python#5738, the live-check protocol for weaver 0.27+), contrib pin a22d0330 -> 270d4b02 | 4b6717049 | **Yes**, it ships with the weaver bump. I did not rerun weaver 0.28 against the old 14971cef pin; that part comes from the previous attempt. |
| dev-requirements-conformance: add `opentelemetry-exporter-otlp-common` and `opentelemetry-exporter-http-transport` from the same commit | b48a861f5 | **Yes**, ships with the pin bump, so the git-pinned otlp stack still resolves at 2fa016b2. |
| `conformance.py` `_experiment_collect_all` (collect every failure, dump JSON) | b48a861f5 | **Investigation only.** Revert it. Real PRs keep `_check_violations` + `scenario.validate`. |
| SEMCONV_GENAI_REF -> e6658f14 | 21305e821 | Yes, but only together with the util metric migration (section 5). |

## 3. Violations, grouped by semconv PR

Every unexpected violation and expected_metrics mismatch across all 14 envs reduces to the 4 distinct raw violations below (de-duplicated on id + message + signal):

```
('missing_metric',    'Metric does not exist in the registry.', signal 'gen_ai.client.token.usage')
('missing_attribute', "Attribute 'gen_ai.token.type' does not exist in the registry.", signal 'gen_ai.client.token.usage')
('missing_metric',    'Metric does not exist in the registry.', signal 'gen_ai.client.operation.time_to_first_chunk')
('missing_metric',    'Metric does not exist in the registry.', signal 'gen_ai.client.operation.time_per_output_chunk')
```

The expected_metrics mismatches are the same root cause seen from the scenario side, e.g.:
`Expected metrics ['gen_ai.client.operation.duration', 'gen_ai.client.token.usage'] but weaver only saw ['gen_ai.client.operation.duration'] — missing ['gen_ai.client.token.usage']`.
Weaver drops samples for metrics that are not in the registry, so `seen_metrics` no longer contains them. No expected_spans mismatch occurred anywhere, and no `unmet_expected` occurred on the new pin without the rego edit.

### semantic-conventions-genai #374: usage metrics broken down by modality, reasoning and cache (8ffdf56)
- Removes `gen_ai.client.token.usage` and `gen_ai.token.type`, confirmed with `git log -S`. They are replaced by counters `gen_ai.client.inference.usage.{input_tokens,output_tokens,cache_read.input_tokens,cache_write.input_tokens,reasoning.output_tokens}` (required attribute `gen_ai.token.modality`) and histograms `gen_ai.client.inference.operation.{input,output}_tokens`.
- Affects: openai, google-genai, agno, smolagents, anthropic, bedrock, dspy, langchain, portkey, ollama (all 10 failing envs).
- Fix owner: **util-genai** (`_instruments.py` creates `token_usage`; recording happens in `_invocation.py` / `_inference_invocation.py` / `_fetch_response_invocation.py`). The installed `opentelemetry-semantic-conventions` (2fa016b2) has no constants for the new names, so util needs private `Final` constants, as it already does for `_GEN_AI_INVOKE_AGENT_DURATION`. Instrumentations own only their scenario `expected_metrics` plus the unit tests that assert the old names (23 test files across 11 packages, plus the smolagents README; `grep -rl gen_ai.client.token.usage instrumentation`).
- Related, not flagged by live-check: #602 (positive contributions only), #548, #536 (failed operations), #679 (no modality on reasoning metrics). The util implementation has to follow these.

### #578: embeddings duration and input-token metrics (2025935)
- Embedding scenarios currently fail only via #374. The replacement for embeddings is `gen_ai.client.embeddings.operation.input_tokens`, not the inference.* metrics, and `gen_ai.client.embeddings.duration` supersedes `gen_ai.client.operation.duration` for embeddings ("SHOULD NOT be reported"). Weaver does not enforce the SHOULD NOT.
- Affects the embedding scenarios in openai, google-genai, agno, bedrock, dspy (HostedEmbedder), ollama. Fix owner: **util-genai** (embedding invocation metrics).

### #521: gen_ai.client.inference.duration and inference signals moved to one file (4f85037)
- Renames `gen_ai.client.operation.time_to_first_chunk` / `time_per_output_chunk` to `gen_ai.client.inference.time_to_first_chunk` / `time_per_output_chunk` (confirmed with `git log -S`). It also introduces `gen_ai.client.inference.duration`, which should replace `gen_ai.client.operation.duration` for inference. The old metric still exists, so weaver does not flag it.
- Chunk-metric violations: anthropic, bedrock, langchain, ollama, openai, portkey, smolagents (streaming scenarios). ollama emits them but does not list them in expected_metrics.
- Fix owner: **util-genai** (`_instruments.py` `time_to_first_chunk` / `time_per_output_chunk`). Instrumentations: streaming scenario expected_metrics.

### #571: invoke_workflow folded into gen_ai.invoke_agent.internal (aa3b7f4)
- Removes `gen_ai.workflow.name` from the registry. The `invoke_workflow` operation value remains, named by `gen_ai.agent.name`.
- It produces **no registry violation** on this branch, because #1103 already stopped emitting `gen_ai.workflow.name`. The two violations #1103 declares come from **this repo's** `policies/genai_span_validation.rego`, which still keys/expects `gen_ai.workflow.name` for `invoke_workflow`:
  ```
  genai_expected_attribute_missing: Span 'invoke_workflow LangGraph' (operation 'invoke_workflow') is missing expected attribute 'gen_ai.workflow.name'
  genai_span_name_format: invoke_workflow span name should be 'invoke_workflow' or 'invoke_workflow <value of gen_ai.workflow.name>', got 'invoke_workflow LangGraph'
  ```
  (The same pair appears for agno x3, llama_index `AgentWorkflow`, and openai_agents `Agent workflow`.)
- So the pin bump alone does **not** make them "declared but no longer seen". The rego edit does (re-check above).
- Fix owner: **repo policy** (`policies/genai_span_validation.rego`) plus scenario cleanup in agno, langchain, llama_index, openai_agents.

### PRs with no failing signal on this run
- #636 (execute_tool span type renamed to `gen_ai.execute_tool`): no violation. Span names and attributes are unchanged.
- #634 (structured content attributes required): no violation. Content capture is off in the scenarios (`opt_in_attribute_not_present` for input/output messages), so this is unexercised rather than proven conformant.
- #658/#659/#667/#668/#669/#670 (exception events): no violation. The scenarios have no error paths, so this is unexercised.
- #513 (`gen_ai.request.stream` on fetch_response / invoke_agent client spans): information-level only (`Conditionally required attribute 'gen_ai.request.stream' is not present.`). It was already reported on the old registry.
- #643/#644/#645 (retrieval excluded from operation.duration, new internal retrieval span + `gen_ai.retrieval.duration`, retrieval.client deprecated): all retrieval scenarios pass. The new metric and span kind are not enforced by live-check.
- #635 (`gen_ai.client.invoke_agent.duration` for remote agents): not enforced. The bedrock InvokeAgent scenarios pass.
- #490 (`*_ref` external content references): new information-level `opt_in_attribute_not_present` for `gen_ai.{input,output}.messages_ref`, `system_instructions_ref`, `tool.definitions_ref`.

## 4. What #1103 must delete once the pin moves past #571

The pin is not the trigger; the rego is. Once `policies/genai_span_validation.rego` maps `invoke_workflow` to `gen_ai.agent.name`, these become "declared but no longer seen" and must be deleted (verified):

1. `instrumentation/opentelemetry-instrumentation-genai-agno/tests/conformance/workflow.py`: the whole `expected_violations` tuple (2 entries + comment) and the `ExpectedViolation` import.
2. `.../genai-agno/tests/conformance/workflow_background.py`: same.
3. `.../genai-agno/tests/conformance/workflow_streaming.py`: same.
4. `.../genai-langchain/tests/conformance/workflow.py`: the 2 `gen_ai.workflow.name` entries + comment (other entries in that tuple stay).
5. `.../genai-llama-index/tests/conformance/workflow.py`: the whole `expected_violations` tuple and the `ExpectedViolation` import.
6. `.../genai-openai-agents/tests/conformance/orchestration.py`: the 2 `gen_ai.workflow.name` entries + comment (the other entries stay, e.g. the `gen_ai.tool.call.id` gap).

Rego edit (2 lines):
```
-_span_name_keyed_attr["invoke_workflow"]   := "gen_ai.workflow.name"
+_span_name_keyed_attr["invoke_workflow"]   := "gen_ai.agent.name"
 _invoke_workflow_expected := {
 	"gen_ai.operation.name",
-	"gen_ai.workflow.name",
+	"gen_ai.agent.name",
 }
```
(The comment above `_invoke_workflow_expected` needs the same rename.) The rego is repo-owned and `gen_ai.agent.name` exists in the old registry too, so #1103 can make this edit itself and drop the ExpectedViolations without waiting for the pin. That inference is untested on the old pin.

## 5. Recommended PR split and order

1. **Harness bump**: WEAVER_VERSION v0.28.0 + opentelemetry-python/contrib pins + the two extra dev-requirements lines. Old SEMCONV ref. Verified green for openai and google-genai on the old registry; run all 14 before opening.
2. **#1103 + rego** (or a small follow-up right after #1103): update `policies/genai_span_validation.rego` for invoke_workflow -> `gen_ai.agent.name` and delete the 6 scenario ExpectedViolations from section 4. Independent of the pin.
3. **Pin bump + util metric migration (one atomic PR)**: SEMCONV_GENAI_REF -> e6658f14. util-genai replaces `gen_ai.client.token.usage` with the #374 inference usage/operation metrics and the #578 embeddings input-token metric, and renames the #521 chunk metrics. Update every scenario's expected_metrics and the instrumentation/util unit tests that assert old names. This cannot be split without breaking CI: old names fail the new registry (`missing_metric`), and new names would fail the old registry. Splitting would need ExpectedViolations in about 60 scenarios. Needs `check-util-api` (private names only, should pass) and a util changelog fragment; instrumentation `-oldest` envs are unaffected because they use released util.
4. **Follow-ups on the new pin** (each green on its own, not enforced by live-check): `gen_ai.client.inference.duration` (#521), `gen_ai.client.embeddings.duration` (#578), `gen_ai.client.invoke_agent.duration` for remote agents (#635), `gen_ai.retrieval.duration` / internal retrieval span (#644/#645), `gen_ai.request.stream` (#513). Do util first, then per-instrumentation expected_metrics.
5. Drop the experiment-only `conformance.py` change; it must not ship.
