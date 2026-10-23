# Experiment architecture

This is a local tool for reproducible prompt-optimization experiments on emotion
classification and financial-news summarization. Candidate generation is a
hypothesis; isolated evaluation and explicit gates decide whether to adopt it.
The current implementation has offline coverage, but no verified current model
run or live Phoenix evidence. See the [experiment record](../artifacts/experiment-summary.md)
and executable reproduction instructions in [Japanese](../README.md) or
[Korean](../README.ko.md).

## Components and lifecycle

```text
YAML → validate config/key → load dataset → deterministic Train/Validation/Test
                                               ↓
                                   baseline Validation
                                               ↓
       current prompt → Train inference/evaluation → optimizer → candidate
              ↑                                               ↓
              └──── accepted only ← gates ← candidate Validation
                                               ↓
                            final selected prompt → Test once → local artifacts
```

[cli.py](../src/prompt_learning/cli.py) binds replaceable components in the shared
[runner](../src/prompt_learning/pipeline/experiment.py). The runner verifies that
input identities are disjoint across subsets and that every result retains its
examples. `max_iterations` counts candidate attempts after baseline Validation,
including failed candidate generation. Each attempt starts from the currently
accepted prompt. An invalid or rejected candidate leaves that prompt in place.
Final Test runs once, after all attempts, using only the selected prompt.

| Component | Responsibility |
|---|---|
| [config](../src/prompt_learning/config.py) | Parse local YAML and types; CLI also validates implemented policy and split minima before external setup |
| [datasets](../src/prompt_learning/datasets/) | Normalize sources, balance/deduplicate, assign stable identities and deterministic splits |
| [tasks](../src/prompt_learning/tasks/) | Validate placeholders, render inputs, retain outputs and failed examples |
| [evaluators](../src/prompt_learning/evaluators/) | Compute task metrics and Train feedback |
| [optimizer](../src/prompt_learning/optimization/optimizer.py) | Send only Train output/feedback columns to Prompt Learning; require nonempty text and the task placeholder |
| [selection](../src/prompt_learning/optimization/selection.py) | Apply Validation gates, report reasons and per-example regressions |
| [gateway](../src/prompt_learning/gateway.py) | Normalize model calls, cache validated responses, retry transient transport errors |
| [artifacts](../src/prompt_learning/artifacts.py) | Write local evidence with explicit export fields and atomic file replacement |
| [Phoenix adapter](../src/prompt_learning/observability/phoenix.py) | Optional metadata tracing, dataset identity sync and aggregate experiment records |

## Data and evaluation boundaries

The default [emotion](../configs/emotion.yaml) and [financial](../configs/financial.yaml)
configs use seed 42 and Train/Validation/Test ratios 0.6/0.2/0.2. Emotion samples up
to 20 examples for each of six labels and uses stratified splits; at least five
per label are required. Financial data is deduplicated by normalized article hash,
then split with a seeded allocation. At least five unique valid articles and a
nonempty allocation to each subset are required. The small README configs use
five examples per label or five financial articles and one candidate attempt.

Train diagnoses failures and supplies candidate-generation feedback. Validation
chooses candidates. Neither Train nor repeatedly consulted Validation is an
unbiased final performance estimate. Test must not inform candidate generation,
selection, or subsequent threshold tuning. Looking at Test and revising the
experiment requires a new untouched test set for a new generalization claim.

Emotion evaluation is deterministic: accuracy, macro F1, per-class
precision/recall/F1, confusion matrix and invalid-output rate. Financial evaluation
uses a JSON-constrained judge for financial figures, conciseness and faithfulness.
Only financial figures can be `not_applicable` when there are no material source
figures; applicable scores average on a 1–5 scale and are multiplied by 20.
Deterministic checks cover blank output, 40–800-character length, schema validity
and unsupported currency/percentage expressions. These checks do not prove the
absence of other factual errors.

Emotion selection requires nondecreasing macro F1, accuracy loss no greater than
0.01, no increase in invalid-output rate, and no class recall loss of 0.05 or more.
Financial selection requires nondecreasing overall score and mean faithfulness,
no increase in severe hallucinations, and at most 10% of examples losing 20 points
or more. Faithfulness 1 or an unsupported currency/percentage expression is a
severe hallucination. Complete metric ties prefer a shorter prompt, then the
incumbent. YAML cannot silently change these implemented policies: unsupported
values fail CLI validation.

Rejected candidates are useful evidence: a score average can hide class-level
or individual-example harm. The candidate prompt, aggregate Validation metrics,
reason codes and selected regression fields remain available for review. Missing
placeholders and empty optimizer output instead produce categorized history
entries; there is no invented prompt or candidate metric file for those attempts.

## Calls, cache and failure behavior

The OpenAI client disables nested SDK retries. The shared gateway retries
connection/time-out errors, HTTP 429 and HTTP 5xx with exponential backoff, up to
three retries (four attempts). Authentication, permission and bad requests fail
immediately. Invalid completed output fails validation without automatic schema
repair retries. Recoverable example failures retain their denominator; a subset
failure rate over 5% aborts the run. With tiny subsets, one failure can exceed that
limit. Artifact-write failure also fails the run; partial files can remain.

The SQLite cache hashes task, model, messages, temperature and parameters. It
stores response text and token counts, and revalidates cached text. It does not
store API keys or plaintext request metadata. Response text may contain model
copies of source content: keep `.cache/` local. Cached reads are not new API
measurements. Prompt Learning's meta-model requests use its SDK directly, outside
the shared gateway, its cache and its retry/accounting boundary.

`ModelResponse` and task frames carry token usage and elapsed gateway latency;
`estimate_cost_usd` accepts configured per-million-token prices. However, the
current CLI does not persist a complete usage/cost/latency ledger or export these
measurements for all model calls. `pricing: {}` means cost is unavailable, not
zero. Manifest timestamps measure the run interval, not isolated model latency.
No complete cost or latency claim should be inferred from these artifacts.

## Local artifact contract

The default root is `runs`, relative to the working directory, rather than the
committed historical `artifacts` directory. Each run owns a new UUID directory.
`artifact_root` and `cache_path` may be changed in YAML. Keep the executed YAML
alongside the results: the manifest records important settings and an evaluation
policy hash, but it is not a full copy of the input YAML or dependency lockfile.

| File under `runs/<task>/<run-id>/` | Meaning |
|---|---|
| `manifest.json` | Run/task ID, UTC start/end, configured revision, model names, seed, split ratios, input hashes, pricing and policy hash |
| `baseline_prompt.txt`, `selected_prompt.txt` | Initial and final selected prompt text |
| `iteration_0_prompt.txt`, `iteration_0_validation_metrics.json` | Baseline Validation evidence |
| `iteration_N_prompt.txt`, `iteration_N_validation_metrics.json` | Generated candidate and its Validation metrics, accepted or rejected |
| `iteration_history.csv` | Iteration, prompt version/hash, acceptance and categorized reasons |
| `validation_metrics.json` | Selected prompt's Validation aggregate metrics |
| `test_metrics.json` | Final selected prompt's Test aggregate metrics |
| `regressions.csv` | Example IDs and task-specific regression fields per iteration |
| `confusion_matrix.png` | Emotion Test matrix; label order is in `test_metrics.json` |
| `report.md` | Aggregate Validation/Test metrics and iteration decisions |

Metrics and regression exports use allowlists and do not reproduce article or
reference text. Prompts are saved verbatim; generated few-shot examples can still
contain source excerpts. Review prompts, traces and screenshots before sharing.
Do not commit `.env`, keys, raw source/reference records, cache databases, runtime
directories or Phoenix volume contents. Commit only reviewed prompt versions,
hashes, aggregate metrics, regression summaries and redacted evidence.

For replay, pin immutable Hugging Face commits rather than `main`, model snapshots
rather than moving aliases where available, and record dependency versions and
the code commit. The manifest records the configured dataset revision: it does
not automatically resolve `main` to its commit. The upstream optimizer Git
dependency and several packages are unpinned; installation is not bit-for-bit
reproducible. Fixed seed and split hashes reproduce data selection under fixed
inputs, but do not guarantee identical external-model outputs.

Test is evaluated only for the selected prompt. This runner does not measure
baseline-versus-selected Test improvement in the same run. A separate preplanned
comparison and sufficient independent data are necessary for that claim.

## Optional Phoenix

## Current experiment comparison

The recommended emotion workflow uses 100 examples per label and repeats the
run with seeds 42, 123 and 2024. `prompt-learning compare` reads only completed
run artifacts and writes a per-run CSV/Markdown table plus a summary CSV with
mean, min, max and population standard deviation. It never fabricates missing
metrics. Token, latency and cost fields are added only when the run artifact
contains them; pricing must be configured from a verified source.

The READMEs contain exact install/start/run/stop commands. `phoenix.enabled` in
YAML defaults to false; `PHOENIX_ENABLED` overrides it. Disabled means no observer
network calls. Enabled observer failures produce sanitized warnings and fall
back to local execution; they do not alter selection decisions.

[Compose](../docker-compose.yml) binds HTTP/UI port 6006 and gRPC port 4317 to
loopback and persists data in a named Docker volume. `PHOENIX_HTTP_PORT` and
`PHOENIX_GRPC_PORT` override host ports; update `PHOENIX_COLLECTOR_ENDPOINT` to the
HTTP base URL when changing ports. `PHOENIX_OTLP_ENDPOINT` overrides the full
trace URL. `PHOENIX_API_KEY`, if needed, is transport-only. Set `PHOENIX_IMAGE` to
a verified release/digest instead of the default moving `latest` for reproducible
server setup.

Spans reflect dataset load, baseline Validation, Train inference, feedback,
optimization, candidate Validation, selection and final Test. Exports contain
allowlisted identities, prompt hashes and numeric aggregates. Dataset sync uses
hashes and IDs, and experiment records contain aggregate metadata; this is not
per-example Phoenix task replay. The optional client/server API combination has
offline adapter coverage, not a verified live compatibility matrix. No current
screen capture is supplied.

## Verification and remaining limits

The default suite uses fake external clients and does not download datasets or
call paid models. Documentation checks resolve local links, execute the marked
offline shell examples with fixed revision fixtures, validate their generated
configs, and run each documented experiment command through missing-key
preflight in a temporary directory. They do not assert human prose.

LLM-judge bias and prompt injection, small samples, repeated Validation selection,
moving dependencies/models, incomplete accounting and unverified live integration
limit conclusions. Compare a sample with human judgments before trusting financial
scores. The [experiment record](../artifacts/experiment-summary.md) distinguishes
current verification from legacy material and records skipped checks.
