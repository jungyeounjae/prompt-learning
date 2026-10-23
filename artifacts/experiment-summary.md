# Experiment evidence record

Recorded on 2026-10-23. This repository is a reproducible prompt-optimization
experiment tool. The current shared pipeline has **no verified real model
execution yet**. No new model metrics, cost, latency, dataset commit, acceptance
outcome or Phoenix screenshot is claimed here.

The authorized worktree had no exported usable `OPENAI_API_KEY` and no local
`.env`. Credentials in other workspaces were not read or used. Paid experiments,
dataset downloads and live Phoenix integration were therefore not performed.
Offline tests use deterministic fixtures and fake external boundaries; their
results are software verification, not evidence of improved model performance.

## Reproduce the planned small experiments

Run from the repository root in an installed Python 3.12+ environment. Follow the
complete setup, secure key input, dataset-commit resolution and offline config
generation in [Korean](../README.ko.md) or [Japanese](../README.md). These generate
`runs/configs/emotion.yaml` and `runs/configs/financial.yaml` from the defaults,
preserving the split and approval policy. They use seed 42, one candidate attempt,
all model roles set to `gpt-4o-mini`, emotion sampling of five examples per label
(30 total), and five unique financial articles (3/1/1 Train/Validation/Test).

```bash
export PHOENIX_ENABLED=false
python -m prompt_learning.cli run --config runs/configs/emotion.yaml
python -m prompt_learning.cli run --config runs/configs/financial.yaml
```

For each actual run, retain its executed YAML, code commit, dependency versions,
model snapshots/access, immutable dataset commits and UTC start/end. Runtime
results appear in `runs/<task>/<run-id>/`; see the
[artifact contract](../docs/architecture.md). Review aggregate metrics, prompt
versions/hashes, rejection reasons and regressions before curating them here.
If Phoenix is used, retain only reviewed screenshots with credentials, account
details and third-party source/reference text redacted.

The minimal financial Validation/Test sets have one article each. These are
execution checks and cannot support a reliable generalization estimate.
Zero accepted candidates is a valid result when the gates prevent regressions.
Final Test evaluates only the selected prompt, so the run alone cannot establish
a baseline-versus-selected Test improvement.

## Current software verification

Local checks on Python 3.12.15 passed: seven documentation checks and the full
346-test offline suite; installed CLI help without a key; Compose configuration;
and whitespace validation. These are software checks, not model or Phoenix
quality results.

`python -m build` could not run because the environment lacks the `build` frontend.
As an offline fallback, the installed `setuptools.build_meta` backend successfully
created both the sdist and wheel under ignored `dist/`. The fallback does not
verify frontend build isolation. Re-run the documented install step to obtain
`build`, then run:

```bash
python -m pytest tests/test_documentation.py -v
python -m pytest -v
python -m build
prompt-learning --help
docker compose config
git diff --check
```

The exact backend fallback used was:

```bash
python -c 'from setuptools.build_meta import build_sdist, build_wheel; print(build_sdist("dist")); print(build_wheel("dist"))'
```

Cost and latency remain unmeasured. Empty configured pricing means unavailable,
not free. The current CLI does not persist a complete model usage/cost/latency
ledger, and optimizer SDK calls bypass shared gateway accounting. A complete
measured experiment needs additional accounting evidence; UTC run duration is
not API latency. See [implementation limits](../docs/architecture.md).

## Legacy evidence — separate pipeline, not current results

The following committed files predate isolated Train/Validation/Test selection:

- [Emotion prompt](emotion/optimized_prompt.txt): legacy generated prompt; no
  independent Test verification or current-pipeline provenance.
- Financial [iteration 1](financial/iteration_1_prompt.txt),
  [iteration 2](financial/iteration_2_prompt.txt),
  [iteration 3](financial/iteration_3_prompt.txt) and
  [selected prompt](financial/optimized_prompt.txt): legacy prompt versions.
- [Financial aggregate history](financial/iteration_history.csv): historical
  scores from the retired experiment loop, not isolated current Validation/Test
  results. These numbers do not establish current performance improvement.

These files are retained for historical context, not replayed as a new run.
Their old dataset/model aliases, split provenance, pricing, timing and dependency
versions are insufficient for an exact reconstruction. No new article or
reference text is added to this record. Generated legacy prompts may include
source-derived examples; inspect their contents and usage rights before sharing.
