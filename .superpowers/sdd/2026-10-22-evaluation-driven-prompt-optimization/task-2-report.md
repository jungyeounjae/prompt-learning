# Task 2 implementation report

Status: complete; implementation and report committed together.

## Scope and interfaces

Created the six files named in the task brief:

- `src/prompt_learning/datasets/__init__.py`
- `src/prompt_learning/datasets/base.py`
- `src/prompt_learning/datasets/emotion.py`
- `src/prompt_learning/datasets/financial_news.py`
- `tests/datasets/test_emotion.py`
- `tests/datasets/test_financial_news.py`

Inspected Task 1's `models.py` and `config.py` before implementation. Their public behavior and files were preserved. No out-of-scope interface correction was needed.

Public functions are `split_emotion(frame, *, seed, samples_per_label=20, ratios=SplitConfig())`, `split_financial(frame, *, seed, sample_count=60, ratios=SplitConfig())`, `load_emotion_dataset(config)`, and `load_financial_dataset(config)`. The optional ratios parameter is typed as `SplitConfig`. They return the existing `DatasetSplits` object.

## Implementation

Shared content identities use SHA-256 of NFKC Unicode-normalized, casefolded text with whitespace collapsed. Both adapters replace source IDs with these identities and deduplicate before sampling, preventing repeated source IDs or equivalent content from crossing subsets. Inputs are copied, never mutated. Blank, missing, or non-string source text is excluded; financial rows also require nonblank string summaries. No source text is written as an artifact.

Emotion maps integer labels 0–5 to the existing ordered six-label mapping, also accepts canonical `expected_label` values, rejects unknown labels and conflicting duplicate annotations, and names any label with fewer than five unique examples. Sampling is balanced and deterministic. Following the parent agent's explicit ruling, when every label has at least five rows but some have fewer than requested, all labels are capped to the smallest available class. When sufficient rows exist, exactly the requested count per class is retained. The two stratified sklearn splits use 60/40 then 50/50 and the supplied seed.

Financial normalization accepts case-insensitive article/content/text/document/source and reference_summary/summary/highlights/target/abstract aliases. It deduplicates articles before seeded shuffle and retains up to `sample_count` unique articles. A deterministic largest-remainder allocation assigns every retained row to exactly one subset, resolving ties in train/validation/test order. At least five unique articles and a requested count of at least five are required.

Both loaders import Hugging Face `datasets` lazily, pass the configured dataset name, revision, and split unchanged, and pass the seed, configured split ratios, and task-specific sample limit to the pure splitter. They load the full configured source split before deduplication, avoiding accidental duplicate/label shortages caused by prefix slicing. No network or paid API was used during testing.

Plan-interface correction requested by the parent before commit: the brief's pure split signatures omitted configurable split ratios even though Task 1 exposes YAML `split` settings. Added the optional typed `ratios` keyword without breaking the brief's existing calls. Both loaders pass `config.split`; defaults remain 60/20/20. Ratio validation matches the config loader's finite/nonnegative/sum requirements, and zero-ratio subsets are supported. Emotion retains the two-stage stratified method with the appropriate relative holdout fractions; financial uses configured quotas in its largest-remainder calculation. Extreme emotion ratios that cannot retain sufficient examples for stratification raise sklearn's descriptive `ValueError` rather than silently losing classes.

## TDD evidence

All commands used `/Users/yeounjae_jung/dev/prompt-learning/.venv/bin/python` in the assigned worktree.

1. Added emotion behavior tests before the emotion implementation. Actual RED from `-m pytest tests/datasets/test_emotion.py -v`:

   ```text
   collected 0 items / 1 error
   ModuleNotFoundError: No module named 'prompt_learning.datasets'
   1 error in 0.11s
   ```

   This is the missing-adapter collection failure explicitly expected by the task brief. After implementing the pure adapter, actual GREEN was `11 passed in 18.64s`.

2. Added financial tests before its implementation. Actual RED from `-m pytest tests/datasets/test_financial_news.py -v`:

   ```text
   collected 0 items / 1 error
   ModuleNotFoundError: No module named 'prompt_learning.datasets.financial_news'
   1 error in 0.10s
   ```

3. Loader sequencing correction: the first emotion implementation briefly included its loader before a loader test. I removed that loader, added the loader behavior test, observed RED, and then reimplemented it. Actual RED from `-m pytest tests/datasets/test_emotion.py -k loader -v`:

   ```text
   collected 12 items / 11 deselected / 1 selected
   FAILED test_emotion_loader_respects_revision_split_seed_and_sample_size
   ImportError: cannot import name 'load_emotion_dataset'
   1 failed, 11 deselected in 1.54s
   ```

4. After implementing financial normalization/splitting and the loaders, `-m pytest tests/datasets -v` reported `31 passed in 1.54s`.

5. Strengthened the existing emotion duplicate test to retain all five examples per class, so omission of deduplication cannot hide behind random subsampling. Added a regression assertion for the parent-approved balanced cap behavior. These passed against the implemented behavior; they supplement the initial RED/GREEN cases rather than representing additional feature implementation.

6. Before implementing the parent-requested ratio extension, added non-default ratio tests and changed loader tests to assert configured 50/25/25 allocations. Actual RED from `-m pytest tests/datasets -k 'configurable or loader' -v`: `4 failed, 30 deselected in 1.87s`, all due to unexpected `ratios` keyword arguments. Added zero-ratio and invalid-ratio coverage before implementing that support; actual RED from `-m pytest tests/datasets -k ratio -q`: `13 failed, 32 deselected in 1.54s`, again because `ratios` was not supported. After implementing the optional parameter, dynamic allocation, validation, and loader propagation, actual GREEN from `-m pytest tests/datasets -v`: `45 passed in 1.56s`.

## Verification and self-review

Final complete suite command: `-m pytest`.

```text
collected 72 items
tests/datasets/test_emotion.py .....................
tests/datasets/test_financial_news.py ........................
tests/test_config.py ...........................
72 passed in 1.45s
```

`git diff --check` exited zero. There were no failing tests, warnings, external requests, paid calls, generated third-party text artifacts, or Task 1 changes.

The 45 dataset cases cover stable row order for repeated seeds; different selection/order for other seeds; exact default counts; configured ratios; zero-ratio subsets; invalid ratio validation; emotion stratification; minimum-size splits; balanced capping; canonical label input; unknown/deficient/conflicting labels; normalized duplicate isolation; input immutability; article aliases; missing references and blank rows; largest-remainder totals and tie-breaking for several small sizes; invalid sampling limits; and loader revision/split/seed/limit/ratios propagation with local fake data sources. Fake loaders are restricted to the external HF boundary; split behavior remains real.

Self-review checked that deduplication precedes sampling, IDs identify content rather than source positions, every retained row belongs to exactly one subset, dataframes are copied, and source text is not persisted by adapter code.

## Limitations and integration notes

- Configurable emotion ratios need enough examples to stratify all six labels in each nonempty subset. Minimum five examples per label guarantees the default 60/20/20 case, but cannot guarantee arbitrarily small holdout proportions.
- The HF loader normally maintains its own dataset cache. No real loader was invoked here and adapter code itself writes no files.
- Reproducibility assumes the same ordered input and dependency behavior. Source revision pinning is the caller/configuration responsibility; loaders pass the configured revision through.
- NFKC/casefold/whitespace identity is deliberately conservative: text differing only by those forms is treated as the same content. The first valid financial summary is retained for duplicate articles; conflicting emotion labels fail explicitly.
