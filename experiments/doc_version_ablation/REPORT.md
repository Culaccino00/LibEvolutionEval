# Doc-version ablation on the executable benchmark

Question: is documentation *version correctness* a primary factor in the pass
rate of the executable LibEvolutionEval benchmark?

## Setup

- Tasks: all 212 executable cases (167 Matplotlib 3.8.3, 45 PyTorch 2.2.0).
  Each task asks the model to fill the `<INSERT>` fragment inside a small
  function; the completion is executed and must pass semantic assertions in the
  pinned target environment.
- Model: `deepseek-v4-flash` (reasoning mode on — responses carry
  `reasoning_content`), temperature 0.0, max_tokens 8192 (endpoint cap; a 256k
  output budget is not supported by the endpoint and unnecessary for
  one-fragment completions).
- Four documentation configurations, docs taken from the packaged API
  snapshots (`data/package_apis.tar.gz`), labeled with their version:
  - `correct`: doc of the new-API callee from the target version snapshot;
  - `wrong`: doc of the old-API callee from the case's source version snapshot;
  - `none`: no documentation;
  - `mixed`: both of the above, order seeded-shuffled.
- Generation: `run_experiment.py` (responses cached under `responses/`).
  Three retry rounds cleared transient empty completions (reasoning tokens
  occasionally exhaust the 8192 budget and return empty `content`); residual
  empty completions are 10/7/15/11 out of 212 (correct/wrong/none/mixed) and
  count as failures.
- Scoring: `score_configs.py` runs `executable_eval.score_predictions` per
  library under the matching pinned venv.

## Headline results (all 212 tasks)

| config  | matplotlib (167) | torch (45) | total (212) |
|---------|------------------|------------|-------------|
| correct | 79 (47.3%)       | 37 (82.2%) | 116 (54.7%) |
| wrong   | 68 (40.7%)       | 13 (28.9%) | 81 (38.2%)  |
| none    | 70 (41.9%)       | 34 (75.6%) | 104 (49.1%) |
| mixed   | 79 (47.3%)       | 21 (46.7%) | 100 (47.2%) |

## Controlled comparison (187 tasks with a non-empty completion in all 4 configs)

| config  | matplotlib (145) | torch (42) | total (187) |
|---------|------------------|------------|-------------|
| correct | 71 (49.0%)       | 36 (85.7%) | 107 (57.2%) |
| wrong   | 63 (43.4%)       | 13 (31.0%) | 76 (40.6%)  |
| none    | 67 (46.2%)       | 34 (81.0%) | 101 (54.0%) |
| mixed   | 74 (51.0%)       | 21 (50.0%) | 95 (50.8%)  |

Failures attributed to old-API usage (completion calls the removed/renamed
old callee, common subset):

| config  | matplotlib | torch | total |
|---------|-----------|-------|-------|
| correct | 11        | 0     | 11    |
| wrong   | 27        | 16    | 43    |
| none    | 16        | 2     | 18    |
| mixed   | 11        | 11    | 22    |

Non-version-related failures are constant at ~68 across configs, i.e. the
entire pass-rate difference between configurations comes from
version-sensitive behavior.

## Conclusions

1. **Version correctness is the differentiating factor, and it is large on
   PyTorch**: correct docs 85.7% vs wrong docs 31.0% (a 55-point swing).
   Wrong-version docs nearly quadruple old-API failures (43 vs 11 overall).
2. **Stale context actively hurts below the no-doc baseline**: on torch,
   `wrong` (31.0%) is far below `none` (81.0%) — a baseline that retrieves
   outdated documentation is *worse than no retrieval at all*.
3. **Mixed context is contaminated by the stale half**: on torch, `mixed`
   lands at 50.0%, halfway between correct and wrong, with 11 old-API
   failures vs 0 for correct docs. Version-resolved retrieval (Memix) rather
   than "more context" is what matters.
4. **Matplotlib shows a weaker doc effect** (correct 49.0% vs wrong 43.4%)
   because the model's prior knowledge of matplotlib is strong; even so,
   wrong docs double old-API failures (27 vs 11) and `none` beats `wrong`.

## Caveats

- Doc hit rate is not 100%: for 34/212 cases the target snapshot has no entry
  for the new callee (`correct` then shows "(no matching documentation
  found)"); wrong-doc lookup misses 11. This slightly compresses the
  correct/wrong gap toward zero.
- Single model, single run at temperature 0; reasoning-mode empties (<7%,
  evenly spread) count as failures in every config.
- The matplotlib signal may grow with tasks sampled from less memorized APIs.

## Reproduce

```bash
python3 experiments/doc_version_ablation/run_experiment.py --dry-run
python3 experiments/doc_version_ablation/run_experiment.py --run --workers 8
python3 experiments/doc_version_ablation/score_configs.py
```
