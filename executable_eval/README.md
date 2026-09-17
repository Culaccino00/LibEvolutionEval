# Executable version evaluation

This extension turns a reviewed subset of LibEvolutionEval into executable tests. It is intentionally narrower than a cross-version compatibility matrix. Every retained case establishes exactly two facts in one pinned target environment:

1. a completion that uses the older API fails for the expected version-related reason; and
2. a completion that uses the current API runs and passes a small semantic assertion.

This makes the score suitable for comparing a baseline that may receive stale context with Memix configured to resolve documentation for the target workspace version.

## Included sets

The `cases/` directory contains 212 manually reviewed cases across two libraries.

### Matplotlib 3.8.3 (167 cases)

- `matplotlib_3_8_3.json`: the initial 22 cases;
- `matplotlib_3_8_3_additional.json`: 38 widget, plotting, layout, font, subplot, and rendering migrations;
- `matplotlib_3_8_3_extended.json`: 31 API, constructor, ticker, widget, and rendering changes;
- `matplotlib_3_8_3_events.json`: 10 canvas-event dispatch migrations;
- `matplotlib_3_8_3_misc.json`: 8 date, font, text-path, picking, and colorbar changes;
- `matplotlib_3_8_3_mplot3d_collections.json`: 16 mplot3d and collection migrations;
- `matplotlib_3_8_3_backends_widgets.json`: 18 mathtext, backend, and widget changes;
- `matplotlib_3_8_3_toolkits_art.json`: 9 axisartist, axes_grid1, offsetbox, patches, and projections changes;
- `matplotlib_3_8_3_core_round2.json`: 15 pyplot, figure, image, legend, colorbar, and layout changes.

Together they cover removed and moved APIs, renamed and removed parameters, keyword-only migrations, class, method and function replacements, renamed attributes, event-dispatch changes, changed argument values, and renamed style resources. Representative changes include:

- `hist`: `normed` was removed in favor of `density`;
- `DivergingNorm` was removed in favor of `TwoSlopeNorm`;
- `LogLocator.base` was replaced by `set_params(base=...)`;
- canvas event helper methods were replaced by explicit event construction and dispatch;
- `plt.gca(projection=...)` keyword arguments were removed in favor of `plt.subplot(projection=...)`;
- `Collection.set_offset_transform` was renamed from `transOffset`; and
- bundled `seaborn-*` style names were renamed to `seaborn-v0_8-*`.

### PyTorch 2.2.0 (45 cases)

- `torch_2_2_0_legacy_linalg.json`: 15 removals of the pre-`torch.linalg` / legacy FFT APIs (`torch.eig`, `torch.solve`, `torch.lstsq`, `torch.matrix_rank`, `torch.fft`/`ifft`/`rfft`/`irfft` and their `Tensor` methods);
- `torch_2_2_0_distributed.json`: 17 removals of `*_multigpu` collectives, FSDP state-dict hooks, `params_with_grad`, and the 1.x `torch.distributed.tensor.parallel` style helpers;
- `torch_2_2_0_misc_round2.json`: 5 removals including `BufferedShuffleDataset`, `Graph.flatten_inps`/`unflatten_outs`, `check_compiler_abi_compatibility`, and `ProcessGroupRpcBackendOptions`;
- `torch_2_2_0_core_misc.json`: 2 `torch.symeig` / `Tensor.symeig` removals;
- `torch_2_2_0_quant_core.json`: 2 removals (`add_observer_`, `get_observer_dict`);
- `torch_2_2_0_nn_quantized.json`: 2 removals (`nn.quantized.ReLU`, `nn.quantized.functional.relu`);
- `torch_2_2_0_onnx.json`: 1 removal (`JitScalarType.from_name`);
- `torch_2_2_0_sig_core.json`: 1 signature change (`fake_quantize_per_channel_affine` gained a required `axis` argument).

Source versions vary per case: each records the last version in which the old API worked (mostly 1.6.0–2.0.0). Representative migrations include `torch.solve` → `torch.linalg.solve`, legacy `torch.fft(x, 1)` → `torch.fft.fft`, `all_gather_multigpu` → `all_gather`, and `make_input_shard_1d` → `distribute_tensor(..., [Shard(0)])`.

Each case records the source and target versions, both reference completions, the expected old-API error, assertions over the result, and provenance from the packaged API snapshots. Archived signatures are included when they are available in the packaged snapshots.

## Prepare the target environments

From the repository root:

```bash
python3 -m venv .venv-mpl383
.venv-mpl383/bin/python -m pip install \
  -r executable_eval/requirements-matplotlib-3.8.3.txt

python3 -m venv .venv-torch220
.venv-torch220/bin/python -m pip install \
  -r executable_eval/requirements-torch-2.2.0.txt
```

The exact library version is asserted inside every case. A different environment will therefore not produce a valid score.

## Validate the cases

Run both reference completions for every case of one library:

```bash
python3 -m executable_eval.validate_cases \
  --cases executable_eval/cases --library matplotlib \
  --python .venv-mpl383/bin/python

python3 -m executable_eval.validate_cases \
  --cases executable_eval/cases --library torch \
  --python .venv-torch220/bin/python
```

A case is valid only if the old completion exits with the recorded API error and the new completion exits successfully after its assertions run. Always match the `--library` filter to the interpreter: Matplotlib cases only validate under Matplotlib 3.8.3 and PyTorch cases only under PyTorch 2.2.0.

## Export model-facing tasks

Do not pass the case file itself to a model, because it contains reference answers and hidden assertions. Export a clean JSONL input first:

```bash
python3 -m executable_eval.export_tasks \
  --cases executable_eval/cases --library matplotlib \
  --output results/matplotlib_3_8_3_tasks.jsonl

python3 -m executable_eval.export_tasks \
  --cases executable_eval/cases --library torch \
  --output results/torch_2_2_0_tasks.jsonl
```

Use exactly these prompts for both systems. The experimental difference should be the memory supplied to the model: the baseline receives its normal or stale context, while Memix resolves the context for the target version.

## Score Memix or a baseline

Write one JSON object per line using this format:

```json
{"id":"matplotlib_hist_normed_removed","completion":"plt.hist(values, bins=2, density=True)"}
```

Then run, once per library with its matching interpreter:

```bash
python3 -m executable_eval.score_predictions \
  --cases executable_eval/cases --library matplotlib \
  --predictions path/to/predictions.jsonl \
  --python .venv-mpl383/bin/python \
  --json-report results/executable-score-matplotlib.json
```

Missing predictions count as failures. Unknown or duplicate IDs are rejected. In addition to the overall pass rate, the scorer reports pass rates by change type so that narrow categories do not remain hidden inside one aggregate number.

The repository includes per-library smoke-test inputs: `examples/reference_new_predictions_{matplotlib,torch}.jsonl` (expected 167/167 and 45/45) and `examples/reference_old_predictions_{matplotlib,torch}.jsonl` (expected 0/167 and 0/45). Regenerate them after changing the cases with:

```bash
python3 -m executable_eval.generate_reference_predictions \
  --cases executable_eval/cases \
  --output-dir executable_eval/examples --library matplotlib
python3 -m executable_eval.generate_reference_predictions \
  --cases executable_eval/cases \
  --output-dir executable_eval/examples --library torch
```

The completion is executed as Python code. Only run predictions and case files from sources you trust; the timeout is not a security sandbox.

## Find more candidates

The discovery command compares the API snapshots already shipped in `data/package_apis.tar.gz` without extracting the archive:

```bash
python3 -m executable_eval.discover_candidates \
  --old-version 3.5.2 \
  --new-version 3.8.3 \
  --kind signature_changed

python3 -m executable_eval.discover_candidates \
  --library torch \
  --old-version v_1.10.0 \
  --new-version v_2.2.0 \
  --kind removed
```

Note the `v_` prefix in the packaged PyTorch snapshot names.

Its output is only a review queue. A signature difference or apparent removal is not automatically a good task: compatibility aliases, ignored keyword arguments, documentation-page moves, and behavior that cannot be checked reliably should be excluded. Execution in the pinned target environment is the ground truth — most snapshot "removals" turn out to be documentation moves whose API still runs. A candidate should enter the benchmark only after it has a minimal assertion and passes the old-fails/new-passes validator in the pinned target environment.

`candidate_review.json` records candidates that were executed but rejected because their historical API still works in the target version (currently over 1300 entries). This prevents apparent documentation differences from being counted as genuine version-sensitive tasks, and should be checked before re-reviewing a candidate.
