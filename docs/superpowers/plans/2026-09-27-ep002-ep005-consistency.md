# EP002–EP005 Consistency Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the public EP002 → EP003 → EP004 → EP005 path internally consistent, reproducible, and aligned with the channel without making the teaching examples harder to understand.

**Architecture:** Keep minimal examples thin, but route them through the same reusable contracts that define official experiments. Harden EP002 at the accepted-data boundary, correct EP003 timestamps without changing labels or metrics, scope EP004 decision-relevant diagnostics to the same train partition used downstream, and make EP005 minimal/full paths share dataset and split preparation. Add optional library-level manifest verification and require it in the full experiment CLIs.

**Tech Stack:** Python 3.11, pandas, numpy, scikit-learn, pytest, TOML configs, GitHub Actions.

**Spec:** `docs/superpowers/specs/2026-09-27-ep002-ep005-consistency-design.md`

## Global Constraints

- Canonical ownership stays: EP002 = data, EP003 = target, EP004 = feature set, EP005 = baseline.
- Minimal examples stay readable; removing detail must not change sample, split, target, or feature semantics.
- Source H1 `timestamp` is the MT5 bar-opening timestamp in UTC.
- EP003 `decision_timestamp` is the nominal close of bar `t`; `target_timestamp` is the nominal close of bar `t+1`.
- The frozen EP004 selected feature list must not change.
- Cyclical feature timestamps must not be shifted merely because EP003 decision timestamps are corrected.
- B0/B1/B2 keep the same 60/20/20 chronological protocol with target-boundary purging.
- Final test model performance remains locked and unevaluated.
- No broker credentials, large raw datasets, or private absolute paths are added to source-controlled reports.

## Review Focus

1. **Request with zero eligible completed bars:** must reject without writing `accepted.csv`, rather than creating an empty accepted dataset.
2. **Malformed timestamp outside an otherwise valid response:** must not silently pass because its range membership cannot be established.
3. **Manifest matches content but snapshot is copied to another local path:** verification should succeed by hash and record sanitized source identity, not require the original absolute path.
4. **ZERO targets near a split boundary:** EP004 train-only correlation scope and EP005 split must use the exact same eligible downstream train membership.
5. **Uniform +1h timestamp correction:** labels, features, partition membership, purge counts, and B0/B1/B2 metrics must remain identical while reported boundaries shift by exactly one hour.

---

### Task 1: Make public teaching scripts syntactically testable

**Files:**
- Create: `tests/test_public_scripts_compile.py`
- Modify: `examples/ep002_mt5_minimal.py`
- Modify: `.github/workflows/tests.yml`

**Interfaces:**
- Consumes: repository Python files under `examples/` and `experiments/`.
- Produces: a CI invariant that every public teaching/experiment script parses before pytest runs.

- [ ] **Step 1: Write the failing compile test**

Create `tests/test_public_scripts_compile.py` with a parametrized test that discovers `examples/**/*.py` and `experiments/**/*.py`, calls `compile(path.read_text(...), str(path), "exec")`, and asserts that at least one script was discovered.

The test must fail on the current literal `\n\n` text in `examples/ep002_mt5_minimal.py`.

- [ ] **Step 2: Run the test and verify the current failure**

Run:

```bash
pytest tests/test_public_scripts_compile.py -v
```

Expected: FAIL with `SyntaxError` pointing at `examples/ep002_mt5_minimal.py`.

- [ ] **Step 3: Fix only the broken minimal MT5 source**

Replace the literal escape text with real newlines so the visible sequence remains:

```python
mt5.symbol_select(symbol, True)

rates = mt5.copy_rates_range(...)
```

Do not add dependency injection or framework abstractions to this example.

- [ ] **Step 4: Run the compile test and verify it passes**

Run:

```bash
pytest tests/test_public_scripts_compile.py -v
```

Expected: PASS.

- [ ] **Step 5: Add an explicit compile step to GitHub Actions**

Add a step before pytest that runs:

```bash
python -m compileall -q examples experiments
```

- [ ] **Step 6: Commit**

```bash
git add tests/test_public_scripts_compile.py examples/ep002_mt5_minimal.py .github/workflows/tests.yml
git commit -m "test: compile public episode scripts in CI"
```

---

### Task 2: Harden the EP002 accepted-data boundary

**Files:**
- Modify: `tests/data/test_ep002_pipeline.py`
- Modify: `tests/data/test_snapshot.py`
- Modify: `src/trading_ai/data/time_policy.py`
- Modify: `src/trading_ai/data/validation.py`
- Modify: `src/trading_ai/data/pipeline.py`
- Modify: `src/trading_ai/data/snapshot.py`

**Interfaces:**
- Consumes: `MarketDataRequest`, raw MT5-shaped DataFrame.
- Produces: `TimePolicyResult.accepted_range`, explicit coverage metadata, and an acceptance decision that cannot write invalid/empty/misaligned H1 snapshots.

- [ ] **Step 1: Add failing pipeline tests for malformed, empty, and missing-boundary inputs**

In `tests/data/test_ep002_pipeline.py`, add:

```python
def test_malformed_source_timestamp_rejects_run(tmp_path):
    raw = _good()
    raw.loc[1, "timestamp"] = "not-a-time"
    result = process_h1_response(...)
    assert result["status"] == "rejected"
    assert "invalid_timestamp" in {i["code"] for i in result["validation_report"]["issues"]}
    assert not (run_dir / "accepted.csv").exists()
```

Add `test_malformed_timestamp_outside_otherwise_valid_response_still_rejects` by appending a row whose timestamp is unparseable while the normal requested rows remain valid; assert `invalid_timestamp` and no `accepted.csv`.\n\nAdd `test_empty_eligible_range_rejects_run` using raw rows only outside `[start, end)`; assert issue code `empty_eligible_range`.

Add `test_missing_requested_start_boundary_rejects_run` using rows at 07:00, 09:00, 10:00, 11:00 for the existing 08:00 request; assert `coverage_ok is False` and `requested_coverage_incomplete`.

Add `test_missing_required_completed_end_boundary_rejects_run` using 08:00–10:00 only; assert the same coverage issue.

- [ ] **Step 2: Run the new EP002 tests and verify they fail**

Run:

```bash
pytest tests/data/test_ep002_pipeline.py -v
```

Expected: the new tests FAIL against the current min/max coverage and filtering behavior.

- [ ] **Step 3: Make coverage boundary-based in `apply_h1_time_policy`**

Keep signature:

```python
def apply_h1_time_policy(
    raw: pd.DataFrame,
    request: MarketDataRequest,
) -> TimePolicyResult
```

Add coverage fields:

```text
first_eligible_timestamp
last_eligible_timestamp
required_start_timestamp
required_start_present
last_required_open_timestamp
required_completed_end_present
coverage_ok
```

Compute `required_start_present` by exact membership of `request.start` in eligible timestamps, not by comparing the raw minimum.

Compute `required_completed_end_present` by exact membership of `request.last_required_open` when it is not `None`.

- [ ] **Step 4: Reject malformed timestamps and empty candidates in the pipeline**

Keep `process_h1_response(...)` signature unchanged.

Before `accepted = not issues`:

- append `invalid_timestamp` when `time_result.exclusions["invalid_timestamp"] > 0`;
- append `empty_eligible_range` when `candidate.empty`;
- append `requested_coverage_incomplete` when `coverage_ok` is false.

Do not turn ordinary `before_start`, `at_or_after_end`, or `incomplete_by_cutoff` exclusions into errors.

- [ ] **Step 5: Add failing H1-alignment tests**

In `tests/data/test_snapshot.py`, add:

```python
def test_half_hour_timestamp_is_rejected():
    frame = _frame()
    frame.loc[1, "timestamp"] = pd.Timestamp("2026-01-01T01:30:00Z")
    with pytest.raises(ValueError, match="H1"):
        validate_h1_snapshot(frame)
```

Add a test where timestamps are hour-aligned but a consecutive delta is not exactly one hour; assert rejection.

In `tests/data/test_ep002_pipeline.py`, add a candidate with a half-hour timestamp and assert issue code `h1_timestamp_misaligned`.

- [ ] **Step 6: Run the alignment tests and verify they fail**

Run:

```bash
pytest tests/data/test_snapshot.py tests/data/test_ep002_pipeline.py -v
```

Expected: new alignment tests FAIL.

- [ ] **Step 7: Enforce H1 alignment in validation and accepted snapshots**

In `audit_h1_records(frame)`, add issue code `h1_timestamp_misaligned` when any parsed timestamp is not aligned to minute/second/microsecond/nanosecond zero.

In `validate_h1_snapshot(frame)`:

- reject non-hour-aligned timestamps;
- require every non-initial timestamp delta to equal exactly `pd.Timedelta(hours=1)`.

Use error messages containing `H1` so the new tests are stable.

- [ ] **Step 8: Add review-focus tests for normal exclusions**

Add:

- `test_out_of_range_rows_remain_normal_exclusions`: valid 07:00 plus 08:00–11:00 must still accept and report one `before_start`.
- `test_cutoff_with_no_completed_bar_rejects_empty_candidate`: cutoff before the first requested H1 bar completes must reject with `empty_eligible_range`.
- keep existing `test_incomplete_current_bar_is_excluded` passing.

- [ ] **Step 9: Run EP002 data tests**

Run:

```bash
pytest tests/data/test_ep002_pipeline.py tests/data/test_snapshot.py -v
```

Expected: PASS.

- [ ] **Step 10: Commit**

```bash
git add tests/data/test_ep002_pipeline.py tests/data/test_snapshot.py src/trading_ai/data/time_policy.py src/trading_ai/data/validation.py src/trading_ai/data/pipeline.py src/trading_ai/data/snapshot.py
git commit -m "fix: enforce the EP002 accepted H1 contract"
```

---

### Task 3: Verify EP002 provenance in downstream reports

**Files:**
- Modify: `tests/data/test_snapshot.py`
- Modify: `src/trading_ai/data/snapshot.py`
- Modify: `src/trading_ai/targets/report.py`
- Modify: `src/trading_ai/features/report.py`
- Modify: `src/trading_ai/experiments/baseline_dataset.py`
- Modify: `src/trading_ai/experiments/baseline_run.py`
- Modify: `experiments/ep003_forecast_horizon/analyze_target.py`
- Modify: `experiments/ep004_feature_engineering/analyze_features.py`
- Modify: `experiments/ep005_baselines/run_baselines.py`

**Interfaces:**
- Produces:
  - `dataset_manifest(frame, path, *, source_manifest_path=None) -> dict[str, Any]`
  - internal verification of an EP002 accepted manifest when `source_manifest_path` is provided.
- Downstream library functions keep manifest optional for synthetic/unit use.
- Full experiment CLIs require `--manifest` so official runs are provenance-verified.

- [ ] **Step 1: Write failing provenance tests**

In `tests/data/test_snapshot.py`, create a helper that writes a small valid `accepted.csv` and an EP002-shaped `manifest.json`.

Add:

```python
def test_dataset_manifest_verifies_matching_ep002_manifest(tmp_path):
    metadata = dataset_manifest(frame, csv_path, source_manifest_path=manifest_path)
    assert metadata["path"] == "accepted.csv"
    assert metadata["sha256"] == expected_csv_sha
    assert metadata["source_manifest"]["status"] == "accepted"
    assert metadata["source_manifest"]["accepted_dataset_sha256"] == expected_csv_sha
    assert metadata["source_manifest"]["manifest_sha256"]
```

Add `test_tampered_snapshot_fails_manifest_verification`; mutate the CSV after writing the manifest and assert `ValueError` matching `hash`.

Add `test_rejected_manifest_cannot_verify_accepted_snapshot`; set status to `rejected` and assert `ValueError`.\n\nAdd `test_manifest_without_accepted_dataset_artifact_is_rejected`; remove `files.accepted_dataset` and assert `ValueError`.

Add `test_dataset_manifest_does_not_expose_absolute_local_path`; assert the temporary directory string is absent from `json.dumps(metadata)`.

Add the review-focus case where the verified CSV is copied to a different directory/name but retains the same bytes; verification must succeed by hash.

- [ ] **Step 2: Run provenance tests and verify they fail**

Run:

```bash
pytest tests/data/test_snapshot.py -v
```

Expected: FAIL because `dataset_manifest` has no `source_manifest_path` support and currently records the supplied path.

- [ ] **Step 3: Extend `dataset_manifest`**

Implement exact signature:

```python
def dataset_manifest(
    frame: pd.DataFrame,
    path: str | Path,
    *,
    source_manifest_path: str | Path | None = None,
) -> dict[str, Any]:
```

Always record only `Path(path).name` in the public `path` field.

When `source_manifest_path` is provided:

- load JSON;
- require `status == "accepted"`;
- require `files.accepted_dataset.sha256`;
- compute the actual snapshot SHA-256;
- require exact hash equality;
- add `source_manifest` with:
  - `contract_id`;
  - `status`;
  - `manifest_sha256`;
  - `accepted_dataset_sha256`;\n  - `accepted_dataset_artifact` from the manifest entry (stored as a manifest-relative filename, never a local absolute path).

Do not expose the manifest's local filesystem location.

- [ ] **Step 4: Thread optional provenance through library report APIs**

Use these signatures:

```python
def analyze_h1_direction_target(
    path: str | Path,
    *,
    source_manifest_path: str | Path | None = None,
) -> dict[str, Any]
```

```python
def analyze_feature_contract(
    snapshot_path: str | Path,
    contract_path: str | Path,
    *,
    source_manifest_path: str | Path | None = None,
    experiment_contract_path: str | Path = "configs/experiments/ep005_baselines.toml",
) -> dict[str, Any]
```

```python
def assemble_episode005_dataset(
    snapshot_path: str | Path,
    feature_contract_path: str | Path,
    *,
    source_manifest_path: str | Path | None = None,
) -> BaselineDataset
```

```python
def run_episode005_validation_baselines(
    snapshot_path: str | Path,
    experiment_contract_path: str | Path = "configs/experiments/ep005_baselines.toml",
    *,
    source_manifest_path: str | Path | None = None,
) -> dict[str, Any]
```

Each should pass the manifest path into `dataset_manifest`; no report should add an absolute source path separately.

- [ ] **Step 5: Require `--manifest` in full experiment CLIs**

Add required argparse option:

```text
--manifest path/to/ep002-run/manifest.json
```

to EP003, EP004, and EP005 CLI scripts, and pass it into the library API.

This requirement applies to full recorded experiment entry points, not unit-test helper calls.

- [ ] **Step 6: Run targeted report/baseline tests**

Run:

```bash
pytest tests/data/test_snapshot.py tests/targets tests/features tests/experiments -v
```

Expected: PASS after adapting any call sites required by the new optional keyword.

- [ ] **Step 7: Commit**

```bash
git add tests/data/test_snapshot.py src/trading_ai/data/snapshot.py src/trading_ai/targets/report.py src/trading_ai/features/report.py src/trading_ai/experiments/baseline_dataset.py src/trading_ai/experiments/baseline_run.py experiments/ep003_forecast_horizon/analyze_target.py experiments/ep004_feature_engineering/analyze_features.py experiments/ep005_baselines/run_baselines.py
git commit -m "feat: verify EP002 provenance downstream"
```

---

### Task 4: Correct EP003 decision and target timestamps without changing labels

**Files:**
- Modify: `tests/targets/test_direction.py`
- Modify: `tests/experiments/test_baseline_run.py`
- Modify: `src/trading_ai/targets/direction.py`
- Modify: `examples/ep003_target_minimal.py`

**Interfaces:**
- Consumes: source bar-opening `timestamp`.
- Produces:
  - `decision_timestamp = timestamp + 1h`;
  - `target_timestamp = timestamp.shift(-1) + 1h`;
  - unchanged `future_return_1h` and `target_h1_direction`.

- [ ] **Step 1: Change target tests to the nominal-close contract**

In `test_h1_direction_target_is_aligned_to_decision_bar`, assert:

```python
assert result["decision_timestamp"].iloc[0] == frame["timestamp"].iloc[0] + pd.Timedelta(hours=1)
assert result["target_timestamp"].iloc[0] == frame["timestamp"].iloc[1] + pd.Timedelta(hours=1)
```

Keep existing label and return assertions unchanged.

Add a test that stores expected labels/returns from the explicit close-to-close formula and asserts the corrected timestamp representation does not alter them.

- [ ] **Step 2: Run direction tests and verify they fail**

Run:

```bash
pytest tests/targets/test_direction.py -v
```

Expected: FAIL on the new +1h assertions.

- [ ] **Step 3: Correct timestamps in `build_h1_direction_target`**

Preserve the exact one-hour eligibility check on **source opening timestamps**.

Set:

```python
decision_timestamp = timestamp + pd.Timedelta(hours=1)
target_timestamp = timestamp.shift(-1) + pd.Timedelta(hours=1)
```

Do not use corrected close timestamps to decide whether source rows are consecutive.

- [ ] **Step 4: Update the minimal EP003 example**

Keep the visible formula:

```python
future_return = df["close"].shift(-1) / df["close"] - 1.0
```

but report decision/target timestamps using the same nominal-close semantics as the reusable implementation.

- [ ] **Step 5: Add the uniform-shift baseline regression test**

In `tests/experiments/test_baseline_run.py`:

1. write the existing synthetic snapshot;
2. run `run_episode005_validation_baselines` normally;
3. monkeypatch `trading_ai.experiments.baseline_dataset.build_h1_direction_target` with a wrapper that calls the real corrected implementation then subtracts one hour from only `decision_timestamp` and `target_timestamp`;
4. run the baseline report again;
5. assert:
   - `validation_metrics` are exactly equal;
   - train/validation/test row counts are equal;
   - purge counts are equal;
   - corrected `validation_start` and `test_start` are exactly one hour later than the legacy-style timestamps.

This pins Review Focus item 5.

- [ ] **Step 6: Run target and baseline regression tests**

Run:

```bash
pytest tests/targets/test_direction.py tests/experiments/test_baseline_run.py -v
```

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add tests/targets/test_direction.py tests/experiments/test_baseline_run.py src/trading_ai/targets/direction.py examples/ep003_target_minimal.py
git commit -m "fix: use nominal H1 close timestamps for EP003"
```

---

### Task 5: Share EP005 sample and split preparation with the minimal baseline

**Files:**
- Modify: `tests/experiments/test_baseline_run.py`
- Modify: `src/trading_ai/experiments/baseline_dataset.py`
- Modify: `src/trading_ai/experiments/baseline_run.py`
- Modify: `examples/ep005_baseline_minimal.py`

**Interfaces:**
- Produces:
  - `PreparedEpisode005Split` dataclass with `dataset`, `split`, and `contract`;
  - `prepare_episode005_split(snapshot_path, experiment_contract_path=..., *, source_manifest_path=None) -> PreparedEpisode005Split`.
- Full baseline runner and minimal B0 example both consume this helper.

- [ ] **Step 1: Write a failing shared-preparation test**

In `tests/experiments/test_baseline_run.py`, add a test that imports `prepare_episode005_split` and asserts:

```python
prepared = prepare_episode005_split(snapshot)
report = run_episode005_validation_baselines(snapshot)

assert len(prepared.split.train) == report["split"]["train_rows"]
assert len(prepared.split.validation) == report["split"]["validation_rows"]
assert len(prepared.split.test) == report["split"]["test_rows"]
assert prepared.split.purged_train_rows == report["split"]["purged_train_boundary_rows"]
assert prepared.split.purged_validation_rows == report["split"]["purged_validation_boundary_rows"]
```

- [ ] **Step 2: Run the test and verify it fails**

Run:

```bash
pytest tests/experiments/test_baseline_run.py -v
```

Expected: FAIL because `prepare_episode005_split` does not yet exist.

- [ ] **Step 3: Implement shared preparation in `baseline_dataset.py`**

Add:

```python
@dataclass(frozen=True)
class PreparedEpisode005Split:
    dataset: BaselineDataset
    split: ChronologicalSplit
    contract: dict[str, Any]
```

Add exact function:

```python
def prepare_episode005_split(
    snapshot_path: str | Path,
    experiment_contract_path: str | Path = "configs/experiments/ep005_baselines.toml",
    *,
    source_manifest_path: str | Path | None = None,
) -> PreparedEpisode005Split
```

It must:

- read the EP005 experiment contract;
- resolve the feature contract relative to repository/config location using the existing rule;
- validate train + validation + test fractions sum to 1;
- call `assemble_episode005_dataset`;
- call `chronological_split`;
- return all three values without fitting a model.

- [ ] **Step 4: Refactor the full baseline runner to consume the helper**

`run_episode005_validation_baselines` must use `prepare_episode005_split` and remove duplicate contract/split assembly.

Model fitting/evaluation stays unchanged.

- [ ] **Step 5: Refactor the minimal EP005 example to consume the helper**

The visible model section should remain:

```python
y_train = prepared.split.train["target_h1_direction"]
y_validation = prepared.split.validation["target_h1_direction"]

majority_class = y_train.value_counts().idxmax()
prediction = pd.Series(majority_class, index=y_validation.index)
accuracy = (prediction == y_validation).mean()
```

Do not rebuild the target or positional split in the example.

Add optional CLI args:

```text
snapshot path
--manifest
--contract
```

The manifest may remain optional in the minimal teaching script; the full experiment CLI from Task 3 requires it.

- [ ] **Step 6: Add a minimal/full B0 equivalence test**

Run the minimal example with the synthetic snapshot via `subprocess.run`, and assert its printed validation row count and B0 accuracy, rounded to four decimals, match the full report's B0 validation metrics.

- [ ] **Step 7: Run baseline tests**

Run:

```bash
pytest tests/experiments/test_baseline_run.py -v
```

Expected: PASS.

- [ ] **Step 8: Commit**

```bash
git add tests/experiments/test_baseline_run.py src/trading_ai/experiments/baseline_dataset.py src/trading_ai/experiments/baseline_run.py examples/ep005_baseline_minimal.py
git commit -m "refactor: share EP005 sample and split preparation"
```

---

### Task 6: Scope EP004 correlation diagnostics to the downstream train partition

**Files:**
- Create: `tests/features/test_report.py`
- Modify: `src/trading_ai/features/report.py`
- Modify: `experiments/ep004_feature_engineering/analyze_features.py`

**Interfaces:**
- Consumes: `prepare_episode005_split(...)` from Task 5.
- Produces: EP004 report with full-history descriptive availability diagnostics plus train-only correlation/redundancy diagnostics.

- [ ] **Step 1: Write the failing feature-report scope test**

Create `tests/features/test_report.py` with a synthetic H1 CSV long enough for the 24-hour warmup and three-way split.

Call:

```python
report = analyze_feature_contract(snapshot, feature_contract)
prepared = prepare_episode005_split(snapshot)
```

Assert:

```python
assert report["selected_features"] == prepared.dataset.selected_features
assert report["correlation_scope"]["policy"] == "episode005_train_only"
assert report["correlation_scope"]["rows"] == len(prepared.split.train)
assert report["correlation_scope"]["validation_start"] == prepared.split.validation_start.isoformat()
assert report["correlation_scope"]["test_used"] is False
```

Add a review-focus fixture containing a ZERO target immediately before a split boundary; assert the correlation-scope row count still equals the exact prepared train membership rather than a simple first-60%-of-bars approximation.

- [ ] **Step 2: Run the new feature-report tests and verify failure**

Run:

```bash
pytest tests/features/test_report.py -v
```

Expected: FAIL because current correlations use the whole feature frame and no scope metadata exists.

- [ ] **Step 3: Build correlations only from exact EP005 train membership**

In `analyze_feature_contract`:

1. build features as today;
2. call `prepare_episode005_split(snapshot_path, experiment_contract_path, source_manifest_path=...)`;
3. derive source bar-open timestamps from exact train decisions:

```python
train_open_timestamp = (
    pd.to_datetime(prepared.split.train["decision_timestamp"], utc=True)
    - pd.Timedelta(hours=1)
)
```

4. select candidate-feature rows whose `features["timestamp"]` are in that exact set;
5. call `_top_correlations` on only those rows.

Keep `feature_diagnostics` over the whole accepted snapshot as descriptive availability diagnostics.

Add:

```json
"correlation_scope": {
  "policy": "episode005_train_only",
  "rows": ...,
  "validation_start": "...",
  "test_used": false
}
```

Update `correlation_note` to say the selected list remains predeclared and the redundancy diagnostics use train-only downstream membership.

- [ ] **Step 4: Pass experiment contract from the EP004 CLI**

The CLI already gains `--manifest` in Task 3. Add optional:

```text
--experiment-contract configs/experiments/ep005_baselines.toml
```

and pass it to `analyze_feature_contract`.

- [ ] **Step 5: Run feature tests**

Run:

```bash
pytest tests/features -v
```

Expected: PASS, including the unchanged predeclared selected-feature test.

- [ ] **Step 6: Commit**

```bash
git add tests/features/test_report.py src/trading_ai/features/report.py experiments/ep004_feature_engineering/analyze_features.py
git commit -m "fix: scope EP004 redundancy diagnostics to train"
```

---

### Task 7: Synchronize documentation and run whole-path verification

**Files:**
- Modify: `README.md`
- Modify: `docs/architecture.md`
- Modify: `docs/episodes/ep002-data.md`
- Modify: `docs/episodes/ep003-target.md`
- Modify: `docs/episodes/ep004-features.md`
- Modify: `docs/episodes/ep005-baseline.md`
- Modify: `experiments/ep002_market_data/README.md`
- Modify: `experiments/ep003_forecast_horizon/README.md`
- Modify: `experiments/ep004_feature_engineering/README.md`
- Modify: `experiments/ep005_baselines/README.md`

**Interfaces:**
- Consumes: all corrected contracts from Tasks 1–6.
- Produces: one public learning path whose commands and terminology match the code.

- [ ] **Step 1: Update only contract-relevant documentation**

Document these exact points:

- MT5 H1 `timestamp` is the bar-opening time in UTC.
- EP003 decision timestamp is nominal bar close = source opening + 1h.
- EP003 target timestamp is nominal next-bar close.
- The future-return formula and labels do not change because of that timestamp correction.
- Full EP003/EP004/EP005 recorded-run commands require the originating EP002 `manifest.json`.
- EP004 frozen selected features remain unchanged.
- EP004 redundancy correlations use exact downstream train membership; whole-history availability diagnostics are descriptive only.
- EP005 minimal and full paths share sample/split preparation.
- Final test model performance is not evaluated.
- Official measured values come from full experiment reports, not minimal examples.

Keep the root README concise and put edge-case details in episode docs.

- [ ] **Step 2: Verify documented commands match argparse**

Check each command in the three full experiment READMEs contains the required `--manifest` and any new EP004 `--experiment-contract` option only where needed.

- [ ] **Step 3: Run the complete test suite**

Run:

```bash
pytest -v
```

Expected: all tests PASS.

- [ ] **Step 4: Compile all public scripts**

Run:

```bash
python -m compileall -q examples experiments
```

Expected: exit code 0 and no output.

- [ ] **Step 5: Run whitespace/diff validation**

Run:

```bash
git diff --check main...HEAD
```

Expected: no output.

- [ ] **Step 6: Verify no test metrics appear in the EP005 report contract**

Run the synthetic baseline test/report path and inspect/assert:

```text
report["test"]["evaluated"] == False
"validation_metrics" exists
no "test_metrics" key exists
```

- [ ] **Step 7: Commit documentation**

```bash
git add README.md docs/architecture.md docs/episodes experiments/*/README.md
git commit -m "docs: align EP002-EP005 with corrected contracts"
```

- [ ] **Step 8: Push branch and wait for GitHub Actions**

Push `fix/ep002-ep005-consistency`.

Verify the workflow for the final head commit completes with conclusion `success`.

- [ ] **Step 9: Create the corrective PR**

Title:

```text
Align EP002-EP005 data, time, diagnostics, and baseline contracts
```

PR body must summarize:

- EP002 syntax/acceptance fixes;
- EP003 nominal-close timestamp correction with metric invariance;
- EP004 train-only redundancy diagnostic scope;
- EP005 shared minimal/full sample and split semantics;
- EP002 manifest verification;
- final test still locked;
- real EURUSD reports/video assets intentionally deferred to a separate follow-up.

Do not merge until whole-branch review is complete.
