# EURUSD H1: features, forecast horizons, and uncertainty checks

This package documents a completed, exploratory EURUSD H1 study and its follow-up diagnostics. It does not contain raw quotes, per-origin predictions, model weights, feature matrices, or private market files.

## What the lab asks

Can a compact point-in-time feature set and history length `W` help estimate the average next-`H` observed-bar return and its within-window dispersion? The first study evaluated `W ∈ {16, 32, 64, 96}` together with `H ∈ {1,…,24}` using six registered feature masks and Ridge (`alpha=1`). Follow-up analyses fixed a W16/core Ridge H8 mean forecast and asked whether horizon agreement, forecast magnitude, or a separate variance estimate changed how its historical results should be read.

The study did **not** establish a stable direction forecast or a trading strategy. The feature/target selector abstained in all eight outer fold/track decisions. Later H8 analyses are fixed-model exploratory diagnostics, not evidence that the first grid selected W16/H8.

## Source, schema, and time boundary

- Source identified by the audit: a MetaTrader 5 **demo** EURUSD H1 OHLC export, with UTC bar-open timestamps and OHLC plus tick-volume/spread fields. Those raw quotes are deliberately not included.
- The fitted feature/target-selection run read only the first 10,621 rows, from the `2024-10-16 00:00 UTC` bar open through the `2026-07-03 15:00 UTC` bar open. Its decision/last target endpoint was bounded at `2026-07-03 16:00 UTC`.
- Audited prefix SHA-256: `141318d453eb62941f0cf7d03d6a516770cee52a5131fb15d44fb4c5f0e9a270` (64 hex characters). The earlier full-file checksum is not a substitute for the prefix hash.
- The original v3 report contains a 66-character typo in its displayed prefix digest. This package's report copy uses the verified 64-character value recorded in the run summary and source-cutoff manifest; the original research artifact was left unchanged.
- Input CSV contract: `time_utc,open,high,low,close` are required; optional source fields include `tick_volume,spread`. Timestamps must be unique, strictly increasing, and parse as UTC; OHLC must be finite, positive, and satisfy high/low consistency. The current scripts validate these constraints.
- A row's source timestamp denotes the bar open; the close is available only after that bar completes. The first forecast origin therefore needs completed data through the origin bar close.

The run treats adjacent **observed** rows as one transition, even across a long timestamp gap. It never interpolates, resets history, divides a return by elapsed calendar time, or uses future timestamps/gap masks as predictors. The core includes `log(open_i / close_(i−1))`; elapsed durations and gap counts remain audit/label metadata. On the 10,621-row prefix, 98 adjacent intervals were not exactly one calendar hour; the cause of each was not inferred.

## Target and feature contract

For the next `H` observed close-to-close transitions, define `r_j = log(C_(i+j) / C_(i+j−1))`, `y_mean(H) = mean(r_1,…,r_H)`, and `y_var(H) = mean((r_j−y_mean)^2)` with `ddof=0`. H=1 is mean-only because its variance target is exactly zero. Features include past OHLC log-ratios, same-bar and lagged OHLC cross-ratios, log-price differences of orders 1–3, and causal univariate/mixed moments. No log is applied to a potentially negative price difference. Feature windows use completed observed transitions only.

W and H were evaluated jointly on one common cohort of 10,501 origins. The selector used four chronological outer folds, two inner validation windows within each outer fold, H24 target-end purging, train-only standardization, and a frozen positive-inner-skill rule. The 6,163 outer validation origins were used for the historical follow-ups and have been examined; they are **not an untouched holdout**.

## Results at a glance

- The v3 selector completed 4,608 inner Ridge fits in 351.7 process-CPU seconds on one CPU thread. It recorded a W×H Pareto frontier, but all eight outer fold/track decisions abstained; there was no selected outer winner.
- Across pooled inner candidate rows for H2–24, mean-return skill versus a zero baseline had median `−12.54%`; only `60/4,416` candidate rows were positive. This is a pooled, overlapping, exploratory distribution, not an outer score for a chosen model.
- A separate fixed W16/core Ridge H8 follow-up had direction accuracy `3,118/6,151 = 50.69%` after excluding 12 exact-zero realized targets. Horizon agreement did not improve the H8 result: H4/H8 agreement accepted 74.9% of origins and scored 50.5% on accepted versus 51.3% on rejected origins.
- Relative strength bins were not monotonic: H8 direction hit rates ranged from 48.3% to 51.9%. Weak predicted movement did not reliably imply a small realized move.
- Four variance-only Ridge fits estimated the next-eight-observed-return **within-window** population variance. Its validation MAE was `5.394e−7`, versus `7.523e−7` for the fold train-mean baseline: a 28.3% reduction in variance MAE. This is not direction accuracy, trading profit, or uncertainty of the H8 mean.
- In a separate no-fit Gaussian-reference check, `N(predicted H8 mean per observed transition, predicted within-window variance)` covered 79.97% of individual observed returns at ±1 sigma and 95.99% at ±2 sigma across overlapping H8 windows. Those percentages are not probabilities of a trade, nor independent samples; the normal curve is a working assumption, not a consequence of “market noise.”

All of these are exploratory historical findings. No profitability, statistical significance, live signal, or stable directional edge is claimed. Variance was more useful than its simple baseline by MAE in this sample; a stable directional forecast remains unestablished.

## Reproduction

The repository root declares NumPy, pandas, scikit-learn, and pytest in `pyproject.toml`; the v3 fit runner also imports SciPy and threadpoolctl, provided through the scientific Python stack. The no-fit tests use deterministic fixtures only:

```bash
python -m pip install -e '.[dev]'
python -m pytest -q experiments/eurusd_h1_feature_target_lab/tests
```

### From the authorized CSV to v3

The packaged scripts provide a path to rebuild the feasibility report, feature schema, point-in-time matrices, targets, purged splits, and v3 Ridge selection run **only** when supplied the matching 10,621-row prefix. The fit runner checks the pinned prefix SHA before reading the prefix; a different prefix fails closed and requires a new versioned protocol. This CSV-to-v3 command sequence was not run during package preparation. The input file stays outside this package.

```bash
CSV_PATH=/path/to/authorized/EURUSD_Model_Error_Lab_H1_UTC_20241016_20261017.csv
python experiments/eurusd_h1_feature_target_lab/run_feasibility_v3.py \
  --source "$CSV_PATH" \
  --output experiments/eurusd_h1_feature_target_lab
python experiments/eurusd_h1_feature_target_lab/run_exploratory_fit_v3.py \
  --source "$CSV_PATH"
```

This chain writes the large feature/target and prediction artifacts under the package's local `results/` folder. It does not copy the CSV into the repository. The v3 fit is a 4,608-fit market experiment; the commands above are documentation and were **not** run while preparing this package.

### Downstream historical analyses and exact dependencies

The package includes the upstream scripts and frozen protocols. The scripts are hash-pinned to the original research runs where applicable; the aggregate evidence in `evidence/` is reviewable without those omitted files.

| Stage | What the packaged code needs | What is included / reproducibility boundary |
|---|---|---|
| Schema, feasibility, v3 fit | Authorized CSV with the exact pinned prefix; v3 protocol and source code | Code/protocol/tests are included. The exact-prefix route is wired in code but was not executed here. A changed prefix needs a new protocol. |
| H4/H8/H24 agreement | A completed v3 run: feature matrix/catalog, target manifest, outer split membership, run summary, and v3 protocol; the fixed 12-fit agreement protocol | `run_horizon_agreement.py` is included with `--source-run`. Its fit was not rerun for this package. |
| H8 strength grading | Agreement outer predictions, model coefficients/scalers, fit index/run summary, plus v3 features/targets/splits | `grade_strength_bins.py` and its hash-pinned protocol are included. Historical aggregate bins are included; upstream per-origin/parameter artifacts are not. Its hash checks intentionally reject substituted upstream outputs until a new analysis protocol locks their hashes. |
| Mean/variance cells | V3 features/catalog/targets/splits and the saved agreement H8 predictions, fit metadata, coefficients/scalers and run summaries, plus matching inventory and protocol | `run_joint_utility.py` and frozen inventory/protocol are included. The historical aggregate tables are included, but required upstream binaries/per-origin outputs are excluded. A fresh run needs those inputs and a protocol/inventory frozen for them. |
| Gaussian coverage | Matching joint-run predictions, H1–H8 target manifest and the corresponding artifact inventory | No-fit diagnostic code/protocol and aggregate coverage are included. Per-origin predictions and nested targets are excluded. |
| Review saved results | Sanitized reports and aggregate metric tables | Included. These support review of the reported figures but do not reconstruct every per-origin metric, coefficient, threshold, or prediction. |

For a fresh v3 run, the agreement command is:

```bash
python experiments/eurusd_h1_feature_target_lab/run_horizon_agreement.py \
  --source-run experiments/eurusd_h1_feature_target_lab/results/<v3_run_id>
```

This is a new run ID, not a byte-for-byte replay of the saved historical run. The strength and joint utility protocols pin particular historical artifact hashes. To analyze newly generated upstream outputs, freeze new, versioned protocols and inventories **before** calculating bins or fitting a variance head; do not edit these archived protocols after looking at outcomes.

The preflight command for the archived joint-utility inputs is:

```bash
python experiments/eurusd_h1_feature_target_lab/run_joint_utility.py --preflight-only
```

The four variance fits are run only by omitting `--preflight-only`; that fit was not run during this packaging task. After a successful joint run, the Gaussian command takes its run directory, matching v3 run, and inventory:

```bash
python experiments/eurusd_h1_feature_target_lab/gaussian_band_diagnostic.py \
  experiments/eurusd_h1_feature_target_lab/results/<joint_run_id> \
  --v3-run experiments/eurusd_h1_feature_target_lab/results/<v3_run_id> \
  --artifact-inventory experiments/eurusd_h1_feature_target_lab/artifact_inventory.json
```

**Missing from this package:** the private CSV, feature matrix, target/split manifests, saved outer predictions, fitted coefficients/scalers, checkpoints, and per-origin diagnostic outputs. These remain in the existing local research artifacts, which are untracked and were not copied. Thus the exact historical downstream runs cannot be rebuilt from a clean clone using this package alone. The package does not promise a one-command reproduction of all reported results; it states which code and inputs each stage needs.

## Files

- `VOICEOVER_EN.md`, `SCENE_PLAN_UK.md`, `CREATIVE_BRIEF.md`: video package.
- `EVIDENCE_MAP.md`: script-claim-to-evidence map.
- `feature_target_selection_v3.py`, `run_feasibility_v3.py`, `run_exploratory_fit_v3.py`: point-in-time feature/target and bounded Ridge experiment code.
- `run_horizon_agreement.py`, `grade_strength_bins.py`: saved-prediction agreement and strength analyses.
- `run_joint_utility.py`, `gaussian_band_diagnostic.py`: later variance and no-fit Gaussian diagnostic code.
- `tests/`: synthetic contract and no-fit helper tests.
- `protocol_feature_target_selection_v3.json`, `horizon_agreement_protocol.json`, `strength_analysis_protocol.json`, `protocol_frozen_pre_fit.json`, `gaussian_diagnostic_protocol.json`, `artifact_inventory.json`: frozen protocols and input checksum inventory.
- `evidence/`: sanitized reports, run receipts, and aggregate tables. No raw market rows, timestamps-per-prediction, predictions, weights, matrices, caches, or binary arrays are included.

## Scope and limitations

Returns and overlapping target windows are dependent. The selected historical prefix has sparse weekend observations and no Sunday UTC origins; the analysis does not make iid significance claims. A small mean return can conceal larger opposing movements. Within-window variance is distinct from conditional variance of the mean forecast error and from terminal/cumulative-return variance. Marginal normality would not prove independence. Bid/spread coverage is incomplete, there is no separate execution backtest, and all bins are descriptive rather than trade thresholds. Do not use these results as investment advice or as a live trading signal.
