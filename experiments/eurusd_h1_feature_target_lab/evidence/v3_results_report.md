# EURUSD H1 feature/target selection v3 — results

**Status: complete exploratory run; all eight outer fold/track selections abstained.** No test rows were read. V3 is a historical benchmark; the earlier v2 run is unchanged. Because both the gap policy and estimand/sample changed, v3–v2 differences are not a one-factor gap-policy effect.

## Data and protocol

- Source: previously audited MetaTrader 5 demo EURUSD H1 OHLC export. Read exactly 10,621 rows of the pre-test prefix, 2024-10-16 00:00 UTC through 2026-07-03 15:00 UTC bar-open. Recomputed prefix SHA-256: `141318d453eb62941f0cf7d03d6a516770cee52a5131fb15d44fb4c5f0e9a270`; prior verified full-file hash was retained without rehashing the later rows. The run summary independently records this verified 64-character value.
- Validation: finite positive OHLC, consistent high/low bounds, unique strictly ordered timestamps. The prefix contains 98 non-one-hour adjacent-row intervals. Their cause is not inferred.
- Target: `r_j=log(C[i+j]/C[i+j-1])` over the next H observed transitions; `y_mean=mean(r_1..r_H)`, `y_var=mean((r_j-y_mean)^2), ddof=0`. No interpolation, gap exclusion, history reset, or division by elapsed time. H=1 variance is exactly zero and excluded from variance fitting/scoring.
- Input: W=16/32/64/96 latest completed observed transitions, each requiring W+1 rows/closes. Fixed core includes `log(open_i/close_(i-1))`. Gap returns/differences/moments use adjacent observed rows; future timestamps, elapsed hours and gap flags are audit/label metadata only.
- Every W/H/mask used the common W96/H24-origin cohort: 10,501 origins overall. Outer validation counts F1–F4: 1,547; 1,534; 1,481; 1,601. The weekday totals across these non-overlapping validation folds were Mon–Sun: 1,196 / 1,224 / 1,248 / 1,209 / 1,237 / 49 / 0. Thursday had 1,209 of 6,163 validation origins (19.6%); UTC hours were near-balanced within each fold. This is no longer Thursday-dominated.
- Elapsed durations are audit metadata. For W96 history, elapsed median is 144h (p95 149h, max 170h). Recomputed from each row of `target_manifest.csv.gz` on the 10,501-origin common cohort, H24 target duration is min/median/p95/max **24/24/72/77h**. The 2,308 H24 origins crossing a non-one-hour transition (22.0%) have min/median/p95/max **25/72/73/77h**; the other 8,193 are exactly 24h. Thus 72h is the conditional median of the gap-crossing subset, not all H24 origins. The frozen `elapsed_time_audit.csv` aggregates H24 as n=10,501, median=72h, p95=72h; that conflicts with its per-origin manifest. The frozen audit file and its recorded hash were left unchanged; this narrative uses the explicit per-origin recomputation. These counts do not assign weekend/holiday/feed-gap causes.
- Frozen protocol SHA-256: `bbbf1f7ccc0f38f8233240e62d24e77459f4913da69fa92545dce46b9def2332`. Alpha=1 Ridge, train-only input/target scalers, 4 W × 24 H × 6 masks, two inner windows in each of four outer folds. The longest H24 endpoint was purged at every boundary so all candidates shared origins.

## Run and validation quality

- Completed 4608/4608 inner fits in 351.7s process CPU (351.8s wall), single-thread CPU. Budget: 900s. No budget stop.
- Inner validation skill distributions below pool candidate-fit rows and are descriptive; folds and overlapping targets are dependent. Mean skill is relative to zero baseline; variance skill is relative to the horizon-specific train-mean variance baseline.

| Track | Validation rows | Median skill | Mean skill | Best row | Positive rows |
|---|---:|---:|---:|---:|---:|
| H1 mean vs zero | 192 | -0.0476 | -0.1916 | -0.0004 | 0/192 (0.0%) |
| H2–24 mean vs zero | 4,416 | -0.1254 | -0.3884 | +0.0405 | 60/4,416 (1.4%) |
| H2–24 variance vs train mean | 4,416 | +0.0240 | -0.2723 | +0.4599 | 2,343/4,416 (53.1%) |

- Median exact sign accuracy: H1 50.3%; H2–24 50.1%. All 192 H1 model rows had negative mean skill. For H2–24 mean, only 60/4,416 rows had positive skill (max +4.05%). Variance median skill was +2.40%, but mean skill was −27.23%, with a long negative tail (minimum −13.55). These are not selected-model or outer scores.
- Raw variance predictions below zero before the registered `max(0, raw)` projection: 560,320/6,733,296 across repeated H2–24 candidate-validation rows (8.32%). Every fit’s count/fraction is in `inner_metrics.csv`; primary variance MAE/RMSE use clipped values.

## W×H frontier and selection

- Saved non-dominated frontier sizes (H≥2 mean/variance skill): F1=6, F2=5, F3=4, F4=5. The H1 mean-only frontier has one point per fold.
- H≥2 frontier W×H: F1 `(16,3) (16,5) (16,6) (16,7) (16,8) (64,7)`; F2 `(16,5) (16,6) (16,7) (16,8) (64,5)`; F3 `(16,5) (16,7) (16,12) (16,13)`; F4 `(16,5) (16,7)` plus `(16,7) (16,10) (16,12)` with `full_minus_single_series_moments`. Every listed frontier point is retained in `pareto_frontiers.csv` with both median skills.
- H1 frontier: F1 W16/core, F2 W16/core, F3 W32/core, F4 W16/core; each has negative median mean skill.
- For a concrete view of the retained H≥2 frontier, the table gives each fold's marginal frontier extreme by median mean skill and by median variance skill. Skills are baseline-relative fractions; each value is the median across the two inner validation windows. These marginal extremes are descriptive coordinates, **not a jointly selected model or winner**.

| Outer fold | Highest median mean skill on frontier: W/H, mask | Mean skill | Variance skill there | Highest median variance skill on frontier: W/H, mask | Mean skill there | Variance skill |
|---|---|---:|---:|---|---:|---:|
| F1 | 16/3, `core` | -0.003226 | -0.006190 | 64/7, `core` | -0.031127 | +0.027438 |
| F2 | 64/5, `core` | -0.002335 | +0.109086 | 16/8, `core` | -0.010626 | +0.129773 |
| F3 | 16/13, `core` | -0.006987 | +0.306035 | 16/7, `core` | -0.009389 | +0.336022 |
| F4 | 16/12, `full_minus_single_series_moments` | +0.000630 | +0.283694 | 16/7, `core` | -0.007636 | +0.316980 |
- Eligibility was frozen as positive mean skill in both inner windows for H1, and positive mean plus variance skill in both inner windows for H≥2. **No candidate passed in any fold/track: 8 abstentions, zero eligible candidates, zero outer refits.** Thus no outer validation score or selected winner is claimed. Pareto status does not imply eligibility.

## Conditional group-drop results

Definition: Δ = skill(full) − skill(full without group); positive favors retaining the group. Denominator is 768 paired mean comparisons/group and 736 variance comparisons/group (H1 variance excluded). These are conditional Ridge ablations, not neural-network importance or significance tests.

| Optional group | Mean Δ mean / median / positive | Variance Δ mean / median / positive |
|---|---:|---:|
| `lagged_cross_ratios` | -0.000365 / -0.000004 / 353/768 (46.0%) | -0.000331 / -0.000023 / 268/736 (36.4%) |
| `log_price_differences` | +0.000644 / -0.000474 / 268/768 (34.9%) | -0.003231 / -0.000271 / 303/736 (41.2%) |
| `mixed_central_moments` | -0.189523 / -0.073786 / 184/768 (24.0%) | -0.202808 / -0.064976 / 30/736 (4.1%) |
| `single_series_moments` | -0.370865 / -0.115109 / 18/768 (2.3%) | -0.501518 / -0.140642 / 118/736 (16.0%) |

The raw gap feature is present in every mask, so this run does not isolate its conditional contribution. The two moment groups have negative average Δ in both heads; log-price differences have positive mean Δ for the mean target but a negative median and only 34.9% positive comparisons, so that result is heterogeneous. The frontier and group-drop rankings are not universal feature recommendations.

## Checks and artifacts

- 8 v3 formula/leakage contract tests passed before fitting. Post-run checks confirmed 4,608 unique fit records (192 one-output H1 and 4,416 two-output H2–24), 12/12 boundary receipts, one train/validation hash per split across all candidates, all 8 abstentions, H1 zero variance, group-drop denominators, and prediction gzip integrity. `predictions.csv.gz` separates actual targets and predictions and carries elapsed/gap audit columns.
- Full local run artifacts: `inner_metrics.csv`, `predictions.csv.gz`, `model_coefficients_and_scalers.npz`, `feature_matrices.npz`, `target_manifest.csv.gz`, `history_window_audit.csv.gz`, `group_drop_deltas.csv`, `group_drop_overall_summary.csv`, `pareto_frontiers.csv`, `selection_by_outer_fold.csv`, `boundary_receipts.json`, and `run_summary.json`.
- No profitability, live trading signal, or statistical significance is claimed. This study is not a clean unseen holdout because the same pre-test historical period had already been examined in prior work.
