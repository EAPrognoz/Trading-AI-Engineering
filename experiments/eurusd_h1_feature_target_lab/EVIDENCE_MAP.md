# Evidence map — claims allowed in the video

All source files below are local run artifacts captured on 2026-10-22. The experiment windows are previously examined historical validation, not an untouched test. The package intentionally omits raw EURUSD quotes and per-origin model outputs.

| Script claim | Evidence file(s) | Safe wording / boundary |
|---|---|---|
| MT5 demo EURUSD H1 source; prefix rows, range, checksum, 98 non-one-hour transitions | `evidence/source_cutoff_manifest.json`, `evidence/v3_results_report.md` | Demo export, audited prefix only; do not infer gap causes. |
| W16/32/64/96 × H1–24, six feature masks, 10,501 common origins, 4,608 inner fits, train-only scaling, H24 purge | `evidence/v3_results_report.md`, `evidence/v3_results_summary.json`, `evidence/sample_feasibility_by_W_H.csv` | Ridge alpha 1; chronological; candidates share origins. |
| Train-only input/target scaling; same training transform applied to validation; no scaling ablation | `protocol_feature_target_selection_v3.json`, `run_exploratory_fit_v3.py` | Explain the control; do not claim an empirical benefit from normalization. |
| H1 variance is zero; target uses next observed returns and ddof=0 | `evidence/v3_results_report.md`, `protocol_feature_target_selection_v3.json` | H1 mean-only; no calendar-time division or synthetic bars. |
| v3 selector abstained all eight outer fold/track decisions | `evidence/v3_results_report.md`, `evidence/selection_by_outer_fold.csv`, `evidence/v3_run_summary.json` | No outer selected model or score; Pareto frontier is descriptive. |
| H8 direction hit 3,118/6,151 = 50.69%, 12 exact-zero targets | `evidence/horizon_agreement_report.md`, `evidence/horizon_agreement_metrics_by_fold.csv` | Zero targets excluded from direction accuracy and counted separately. |
| H4/H8 agreement accepted 74.9%, hit 50.5% accepted vs 51.3% rejected | `evidence/horizon_agreement_report.md`, `evidence/horizon_agreement_metrics_by_fold.csv` | Same H8 forecast; correlated filtering, not prediction improvement. |
| Strength bins 48.3%–51.9%, not monotonic | `evidence/prediction_strength_report.md`, `evidence/strength_bin_metrics.csv` | Fold-relative train quantiles; don't recommend a post-hoc bin. |
| Variance MAE 5.393904e−7 vs 7.522814e−7 baseline, skill 28.3% | `evidence/mean_variance_results_report.md`, `evidence/variance_overall_metrics.csv`, `evidence/coverage_and_baseline.csv` | This is variance-target MAE reduction, not directional accuracy/profit. |
| Variance is next-eight observed-return within-window dispersion | `evidence/mean_variance_results_report.md`, `protocol_frozen_pre_fit.json` | Not H8 mean-error variance or terminal-return variance. |
| ±sigma / ±2sigma individual-return coverage 79.97% / 95.99% | `evidence/gaussian_coverage_results.md`, `evidence/gaussian_band_diagnostic.csv`, `gaussian_diagnostic_protocol.json` | 49,304 returns are nested in overlapping H8 windows; not independent trials or trade probabilities. Gaussian is an assumption. |

Evidence source paths in the working tree:

- v3: `experiments/eurusd_h1_feature_target_selection_v3/results/eurusd_h1_feature_target_v3_20261022T151237Z/`
- Agreement: `experiments/eurusd_h1_horizon_agreement/results/eurusd_h1_horizon_agreement_v1_20261022T170250Z/`
- Strength: `experiments/eurusd_h1_prediction_strength/results/eurusd_h1_prediction_strength_v1_20261022T171340Z/`
- Joint mean/variance: `experiments/eurusd_h1_mean_variance_joint_utility/results/eurusd_h1_mean_variance_joint_utility_v2_20261022T185128Z/`
