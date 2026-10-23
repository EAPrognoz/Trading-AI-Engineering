# EURUSD H1 H8 prediction-strength grading

Run: `eurusd_h1_prediction_strength_v1_20261022T171340Z`. No model was fit; all H8 training predictions were reconstructed from saved coefficients/scalers. No test rows after July 3 were read.

## Frozen method

Strength is `abs(predicted mean H8 log return per observed transition)`. Displayed cumulative magnitude is `abs(8 × mean × 10,000)` in bp of log return, not pips. Per-fold q20/q40/q60/q80 edges came from that fold’s in-sample predictions on H24-purged training origins and were applied unchanged to its outer validation predictions. Validation occupancy is not expected to be 20%. Exact threshold ties go to the lower bin; repeated edges remain repeated and may leave bins empty. Pooled bins are fold-relative percentile strata, not a common absolute bp scale.

Zero forecasts and exact-zero realized outcomes are counted separately. Across all 6,163 validation origins, H4, H8 and H24 forecasts were all nonzero; there were 12 exact-zero realized H8 targets. A zero forecast abstains and is excluded from directional accuracy; a zero realized target is separately counted and excluded. Direction prediction is the same saved H8 prediction. Training majority is the existing per-fold majority of nonzero H8 train labels, scored on the same actual-nonzero subset.

## Per-fold train thresholds

| Fold | Train n | q20 bp | q40 bp | q60 bp | q80 bp | Train bin counts (very weak → very strong) |
|---|---:|---:|---:|---:|---:|---|
| F1 | 4242 | 0.7883 | 1.6108 | 2.7704 | 4.7730 | 849/848/848/848/849 |
| F2 | 5813 | 0.6419 | 1.3443 | 2.2678 | 3.7995 | 1163/1162/1163/1162/1163 |
| F3 | 7371 | 0.5322 | 1.1333 | 1.8536 | 3.1406 | 1475/1474/1474/1474/1474 |
| F4 | 8876 | 0.4377 | 0.9070 | 1.4753 | 2.3389 | 1776/1775/1775/1775/1775 |

Threshold edges in both raw log-return units and bp are in `train_strength_thresholds.csv`; the training predictions are in-sample.

## Primary validation bins

Coverage is bin count divided by all outer validation origins in that fold (or all folds for ALL). Predicted/realized magnitudes below are absolute cumulative H8 log returns in bp. `hit` and majority accuracy exclude zero realized H8 targets; zero forecasts are separately counted. Balanced accuracy is blank unless both actual direction classes occur.

| Fold | Bin | n / coverage | zero pred / zero actual | up / down signals | H8 hit / n | balanced acc. | train majority / n | pred magnitude mean / median bp | realized magnitude mean / median / q75 / p90 bp |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| F1 | Very weak | 395 / 25.5% | 0 / 1 | 184 / 211 | 0.482 (190/394) | 0.482 | 0.500 / 394 | 0.382 / 0.380 | 18.127 / 13.638 / 24.718 / 36.768 |
| F1 | Weak | 360 / 23.3% | 0 / 2 | 153 / 207 | 0.472 (169/358) | 0.472 | 0.500 / 358 | 1.201 / 1.201 | 19.418 / 14.236 / 26.514 / 42.113 |
| F1 | Medium | 361 / 23.3% | 0 / 1 | 157 / 204 | 0.461 (166/360) | 0.460 | 0.492 / 360 | 2.114 / 2.087 | 19.621 / 14.900 / 24.934 / 39.571 |
| F1 | Strong | 298 / 19.3% | 0 / 1 | 145 / 153 | 0.488 (145/297) | 0.488 | 0.502 / 297 | 3.578 / 3.494 | 18.516 / 12.759 / 24.024 / 40.902 |
| F1 | Very strong | 133 / 8.6% | 0 / 0 | 94 / 39 | 0.519 (69/133) | 0.527 | 0.481 / 133 | 7.697 / 6.174 | 16.453 / 11.615 / 20.980 / 38.490 |
| F2 | Very weak | 386 / 25.2% | 0 / 0 | 161 / 225 | 0.528 (204/386) | 0.524 | 0.474 / 386 | 0.328 / 0.332 | 12.051 / 9.116 / 16.850 / 26.730 |
| F2 | Weak | 401 / 26.1% | 0 / 1 | 136 / 265 | 0.545 (218/400) | 0.534 | 0.465 / 400 | 0.988 / 0.987 | 12.498 / 9.711 / 16.820 / 27.154 |
| F2 | Medium | 402 / 26.2% | 0 / 1 | 112 / 290 | 0.486 (195/401) | 0.501 | 0.534 / 401 | 1.755 / 1.725 | 11.987 / 9.108 / 16.653 / 26.740 |
| F2 | Strong | 286 / 18.6% | 0 / 1 | 88 / 198 | 0.509 (145/285) | 0.522 | 0.533 / 285 | 2.858 / 2.782 | 14.053 / 11.227 / 19.123 / 30.110 |
| F2 | Very strong | 59 / 3.8% | 0 / 0 | 37 / 22 | 0.542 (32/59) | 0.532 | 0.542 / 59 | 4.917 / 4.368 | 12.685 / 9.387 / 17.979 / 25.739 |
| F3 | Very weak | 299 / 20.2% | 0 / 1 | 148 / 151 | 0.463 (138/298) | 0.463 | 0.477 / 298 | 0.250 / 0.244 | 14.697 / 10.504 / 18.906 / 29.707 |
| F3 | Weak | 336 / 22.7% | 0 / 0 | 156 / 180 | 0.500 (168/336) | 0.496 | 0.440 / 336 | 0.823 / 0.813 | 17.553 / 12.003 / 23.451 / 40.797 |
| F3 | Medium | 301 / 20.3% | 0 / 0 | 138 / 163 | 0.508 (153/301) | 0.501 | 0.412 / 301 | 1.493 / 1.509 | 17.380 / 12.407 / 24.200 / 37.576 |
| F3 | Strong | 292 / 19.7% | 0 / 0 | 157 / 135 | 0.507 (148/292) | 0.514 | 0.414 / 292 | 2.398 / 2.343 | 19.160 / 14.938 / 24.868 / 42.481 |
| F3 | Very strong | 253 / 17.1% | 0 / 0 | 204 / 49 | 0.435 (110/253) | 0.477 | 0.431 / 253 | 5.035 / 4.377 | 21.810 / 18.388 / 31.720 / 43.411 |
| F4 | Very weak | 343 / 21.4% | 0 / 0 | 154 / 189 | 0.528 (181/343) | 0.531 | 0.469 / 343 | 0.230 / 0.244 | 14.574 / 10.801 / 19.444 / 33.127 |
| F4 | Weak | 346 / 21.6% | 0 / 0 | 124 / 222 | 0.558 (193/346) | 0.548 | 0.535 / 346 | 0.673 / 0.673 | 15.146 / 12.849 / 20.961 / 29.546 |
| F4 | Medium | 415 / 25.9% | 0 / 3 | 113 / 302 | 0.544 (224/412) | 0.514 | 0.566 / 412 | 1.179 / 1.171 | 14.409 / 12.067 / 19.332 / 30.032 |
| F4 | Strong | 340 / 21.2% | 0 / 0 | 92 / 248 | 0.559 (190/340) | 0.538 | 0.547 / 340 | 1.855 / 1.849 | 15.414 / 10.867 / 20.514 / 30.917 |
| F4 | Very strong | 157 / 9.8% | 0 / 0 | 86 / 71 | 0.510 (80/157) | 0.509 | 0.497 / 157 | 3.317 / 2.839 | 14.615 / 9.111 / 19.583 / 34.448 |
| ALL | Very weak | 1423 / 23.1% | 0 / 2 | 647 / 776 | 0.502 (713/1421) | 0.501 | 0.481 / 1421 | 0.303 / 0.287 | 14.902 / 11.130 / 19.890 / 31.798 |
| ALL | Weak | 1443 / 23.4% | 0 / 3 | 569 / 874 | 0.519 (748/1440) | 0.513 | 0.485 / 1440 | 0.927 / 0.885 | 16.036 / 11.803 / 21.714 / 34.703 |
| ALL | Medium | 1479 / 24.0% | 0 / 5 | 520 / 959 | 0.501 (738/1474) | 0.492 | 0.507 / 1474 | 1.627 / 1.599 | 15.627 / 11.687 / 20.826 / 33.716 |
| ALL | Strong | 1216 / 19.7% | 0 / 2 | 482 / 734 | 0.517 (628/1214) | 0.512 | 0.501 / 1214 | 2.644 / 2.556 | 16.754 / 12.174 / 22.076 / 35.492 |
| ALL | Very strong | 602 / 9.8% | 0 / 0 | 421 / 181 | 0.483 (291/602) | 0.495 | 0.470 / 602 | 5.163 / 4.411 | 17.856 / 12.967 / 25.344 / 39.049 |

## Agreement filters, same H8 strength bins

Coverage is accepted rows divided by all rows in that fold/bin. The H8 direction remains unchanged. These are descriptive selections, not new models or thresholds.

| Fold | Bin | H4/H8 cov.; hit; actual abs median bp | H8/H24 cov.; hit; actual abs median bp | Unanimous cov.; hit; actual abs median bp |
|---|---|---:|---:|---:|
| F1 | Very weak | 59.2%; 46.6%; 14.255 | 52.9%; 46.9%; 13.520 | 33.7%; 42.1%; 14.123 |
| F1 | Weak | 65.8%; 46.8%; 14.229 | 70.0%; 50.0%; 13.337 | 46.9%; 51.5%; 13.596 |
| F1 | Medium | 78.7%; 44.5%; 14.740 | 75.3%; 45.4%; 14.914 | 60.9%; 43.4%; 14.874 |
| F1 | Strong | 87.9%; 47.5%; 12.965 | 81.5%; 48.8%; 12.204 | 71.8%; 47.4%; 12.759 |
| F1 | Very strong | 97.7%; 52.3%; 11.922 | 90.2%; 51.7%; 11.561 | 88.0%; 52.1%; 11.595 |
| F2 | Very weak | 54.7%; 54.0%; 8.411 | 59.1%; 53.5%; 10.007 | 30.8%; 55.5%; 9.237 |
| F2 | Weak | 70.1%; 51.8%; 9.619 | 70.8%; 53.4%; 10.171 | 50.1%; 52.5%; 10.326 |
| F2 | Medium | 81.6%; 48.3%; 9.369 | 80.8%; 46.6%; 9.229 | 66.4%; 45.5%; 9.878 |
| F2 | Strong | 95.8%; 49.6%; 11.361 | 89.9%; 49.8%; 11.079 | 87.4%; 48.4%; 11.361 |
| F2 | Very strong | 98.3%; 53.4%; 9.314 | 91.5%; 53.7%; 9.314 | 89.8%; 52.8%; 9.242 |
| F3 | Very weak | 53.2%; 46.2%; 9.851 | 50.8%; 46.7%; 11.399 | 28.8%; 47.7%; 10.741 |
| F3 | Weak | 62.8%; 53.1%; 12.131 | 67.0%; 52.9%; 13.342 | 43.2%; 56.6%; 13.973 |
| F3 | Medium | 75.1%; 53.5%; 12.523 | 76.4%; 50.0%; 12.392 | 58.5%; 53.4%; 12.523 |
| F3 | Strong | 85.6%; 53.2%; 14.938 | 82.5%; 52.3%; 14.995 | 71.9%; 54.8%; 14.938 |
| F3 | Very strong | 94.1%; 44.1%; 18.691 | 91.3%; 42.9%; 16.927 | 86.2%; 43.6%; 17.026 |
| F4 | Very weak | 59.2%; 51.2%; 9.651 | 59.2%; 50.2%; 11.167 | 36.4%; 47.2%; 10.703 |
| F4 | Weak | 69.7%; 56.4%; 11.608 | 71.4%; 55.5%; 12.949 | 50.6%; 56.6%; 11.087 |
| F4 | Medium | 80.5%; 53.2%; 12.307 | 79.8%; 55.2%; 12.087 | 66.3%; 54.4%; 12.386 |
| F4 | Strong | 90.9%; 56.0%; 10.811 | 89.7%; 56.7%; 10.821 | 83.2%; 56.9%; 10.811 |
| F4 | Very strong | 93.6%; 49.0%; 8.947 | 90.4%; 50.7%; 9.029 | 85.4%; 49.3%; 8.764 |
| ALL | Very weak | 56.7%; 49.6%; 10.688 | 55.7%; 49.6%; 11.403 | 32.5%; 47.9%; 11.212 |
| ALL | Weak | 67.2%; 52.0%; 11.433 | 69.9%; 52.9%; 11.857 | 47.8%; 54.1%; 11.679 |
| ALL | Medium | 79.2%; 49.8%; 12.074 | 78.3%; 49.4%; 11.655 | 63.4%; 49.1%; 12.166 |
| ALL | Strong | 90.0%; 51.7%; 12.209 | 86.0%; 52.2%; 12.008 | 78.7%; 52.1%; 12.134 |
| ALL | Very strong | 95.2%; 48.2%; 12.859 | 90.9%; 47.9%; 12.844 | 86.7%; 47.9%; 12.346 |

The hypothesis that a weaker H8 mean forecast implies a smaller realized move is evaluated by realized absolute-magnitude distributions in the primary table, not by selecting a best bin. Small mean returns can reflect cancellation of larger opposite-signed moves.

## Optional path diagnostics

Not computed: the saved permitted-prefix artifacts contain derived features, targets, predictions, and parameters, but no raw OHLC close/high/low series. The next-eight-observed-bar both-side crossing and excursion measures therefore cannot be reconstructed; no external retrieval was attempted. Even if available, visiting both sides of origin close would not establish swing order or oscillation.

## Limits and artifacts

This is an exploratory historical analysis on previously examined folds; the pooled “strong” bin is relative to each fold’s training prediction distribution. No strategy recommendation, probability-confidence claim, statistical significance, or profitability claim is made. Bid/spread data are incomplete.

- `train_strength_thresholds.csv`: exact train-derived edges and train-bin occupancy.
- `train_in_sample_strength_predictions.csv`: saved-coefficient train predictions used only to set edges.
- `outer_validation_strength_bins.csv`: origin-level bins, predictions, actuals, signs, gap audit, and filter flags.
- `strength_bin_metrics.csv`: complete primary per-fold and pooled table.
- `agreement_filters_by_strength_bin.csv`: complete secondary pair/unanimity table.
- `path_diagnostic_status.json`: path-data availability blocker.
- `zero_prediction_outcome_counts.json`: exact-zero forecast/outcome totals by horizon.
