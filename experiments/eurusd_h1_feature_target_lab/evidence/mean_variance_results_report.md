# EURUSD H1 H8 mean/variance joint utility — primary results

Run `eurusd_h1_mean_variance_joint_utility_v2_20261022T185128Z`. Exactly four variance-only Ridge fits; saved H8 mean forecasts reused unchanged. No post-cutoff rows were read. This is an exploratory historical benchmark, not an untouched confirmatory holdout.

Variance target is ddof=0 dispersion of the next eight observed log returns around their realized mean. It is within-window dispersion, not uncertainty of the H8 mean. `sigma=sqrt(predicted variance)` applies only when raw variance>0. Raw<=0 remains separate; variance MAE/calibration uses max(0,raw).

### Full H8 mean baseline by fold

| Fold | n | H8 hits / n | hit rate | balanced accuracy | raw var positive / negative / exact zero |
|---|---:|---:|---:|---:|---:|
| F1 | 1547 | 739/1542 | 47.92% | 47.91% | 1547/0/0 |
| F2 | 1534 | 794/1531 | 51.86% | 51.89% | 1534/0/0 |
| F3 | 1481 | 717/1480 | 48.45% | 48.98% | 1481/0/0 |
| F4 | 1601 | 868/1598 | 54.32% | 53.53% | 1601/0/0 |
| ALL | 6163 | 3118/6151 | 50.69% | 50.36% | 6163/0/0 |

### Train-only cutoffs

| Fold | n train | abs mean q33/q67 cumulative bp | sigma q33/q67 | ratio q33/q67 | mean/sigma/ratio train bin counts |
|---|---:|---:|---:|---:|---|
| F1 | 4242 | 1.3446/3.2761 | 0.00092767274/0.0011349414 | 0.0171/0.0399 | 1414/1414/1414; 1414/1413/1414; 1414/1413/1414 |
| F2 | 5813 | 1.1098/2.6771 | 0.00088945243/0.0010679516 | 0.0144/0.0344 | 1938/1937/1938; 1938/1937/1938; 1938/1937/1938 |
| F3 | 7371 | 0.9280/2.1767 | 0.00080053216/0.00099133665 | 0.0136/0.0309 | 2457/2457/2457; 2457/2456/2457; 2457/2456/2457 |
| F4 | 8876 | 0.7398/1.6881 | 0.0007833383/0.00097334395 | 0.0109/0.0248 | 2959/2958/2959; 2959/2958/2959; 2959/2958/2959 |

Mean edges use all train predictions; sigma and ratio edges use positive raw variance train predictions only. Ties go to lower bin; repeated edges and empty cells remain unchanged. ALL bins are fold-relative, not common absolute thresholds.

### Aggregate positive-variance 3×3 grid

Coverage uses all validation origins; cells include raw variance>0 only. Gross directional outcome is actual cumulative H8 log return × sign(saved mean forecast), in bp; median/p10 is descriptive gross outcome, not profit or max adverse excursion.

| mean × sigma | n / full coverage | zero actual | buy/sell | hit rate (n) | balanced acc. | majority acc. (n) | pred abs mean median | sigma median | realized abs bp median/p90 | gross bp median/p10 | var MAE / train mean MAE | var skill | calib ratio |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| weak x small | 1542/25.0% | 3 | 630/912 | 49.4% (760/1539) | 49.0% | 47.9% (1539) | 6.05903e-06 | 0.00072715 | 10.455/30.389 | -0.381/-22.188 | 3.8431626e-07/7.7480756e-07 | 0.5040 | 1.2027 |
| weak x medium | 652/10.6% | 1 | 304/348 | 52.8% (344/651) | 52.8% | 47.8% (651) | 6.12325e-06 | 0.000932414 | 12.518/34.813 | 1.112/-26.477 | 6.1993439e-07/7.3833198e-07 | 0.1604 | 1.4780 |
| weak x large | 205/3.3% | 1 | 103/102 | 55.9% (114/204) | 55.9% | 50.0% (204) | 6.12423e-06 | 0.00111813 | 11.815/44.091 | 1.881/-23.118 | 8.7746499e-07/6.697266e-07 | -0.3102 | 1.6111 |
| medium x small | 1502/24.4% | 2 | 425/1077 | 52.6% (789/1500) | 51.8% | 53.6% (1500) | 1.95979e-05 | 0.000726313 | 11.327/32.385 | 1.277/-23.279 | 4.0979985e-07/7.7270604e-07 | 0.4697 | 1.1315 |
| medium x medium | 619/10.0% | 4 | 296/323 | 48.1% (296/615) | 48.1% | 44.7% (615) | 1.9593e-05 | 0.000930886 | 12.979/35.713 | -1.433/-29.183 | 6.8136431e-07/7.8807315e-07 | 0.1354 | 1.2012 |
| medium x large | 249/4.0% | 0 | 129/120 | 48.2% (120/249) | 48.5% | 47.8% (249) | 1.87549e-05 | 0.00113106 | 15.215/43.809 | -1.215/-34.032 | 8.5773299e-07/6.5003514e-07 | -0.3195 | 1.6844 |
| strong x small | 567/9.2% | 0 | 200/367 | 52.7% (299/567) | 52.0% | 51.5% (567) | 3.69064e-05 | 0.000719102 | 10.913/30.976 | 0.945/-20.963 | 3.8616188e-07/7.1715058e-07 | 0.4615 | 1.1134 |
| strong x medium | 427/6.9% | 1 | 266/161 | 49.3% (210/426) | 50.5% | 45.5% (426) | 4.18634e-05 | 0.00094093 | 12.859/39.169 | -0.260/-30.256 | 6.3620691e-07/7.2926746e-07 | 0.1276 | 1.4622 |
| strong x large | 400/6.5% | 0 | 286/114 | 46.5% (186/400) | 48.9% | 46.5% (400) | 4.75652e-05 | 0.00116674 | 15.464/39.755 | -1.605/-34.333 | 1.0152405e-06/7.3642145e-07 | -0.3786 | 1.5301 |

### Per-fold stability and sparse counts

The complete fold/cell means, medians, H8 hit, gross outcome median/p10, variance MAE/calibration, zero counts and train-majority comparisons are in `mean_sigma_grid_metrics.csv`. Empty cells are preserved.

| Fold | mean × sigma | n | H8 hit rate | gross median/p10 bp | variance MAE skill |
|---|---|---:|---:|---:|---:|
| F1 | weak x small | 413 | 47.0% (193/411) | -2.146/-30.847 | 0.3751 |
| F1 | weak x medium | 200 | 51.3% (102/199) | 0.344/-26.334 | 0.1715 |
| F1 | weak x large | 34 | 58.8% (20/34) | 3.475/-16.372 | -0.3928 |
| F1 | medium x small | 347 | 46.7% (162/347) | -1.017/-31.320 | 0.3376 |
| F1 | medium x medium | 171 | 43.5% (74/170) | -5.602/-33.267 | 0.0949 |
| F1 | medium x large | 52 | 48.1% (25/52) | -1.199/-23.992 | -0.4441 |
| F1 | strong x small | 134 | 53.7% (72/134) | 1.204/-19.850 | 0.3826 |
| F1 | strong x medium | 116 | 45.2% (52/115) | -2.630/-20.738 | 0.1560 |
| F1 | strong x large | 80 | 48.8% (39/80) | -0.472/-34.333 | -0.4628 |
| F2 | weak x small | 493 | 51.0% (251/492) | 0.429/-17.824 | 0.5829 |
| F2 | weak x medium | 136 | 57.4% (78/136) | 2.435/-20.217 | 0.1792 |
| F2 | weak x large | 28 | 50.0% (14/28) | -1.240/-25.736 | -0.2640 |
| F2 | medium x small | 510 | 50.3% (256/509) | 0.085/-20.454 | 0.5590 |
| F2 | medium x medium | 114 | 57.5% (65/113) | 2.325/-20.817 | 0.2038 |
| F2 | medium x large | 24 | 58.3% (14/24) | 5.633/-19.538 | -0.2740 |
| F2 | strong x small | 135 | 47.4% (64/135) | -1.464/-21.841 | 0.5400 |
| F2 | strong x medium | 70 | 57.1% (40/70) | 2.507/-30.321 | 0.1812 |
| F2 | strong x large | 24 | 50.0% (12/24) | -1.535/-37.438 | -0.3376 |
| F3 | weak x small | 283 | 49.5% (140/283) | -0.172/-17.879 | 0.6068 |
| F3 | weak x medium | 143 | 42.7% (61/143) | -3.926/-29.751 | 0.1216 |
| F3 | weak x large | 106 | 53.3% (56/105) | 1.000/-24.906 | -0.2416 |
| F3 | medium x small | 232 | 53.4% (124/232) | 1.764/-25.651 | 0.5235 |
| F3 | medium x medium | 160 | 49.4% (79/160) | -0.766/-29.992 | 0.1367 |
| F3 | medium x large | 110 | 45.5% (50/110) | -3.472/-47.497 | -0.2181 |
| F3 | strong x small | 104 | 49.0% (51/104) | -0.634/-27.940 | 0.4802 |
| F3 | strong x medium | 129 | 48.1% (62/129) | -1.998/-42.200 | 0.0543 |
| F3 | strong x large | 214 | 43.9% (94/214) | -2.341/-35.314 | -0.2752 |
| F4 | weak x small | 353 | 49.9% (176/353) | -0.086/-22.177 | 0.4988 |
| F4 | weak x medium | 173 | 59.5% (103/173) | 3.162/-26.348 | 0.1590 |
| F4 | weak x large | 37 | 64.9% (24/37) | 5.118/-18.776 | -0.4689 |
| F4 | medium x small | 413 | 60.0% (247/412) | 4.218/-18.678 | 0.4720 |
| F4 | medium x medium | 174 | 45.3% (78/172) | -2.324/-24.541 | 0.1664 |
| F4 | medium x large | 63 | 49.2% (31/63) | -0.514/-23.537 | -0.4181 |
| F4 | strong x small | 194 | 57.7% (112/194) | 3.029/-15.399 | 0.4544 |
| F4 | strong x medium | 112 | 50.0% (56/112) | -0.001/-19.552 | 0.1512 |
| F4 | strong x large | 82 | 50.0% (41/82) | -0.553/-27.750 | -0.6320 |

### Raw nonpositive variance subset

Negative raw variance is invalid; exact zero is mathematically possible but does not demonstrate certainty. Full fold/mean-tercile details are in `raw_nonpositive_variance_metrics.csv`.

| Fold | raw<0 n | raw=0 n | raw<=0 n | full-fold coverage | H8 hit rate (n) |
|---|---:|---:|---:|---:|---:|
| F1 | 0 | 0 | 0 | 0.0% | — |
| F2 | 0 | 0 | 0 | 0.0% | — |
| F3 | 0 | 0 | 0 | 0.0% | — |
| F4 | 0 | 0 | 0 | 0.0% | — |
| ALL | 0 | 0 | 0 | 0.0% | — |

### Positive-variance abs-mean/sigma ratio terciles

| Fold | ratio group | n | H8 hit rate | gross median/p10 bp | variance MAE skill |
|---|---|---:|---:|---:|---:|
| F1 | low | 580 | 49.5% (286/578) | -0.510/-27.092 | 0.2348 |
| F1 | medium | 555 | 44.5% (246/553) | -2.882/-32.115 | 0.2057 |
| F1 | high | 412 | 50.4% (207/411) | 0.172/-23.232 | 0.1631 |
| F2 | low | 551 | 54.0% (297/550) | 1.360/-19.382 | 0.4326 |
| F2 | medium | 620 | 50.9% (315/619) | 0.258/-18.377 | 0.4675 |
| F2 | high | 363 | 50.3% (182/362) | 0.171/-25.475 | 0.4626 |
| F3 | low | 503 | 48.2% (242/502) | -0.855/-25.987 | 0.2019 |
| F3 | medium | 486 | 46.9% (228/486) | -1.324/-30.596 | 0.2671 |
| F3 | high | 492 | 50.2% (247/492) | 0.127/-36.514 | 0.0810 |
| F4 | low | 510 | 52.4% (267/510) | 1.077/-22.530 | 0.2803 |
| F4 | medium | 562 | 54.8% (307/560) | 2.217/-23.032 | 0.2597 |
| F4 | high | 529 | 55.7% (294/528) | 2.135/-19.217 | 0.3418 |
| ALL | low | 2144 | 51.0% (1092/2140) | 0.432/-24.086 | 0.2905 |
| ALL | medium | 2223 | 49.4% (1096/2218) | -0.340/-25.634 | 0.3017 |
| ALL | high | 1796 | 51.9% (930/1793) | 0.729/-26.744 | 0.2502 |

No post-hoc cell is recommended. No IID significance or profitability is claimed. A normal distribution would be an extra assumption and requires checking coverage for the exact target; within-window variance is not H8-mean uncertainty.

- `mean_sigma_grid_metrics.csv`: full per-fold and aggregate 3×3 results.
- `raw_nonpositive_variance_metrics.csv`: raw negative, exact-zero and combined nonpositive results.
- `mean_sigma_ratio_metrics.csv`: positive-variance ratio terciles.
- `variance_overall_metrics.csv`: full, positive, nonpositive, negative, and exact-zero variance cohorts.
- `train_prediction_thresholds.csv`: fold-specific train-only cutoffs.
- `outer_predictions_mean_variance.csv`: predictions, actual targets and audit fields separately.
- `variance_fit_index.csv`, `variance_coefficients_and_scalers.npz`: four fitted variance heads.
