# EURUSD H1 horizon agreement — exploratory outer results

Run: `eurusd_h1_horizon_agreement_v1_20261022T170250Z`. This is an exploratory historical benchmark on a previously examined pre-test prefix, not an untouched confirmation period.

## Protocol and model

Frozen W16/core Ridge (`alpha=1`, intercept, Cholesky), with train-only X and target standardization. Three mean-only outputs were fit for H=4, 8, and 24 observed transitions in each of the four fixed outer folds (12 fits total). Training origins were identical across the three horizons within each fold and purged by the H24 endpoint. All outer validation origins had all three forecasts; all gaps were retained. No test rows after the development prefix were read.

For each accepted subset, the directional prediction is still sign(H8 forecast); the agreement rule only filters origins. Therefore accepted-set H8 accuracy is the conditional accuracy of the same H8 model, not evidence that combining forecasts changed or improved the prediction. Agreement is correlated filtering, not independent confirmation. Exact-zero forecasts abstain and exact-zero H8 outcomes are counted separately and excluded from direction accuracy.

## Accepted-set and rejected-set H8 results

Coverage is accepted origins divided by all joint outer-validation origins in that fold. H8 accuracy uses rows with a nonzero H8 forecast and nonzero realized H8 target. Balanced accuracy is omitted when both realized direction classes are not present. Baselines are scored on the accepted subset only; majority is computed from nonzero training H8 labels per fold, with ties abstaining. Weekday and gap counts are descriptive metadata. There were no exact-zero forecasts for H4, H8, or H24, so every rejected origin was rejected for sign disagreement; exact-zero forecasts remain abstentions by the frozen rule.

| Fold | Filter | Accepted / joint | Coverage | Accepted H8 hit (n) | Rejected H8 hit (n) | Accepted majority | Accepted always-up | Accepted weekday Mon..Sun | Accepted H8 gap-crossings |
|---|---|---:|---:|---:|---:|---:|---:|---|---:|
| F1 | `H8_only` | 1547/1547 | 1.000 | 0.479 (739/1542) | — (0/0) | 0.497 (766/1542) | 0.497 (766/1542) | 299/312/312/304/307/13/0 | 120/1547 |
| F1 | `agreement_H4_H8` | 1147/1547 | 0.741 | 0.470 (538/1145) | 0.506 (201/397) | 0.498 (570/1145) | 0.498 (570/1145) | 218/229/234/239/215/12/0 | 91/1147 |
| F1 | `agreement_H8_H24` | 1096/1547 | 0.708 | 0.482 (526/1092) | 0.473 (213/450) | 0.511 (558/1092) | 0.511 (558/1092) | 209/218/214/212/231/12/0 | 90/1096 |
| F1 | `unanimous_H4_H8_H24` | 853/1547 | 0.551 | 0.470 (400/851) | 0.491 (339/691) | 0.505 (430/851) | 0.505 (430/851) | 160/163/170/176/172/12/0 | 73/853 |
| F2 | `H8_only` | 1534/1534 | 1.000 | 0.519 (794/1531) | — (0/0) | 0.501 (767/1531) | 0.501 (767/1531) | 299/312/312/288/311/12/0 | 120/1534 |
| F2 | `agreement_H4_H8` | 1152/1534 | 0.751 | 0.508 (584/1150) | 0.551 (210/381) | 0.508 (584/1150) | 0.508 (584/1150) | 234/229/236/209/233/11/0 | 94/1152 |
| F2 | `agreement_H8_H24` | 1148/1534 | 0.748 | 0.507 (581/1146) | 0.553 (213/385) | 0.502 (575/1146) | 0.502 (575/1146) | 225/241/231/206/238/7/0 | 92/1148 |
| F2 | `unanimous_H4_H8_H24` | 890/1534 | 0.580 | 0.497 (441/888) | 0.549 (353/643) | 0.514 (456/888) | 0.514 (456/888) | 181/186/178/156/183/6/0 | 71/890 |
| F3 | `H8_only` | 1481/1481 | 1.000 | 0.484 (717/1480) | — (0/0) | 0.435 (644/1480) | 0.435 (644/1480) | 299/288/288/288/307/11/0 | 113/1481 |
| F3 | `agreement_H4_H8` | 1084/1481 | 0.732 | 0.502 (544/1083) | 0.436 (173/397) | 0.436 (472/1083) | 0.436 (472/1083) | 222/211/201/217/225/8/0 | 81/1084 |
| F3 | `agreement_H8_H24` | 1079/1481 | 0.729 | 0.491 (530/1079) | 0.466 (187/401) | 0.437 (471/1079) | 0.437 (471/1079) | 215/219/207/209/220/9/0 | 80/1079 |
| F3 | `unanimous_H4_H8_H24` | 835/1481 | 0.564 | 0.511 (427/835) | 0.450 (290/645) | 0.432 (361/835) | 0.432 (361/835) | 165/168/154/170/171/7/0 | 63/835 |
| F4 | `H8_only` | 1601/1601 | 1.000 | 0.543 (868/1598) | — (0/0) | 0.528 (843/1598) | 0.472 (755/1598) | 299/312/336/329/312/13/0 | 104/1601 |
| F4 | `agreement_H4_H8` | 1234/1601 | 0.771 | 0.537 (662/1233) | 0.564 (206/365) | 0.538 (663/1233) | 0.462 (570/1233) | 232/241/268/253/233/7/0 | 73/1234 |
| F4 | `agreement_H8_H24` | 1228/1601 | 0.767 | 0.543 (665/1225) | 0.544 (203/373) | 0.536 (657/1225) | 0.464 (568/1225) | 220/243/265/249/241/10/0 | 79/1228 |
| F4 | `unanimous_H4_H8_H24` | 992/1601 | 0.620 | 0.539 (534/991) | 0.550 (334/607) | 0.545 (540/991) | 0.455 (451/991) | 182/196/217/201/191/5/0 | 59/992 |
| ALL | `H8_only` | 6163/6163 | 1.000 | 0.507 (3118/6151) | — (0/0) | 0.491 (3020/6151) | 0.477 (2932/6151) | 1196/1224/1248/1209/1237/49/0 | 457/6163 |
| ALL | `agreement_H4_H8` | 4617/6163 | 0.749 | 0.505 (2328/4611) | 0.513 (790/1540) | 0.496 (2289/4611) | 0.476 (2196/4611) | 906/910/939/918/906/38/0 | 339/4617 |
| ALL | `agreement_H8_H24` | 4551/6163 | 0.738 | 0.507 (2302/4542) | 0.507 (816/1609) | 0.498 (2261/4542) | 0.478 (2172/4542) | 869/921/917/876/930/38/0 | 341/4551 |
| ALL | `unanimous_H4_H8_H24` | 3570/6163 | 0.579 | 0.505 (1802/3565) | 0.509 (1316/2586) | 0.501 (1787/3565) | 0.476 (1698/3565) | 688/713/719/703/717/30/0 | 266/3570 |

`H8_only` is the single-horizon reference; its accepted rows are exactly the origins where the H8 forecast is nonzero. H4/H8, H8/H24, and unanimous rules are separate abstention-aware agreement filters; no two-of-three majority rule was evaluated.

Pooled H8 hit rates were 50.7% overall (3,118/6,151 evaluated directions), versus 49.1% for the fold-specific training-majority baseline and 47.7% for always-up. H4/H8 agreement accepted 74.9% of origins and had 50.5% H8 accuracy (2,328/4,611), compared with 51.3% on rejected origins (790/1,540). H8/H24 accepted 73.8%, with 50.7% accepted and 50.7% rejected accuracy. Unanimous agreement accepted 57.9%, with 50.5% accepted and 50.9% rejected accuracy. There is no consistent directional-hit improvement from agreement; these dependent historical folds do not support significance claims.

Across all outer validation origins, accepted/rejected H8 target-path gap crossings were: H4/H8 339/118, H8/H24 341/116, and unanimous 266/191. These are future-label audit metadata, not model inputs. The H8-only reference had 2,639 positive and 3,524 negative forecasts; realized H8 target was exactly zero on 12 origins, reported separately and excluded from accuracy.

Accepted H8 buy/sell counts were H4/H8 2,026/2,591, H8/H24 1,711/2,840, and unanimous 1,392/2,178. Pooled descriptive UTC weekday counts and H8 gap crossings for accepted/rejected groups are:

| Filter | Accepted Mon..Sun | Rejected Mon..Sun | H8 gap crossings accepted/rejected |
|---|---|---|---:|
| H4/H8 | 906/910/939/918/906/38/0 | 290/314/309/291/331/11/0 | 339/118 |
| H8/H24 | 869/921/917/876/930/38/0 | 327/303/331/333/307/11/0 | 341/116 |
| Unanimous | 688/713/719/703/717/30/0 | 508/511/529/506/520/19/0 | 266/191 |

Weekday distributions use the UTC decision timestamp (Monday through Sunday). Per-fold accepted/rejected weekday counts, actual-zero counts, and gap fractions are in `agreement_metrics_by_fold.csv`; pooled rejection reasons and zero/gap counts are in `posthoc_prediction_audit.json`.

## Outputs and limits

- `outer_predictions.csv` contains one row per origin with separate horizon predictions, actual labels, and timestamps.
- `agreement_metrics_by_fold.csv` contains accepted and rejected counts, coverage, direction hits, zero counts, baselines, weekday distribution, and H8 gap-crossing audits for every filter, fold, and pooled descriptive cohort.
- `fit_index.csv` and `model_coefficients_and_scalers.npz` retain all 12 fits and train-only scaling parameters.
- `illustrative_examples.json` contains real rows from the saved outer predictions; no timestamp was invented.
- `posthoc_prediction_audit.json` records per-filter rejection reasons, forecast-zero counts, and pooled accepted/rejected gap audits.
- Overlapping horizons/origins are dependent; no IID significance test or profitability claim is made. Results are conditional on the fixed historical folds and filters.
