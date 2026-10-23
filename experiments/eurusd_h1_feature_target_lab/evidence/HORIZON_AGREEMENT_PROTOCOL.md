# EURUSD H1 horizon agreement — frozen pre-fit protocol

Status: **frozen before the 12 bounded outer fits**. This is an exploratory historical benchmark on a pre-test prefix that has already been examined; it is not an untouched confirmation.

## Data and fit plan

Use the v3 audited MT5 demo EURUSD H1 pre-test prefix (10,621 rows; source-prefix SHA-256 recorded in `protocol_frozen_pre_fit.json`). No data after the development cutoff is read. Reuse the saved causal W16 feature matrix and fixed `core` columns: own OHLC log ratios, same-bar cross ratios, and the observed-transition gap feature. Keep all observed bars and gaps; do not interpolate, reset histories, divide gap returns by elapsed time, or use future timestamps/gap flags as model features.

Fit 12 deterministic CPU Ridge models: H=4, 8, 24 observed transitions for each of the four existing outer folds. Use alpha=1, intercept, Cholesky solver; independently standardize inputs and each H target using only the shared H24-purged training origins, then inverse-transform predictions. All three horizons use identical training origins and identical validation origins within each fold. No tuning, seed search, or new grid.

Fold intervals (UTC, end exclusive): F1 2025-07-01 to 2025-10-01; F2 2025-10-01 to 2026-01-01; F3 2026-01-01 to 2026-04-01; F4 2026-04-01 to 2026-07-04. A training origin is eligible only when its H24 target endpoint is strictly before fold start. A validation origin's decision timestamp must be in the interval and its H24 endpoint strictly before fold end.

## Targets and agreement rules

For observed transitions, `r_j = log(C[i+j] / C[i+j-1])`; for each H, the target is the mean of the next H returns per observed transition. H8 is the sole directional anchor for all comparisons. An exact-zero forecast abstains; an exact-zero realized H8 target is counted separately and excluded from binary accuracy.

Compare H8-only predictions with three filters: H4/H8 agreement, H8/H24 agreement, and unanimous H4/H8/H24 agreement. Accept only when the required forecasts are all nonzero and have the same sign. On accepted origins the prediction remains sign(H8 forecast); the filter only selects examples. Do not test a “2 of 3” majority rule: among three nonzero predictions it is automatic and does not represent abstention-aware agreement.

Report per fold and overall: common validation origins, accepted/rejected counts and coverage, accepted H8 buys/sells, H8 hit on accepted/rejected/all active forecasts, forecast-zero abstentions, realized-zero targets, balanced accuracy only when both realized direction classes occur, and train-H8-majority plus always-up baselines on each accepted subset. Majority is computed from nonzero H8 training labels per fold; exact tie means abstain. Also summarize UTC weekday coverage and H8 gap-crossing counts for accepted/rejected groups as retrospective metadata only. Do not give IID significance claims or profitability claims; target windows overlap.

## Saved predictions check

The previous v3 inner `predictions.csv.gz` exists but contains inner-validation predictions only. Its `outer_predictions.csv.gz` is header-only (281 bytes), v3 has zero outer refits, and all eight prior outer selection tracks abstained. Therefore saved inner predictions are not a substitute: this protocol defines a new fixed exploratory outer track with up to 12 mean-only refits.

Protocol source of truth: `protocol_frozen_pre_fit.json`. The full hash is recorded in the new run metadata.
