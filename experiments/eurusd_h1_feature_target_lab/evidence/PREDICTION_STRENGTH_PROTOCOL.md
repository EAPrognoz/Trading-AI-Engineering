# EURUSD H1 prediction-strength grading — frozen analysis protocol

**Status:** frozen before strength-bin calculation. No model fit is permitted in this analysis.

## Source and primary quantity

Use only outer predictions from `eurusd_h1_horizon_agreement_v1_20261022T170250Z`, its saved H8 Ridge weights/scalers, W16/core feature matrix, split membership, and target manifest. The source is the previously examined pre-test prefix; test rows after July 3 are excluded.

Primary direction remains the saved H8 prediction. Strength is `abs(predicted mean observed-transition log return)`. For display, show `abs(8 × mean × 10,000)` as expected cumulative log return in basis points (bp of log return, not pips). Reconstruct each fold's training predictions from its saved H8 coefficients/intercept and scaler on the existing H24-purged train origins; do not refit.

## Five bins and edge rules

For each fold, obtain linearly interpolated 20th/40th/60th/80th percentiles of absolute H8 training predictions, then apply those fixed thresholds to that fold's outer validation predictions. Bins are right-closed: `(-inf,q20]`, `(q20,q40]`, `(q40,q60]`, `(q60,q80]`, `(q80,+inf)`. A value exactly on an edge belongs to the lower bin. Repeated edges remain repeated and can yield empty bins; do not merge bins or split ties. Training predictions are in-sample fitted values, so validation occupancy need not be 20% in each bin. Pooled `ALL` bins combine fold-specific rank strata: “strong” means a fold-relative train-percentile bin, not one shared absolute bp threshold.

## Measures

For each fold/bin and pooled bin, report counts and coverage, H8 up/down forecast counts, exact-zero actual count, directional hit and denominator, balanced accuracy only if both realized direction classes appear, and the saved fold-specific train-majority accuracy on the same realized-nonzero subset. Report predicted absolute cumulative log-return magnitude and realized `abs(8 × actual mean × 10,000)` in bp, including mean/median, realized q25/q75 and p90. Count exact-zero forecasts separately from exact-zero realized targets in every bin. A zero forecast remains in the zero-strength bin but is an abstention and is excluded from direction accuracy; a zero outcome is separately counted and excluded from binary accuracy.

For H4/H8, H8/H24 and unanimous filters, use the already-defined agreement flags and the same H8-strength bins. Report accepted count, within-bin coverage, H8 hit, and realized absolute H8 magnitude; do not derive new thresholds. Do not select a “best” bin or present it as a strategy.

## Optional path diagnostic and limits

If raw OHLC for the permitted prefix is already locally available, use exactly the next eight observed bars and exclude the origin bar. Describe whether any future High exceeds origin Close and any future Low falls below it, plus max-up `max(log(High/Close_origin))*10,000` bp and max-down `min(log(Low/Close_origin))*10,000` bp. Both-side crossing means both levels were visited; it does not establish order, repeated swings, or proximity to zero, so do not call it oscillation. These are realized diagnostics only, never features, filters, or confidence probabilities. The initial local inventory found no raw OHLC in the saved agreement artifacts; do not retrieve it externally for this analysis.

This is a historical exploratory analysis on previously examined folds, not independent confirmation. H8 outcomes overlap, so do not claim IID significance or profitability; bid/spread data are incomplete.
