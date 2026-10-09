# EP007: Three five-observation forecast questions

This folder contains the reproducible experiment behind [Episode 007](https://youtu.be/WkEaI-4A55w).
The internal experiment identifier remains `ep008-joint-five-observation` and its frozen run ID is
`ep008-9f1cb85ac383e69b9704c537`.

The example uses the European Central Bank's daily reference rates, quoted as **USD per 1 EUR**.
The exact public archive is included at `data/raw/ecb_eurofxref_hist/eurofxref-hist.zip` with its
provenance and quality summary. The archive SHA-256 is
`e39ffa4c4e8cf3207f2b2589ce5ed5aea56253b35e33c66928d1f7ed76a9d941`.
Data source: [ECB euro foreign exchange reference rates](https://www.ecb.europa.eu/stats/eurofxref/).

## Run the saved-result audit

From the repository root:

```bash
python -m pip install -e .
python experiments/ep008_joint_five_observation_predictability/audit_saved.py
```

The audit checks the archive hash and safe ZIP member, rebuilds labels for the frozen prediction
IDs, and recomputes the listed metrics from the saved predictions. It does **not** fit a model or
run the validate, freeze, predict, or score phases. The committed predictions, frozen IDs, config,
and run manifests are in `runs/ep008-9f1cb85ac383e69b9704c537/`. The plots and chart data are in
`presentation/runs/ep008-9f1cb85ac383e69b9704c537/`.

The archive contains 6,654 unique published observations from 2000-01-03 through 2025-12-31.
The USD values are numeric, finite, and positive; dates are unique. The period has 2,843 dates
without a published rate, including 129 Monday-to-Friday dates. ECB publication calendars include
holidays, so those gaps are retained as gaps. No forward fill is applied.

## What was measured

At each decision date, features use information then published. Labels aggregate the next five
published observations. The frozen 2026 evaluation contains 186 decision/endpoint pairs, with
endpoints through 2026-09-30. These five-observation targets overlap, so the rows are dependent and
the test has limited power. This is an after-the-fact audit of a completed saved run; its labels
have already been read for that evaluation and the audit is not a fresh untouched test.

| Question | Frozen reference | Saved model result | Interpretation |
|---|---:|---:|---|
| Five-observation return, MAE (bps) | Zero forecast: 64.7855 | Linear: 65.2824; boosting: 64.9899 | Neither model improves on the zero forecast in this sample; paired 95% intervals include zero. |
| Movement, QLIKE | EWMA: 0.3615 | Boosting: 0.3713; linear: 203,184.8040 | Boosting is close to EWMA. Three linear predictions hit the 0.0001 bps² scoring floor, sharply worsening QLIKE. |
| Return quantiles, mean pinball (bps) | EWMA-scaled empirical: 17.4470 | Linear: 17.0168; boosting: 16.5899 | Both paired intervals include zero; learned 90% bands are wider and coverage is not proof of calibration. |

These are descriptive measurements from one saved evaluation. ECB reference rates are
informational and non-executable; they are not bid/ask prices, fill prices, or a trading strategy.
This is an ML engineering demonstration, not a profitability claim or investment advice.

## Reproduction artifacts

- Frozen config: [`configs/experiments/ep008_joint_five_observation_predictability.toml`](../../configs/experiments/ep008_joint_five_observation_predictability.toml)
- Saved predictions and IDs: [`runs/ep008-9f1cb85ac383e69b9704c537/`](runs/ep008-9f1cb85ac383e69b9704c537/)
- Presentation outputs: [`presentation/runs/ep008-9f1cb85ac383e69b9704c537/`](presentation/runs/ep008-9f1cb85ac383e69b9704c537/)
- Separate post-run audit: [`audit_saved.py`](audit_saved.py)
