# Episode 005 — Baseline

Status: **published**.

Video: [Bitcoin Baselines: Can Logistic Regression Beat Always Up?](https://www.youtube.com/watch?v=iEr_WGBUkxQ).

Episode 005 is the repository's **baseline** stage. It consumes, rather than
redefines:

- the accepted/reproducible data boundary from Episode 002;
- the H1 next-hour direction target from Episode 003;
- the frozen point-in-time feature set from Episode 004.

The teaching order is:

```text
start with the easiest benchmark
        ↓
measure it honestly
        ↓
add a naive market baseline
        ↓
add a simple learned baseline
        ↓
only then consider a more complex model
```

## Start here

```powershell
python examples/ep005_baseline_minimal.py .local/ep002-mt5/accepted.csv
```

The minimal example shows B0: find the majority class in training data and use it
as the reference prediction on validation data. It uses the same target,
feature-warmup eligibility, chronological split, and boundary purge as the full
experiment; only the model shown to the viewer is simplified.

That is enough to teach the key question:

> What does a more complicated model actually have to beat?

## Full baseline ladder

1. **B0 — majority class**
2. **B1 — previous-hour direction**
3. **B2 — standardized Logistic Regression**

The full experiment uses chronological train, validation, and locked test
partitions. Samples whose `target_timestamp` crosses a partition boundary are
purged.

Episode 005 reports the validation benchmark while the final 20% test partition
remains locked. Model performance on that reserved test partition is not
evaluated during baseline development.

`baseline_report.json` records the dataset identity, Episode 003 target
contract, Episode 004 feature contract, split boundaries, and B0/B1/B2 metrics.

Measured values used in the final Episode 005 video must come from
`experiments/ep005_baselines/`, not from the minimal teaching example.

## Separate BTC bundle validation

The snapshot example above is the existing EURUSD H1 teaching route. The BTC
full runner accepts `--bundle-manifest` and requires the explicit
`--contract configs/experiments/ep005_btc_mtf_baselines.toml`; its default
contract remains the EURUSD H1 config. EP005 uses the same hash-verified
H1/H4/D1 bundle and canonical prepared sample/split path used by EP004's
train-only diagnostics. The one-hour H1 target is unchanged; H4/D1 are input
streams with native-bar feature lookbacks.

The BTC contract fixes a chronological 60/20/20 split with boundary purging.
B0 and B2 fit from train, while B1 uses the previous H1 direction. The BTC
report contains validation metrics only. The final test stays `locked` with
`evaluated=false`. On BTC validation, Logistic Regression scored 11/27 and
Always-Up scored 13/27. This is a small classification comparison, not a
profitability result.
