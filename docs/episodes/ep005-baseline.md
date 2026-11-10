# Episode 005 — Baseline

Status: **published**.

Video: [Bitcoin Baselines: Can Logistic Regression Beat Always Up?](https://www.youtube.com/watch?v=iEr_WGBUkxQ).

The published Bitcoin H1/H4/D1 benchmark is on
`release/btc-mtf-foundation-20261002`. Use its [BTC run guide](https://github.com/EAPrognoz/Trading-AI-Engineering/blob/release/btc-mtf-foundation-20261002/experiments/ep005_baselines/README.md)
with your own authorized broker data. The minimal B0 example below teaches the
baseline idea; it does not reproduce the published BTC result.

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
as the reference prediction on validation data.

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
remains locked.

`baseline_report.json` records the dataset identity, Episode 003 target
contract, Episode 004 feature contract, split boundaries, and B0/B1/B2 metrics.

The published Episode 005 values come from the full BTC experiment described
in the [release-branch run guide](https://github.com/EAPrognoz/Trading-AI-Engineering/blob/release/btc-mtf-foundation-20261002/experiments/ep005_baselines/README.md), not from the minimal teaching example.
