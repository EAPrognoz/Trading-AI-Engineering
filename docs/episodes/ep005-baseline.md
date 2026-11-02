# Episode 005 — Baseline

Status: **upcoming**.

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
