# Episode 005 — Baseline

Episode 005 is the repository's **baseline** stage.

It consumes, rather than redefines:

- the accepted/reproducible data boundary from Episode 002;
- the H1 next-hour direction target from Episode 003;
- the frozen point-in-time feature set from Episode 004.

## Baseline ladder

1. **B0 — majority class**
2. **B1 — previous-hour direction**
3. **B2 — standardized Logistic Regression**

All three are evaluated under the same target, eligible samples, time boundaries,
metrics, and (for B2) frozen Episode 004 features.

## Chronology before convenience

The experiment does not randomly shuffle the time series. It uses chronological
train, validation, and test partitions. A sample is purged at a partition
boundary when its `target_timestamp` falls in the next partition.

## Locked test policy

Episode 005 uses train data to fit B0/B2 and reports the validation benchmark.
The final 20% test partition is created but not evaluated during baseline
development.

## Output

`baseline_report.json` records:

- dataset hash and timestamp coverage;
- Episode 003 target contract ID;
- Episode 004 feature contract ID;
- selected feature names;
- chronological split boundaries and purge counts;
- train diagnostics;
- validation metrics for B0/B1/B2;
- an explicit statement that the test partition was not evaluated.

Measured values, not illustrative performance numbers, are the source for the
Episode 005 video.
