# Episode 005 — Baseline Model

With the Episode 003 H1 target and Episode 004 baseline feature set now frozen,
Episode 005 can measure what a more complex trading model will have to beat.

## Prediction task

At the close of completed H1 bar `t`, predict whether the next close-to-close
return is UP or DOWN.

## Baseline ladder

1. **B0 — majority class**
2. **B1 — previous-hour direction**
3. **B2 — standardized Logistic Regression**

All three are evaluated under the same target, eligible samples, feature
contract where applicable, time boundaries, and metrics.

## Chronology before convenience

The experiment does not randomly shuffle the time series. It uses chronological
train, validation, and test partitions. A sample is purged at a partition
boundary when its `target_timestamp` falls in the next partition.

This means the last training decision cannot borrow the first validation close
to create its label.

## Locked test policy

Episode 005 uses train data to fit B0/B2 and reports the validation benchmark.
The final 20% test partition is created but not evaluated.

Publishing the test metrics now would make the supposedly unseen test result
part of future model-development decisions. The repository therefore keeps it
locked until a later candidate-model comparison.

## Output

`baseline_report.json` records:

- dataset hash and timestamp coverage;
- EP003 target and EP004 feature contract IDs;
- selected feature names;
- chronological split boundaries and purge counts;
- train diagnostics;
- validation metrics for B0/B1/B2;
- an explicit statement that the test partition was not evaluated.

The Episode 005 video should use measured values from this report, not
illustrative performance numbers.
