# Episode 005 — Baseline gate

Episode 005 is intentionally blocked behind two repository milestones:

- Episode 003 target / forecast-horizon contract;
- Episode 004 point-in-time feature contract and frozen baseline feature set.

No baseline experiment should be treated as valid until both layers are
implemented and tested.

Planned Episode 005 task:

- timeframe: H1;
- target: direction of the next one-hour return;
- baselines: majority class, previous-hour direction, Logistic Regression;
- evaluation: chronological train / validation / test partitions;
- primary report: predictive metrics plus reproducibility metadata.

Measured baseline numbers will be committed only after an executable experiment
has been run against an identified dataset snapshot.
