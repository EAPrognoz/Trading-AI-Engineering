# Trading AI Engineering

Building an AI-powered trading system from market data to models, evaluation, risk, and execution — one reproducible engineering step at a time.

This repository accompanies the **Trading AI Engineering** YouTube series.

## Engineering path

1. **Episode 002 — Market data**: establish a reproducible data boundary.
2. **Episode 003 — Forecast horizon / target**: define what the model is asked to predict.
3. **Episode 004 — Features**: construct point-in-time inputs and freeze a defensible feature set.
4. **Episode 005 — Baselines**: measure what a more complex model must beat.
5. Later stages: candidate models, validation, backtesting, risk, execution, and autonomous agents.

The repository is being formalized after Episodes 002–004 were published. Commit dates therefore reflect when the engineering artifacts were codified, not the original video publication dates.

## Current experiment contract

The first baseline experiment is now grounded by two pre-model contracts:

- **EP003 target**: at H1 bar close `t`, predict the sign of `close[t+1] / close[t] - 1`;
- **EP004 features**: use only point-in-time inputs available at or before bar close `t`, with a predeclared baseline-v1 feature list.

Episode 005 baseline code is intentionally kept behind these contracts.

## Principles

- no future information in model inputs;
- chronological evaluation for time-series experiments;
- reproducible configs, manifests, and reports;
- simple baselines before complex models;
- negative results stay visible;
- no profitability claims.

## Development

```bash
python -m venv .venv
# activate the environment
python -m pip install -e ".[dev]"
pytest
```

Large broker datasets are not committed. Use validated local snapshots and keep
their hashes and coverage in experiment reports.

See `docs/architecture.md` and the episode notes under `docs/episodes/`.
