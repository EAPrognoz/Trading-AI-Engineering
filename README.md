# Trading AI Engineering

Building an AI-powered trading system from market data to models, evaluation, risk, and execution — one reproducible engineering step at a time.

This repository accompanies the **Trading AI Engineering** YouTube series.

## Published-series mapping

1. **Episode 002 — Data pipeline**  
   Python + MetaTrader 5 market-data request, time policy, validation, raw preservation, accepted/rejected runs, and manifests.
2. **Episode 003 — Target**  
   Define the prediction problem and forecast horizon; the current downstream contract is H1 next-hour direction.
3. **Episode 004 — Feature set**  
   Build point-in-time candidate features, inspect redundancy/quality, and freeze the feature set used by the first baseline experiment.
4. **Episode 005 — Baseline**  
   Measure B0/B1/B2 under one chronological evaluation protocol before introducing a more complex candidate model.

The repository is being formalized after Episodes 002–004 were published. Commit dates therefore reflect when engineering artifacts were codified, not the original video publication dates.

## Start here: Episode 002 is simple first

To prove Python can read H1 bars from a local MetaTrader 5 terminal, start with:

```powershell
pip install MetaTrader5 pandas
python examples/ep002_mt5_minimal.py
```

The first example is intentionally short: initialize MT5, select a symbol, call `copy_rates_range`, convert the result to a DataFrame. Validation and reproducibility are added only after that basic connection is clear.

See `experiments/ep002_market_data/README.md` for the full Episode 002 pipeline.

## Current experiment lineage

```text
EP002 accepted H1 snapshot + manifest
        ↓
EP003 target: H1 next-hour direction
        ↓
EP004 frozen point-in-time feature set
        ↓
EP005 B0 / B1 / B2 validation baselines
```

The downstream experiment is valid only when each upstream contract is explicit and reproducible.

## Principles

- no future information in model inputs;
- explicit market-data request and time semantics;
- raw responses are preserved separately from accepted datasets;
- unresolved data problems do not silently become successful exports;
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

Large broker datasets are not committed. Use validated local snapshots and keep their hashes, request contract, provenance, environment, and timestamp coverage in run manifests.

See `docs/architecture.md` and the episode notes under `docs/episodes/`.
