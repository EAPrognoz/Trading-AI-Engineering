# Trading AI Engineering

Building an AI-powered trading system from market data to models, evaluation, risk, and execution — one reproducible engineering step at a time.

This repository accompanies the **Trading AI Engineering** YouTube series.

## Watch → run → inspect

Each stage has the same learning order:

```text
watch the episode
      ↓
run the smallest example
      ↓
inspect the full engineering implementation
```

| Episode | Stage | Video | Smallest example | Full implementation |
|---|---|---|---|---|
| 001 | System map | [Watch](https://www.youtube.com/watch?v=YRGF0TWYnFo) | `docs/architecture.md` | repository architecture |
| 002 | Data pipeline | [Watch](https://youtu.be/kltuqw7vKrY) | `examples/ep002_mt5_minimal.py` | `experiments/ep002_market_data/` |
| 003 | Target | [Watch](https://youtu.be/VfnH1q76fj4) | `examples/ep003_target_minimal.py` | `experiments/ep003_forecast_horizon/` |
| 004 | Feature set | [Watch](https://youtu.be/SNOTNSZoNQY) | `examples/ep004_feature_set_minimal.py` | `experiments/ep004_feature_engineering/` |
| 005 | Baseline | upcoming | `examples/ep005_baseline_minimal.py` | `experiments/ep005_baselines/` |

See [docs/series-map.md](docs/series-map.md) for the video-to-code map.

## Start with Episode 002: getting bars is simple

On Windows, open MetaTrader 5, log in to a demo account, then:

```powershell
pip install MetaTrader5 pandas
python examples/ep002_mt5_minimal.py
```

The first example does only four conceptual things:

```text
initialize MT5
→ select symbol
→ copy H1 bars
→ DataFrame
```

Only after that connection is clear do we add request semantics, validation,
raw-response preservation, rejection evidence, and manifests.

## Published-series lineage

```text
EP002 DATA
accepted H1 snapshot + manifest
        ↓
EP003 TARGET
H1 next-hour direction
        ↓
EP004 FEATURE SET
frozen point-in-time inputs
        ↓
EP005 BASELINE
B0 / B1 / B2
```

Episode 005 consumes the target and feature set created by Episodes 003 and 004;
it does not redefine them.

## Minimal examples vs measured experiments

The files under `examples/` are deliberately short teaching examples. They show
one idea at a time and may omit safeguards that would distract from that idea.

Measured results used in the video series must come from the full
`experiments/` implementation and its recorded artifacts, not from the minimal
examples.

## Principles

- make the first working example easy to understand;
- then make assumptions explicit;
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

Large broker datasets are not committed. Use validated local snapshots and keep
their hashes, request contract, provenance, environment, and timestamp coverage
in run manifests.
