# Trading AI Engineering — video-to-code map

The repository follows the same order as the published series.

The learning pattern is deliberately consistent:

```text
WATCH THE IDEA
      ↓
RUN THE SMALLEST EXAMPLE
      ↓
OPEN THE FULL ENGINEERING VERSION
```

| Episode | Stage | Video | Start here | Full implementation |
|---|---|---|---|---|
| 001 | System map | https://www.youtube.com/watch?v=YRGF0TWYnFo | `docs/architecture.md` | repository architecture |
| 002 | Data pipeline | https://youtu.be/kltuqw7vKrY | `examples/ep002_mt5_minimal.py` | `experiments/ep002_market_data/` |
| 003 | Target | https://youtu.be/VfnH1q76fj4 | `examples/ep003_target_minimal.py` | `experiments/ep003_forecast_horizon/` |
| 004 | Feature set | https://youtu.be/SNOTNSZoNQY | `examples/ep004_feature_set_minimal.py` | `experiments/ep004_feature_engineering/` |
| 005 | Bitcoin Baseline | https://www.youtube.com/watch?v=iEr_WGBUkxQ | `examples/ep005_baseline_minimal.py` | `experiments/ep005_baselines/` |

## Episode 002 — data

First prove that Python can ask the local MetaTrader 5 terminal for H1 bars.
Then add time semantics, validation, raw preservation, rejection evidence, and
the manifest.

## Episode 003 — target

First calculate the next-hour return and convert it to a target. Then make the
decision timestamp, target timestamp, zero-return policy, and gap policy
explicit.

## Episode 004 — feature set

First transform bars into a few understandable model inputs. Then inspect the
candidate universe and freeze the feature set before baseline comparison.

## Episode 005 — baseline

First build the majority-class reference. Then compare B0, B1, and Logistic
Regression under one chronological protocol while keeping the final test set
locked.
