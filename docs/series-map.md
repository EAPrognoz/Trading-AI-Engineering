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
| 005 | Bitcoin H1 baseline | [Watch](https://www.youtube.com/watch?v=iEr_WGBUkxQ) | `examples/ep005_baseline_minimal.py` (teaching example) | [BTC H1/H4/D1 run guide](https://github.com/EAPrognoz/Trading-AI-Engineering/blob/release/btc-mtf-foundation-20261002/experiments/ep005_baselines/README.md) |
| 006 | Bitcoin baseline stability | [Watch](https://www.youtube.com/watch?v=rxJymSciu7A) · [CodeLab](https://www.youtube.com/watch?v=IDaKQQjM79k) | [CodeLab audit notebook](https://github.com/EAPrognoz/Trading-AI-Engineering/blob/release/ep006-btc-stability-20261003/experiments/ep006_baselines/EP006_Bitcoin_Baseline_Stability_CodeLab.ipynb) (requires a saved run) | [EP006 run guide](https://github.com/EAPrognoz/Trading-AI-Engineering/blob/release/ep006-btc-stability-20261003/experiments/ep006_baselines/README.md) |
| 007 | Three forecast questions | [Watch](https://youtu.be/6me2vpiVD4M) | [Audit the saved ECB run](https://github.com/EAPrognoz/Trading-AI-Engineering/blob/release/ep007-three-forecast-questions-20261009/experiments/ep008_joint_five_observation_predictability/audit_saved.py) | [Experiment, results, and reproduction guide](https://github.com/EAPrognoz/Trading-AI-Engineering/blob/release/ep007-three-forecast-questions-20261009/experiments/ep008_joint_five_observation_predictability/README.md) |

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
locked. The minimal B0 example is for teaching; reproduce the published Bitcoin
H1/H4/D1 benchmark using the [BTC run guide](https://github.com/EAPrognoz/Trading-AI-Engineering/blob/release/btc-mtf-foundation-20261002/experiments/ep005_baselines/README.md) on
`release/btc-mtf-foundation-20261002` with your own authorized broker data.

## Episode 006 — Bitcoin baseline stability

Compare the baseline family across four consecutive validation windows while
keeping the reserved test unevaluated. The [run guide](https://github.com/EAPrognoz/Trading-AI-Engineering/blob/release/ep006-btc-stability-20261003/experiments/ep006_baselines/README.md) is on
`release/ep006-btc-stability-20261003` and requires your own local data and
private config. The CodeLab audits an existing saved run; it does not fetch
bars, fit models, or evaluate the reserved test.

## Episode 007 — three forecast questions

Use the [public ECB reproduction guide](https://github.com/EAPrognoz/Trading-AI-Engineering/blob/release/ep007-three-forecast-questions-20261009/experiments/ep008_joint_five_observation_predictability/README.md) for five-observation return,
movement, and conditional-quantile forecasts. The saved-run audit recomputes
metrics without fitting models. The code and artifacts are on
`release/ep007-three-forecast-questions-20261009`; the internal
`ep008_joint_five_observation_predictability` folder name is intentional and
belongs to the published Episode 007. ECB reference rates are not executable
bid/ask prices, and the results are not a profitability claim.
