# Episode 005 — Baseline experiment

Status: **published**.

Video: [Bitcoin Baselines: Can Logistic Regression Beat Always Up?](https://www.youtube.com/watch?v=iEr_WGBUkxQ).

Episode 005 begins only after the Episode 003 target and Episode 004 feature set
are frozen.

For the simple teaching version, start with:

```powershell
python examples/ep005_baseline_minimal.py path/to/ep002-run/accepted.csv
```

That example shows only the majority-class baseline, but it uses the same
eligible samples, feature warmup, chronological split, and boundary purge as the
full experiment.

This directory contains the **full measured benchmark**.

## Baseline ladder

- **B0 — majority class**
- **B1 — previous-hour direction**
- **B2 — Logistic Regression** using the frozen Episode 004 feature set

## Evaluation protocol

The eligible sample sequence is split chronologically:

- 60% train;
- 20% validation;
- 20% locked test.

Samples whose target timestamp crosses from train into validation, or from
validation into test, are purged at the boundary.

Episode 005 reports train diagnostics and validation baseline metrics. The final
test partition is created, but model performance on it is not evaluated during
baseline development.

## Existing EURUSD H1 recorded run

```bash
python experiments/ep005_baselines/run_baselines.py \
  --input path/to/ep002-run/accepted.csv \
  --manifest path/to/ep002-run/manifest.json \
  --output reports/ep005/baseline_report.json
```

The measured report verifies the accepted CSV against its EP002 manifest and is
the source for Episode 005 visuals. Never replace it with numbers produced by the
minimal teaching example or with illustrative results.

## Separate BTC validation run from the same bundle

After EP002 and EP004 use the verified H1/H4/D1 bundle, run EP005 with its
explicit BTC contract. The CLI's default contract is the existing EURUSD H1
config, so `--contract` is required for this BTC path:

```powershell
$runDir = '.local/btc-mtf-<unique-run-id>'
python experiments/ep005_baselines/run_baselines.py `
  --bundle-manifest "$runDir/bundle_manifest.json" `
  --contract configs/experiments/ep005_btc_mtf_baselines.toml `
  --output "$runDir/reports/ep005/baseline_report.json"
```

The bundle loader verifies source identities and SHA-256 hashes before sample
preparation. This runner and EP004 use the same canonical prepared dataset and
chronological purged split. EP003's target remains the next H1 close one hour
ahead; H4/D1 supply native-bar input features. The BTC config fixes 60% train,
20% validation, and 20% locked test. B0 and B2 fit on train, while B1 uses the
previous H1 direction. The BTC report contains validation metrics only. The
test section records `status: locked` and
`evaluated: false`; it contains no test model metrics. Published BTC validation
scored Logistic Regression at 11/27 and Always-Up at 13/27. This is
classification evidence, not a profitability claim.
