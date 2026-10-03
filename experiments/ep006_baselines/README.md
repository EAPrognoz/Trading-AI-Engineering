# Episode 006 Bitcoin baseline stability

EP006 is a research-only study of whether the same Bitcoin H1 direction baseline family used in EP005 remains consistent across four consecutive validation windows. It uses an outer chronological 60/20/20 split, expanding training folds, and a locked reserved test. A validation label is eligible only when its target timestamp is strictly before the next partition boundary.

The experiment reuses the EP003 target, EP004 point-in-time H1/H4/D1 features, and the EP005 baseline estimators. It makes no profitability claim. EP005 reported 11/27 Logistic Regression versus 13/27 Always-Up on validation; that result came from a different sample and is not an EP006 paired retest.

## Public files

- configs/experiments/ep006_btc_stability.toml is a template. It has no source package identifier, broker-data path, time range, or data-derived hashes.
- src/trading_ai/experiments/ep006_stability.py validates a user-completed local config, verifies the named manifest, coverage ledger, source revision, feature contract, and raw stream hashes, and writes audit artifacts.
- experiments/ep006_baselines/run_ep006.py runs the experiment.
- experiments/ep006_baselines/EP006_Bitcoin_Baseline_Stability_CodeLab.ipynb audits a saved run; it does not fetch bars, fit estimators, or score predictions.
- The tests use an owner-authorized local input package when available. Public CI checks the template and code and skips data-dependent checks when private inputs are absent.

## Prepare your private input

The public template cannot run as-is. Keep source files, local pins, and generated reports under ignored .local/. Copy the template to a private config file:

~~~powershell
Copy-Item configs/experiments/ep006_btc_stability.toml .local/ep006_btc_stability.private.toml
~~~

Edit that private copy. Set source to your local package and replace each SET_LOCALLY field with values from your own authorized source package and checkout. The package directory must contain:

- raw_manifest.json
- coverage_reconciliation.json
- H1_raw.csv, H4_raw.csv, and D1_raw.csv

Use a Bitcoin symbol supported by your broker; do not substitute EURUSD. Set the half-open analysis interval and four consecutive UTC validation windows. The first validation window must begin at the outer validation boundary; the last must end at the locked-test boundary. The runner validates these conditions and derives row counts from your local data instead of expecting counts from another user's package.

The local config values are integrity pins supplied by you. They let the runner detect input/config drift; they do not independently certify broker history or prove that a strategy is profitable. Never commit the private config, source files, or run output.

## Run

With Python 3.11 or newer:

~~~powershell
python -m pip install -e ".[dev]"
python experiments/ep006_baselines/run_ep006.py `
  --config .local/ep006_btc_stability.private.toml `
  --output-dir .local/ep006-bitcoin-stability-run
~~~

The output directory must be new and remain under .local/. The runner writes a run manifest, gap ledger, window membership, predictions, metrics, and a Markdown report. It records the source and implementation fingerprints in those local artifacts and marks the reserved test as unevaluated.

## Audit saved outputs with CodeLab

The notebook requires an existing local run. In PowerShell, set:

~~~powershell
$env:EP006_RUN_DIR = ".local/ep006-bitcoin-stability-run"
~~~

Then open experiments/ep006_baselines/EP006_Bitcoin_Baseline_Stability_CodeLab.ipynb. It verifies local artifact hashes, fold membership and purge rules, matches predictions to the saved validation rows, and recomputes metrics. It never reads the locked test.
