# EP002–EP005 consistency design

Date: 2026-09-27  
Repository: `EAPrognoz/Trading-AI-Engineering`  
Branch: `fix/ep002-ep005-consistency`

## Goal

Bring the public repository into one internally consistent learning and experiment path:

```text
EP002 DATA
  ↓
EP003 TARGET
  ↓
EP004 FEATURE SET
  ↓
EP005 BASELINE
```

The repository must keep the channel's teaching rule:

```text
watch the idea
→ run the smallest example
→ inspect the full engineering implementation
```

Minimal examples stay readable. Full implementations stay strict and reproducible. Simplicity may remove distracting implementation detail, but it must not silently change the experiment contract.

## Scope

This change covers six consistency areas:

1. EP002 minimal example and CI syntax coverage.
2. EP002 acceptance, coverage, and H1 timestamp validation.
3. EP003 decision/target timestamp semantics.
4. EP004 diagnostic scope relative to the locked test partition.
5. EP005 minimal/full experiment sample and split consistency.
6. Downstream provenance from EP002 accepted data into later reports.

It does not add a new model, change the frozen Episode 004 feature list, evaluate the final test partition, publish real broker data, or produce video assets.

## EP002 — minimal example and CI

### Problem

The current public `examples/ep002_mt5_minimal.py` contains literal `\n\n` text between statements and therefore does not compile, while CI only runs pytest.

### Design

- Fix the minimal example without adding framework abstractions.
- Keep the visible path limited to:
  - initialize MT5;
  - select symbol;
  - call `copy_rates_range`;
  - shut down MT5;
  - build a DataFrame.
- Extend CI so Python source under `examples/` and `experiments/` is compiled before pytest.
- CI must fail if a public teaching script contains syntax errors even when the unit-test suite does not import it.

## EP002 — accepted data contract

### Problem

The current pipeline can filter invalid or out-of-range rows before record validation, coverage is inferred from raw min/max timestamps, an empty candidate can reach the acceptance decision, and accepted snapshots reject only gaps greater than one hour rather than enforcing exact H1 alignment.

### Design

The pipeline keeps two distinct concepts:

1. **raw evidence** — exactly what the source returned;
2. **candidate accepted range** — records eligible under request time semantics.

Raw evidence remains preserved unchanged.

The acceptance decision must require all of the following:

- raw timestamps that are expected to participate in request validation are parseable;
- the accepted candidate is non-empty when at least one completed bar is required;
- the requested start boundary is actually represented by the eligible H1 sequence;
- the latest required completed H1 opening time is represented;
- accepted timestamps are UTC-hour aligned;
- accepted timestamps are unique and strictly increasing;
- consecutive accepted timestamps differ by exactly one hour for this first narrow accepted-snapshot contract;
- all existing OHLC, finite-value, and non-negative-field checks pass;
- unresolved gaps still produce visible rejection evidence rather than synthetic fill.

Rows outside the requested half-open range may be excluded normally. A row excluded only because its H1 interval is incomplete at the cutoff is not itself a data-quality error.

A malformed timestamp in the source response must not be silently transformed into a successful accepted run.

### Coverage representation

Coverage reporting should distinguish:

- first/last timestamp returned by the source;
- first/last timestamp eligible after range/cutoff policy;
- required start timestamp;
- required latest completed opening timestamp;
- whether those required boundaries are present.

The exact implementation may compute boundary presence directly rather than infer it from minima/maxima.

### Empty candidate

If the request requires completed H1 observations and no eligible rows remain, the run is rejected with an explicit issue code. It must not write `accepted.csv`.

## EP003 — timestamp semantics

### Source-time convention

The market-data `timestamp` remains the MT5 H1 **bar opening timestamp in UTC**.

### Decision-time convention

For a completed H1 bar opening at `t`:

```text
decision_timestamp = t + 1 hour
```

This is the nominal close of the bar whose final OHLC values are used as model inputs.

For the next consecutive H1 bar opening at `t + 1h`:

```text
target_timestamp = t + 2 hours
```

This is the nominal close at which the next close-to-close return is realized.

The supervised return remains:

```text
close[t+1] / close[t] - 1
```

The change is a correction of timestamp meaning, not a change to labels or market values.

### Required regression guarantee

For the same accepted dataset:

- `target_h1_direction` must remain unchanged;
- `future_return_1h` must remain unchanged;
- eligible binary-row count must remain unchanged;
- feature values must remain unchanged;
- split membership and purge counts must remain unchanged except that reported boundary timestamps shift consistently by +1 hour;
- B0/B1/B2 metrics must remain unchanged.

### Minimal example

The EP003 teaching example must use the same timestamp semantics as the reusable target implementation. It should still foreground the simple future-return formula rather than the full contract machinery.

## EP004 — diagnostic scope

### Frozen feature contract

The Episode 004 selected feature list remains unchanged.

Feature formulas remain point-in-time and continue to use the market bar timestamps needed by their definitions. In particular, cyclical hour/day features are not automatically shifted merely because Episode 003 decision timestamps are corrected.

### Locked-test discipline

The repository must distinguish between:

- structural, predeclared feature selection;
- diagnostics that could influence future feature decisions.

The locked test partition must not be used to make feature-selection or redundancy decisions.

The full Episode 004 report should therefore expose a clear diagnostic scope. Preferred design:

- compute the same feature engineering on the accepted history;
- define a development/training diagnostic window from the same chronological experiment boundaries used downstream;
- compute correlation/redundancy statistics used for development decisions only on that allowed scope;
- keep any whole-history availability information clearly descriptive and non-selective.

If implementation constraints make a shared split helper preferable, reuse the same boundary rules rather than create a second incompatible splitter.

No final-test model metrics are introduced.

## EP005 — minimal and full experiment consistency

### Problem

The current minimal B0 example independently rebuilds the target and uses positional 60/20 slicing without the same boundary purge used by the full experiment.

### Design

The minimal example remains pedagogically short but obtains its train and validation labels from the same reusable dataset assembly and chronological split semantics as the full experiment.

The visible teaching idea stays simple:

```python
majority_class = y_train.value_counts().idxmax()
prediction = ...
```

What is simplified is the model, not the sample definition.

The full experiment remains:

- B0 majority class fit on train only;
- B1 previous-hour direction;
- B2 StandardScaler + LogisticRegression fit on train only;
- chronological 60/20/20;
- target-boundary purge;
- validation reporting;
- final test locked and not evaluated.

## Provenance and dataset identity

### Problem

Downstream reports currently compute a hash of whichever CSV path they are given but do not prove that the file is the accepted dataset recorded by the originating EP002 manifest.

### Design

Add a narrow verification path that can consume the EP002 run manifest alongside `accepted.csv`.

For a verified downstream run:

- manifest status must be `accepted`;
- manifest must identify the accepted dataset artifact;
- the accepted CSV hash must match the manifest hash;
- downstream reports must record a stable EP002 run identity or manifest hash;
- reports intended for source control must not expose private absolute local paths.

The implementation must still allow unit tests and intentionally isolated fixtures to use explicit test helpers without pretending those fixtures came from a broker run.

No broker credentials or raw large datasets are committed.

## Documentation

Update the public docs only where the contract changed:

- MT5 H1 source timestamps are bar-open timestamps in UTC;
- EP003 decision/target timestamps represent nominal H1 closes;
- minimal examples share the same experimental semantics as the full path;
- test model performance remains unevaluated;
- official measured results come from the full experiment artifacts.

Keep README concise. Detailed edge-case rules belong in episode or architecture documentation.

## Testing strategy

Use test-driven changes.

Required new regression coverage includes:

### Public-script coverage

- all `examples/*.py` compile;
- all `experiments/**/*.py` compile.

### EP002

- malformed source timestamp cannot produce an accepted run;
- empty eligible candidate cannot produce an accepted run;
- missing requested start boundary is rejected;
- missing required completed end boundary is rejected;
- half-hour/misaligned H1 timestamps are rejected;
- exact consecutive hourly accepted data still passes;
- ordinary out-of-range rows remain normal exclusions;
- incomplete final H1 bar remains a normal cutoff exclusion.

### EP003

- source timestamp is interpreted as opening time;
- decision timestamp is exactly opening + 1h;
- target timestamp is next opening + 1h;
- labels and returns are unchanged by the timestamp-semantic correction;
- gap behavior remains unchanged.

### EP004

- diagnostics used for redundancy/selection exclude locked-test rows;
- selected feature list remains unchanged;
- point-in-time feature values remain unchanged.

### EP005

- minimal and full B0 use the same eligible train/validation membership;
- boundary purge behavior is shared;
- test metrics remain absent;
- timestamp shift alone does not alter B0/B1/B2 metrics.

### Provenance

- matching accepted CSV + EP002 manifest verifies;
- tampered accepted CSV fails verification;
- rejected manifest cannot be used as an accepted downstream source;
- report metadata uses sanitized/stable source identity rather than private absolute paths.

## CI and acceptance criteria

The corrective PR is acceptable only when all of the following are true:

```text
existing tests pass
new regression tests pass
examples compile
experiment scripts compile
git diff --check passes
EP002 cannot accept empty/invalid/misaligned H1 data
EP003 timestamps represent nominal H1 closes
EP003 labels/returns/sample membership remain invariant
EP004 decision diagnostics do not use locked test
minimal EP005 and full EP005 share sample/split semantics
test model metrics remain absent
GitHub Actions is green
```

## Migration of the first real experiment

The local real EURUSD experiment and its reports are not merged as part of the corrective code changes until they have been rerun or verified against the corrected contracts.

Expected invariant from the already observed timestamp correction: data, labels, split membership, purge counts, and metrics remain the same; only decision/split boundary labels move by one hour.

After the corrective PR is green, the real EP002→EP005 reports and dependent video assets can be refreshed and reviewed as a separate handoff.
