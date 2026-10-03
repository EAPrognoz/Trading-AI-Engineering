# Architecture

The repository follows the published engineering sequence of the Trading AI
Engineering series.

```text
EP002  DATA PIPELINE
       request → raw response → time policy → validation
       → accepted snapshot or rejection evidence → manifest
            ↓
EP003  TARGET
       prediction problem + forecast horizon + label timestamps
            ↓
EP004  FEATURE SET
       point-in-time candidate features → diagnostics → frozen set
            ↓
EP005  BASELINE
       chronological split → B0 / B1 / B2 → validation benchmark
            ↓
       candidate models → backtesting → risk → execution
```

## Episode boundaries

### Episode 002 — data

A later experiment consumes an accepted snapshot produced under an explicit
request contract. Raw source data is preserved separately. MT5 H1 timestamps are
UTC bar-opening times. A bad, empty, malformed, misaligned, or unresolved input
must not look like a successful export. Recorded downstream runs verify the
accepted CSV hash against its EP002 manifest.

### Episode 003 — target

The forecast horizon and target are declared before the model. For H1 data the
source timestamp remains the bar opening; `decision_timestamp` is its nominal
close one hour later, and `target_timestamp` is the nominal close of the next
consecutive H1 bar. Future values may construct the supervised label but may not
enter model inputs at decision time.

### Episode 004 — feature set

Features are point-in-time and inspectable. More features are not automatically
better; the first downstream feature set is frozen before baseline fitting.
Whole-history availability diagnostics are descriptive, while redundancy
correlations used for development are restricted to the exact downstream train
membership.

### Episode 005 — baseline

Complexity must justify itself against declared references under one
chronological evaluation protocol. The minimal B0 example and the full B0/B1/B2
experiment share the same eligible samples and chronological split. The final
test partition remains locked and its model performance is not evaluated while
the baseline is being developed.

## Separate BTC H1/H4/D1 path

The existing single-stream H1 snapshot and report route documents the EURUSD
experiment. The BTC route starts with three historical-only EP002 MT5 requests.
Each H1, H4, and D1 stream is validated for its own native cadence and gaps;
only three accepted streams with one exact broker symbol and sufficient common
decision coverage can produce a `bundle_manifest.json`. That manifest binds
each stream's source manifest and accepted CSV by identity and SHA-256. Raw and
acquired BTC data stays under ignored `.local/`.

MT5 timestamps are UTC bar opens. A source bar's nominal close is its open plus
its native duration. At an H1 decision, the latest eligible source bar has
`nominal_close <= decision_timestamp`; later and partial bars cannot enter a
feature. Broker D1 bar opens need not occur at 00:00 UTC. A gap resets that
source's feature segment; no feature bridges an unresolved gap.

EP003 still predicts the next H1 close exactly one hour later. H1/H4/D1 name
the three input streams, while feature windows count native bars on each stream.
EP004's selected BTC features come from
`configs/features/ep004_btc_mtf_features.toml`. Its correlation diagnostics use
the exact train decisions from canonical EP005 split preparation, excluding
validation and the locked test. EP005 consumes the same verified bundle and
prepared samples, reports validation only, and records `test.evaluated=false`.
The verified instrument is `BITCOIN_i`, a broker CFD. Published EP005 reports Logistic Regression at 11/27 and Always-Up at 13/27 on validation; its reserved test remains unevaluated. The BTC H1/H4/D1 bundle route is separate from the EURUSD H1 snapshot path.

## Historical note

Episodes 002–004 were published before this repository was formalized. Git
history records when their engineering contracts were codified; it does not
pretend those commits existed on the original publication dates.
