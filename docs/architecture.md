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

## Historical note

Episodes 002–004 were published before this repository was formalized. Git
history records when their engineering contracts were codified; it does not
pretend those commits existed on the original publication dates.
