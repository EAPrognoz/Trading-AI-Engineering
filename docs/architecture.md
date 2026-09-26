# Architecture

The repository follows the engineering sequence of the Trading AI Engineering
series rather than starting with a model.

```text
market data
    ↓
forecast horizon / target
    ↓
point-in-time features
    ↓
experiment protocol
    ↓
baselines
    ↓
candidate models
    ↓
backtesting / risk / execution
```

## Layer boundaries

### Data

A later experiment consumes a validated snapshot. It must not silently repair
duplicates, invalid timestamps, contradictory OHLC values, or non-finite market
fields.

Large broker snapshots are not committed to Git. Code, configuration, small
synthetic fixtures, hashes, manifests, and experiment reports are.

### Target

Future observations may be used to construct a supervised-learning label. They
may not leak into the feature vector available at the decision timestamp.

### Features

Features are computed from information available at or before the decision
timestamp. Feature construction and feature selection are separate steps.

### Evaluation

Time-series experiments use chronological partitions. Random shuffling is not
the default evaluation protocol.

### Models

Complexity has to justify itself against predeclared baselines. A model does not
receive credit merely for being more sophisticated.

## Historical note

Episodes 002–004 were published before this repository was formalized. The Git
history records when the engineering contracts were codified; it does not
pretend those commits existed on the original publication dates.
