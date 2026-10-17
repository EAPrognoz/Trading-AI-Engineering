# Watch a Neural Network Learn — synthetic experiment

This is a reproducible **synthetic teaching experiment**. It is not market data,
does not model a trading strategy, and supports no claim about market
profitability.

## Frozen protocol

- Generator: `x ~ Uniform(-1, 1)`; `y = sin(pi*x) + 0.25*sin(3*pi*x) + e`,
  with `e ~ Normal(mean=0, standard_deviation=0.05)`.
- One fixed master seed: `20261015`; three independent child random streams
  generate train (128), validation (64), and test (64) once from that same
  generator. Samples are sorted by `x` for readable plots after generation.
- Network: `1 -> 8 tanh -> 1 linear`; weights use seeded Xavier uniform
  initialization and biases start at zero.
- Full-batch MSE, Adam (`lr=0.03`, `beta1=0.9`, `beta2=0.999`, `epsilon=1e-8`),
  batch size 128 and exactly 400 optimizer updates (one update per epoch). No
  early stopping or validation/test-based tuning.
- Checkpoints: epochs `0, 1, 5, 10, 25, 50, 100, 200, 400`. The locked test
  split is first read after the 400-epoch loop and checkpoint capture finish;
  its score is reported for epoch 400.
- A fixed demonstration query at `x=0.25` is an input set aside from all three
  splits. It is not a test observation and is not included in any metric. Its
  clean generator value is included only as a visual reference; it is not used
  for training or checkpoint selection.
- Reported checkpoint metrics are at the predeclared epochs. Final test metrics
  belong to epoch 400; no checkpoint is selected by validation.
- Before training, central finite differences check every parameter's analytic
  gradient at initialization. Every saved metric is recomputed from its saved
  predictions and targets.

No seed search is performed. All splits, including the locked test, are
preserved so the visualization can show the actual experiment.

## Reproduce

This run requires Python 3.11+ and NumPy (already declared by the repository;
the experiment adds no package dependencies). From the repository root:

```bash
python experiments/neural_network_learning_20261015/run_experiment.py
```

Optional output location:

```bash
python experiments/neural_network_learning_20261015/run_experiment.py \
  --output-dir experiments/neural_network_learning_20261015/outputs/seed20261015
```

The run writes split CSVs, all checkpoint weights/predictions/metrics, a
gradient-check record, a compact animation JSON timeline, and a manifest with
configuration, environment, and SHA-256 hashes. Existing episodes are untouched.
After the fixed training finishes, it also computes separately labelled test
comparisons for the untrained network and the constant training-mean predictor;
these diagnostics do not tune or select a model.

## Recorded results

At epoch 400, train MSE is `0.018808`, validation MSE is `0.023916`, and the
locked test MSE is `0.031161` (64 synthetic observations). The constant baseline
predicts the train-target mean and has test MSE `0.500789`; it was calculated
after training and was not used to tune the network or choose a checkpoint.
These results describe this fixed synthetic curve only. They do not measure
trading performance or establish behavior outside this input range or on changing
patterns.
