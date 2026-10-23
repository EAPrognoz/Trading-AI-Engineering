# Gaussian single-return band diagnostic

Run `eurusd_h1_mean_variance_joint_utility_v2_20261022T185128Z`; reconstructed and verified H1..H8 nested targets for 6163 saved validation origins from the allowed pre-test manifest. No fit was run.

Working reference: for an individual observed-transition return within the next-eight-observed-return window, compare to `N(saved H8 mean prediction per observed transition, saved raw H8 within-window variance)`. This is not a distribution for the H8 mean or terminal return.

Bands are inclusive. Coverage is calculated only for positive raw predicted variance; negative and exact-zero predictions are reported separately without epsilon. Equal-origin and pooled-return coverage are both shown. Nominal 68.27% and 95.45% are Gaussian reference values, not guaranteed or claimed results. Marginal normality does not imply independence; no `sqrt(H)` or terminal-variance inference is made.

Target reconstruction checks passed: nested H1..H8 timestamps ordered; each H8 endpoint matches the primary prediction row and is before its fold end; reconstructed H8 mean and ddof=0 variance match saved targets at rtol 1e-10 / atol 1e-14.

| Fold | Origins | Positive variance | raw<0 / raw=0 | returns / band | ±σ micro / origin | ±2σ micro / origin |
|---|---:|---:|---:|---:|---:|---:|
| F1 | 1547 | 1547 | 0 / 0 | 12376 | 79.36% / 79.36% | 95.41% / 95.41% |
| F2 | 1534 | 1534 | 0 / 0 | 12272 | 83.69% / 83.69% | 97.51% / 97.51% |
| F3 | 1481 | 1481 | 0 / 0 | 11848 | 76.75% / 76.75% | 95.11% / 95.11% |
| F4 | 1601 | 1601 | 0 / 0 | 12808 | 79.97% / 79.97% | 95.91% / 95.91% |
| ALL | 6163 | 6163 | 0 / 0 | 49304 | 79.97% / 79.97% | 95.99% / 95.99% |
