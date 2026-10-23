# Supplemental Gaussian return-band diagnostic — frozen protocol

This is a separate, no-fit diagnostic. It does not change the primary 4-variance-fit protocol, model thresholds, or tercile boundaries.

For validation origins with positive raw variance only, use `N(mu_hat_H8, v_hat_H8)` as a working Gaussian reference for **one individual observed-transition return** in the next-eight-observed-return window. It is not a distribution for the H8 mean or cumulative/terminal return.

The saved pre-test target manifest contains mean targets for H=1..8. If `m_H` is the saved mean of the next H observed returns and `m_0=0`, reconstruct `r_H=H*m_H-(H-1)*m_(H-1)`. Before calculating coverage, require all eight labels to be valid/finite, ordered and within the H8 endpoint, H8 to end strictly before the fold boundary, reconstructed H8 mean and population variance (ddof=0) to match saved targets within `atol=1e-14, rtol=1e-10`, and no test rows after July 3. If any check fails, omit coverage and report the failure.

For `raw_variance>0`, set `sigma=sqrt(raw_variance)` and count the eight returns inside inclusive `mu±sigma` and `mu±2sigma` bands. Report (a) per-origin fraction covered, averaged equally across origins, and (b) micro coverage across all individual returns, by fold and overall. Keep raw-negative, exact-zero, and combined nonpositive variance separate; no epsilon or band for nonpositive values.

68.27%/95.45% are normal-distribution references, not guaranteed coverage. This diagnostic does not establish normality, independence, H8-mean uncertainty, or terminal-return coverage. Marginal normality does not imply independent returns; do not apply `sqrt(H)` or use within-window variance as terminal variance. No PnL, probability-confidence, or strategy claim.
