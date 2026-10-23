# Voice-over script — Does a stronger forecast make a better signal?

**Target runtime:** approximately 10.2–11.4 minutes: 1,235 spoken words at 125–135 words per minute, plus about 60–90 seconds of chart-reading pauses. The estimate is from word count; no recording was made.
**Visual key:** blue = completed inputs/features; amber = model forecast; green = observed fact/target.

## 0:00 — Opening: the practical question

Does a stronger market forecast actually make a better signal? And if direction is hard to predict, can a model still learn something useful about how spread out the next moves may be?

In this EURUSD H1 lab, you’ll see how to test features and target definitions without turning a lower error number into a trading promise. Our first answer is a surprise: the selection rule picked no winner.

If careful, reproducible model experiments are useful to you, please like the video and subscribe. Then let’s see what the data can—and cannot—tell us.

## 0:50 — First, what does “the next bar” mean?

Our source is an audited MetaTrader 5 demo export. Its timestamps mark bar opens; a bar’s close is available only after that hour has finished. So the blue input is the completed history. The amber forecast follows it. Green is what happened later.

The feed has gaps. We keep each one as an observed transition: no invented candles, interpolation, reset, or division by calendar hours. So the next twenty-four observed transitions may span more than one calendar day. We record elapsed time for audit, but the model never sees future gap timing.

[ON-SCREEN SOURCE CARD, NOT SPOKEN: CSV name; 10,621-row pretest prefix; UTC open-time range; prefix SHA-256.]

**Evidence:** [source cutoff manifest](evidence/source_cutoff_manifest.json) · [v3 report](evidence/v3_results_report.md)

## 1:45 — What are we asking the model to predict?

For the next H observed close-to-close log returns, our first target is their average. The second is their population variance around that realized average, with ddof zero. The first asks about average signed movement per observed transition. The second describes spread inside that future window.

Those targets answer different questions. Within-window variance is not uncertainty in the mean forecast, and it is not the variance of the final cumulative return. At H equals one, variance is exactly zero by definition, so that horizon is mean-only. We do not add an epsilon and pretend it is confidence.

A small average may also hide larger positive and negative returns that cancel. So our first hypothesis is modest: perhaps history length and feature groups help one target, both, or neither. The validation results have to decide.

**Evidence:** [frozen v3 protocol](protocol_feature_target_selection_v3.json) · [v3 results](evidence/v3_results_report.md)

## 2:55 — Features, normalization, and a paired search

We build features only from completed bars: OHLC log ratios, same-bar and lagged cross ratios, differences of log prices through order three, and causal means, variances, third central moments, skewness, and mixed moments. The core includes log open over the previous close, so a gap that already happened is visible. Future timestamps and gap flags are never inputs.

We pair each history window—sixteen, thirty-two, sixty-four, or ninety-six observed transitions—with horizons one through twenty-four. That is ninety-six W-by-H pairs across six feature masks. Every candidate uses the same 10,501 origins, so changing W or H does not quietly change the sample.

Normalization matters because these features live on different numeric scales. Ridge penalizes coefficient size, so scale affects what that penalty means. For each chronological fold, we fit the input and target scalers on training data only. We keep those training means and scales, then apply those exact same parameters to validation. We never refit the scaler on validation. This is part of the fixed protocol; we did not run a separate normalization ablation.

We used four chronological outer folds and two earlier validation windows inside each fold. The longest H24 target was purged at every boundary. The learner was a simple Ridge baseline, alpha one—not a neural network. The run completed 4,608 inner fits in about 352 seconds of single-thread CPU time.

**Evidence:** [feasibility by W/H](evidence/sample_feasibility_by_W_H.csv) · [Pareto frontier](evidence/pareto_frontiers.csv) · [v3 summary](evidence/v3_run_summary.json)

## 4:55 — The first answer: no selected winner

For horizons two through twenty-four, median mean-target skill versus a zero-change baseline was negative 12.54 percent. Only 60 of 4,416 inner candidate rows had positive mean skill. Variance skill versus a horizon-specific training-mean baseline had a positive median of 2.40 percent, but its average was negative 27.23 percent because of a long negative tail.

These are pooled, overlapping inner-validation results—not scores for a selected model. We kept a Pareto frontier to show tradeoffs. But our frozen rule required positive inner evidence, and all eight outer fold-and-track decisions abstained. No candidate qualified. This search produced no selected-model outer score.

That is useful evidence too. A responsible experiment must be able to stop without a winner. Picking the nicest-looking cell after seeing the table would answer a different question.

**Evidence:** [outer abstentions](evidence/selection_by_outer_fold.csv) · [v3 report](evidence/v3_results_report.md)

## 6:05 — Next question: does stronger direction help?

For a narrower follow-up, we fixed W16, core features, and Ridge for the H8 mean. This was not a winner chosen by the first search. All results are exploratory historical evaluation on an already examined period, not untouched confirmation.

The H8 direction hit rate was 3,118 out of 6,151 nonzero realized outcomes: 50.69 percent. Twelve exact-zero outcomes were counted separately. That result does not establish a stable directional signal.

We then tested a simple idea: perhaps forecasts agreeing at H4 and H8 are more useful. Agreement accepted 74.9 percent of origins, but the unchanged H8 forecast hit 50.5 percent on accepted examples and 51.3 percent on rejected examples. Agreement did not help here. The forecasts share information; matching directions are correlated filtering, not independent confirmation.

**Evidence:** [agreement report](evidence/horizon_agreement_report.md) · [fold metrics](evidence/horizon_agreement_metrics_by_fold.csv)

## 7:10 — Does forecast strength track realized movement?

We sorted absolute H8 forecasts into five relative strength groups using cutoffs from each fold’s training predictions. From very weak to very strong, hit rates were 50.2, 51.9, 50.1, 51.7, and 48.3 percent. They are not monotonic.

The realized move size did not increase cleanly either. “Strong” is relative to a fold’s own training predictions, not one shared threshold. A weak average forecast does not mean the path stayed close to zero; opposite returns can cancel. We should not turn one attractive bin into a rule after seeing the outcomes.

**Evidence:** [strength report](evidence/prediction_strength_report.md) · [strength bins](evidence/strength_bin_metrics.csv)

## 8:00 — Could variance tell us something different?

Next we held that H8 mean forecast fixed and fit four variance-only Ridge models, one per fold. The target was the population dispersion of the next eight observed returns around their realized average. All validation variance predictions were positive.

Variance MAE was 5.394 times ten to the minus seven, compared with 7.523 times ten to the minus seven for the training-mean baseline. That is a 28.3 percent reduction in **variance-forecast MAE** for this target and sample.

It is not a 28.3 percent improvement in direction accuracy or profit. And this within-window variance is not the error variance of the mean forecast. The joint mean-strength and variance groups did not reveal a consistent direction rule either.

**Evidence:** [mean/variance report](evidence/mean_variance_results_report.md) · [joint cells](evidence/mean_sigma_grid_metrics.csv) · [variance summary](evidence/variance_overall_metrics.csv)

## 9:05 — One final check: a Gaussian reference

We used a normal distribution only as a working reference for one individual return inside the next-eight-return window, centered on predicted H8 mean per transition with the predicted within-window variance. “Markets are noisy” does not tell us that returns are normally distributed.

We reconstructed the eight returns from saved nested targets and checked that their mean and ddof-zero variance matched the H8 labels. Coverage was 79.97 percent inside plus or minus one sigma and 95.99 percent inside plus or minus two sigma. The familiar normal references are 68.27 and 95.45 percent.

These are coverage rates for individual returns inside overlapping H8 windows. The 49,304 returns are not independent trials. This does not mean a trade has a 79 or 96 percent chance of success. It is not a distribution for the H8 mean or terminal return, either. Marginal normality would not prove independence.

**Evidence:** [Gaussian protocol](gaussian_diagnostic_protocol.json) · [coverage report](evidence/gaussian_coverage_results.md) · [coverage table](evidence/gaussian_band_diagnostic.csv)

## 10:20 — What we learned, and the next question

The first search did not select a model. Our fixed H8 direction follow-ups stayed near fifty-fifty, so a stable directional signal remains unestablished. The variance estimate did beat its simple baseline on MAE for within-window dispersion. That is a narrower, more defensible finding—not a trading result.

The next lab can ask a new, frozen question and continue the feature and target work. We will keep forecast, realized outcome, and any separate trading test distinct. For now, we know what this evidence supports, and where it stops.

Thanks for watching. The next lab continues from here.
