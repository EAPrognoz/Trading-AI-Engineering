# Creative brief — EURUSD H1 feature and target lab

## Viewer promise

Show how we tested history length and forecast horizon together, what the validation record actually says, and why “some useful variance prediction” is not the same claim as a reliable direction forecast.

## Audience and voice

For technically curious traders and ML learners. Calm, empirical, plain language. Explain observed-bar time and population variance before using those terms repeatedly. Avoid victory language and “AI predicts the market” framing.

## Three evidence-backed hooks

1. **Outcome hook option:** “We tested 96 history-and-horizon combinations. The selection rule chose none—and that is a result.” Evidence: v3 run summary, 10,501 common origins, eight outer abstentions.
2. **Variance distinction:** “The model cut variance-forecast error by 28.3%, while direction stayed around 50.7%.” Evidence: joint utility report; say precisely what the MAE is and what it is not.
3. **Distribution check:** “A Gaussian band covered 79.97% of the next observed returns at one sigma. That does not mean a trade has an 80% chance.” Evidence: no-fit Gaussian diagnostic; include dependence and target caveats.

## Three title and thumbnail options

1. **Title:** *Does a Stronger Forecast Make a Better Signal?*
   - **Thumbnail text:** `STRONGER ≠ BETTER?`
   - **Concept:** amber forecast strength bars beside green observed outcomes that do not rise monotonically. Keep both colors equally prominent; no profit imagery.
2. **Title:** *What Can EURUSD H1 Features Actually Predict?*
   - **Thumbnail text:** `DIRECTION OR DISPERSION?`
   - **Concept:** split chart with an amber mean forecast on one side and a spread band over green bar returns on the other; label the band as historical dispersion.
3. **Title:** *When the Model Picks No Winner*
   - **Thumbnail text:** `NO WINNER`
   - **Concept:** W×H grid with several Pareto markers but no selected cell; a small note says `EXPLORATORY`. No “failed model” or “market solved” framing.

The selected opening hook is: **“Does a stronger market forecast actually make a better signal?”** It promises a test of features/targets and a distinction between movement direction and dispersion, without promising accuracy or returns.

## Narrative spine

Question → audited data and observed-bar clock → point-in-time features and two targets → paired W×H search → abstention instead of a forced winner → fixed-model follow-ups → separate mean direction, within-window variance, and single-return Gaussian-reference results → limitations → next episode continues the search.

## Visual system

- Inputs/features: blue (`#2F80ED`).
- Forecasts: amber (`#F2A900`).
- Observed outcomes/facts: green (`#27AE60`).
- Keep colors consistent in equations, timelines, scatter plots, and labels. Use neutral gray for split boundaries and excluded/unknown metadata.
- Use a timeline animation that reveals the next observed bar after each completed input bar; show calendar gaps as empty space, not fabricated candles.
- Show Ridge / linear baseline flow only. Do not show neural-network training, gradient updates, GPU acceleration, or a live trading account.

## Editorial guardrails

Only claims in `EVIDENCE_MAP.md`. Describe the v3 selection as abstention, not a model win. Later W16/core/H8 analyses are fixed exploratory follow-ups and were not selected by v3. Keep all rows explicitly historical and previously examined. No profit, probability-of-trade, significance, or universal feature-importance claims. A like/subscribe sentence follows the opening hook and is brief.

## Production state

Script and scene plan are prepared only. No voice recording, TTS, editing, rendering, upload, or publication was run.
