# EURUSD V2 replay lab — script and shot plan draft 03

**Revision:** EURUSD-V2-LAB-SCRIPT-DRAFT-03
**Status:** corrected script approved; technical and visual review passed; human audio listen pending
**Format/language:** 16:9 educational Lab episode, English
**Approval:** User approved the exact corrected sentence “The largest drawdown ended at an equity value of 9,984 dollars on August nineteenth.” All remaining narration and the shot plan are preserved.

## Editorial direction

The episode now teaches a practical route from the permitted EURUSD H1 export to an inspectable development model run, then follows the frozen predictions into the separately verified quote replay. The tutorial development smoke is not the final confirmation run. The replay kit reads the frozen predictions and does not retrain the final model.

**Selected hook:** “The model abstained on most hours. Yet its fixed signals produced 74 virtual positions on the saved quote replay. Let’s trace the build from an H1 CSV to the bid and ask fills.”

**Viewer outcome:** inspect the UTC H1 schema, build the contract’s 12 point-in-time features, fit the existing train-only scaler/logistic pipeline in a new development folder, read chronological validation and exported predictions, then reproduce the frozen quote replay.

## Exact narration draft

### 1 — Hook and CTA

The model abstained on most hours. Yet its fixed signals produced seventy-four virtual positions on the saved quote replay. Let’s trace the build from an H1 file to the bid and ask fills.

This is an educational replay on saved demo data. It sent no broker orders.

If you build with Python and MetaTrader, like this video and subscribe for practical engineering.

### 2 — Inspect the permitted export

Start with the permitted EURUSD hourly export. It has twelve thousand four hundred twenty-nine rows and seven columns: time_utc, open, high, low, close, tick_volume, and spread. The manifest identifies time_utc as UTC bar-start time. The reporting interval includes October sixteenth, 2024, and ends before October seventeenth, 2026.

Here is the existing loader checking that schema and the reviewed gaps. The timestamps stay in UTC; each reviewed gap resets rolling features.

### 3 — Build the features and target

The feature contract selects exactly twelve inputs: return_1h, return_6h, return_24h; rolling_vol_6h and rolling_vol_24h; range_pct, body_return, and relative_tick_volume_24h; then hour_sin, hour_cos, dow_sin, and dow_cos.

At each completed bar, the decision time is its close. The target is the direction of the next hourly close-to-close move. The feature builder uses only current and earlier bars, and starts new windows after a data gap. The first training decision is October seventeenth at one a.m. UTC. On screen, we print its twelve feature values before fitting anything.

### 4 — Fit a development model

The existing pipeline is StandardScaler followed by LogisticRegression. The runner fits both on the training partition only. Run it into a new development output folder. This smoke scores train, validation, and diagnostic partitions; it does not score the final confirmation window.

The terminal reports development mode, final confirmation evaluated false, and train-only fit scope. The model file saves the scaler parameters, coefficients, and the exact feature names. No trading profit is calculated by this training command.

### 5 — Read chronological validation and predictions

The partitions move forward in UTC: train ends October 2025, validation ends April 2026, diagnostic ends July, and the untouched final confirmation comes last. This smoke has four thousand seven hundred fifty-seven training rows, two thousand three hundred six validation rows, and one thousand two hundred twenty-two diagnostic rows. It leaves the one-thousand-two-hundred-fifty-one final rows unscored.

The predictions export contains eight thousand two hundred eighty-five development rows. Its first validation decision is October seventeenth, 2025, at midnight UTC. The model predicts up with a raw score of about point five-zero-eight; the fixed candidate band labels it ABSTAIN. Validation accuracy is 51.30 per cent, against 49.87 for the majority baseline and 49.96 for the previous-hour baseline. These are classification summaries, not trading returns, and the raw score is not calibrated.

### 6 — Freeze the replay rules

The saved final predictions use fixed bands: at or below point four-five is short; at or above point five-five is long; the middle is ABSTAIN. Across the final window that makes twenty-one long, eighty-nine short, and one thousand one hundred forty-one ABSTAIN decisions.

The replay permits one virtual one-lot position. A same-side signal does nothing: it neither adds size nor moves the original one-hour target. ABSTAIN also leaves an open position on that target. An opposite signal can reverse at the next eligible quote, but there were zero actual reversals in this window.

### 7 — Follow the quotes and explain profit factor

For a long entry, the replay takes the first eligible ask; for a short, the first eligible bid. Exits use the other side at or after the one-hour target. The observed spread is included. Additional slippage is assumed zero, and historical swap is unavailable.

All seventy-four closed positions stay visible on the moving quote chart: thirty-six wins, thirty-seven losses, and one breakeven. The quote result is plus twelve hundred nine dollars before swap. The win rate is 48.65 per cent, while profit factor is 1.5863: three thousand two hundred seventy-one dollars of gross wins divided by two thousand sixty-two dollars of gross losses. Fewer positions won, but the average win, about 90 dollars and 86 cents, was larger than the average loss, about 55 dollars and 73 cents. That arithmetic explains the pair of metrics; it does not establish a durable edge or an account-wide net result.

### 8 — Track balance, equity, and drawdown

Balance changes when a position closes. Equity marks the open virtual position at each observed quote. Sweep one UTC cursor across both curves. Maximum balance drawdown was 469 dollars, or 4.47 per cent from its earlier balance peak of 10,495 dollars. Maximum equity drawdown was 515 dollars, or 4.91 per cent from its preceding 10,499-dollar equity peak. The largest drawdown ended at an equity value of 9,984 dollars on August nineteenth. No values are interpolated across missing quote time.

### 9 — Inspect two post-hoc examples

Now zoom to the largest winner and largest loser, selected after all outcomes were known. On October first, a short moved from bid 1.12964 to ask 1.12659 for plus 305 dollars. On August nineteenth, another short moved from bid 1.16357 to ask 1.16564 for minus 207 dollars. Label these as post-hoc extremes, then return to all seventy-four positions.

### 10 — State the remaining limits

This quote replay uses a ten-thousand-dollar starting balance, one standard lot, and a hypothetical 1-to-200 leverage scenario. The current demo snapshot says 1-to-100; the replay does not model margin calls or stopouts. Commissions, fees, and taxes are zero by the saved scenario assumptions, not by an account statement. Historical swap terms and rollover timing are unavailable, so after-swap totals remain withheld.

A new feature set or signal band needs a new protocol and a later untouched confirmation window. The current final window has already been evaluated and stays fixed.

### 11 — Reproduce the frozen replay

Run the one-command kit with the same authorized H1 snapshot and saved demo quote packages. It checks input hashes, copies the frozen model and predictions, runs the quote replay, and compares its results with an independent verifier. It does not retrain the final model. The verified run writes every position, signal action, balance and equity curve, metrics, and provenance manifest.

With access to those same authorized files, run the kit and compare the verification report. With another export, follow the same method but expect different numbers. The useful result is an inspectable build and replay. Every displayed value leads back to a file.

## Practical build route and actual code-to-output inserts

The companion walkthrough is h1_model_build_walkthrough.md. It uses the permitted two-year UTC H1 export and existing code only.

| Beat | Existing command/code | Recorded output to show |
|---|---|---|
| Inspect source | Python CSV header/count command in the companion walkthrough; source manifest and prepare_samples validation. | 12,429 rows; time_utc in UTC; columns: time_utc, open, high, low, close, tick_volume, spread. |
| Build features and splits | load_protocol, prepare_samples, partition_samples; build_point_in_time_features; exact feature-contract list. | 12 selected names; eligible rows train 4,757, validation 2,306, diagnostic 1,222, final confirmation 1,251. |
| Fit and score development | experiments/eurusd_h1_model_error_lab/run_lab.py without --run-final-confirmation; output in outputs/development_smoke_20261019/. | run_mode=development; final_confirmation_evaluated=false; scaler and logistic fit scopes both train_only; scored partitions train/validation/diagnostic only. |
| Read validation/export | metrics.json, model.json, predictions.csv. | Validation B2 51.30% accuracy / 51.29% balanced accuracy; 8,285 prediction rows; first validation raw UP score 0.5078778 and candidate action ABSTAIN. |
| Replay the frozen output | One-command replay kit. | Separate run; no model retraining; independent verifier and exact expected-output comparison. |

### Exact feature contract

return_1h, return_6h, return_24h, rolling_vol_6h, rolling_vol_24h, range_pct, body_return, relative_tick_volume_24h, hour_sin, hour_cos, dow_sin, dow_cos.

The feature contract has 12 selected features. return_3h, return_12h, and rolling_vol_12h are candidate features explicitly excluded from this baseline.

### Shot-by-shot plan

All shots are 16:9, data-led, and motion-first. Price and quote lines move continuously. Commands and outputs appear beside the moving chart in short readable inserts; no static table replaces the market timeline.

| Shot | Time | Picture and proof | Narration coverage |
|---|---:|---|---|
| 01 | 0:00–0:23 | Saved bid/ask path moves; all 74 entry/exit brackets build on the chart, losses included. Overlay: 74 VIRTUAL POSITIONS. | Hook, replay scope, Like/Subscribe. |
| 02 | 0:23–0:55 | Terminal shows CSV header, row count, UTC manifest interval, then a chart cursor marks the half-open endpoints. | H1 schema, UTC bar-start meaning, date range. |
| 03 | 0:55–1:45 | Completed-bar cursor reveals past bars only; exact 12 feature names stream into a live 12-column preview. A gap resets the rolling window. | Feature builder, target timing, first training sample output. |
| 04 | 1:45–2:27 | Short code insert shows StandardScaler to LogisticRegression; terminal displays the development command and JSON summary. | Train-only fit and isolated smoke; final partition left unscored. |
| 05 | 2:27–3:18 | UTC split timeline advances; validation score appears beside the moving price line; prediction CSV scrolls to first validation row. | Chronological validation, baselines, raw score and ABSTAIN example. |
| 06 | 3:18–3:58 | Probability needle crosses 0.45/0.55; one-position track shows same-side and ABSTAIN no-op, preserving the initial target. | Frozen decision bands and position policy. |
| 07 | 3:58–5:05 | All 74 positions animate on continuous bid/ask paths. Beside it, gross wins and losses count up to PF 1.5863 and 48.65% win rate. | Quote fills, wins/losses, +$1,209 before swap, why PF exceeds 1 despite a sub-50% win rate. |
| 08 | 5:05–5:47 | Balance/equity/drawdown panels share the same moving UTC cursor; peak-to-trough brackets animate. | Balance versus equity and the two peak-based drawdowns. |
| 09 | 5:47–6:20 | Best and worst positions zoom into bid/ask ticks, then return to the complete overview. | Post-hoc +$305 and −$207 examples. |
| 10 | 6:20–6:52 | Scenario labels remain beside the price path; unknown swap and unmodeled margin are highlighted briefly, then cleared. | Leverage scenario, zero-cost assumptions, future holdout requirement. |
| 11 | 6:52–7:25 | Terminal runs one command; verified output names appear beside a small hash/status panel. End on the full moving chart and CTA. | One-command frozen replay, exact-data caveat, closing. |

**Timing estimate:** 984 spoken words at roughly 140 words per minute is about 7:02, plus roughly 23 seconds of chart holds for a 7:25 total. Re-time after script approval and an actual voice read.

## Evidence and review status

The exact frozen H1 snapshot, model, predictions, and demo quote packages are private workspace inputs and are not included in the public repository. The public CSV replay interface accepts a viewer’s own frozen decisions and bid/ask quotes; its outputs will differ for different inputs. See `experiments/eurusd_h1_model_error_lab/backtest/v2/README.md` for the public schema, export example, assumptions, and synthetic-test scope.

The saved demo quote history has no recorded redistribution rights. Aggregate replay figures in `backtest/v2/public/proof_summary.json` do not make the private input files public.

**Approval record:** The user approved the exact factual correction and requested publication after review. Technical and visual checks passed; a human listen of the regenerated paragraph and final AAC mux remains pending. No platform action has occurred.
