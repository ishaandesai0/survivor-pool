# Vegas + ML Win Probability Model

## The one decision that matters

Don't train a classifier on "did the home team win" from team stats. That model spends its capacity relearning, badly, what the point spread already encodes — and then loses to the spread. The NFL closing line aggregates injuries, weather, sharp money and information no public model has access to.

So the architecture is **market-first, ML as a correction**:

```
closing spread ──► calibrated win probability          (Layer 1: the bar)
       │
       ├──► ridge-inverted into power ratings ──► spreads for ALL 18 weeks
       │                                                (Layer 2: the hard part)
       └──► ML predicts market ERROR, shrunk hard        (Layer 3: the long shot)
```

The ML target is `actual_margin - spread_line`. By construction that's what the line doesn't already know. If your features have no edge, the target is pure noise and the model correctly learns ~0 — an honest answer instead of a false one.

## What you should actually expect

The closing line is close to efficient. Realistic out-of-sample R² on market error is near zero, and most published attempts to beat it don't survive walk-forward validation. **Beating the closing line by 0.3 points of margin would be a genuinely strong result.**

The realistic prize is not accuracy — it's **calibration and full-season coverage**. For survivor you don't need to pick winners better than Vegas. You need probabilities whose 80% bucket actually goes 80%, for all 23 picks, in August, when only Week 1 has posted lines. That's a real gap the market doesn't fill, and it's where your work adds value.

## Layer 1 — spread → win probability

Two conversions, fitted rather than assumed:

- **Probit:** `P(win) = Φ(spread / σ)`, with σ fitted by maximum likelihood. Smooth, extrapolates safely.
- **Isotonic:** monotone empirical fit. Captures NFL **key numbers** — margins pile up on 3 and 7 because of field goals and touchdowns, so the true curve has flat spots a normal CDF smooths away.

Shipped default blends them: isotonic in the dense middle, probit in the sparse tails. Also included is **de-vigging** for moneylines (multiplicative and Shin). At a 4.5% hold a raw 0.85 favourite is really about 0.81 — across 23 picks that gap compounds badly.

## Layer 2 — market-implied power ratings

This solves the problem specific to survivor: you need Week 15 probabilities now, and no book posts them. Invert the market instead.

If `spread_home ≈ rating_home − rating_away + HFA`, a week of lines is a linear system in team ratings. Two details matter:

- **Ridge, not OLS.** One week gives ~16 equations for 32 unknowns. OLS returns garbage; the ridge penalty pulls unseen teams toward a prior derived from market win totals (~2.6 pts per win above .500).
- **σ widens with weeks-ahead.** A Week 15 projection inherits real uncertainty about who those teams even are by then. Skipping this is how these grids end up overconfident about Week 18 — and in survivor, overconfidence costs more than being wrong, because you plan around picks that were never as safe as claimed.

This is exactly what "Week 18 modeled from power ratings" meant on the grid you found.

## Layer 3 — the ML layer

`HistGradientBoostingRegressor` on pre-kickoff features only: rest differential, off-bye, short week, travel and timezone shift, division/primetime, **QB change** (the single biggest one), wind/temp/dome, recent EPA differentials, and the spread itself so the model can find spread-dependent bias.

Output is shrunk by a coefficient fitted from cross-validated predictions. If the model is noise, that coefficient collapses to zero on its own.

## The synthetic harness — run this first

You cannot validate a forecasting pipeline on real data alone, because a null result is ambiguous between "the market is efficient" and "my code is broken." `demo.py` generates seasons from a known process:

| Scenario | Expected | Got |
|---|---|---|
| A. Efficient market | find nothing | t = **−1.38**, shrink 0.00 ✓ |
| B. Backup QB worth 5.5 pts, unpriced | find it | t = **+4.42**, shrink 0.55, log loss 0.6170 → 0.6143 ✓ |
| C. Recover ratings from spreads only | recover them | r = **0.9996**, MAE **0.17 pts**, HFA 1.85 vs 1.90 true ✓ |

**This harness already earned its keep.** My first version judged edges by R², and it failed scenario B — calling a real, known, money-making 5.5-point edge "no edge," because an effect appearing in 10% of games against 13.5 points of noise explains almost no variance. The detector is now a **t-statistic on the shrinkage coefficient**, which passes both scenarios. If you take one thing from this: R² is the wrong statistic for betting-model edges, and a synthetic test with a planted signal is the only way to catch that.

## Data

`nflreadpy` — the current Python loader; `nfl_data_py` is deprecated. `load_schedules()` carries `spread_line`, `total_line`, moneylines and rest days back to 1999. Treat its market fields as closing proxies rather than authenticated book snapshots. CC-BY-4.0.

Suggest training on 2007+ — HFA has drifted down meaningfully since, and older seasons will bias it high.

## Running it

```bash
pip install nflreadpy scikit-learn scipy pandas numpy
python3 demo.py                                    # validate first
python3 pipeline.py --train 2007 2025 --season 2026 --week 1
```

`pipeline.py` writes `win_probs_model.csv` in the same 32×18 shape the survivor code already reads, so it drops straight into `qc_grid.py`, `slots_model.py` and `prize23.py`. It self-checks that every week sums to 100 × games — the same constraint that caught the transcription errors in the screenshot grid.

## Where to actually find edge

Ranked by realistic payoff:

1. **Better future-week projection.** Everyone has this week's line; almost nobody projects Week 15 well. This is your genuine edge in a survivor pool and it needs no market inefficiency at all.
2. **Line shopping / opening lines.** Openers are softer than closers. Modelling the open-to-close move is a real, documented signal — and it's a forecasting problem, not a handicapping one.
3. **QB injury latency.** The market prices a confirmed QB change fast, but there's a window before confirmation. Needs a fast news pipeline more than a better model.
4. **Beating the closing line on fundamentals.** Lowest expected payoff. Try it, but validate walk-forward and believe the null result if you get one.

## Files
`market.py` · `ratings.py` · `residual.py` · `demo.py` · `pipeline.py`
