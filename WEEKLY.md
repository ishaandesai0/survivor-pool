# Weekly Pick Engine

```bash
python3 weekly.py --week 1
python3 weekly.py --week 6 --used LAC,TB,SF,CHI,DET,NE --strikes 1
python3 weekly.py --week 10 --used ... --strikes 1 --objective depth
python3 weekly.py --week 4 --grid win_probs_model.csv     # your own model
```

Run it every Tuesday once lines are posted. It reads the same 32×18 CSV that `pipeline.py` produces, so the moment your Vegas+ML model is live you just point `--grid` at its output and every downstream number updates.

## What it does that a static plan can't

**It re-solves the whole remaining season.** The August plan is a prior, not a commitment. Every strike and every line move changes what's optimal from here. It re-runs the Hungarian assignment over only the slots and teams you still have.

**It searches pairs in double weeks.** In Weeks 5/7/10/12/15 the best two teams individually are often *not* the best pair — taking both can strip two teams from the same stretch of schedule and leave a hole later. It evaluates all pairs directly rather than picking twice greedily.

**Its objective changes as you take strikes.** The budget is `k = 2 − strikes`, and that feeds straight into the Poisson-binomial tail:

| Strikes | Budget | Behaviour |
|---|---|---|
| 0 | 2 losses | Will accept a 70% pick to protect a scarce team |
| 1 | 1 loss | Balanced |
| 2 | 0 losses | **Pure single-elimination** — takes max win probability, ignores future value entirely |

That last switch is real and the tool announces it. At 2 strikes, future value is worth nothing because there is no future to spend it in.

## Reading the output

- **cost** — % of your remaining survival probability given up versus the best available pick. A `*` marks picks that are free (on an optimal path). Anything under ~5% is a legitimate alternative if you have a read the grid doesn't.
- **then** — the next several picks the optimizer would make *given* that choice. Useful for spotting when a tempting pick quietly wrecks the back half.
- **alive at each future pick** — your real horizon. This is the number that should stop you hoarding elite teams.
- **protect** — teams whose loss would hurt most from here. Don't spend these on marginal weeks.

## Worked examples

**Week 1, clean slate:** LAC is free (0.0% cost, 6.90% survival). JAX costs 4.6% — cheap enough to be a real contrarian option if you want separation. LV costs 9.4%.

**Week 5 double, 1 strike, four teams burned:** recommends NE+DET at 3.68%. Note DET+CIN averages a *higher* win% (74% vs 78%… on the pair average) yet scores worse — that's the pair interaction the greedy approach misses.

**Week 9, 2 strikes:** switches to zero-margin mode and takes SEA at 87%. KC at 82% costs 4.0%. Nothing else is close, and future value stops mattering.

## Two honest limits

- **It optimizes survival, not prize equity.** Given only ~1.6 of 200 entrants finish, maximizing depth and survival is a good proxy — but it doesn't model what your 199 rivals are picking. Use `prize23.py` for the dollar view.
- **Garbage in, garbage out.** The engine is exact; the grid isn't. Feed it the model output rather than my screenshot transcription as soon as you have real lines.
