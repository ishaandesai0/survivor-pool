# RUN GUIDE

Windows / PowerShell. All scripts import each other by name — keep them flat in one directory, don't nest them.

Repo: `github.com/ishaandesai0/survivor-pool` (private)

## Files

**Weekly use**
| file | what it does |
|---|---|
| `state.py` | reads `picks.csv`, derives `--used`/`--strikes`, prints the week's commands |
| `picks.csv` | your pick log — the single source of truth |
| `pipeline.py` | pulls nflverse data, builds `win_probs_model.csv` |
| `weekly.py` | ranks this week's picks at the point estimate |
| `weekly_robust.py` | ranks them under grid uncertainty — **prefer this** |

**Analysis**
| file | what it does |
|---|---|
| `stability.py` | which future picks are real information vs placeholder |
| `prize23.py` | expected dollars in the 200-entrant pool |
| `tie_analysis.py` | how 1st place resolves when nobody survives |
| `slots_model.py` | the 23-pick optimal path |
| `strikes_path.py` | P(≤k strikes) optimiser |
| `qc_grid.py` | validate a grid, recover matchups from probability complements |

**Model internals**
`market.py` (spread/moneyline → probability, de-vig) · `ratings.py` (market-implied power ratings) · `residual.py` (ML layer) · `fit_decay.py` (fits projection uncertainty) · `demo.py` (synthetic validation) · `smoke_test.py`

**Docs**
`POOL_STRATEGY.md` · `README_MODEL.md` · `WEEKLY.md`

## Fitted parameters

Measured, not guessed. Re-fit monthly at most.

| param | value | source |
|---|---|---|
| `--base` | 3.717 | `fit_decay.py` — sigma floor, present even 1 week out |
| `--decay` | 0.355 | `fit_decay.py` — sigma pts per week ahead |
| `sigma0` | 11.16 | MLE in `market.py` |
| HFA | 1.64 | fitted in `ratings.py` |

## Setup

Use **python.org** Python, not the Microsoft Store build — the Store version gets auto-updated by Windows and silently breaks its venvs.

```powershell
py -3.14 -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe smoke_test.py
```

Call `.venv\Scripts\python.exe` explicitly. `activate` does not reliably rebind `python` on this setup, and a hardcoded `python3` can silently run a different interpreter than you think.

Expect 14 PASS. Then check the numbers that matter:

```powershell
.venv\Scripts\python.exe demo.py
```

- A. efficient market → t ≈ −1.4, finds nothing
- B. planted 5.5pt QB edge → t ≈ +4.4, finds it
- C. ratings from spreads → r ≈ 0.9996, MAE ≈ 0.17

**If any of those fail, stop.** Exit codes prove a script didn't crash; only these three prove the numerics are still right after a library upgrade.

## Weekly routine

```powershell
.venv\Scripts\python.exe state.py
```

It validates `picks.csv` and prints the exact commands with flags filled in. Run those, pick, then log it:

```powershell
# picks.csv — double weeks (5,7,10,12,15) get two rows with the same week
# week,team,result,notes
# 1,LAC,W,model 83.0 devig 82.4

git add picks.csv win_probs_model.csv
git commit -m "Week 1: LAC (W). 0 strikes."
git push
```

Never type `--used` by hand. It's the one input nothing else can validate — `state.py` catches `LA` for `LAR`, duplicate teams, too many picks in a week, and infeasible paths.

Raise `--posted-through` as line coverage grows; your commitment horizon should extend as it does.

## Why `win_probs_model.csv` is committed

`.gitignore` deliberately does not ignore it. Committing it weekly makes git history a record of what the model believed *before* each slate resolved. Regenerating it in January from updated lines would leak hindsight into any calibration analysis.

## Known issues

- **`mm.report()` calibration is in-sample.** The ±0.000 reliability numbers come from evaluating the isotonic fit on its own training data. Meaningless as printed. Doesn't affect picks; needs a train/test split to be worth reading.
- **`fit_decay.py` only measured to d=14**, with thin data at the edge (80 samples vs ~160). Week 17–18 extrapolation is the softest part of the model. `--fit-weeks 2` extends the range at the cost of noisier ratings.
- **`win_probs_2026.csv`** is a hand transcription of a screenshot with ±5pt noise in Weeks 6–18 and some wrong recovered matchups. Kept only as a fallback and smoke-test fixture. Run everything off `win_probs_model.csv`.

## Troubleshooting

**`No Python at '...WindowsApps...'`** — Store Python was updated or removed and `.venv` points at a dead path. `py -0p` to list interpreters, then rebuild the venv.

**`ModuleNotFoundError: pyarrow`** — `nflreadpy` returns polars and `.to_pandas()` needs it. In `requirements.txt`, but install directly if missing.

**`No feasible path remains`** — `--used` leaves fewer teams than remaining slots. Run `state.py --check`.

**`No posted lines yet`** — books haven't hung the week. Seed from win totals via `prior_from_win_totals()` in `ratings.py`.

**Import errors** — the files got nested. Keep them flat.

**`git ls-files` shows `__pycache__`** — `.gitignore` only affects untracked files. `git rm -r --cached __pycache__`.
