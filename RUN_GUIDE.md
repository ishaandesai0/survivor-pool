# RUN GUIDE

Windows / PowerShell. All scripts import each other by name — keep them flat in one directory.

Repo: `github.com/ishaandesai0/survivor-pool` (private)

Call `.venv\Scripts\python.exe` explicitly. `activate` does not reliably rebind `python` here.

## Weekly routine

```powershell
.venv\Scripts\python.exe state.py
```

Reads `picks.csv`, derives `--used`/`--strikes`, prints the week's commands. **Override its `--posted-through` guess** — see Fitted parameters below.

After the sheet arrives:

```powershell
# paste the results PDF text into sheets\weekN.txt
.venv\Scripts\python.exe field.py parse sheets\weekN.txt --week N
.venv\Scripts\python.exe field.py report --me "Ishaan"
```

Then log the pick and commit:

```powershell
# picks.csv: week,team,result,notes   (double weeks = two rows, same week)
git add picks.csv win_probs_model.csv results_2026.csv sheets
git commit -m "Week N: TEAM (W/L). n strikes."
git push
```

Never type `--used` by hand. `state.py` catches `LA` for `LAR`, duplicate teams, too many picks in a week, and infeasible paths.

## Files

**Weekly**
| file | what it does |
|---|---|
| `state.py` | pool state from `picks.csv`; emits the week's commands |
| `pipeline.py` | pulls nflverse, builds `win_probs_model.csv` |
| `weekly.py` | ranks this week's picks (`--objective survive\|depth`) |
| `weekly_robust.py` | ranks them under grid noise — **prefer this** |
| `field.py` | parses the pool sheet into `field_state.csv` |

**Analysis**
| file | what it does |
|---|---|
| `stability.py` | which future picks are information vs placeholder |
| `ridge_cv.py` | cross-validates `--ridge` |
| `fit_decay.py` | fits `--base` and `--decay` |
| `prize23.py` | expected dollars (assumes 0 strikes — see Known issues) |
| `tie_analysis.py` | how 1st place resolves (same caveat) |
| `slots_model.py` / `strikes_path.py` | 23-pick path optimisers |
| `qc_grid.py` | validates a grid, recovers matchups from probability complements |

**Model internals**: `market.py` · `ratings.py` · `residual.py` · `demo.py` · `smoke_test.py`

**Data**: `win_probs_model.csv` (current grid, committed weekly as an audit trail) · `win_probs_2026.csv` (screenshot transcription, fallback/fixture only) · `picks.csv` · `results_2026.csv` · `sheets/` · `field_state.csv` (gitignored, regenerable)

## Fitted parameters

Measured, not guessed.

| param | value | source | re-fit |
|---|---|---|---|
| `--base` | 3.717 | `fit_decay.py` | rarely |
| `--decay` | 0.355 | `fit_decay.py` | rarely |
| `--ridge` | 0.5 | `ridge_cv.py` (historical arm) | every few weeks — falls as lines accumulate |
| `sigma0` | 11.16 | MLE in `market.py` | automatic |
| HFA | 1.83 | fitted in `ratings.py` | automatic |
| `--posted-through` | **2** | = last week with posted lines | **check every week** |

`--posted-through` is the trap. nflverse carried lookahead lines through Week 16 preseason, then swapped to current-week-only at kickoff — 112 posted games became 45. Read `games from posted lines` in the pipeline output and set it to the last week actually covered. Too high and `stability.py` / `weekly_robust.py` skip perturbing weeks that are really projections, making those picks look far more certain than they are.

## Setup

Use **python.org** Python, not the Microsoft Store build — the Store version gets auto-updated by Windows and breaks its venvs.

```powershell
py -3.14 -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe smoke_test.py
.venv\Scripts\python.exe demo.py
```

Smoke test checks that nothing crashes. `demo.py` checks the numbers, which is the part that matters after a library upgrade:

- A. efficient market → t ≈ −1.4, finds nothing
- B. planted 5.5pt QB edge → t ≈ +4.4, finds it
- C. ratings from spreads → r ≈ 0.9996, MAE ≈ 0.17

**If any fail, stop.**

## Full chain

```powershell
.venv\Scripts\python.exe smoke_test.py
.venv\Scripts\python.exe demo.py
.venv\Scripts\python.exe ridge_cv.py --season 2026 --week N
.venv\Scripts\python.exe pipeline.py --train 2007 2025 --season 2026 --week N --base 3.717 --decay 0.355 --ridge 0.5
.venv\Scripts\python.exe weekly.py --week N --used ... --strikes n --grid win_probs_model.csv --objective depth
.venv\Scripts\python.exe weekly_robust.py --week N --used ... --strikes n --grid win_probs_model.csv --draws 200 --posted-through P
```

Pipeline health check: sigma ~11.16, residual t near 0, ratings sd 2.5–5.0 with no compression warning, HFA 1.6–2.0, no week-sum warnings.

Grid validation (`qc_grid.py` only reads `win_probs_2026.csv`, so do this instead):

```powershell
.venv\Scripts\python.exe -c "import pandas as pd; d=pd.read_csv('win_probs_model.csv').set_index('team'); d.columns=[int(c) for c in d.columns]; print('teams', len(d)); [print(f'  W{w}: {d[w].sum():.1f} vs {100*(d[w].notna().sum()//2)}') for w in range(1,19)]"
```

Every week must match exactly (1600 / 1500 / 1400 / 1300 by bye count).

## Known issues

- **`mm.report()` calibration is in-sample.** The ±0.000 reliability column is isotonic regression scored on its own training data. Meaningless as printed. Needs a train/test split; doesn't affect picks.
- **`prize23.py` and `tie_analysis.py` assume 0 strikes and 23 picks.** Their EV and equity numbers describe a clean entrant, not your actual position. Use `weekly.py` for that.
- **`state.py` guesses `--posted-through` as week+5.** Wrong since the lookahead lines disappeared. Override it.
- **`qc_grid.py` hardcodes `win_probs_2026.csv`.** No `--grid` flag.
- **`fit_decay.py` only measured to d=14**, thin at the edge (80 samples vs ~160). Week 17–18 extrapolation is the softest part of the model.
- **`win_probs_2026.csv`** is a hand transcription with ±5pt noise in Weeks 6–18 and some wrong recovered matchups. Fixture only.

## Troubleshooting

**`No Python at '...WindowsApps...'`** — Store Python updated or removed; `.venv` points at a dead path. `py -0p`, then rebuild.

**`ModuleNotFoundError: pyarrow`** — `nflreadpy` returns polars; `.to_pandas()` needs it.

**`No feasible path remains`** — `--used` leaves fewer teams than slots. `state.py --check`.

**`No posted lines yet`** — books haven't hung the week. Seed from win totals via `prior_from_win_totals()`.

**`git status` shows `__pycache__`** — `.gitignore` only affects untracked files. `git rm -r --cached __pycache__`.

**PowerShell `The '<' operator is reserved`** — you pasted a `<placeholder>`. Substitute a real value.
