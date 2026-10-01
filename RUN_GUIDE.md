# NFL Survivor 2026 — Run Guide

200 entrants · $5,000 · 23 picks · 3 strikes · 10 divisions of 20

Windows / PowerShell. Keep all files flat in one directory — they import each other by name.
Repo: `github.com/ishaandesai0/survivor-pool` (private)

```powershell
cd C:\Users\ishaa\survivor_pool
.venv\Scripts\activate       # prompt must show (.venv)
```

Without `(.venv)` you're on base Python with a different package set.

---

## Weekly routine

**Tuesday** — refit on the new results, get the week's candidates:

```powershell
python ridge_cv.py --season 2026 --week N          # every few weeks
python pipeline.py --train 2007 2025 --season 2026 --week N
python weekly.py --week N --used ... --strikes n --grid win_probs_model.csv --objective depth
python weekly_robust.py --week N --used ... --strikes n --grid win_probs_model.csv --draws 200 --posted-through P
python division_opt.py --week N
```

**Thursday, Friday, Saturday** — check for staleness before submitting:

```powershell
python prekick.py --week N --used ... --strikes n
```

Week 3 showed the cost of skipping these: a QB designation appeared Thursday and cleared Friday, unseen.

**After the sheet posts:**

```powershell
python field.py parse sheets\weekN_sheet.txt --week N
python field.py report --me "Ishaan"
python prize_live.py --me "Ishaan" --week N --sims 1500
```

**Then log and commit:**

```powershell
# picks.csv: week,team,result,notes   (double weeks = two rows, same week)
git add picks.csv win_probs_model.csv results_2026.csv sheets division_roster.csv
git commit -m "Week N: TEAM (W/L). n strikes."
git push
```

`state.py` prints the week's commands with flags derived from `picks.csv` — never type `--used` by hand.

---

## Files

**Weekly** · `state.py` (pool state → commands) · `pipeline.py` (builds the grid) · `weekly.py` (ranks picks) · `weekly_robust.py` (ranks under grid noise — **prefer this**) · `prekick.py` (line movement + injuries) · `field.py` (parses the pool sheet) · `division_opt.py` (optimises for the $150 division prize)

**Periodic** · `ridge_cv.py` (cross-validates `--ridge`) · `fit_decay.py` (fits `--base`/`--decay`) · `stability.py` (which future picks are information vs placeholder) · `prize_live.py` (expected dollars from real field state) · `qc_grid.py` (validates a grid, recovers matchups)

**Checks** · `selfcheck.py` (files intact?) · `smoke_test.py` (everything runs?) · `demo.py` (numbers right?) · `deps.py` (import graph)

**Libraries** · `market.py` · `ratings.py` · `residual.py` · `slots_model.py` · `survivor.py`

**Data** · `win_probs_model.csv` (live grid, committed weekly as an audit trail) · `picks.csv` · `results_2026.csv` · `division_roster.csv` · `sheets/` · `win_probs_2026.csv` (screenshot transcription — smoke-test fixture only) · `field_state.csv` (gitignored, regenerable)

---

## Fitted parameters

All measured, none guessed.

| param | value | source | re-fit |
|---|---|---|---|
| `--base` | 3.717 | `fit_decay.py` | rarely |
| `--decay` | 0.355 | `fit_decay.py` | rarely |
| `--ridge` | 0.5 | `ridge_cv.py`, historical arm | every few weeks |
| `sigma0` | 11.16 | MLE in `market.py` | automatic |
| HFA | ~1.4 | fitted in `ratings.py` | automatic |
| `--posted-through` | = last week with posted lines | pipeline output | **check weekly** |

`--posted-through` is the trap. nflverse carried lookahead lines through Week 16 preseason, then switched to current-week-only at kickoff. Read `sources: posted N` from the pipeline and set it to the last week actually covered. Too high and `stability.py`/`weekly_robust.py` skip perturbing weeks that are really projections.

---

## Setup

Use **python.org** Python, not the Microsoft Store build — the Store version gets auto-updated by Windows and breaks its venvs.

```powershell
py -3.14 -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe selfcheck.py
.venv\Scripts\python.exe smoke_test.py
.venv\Scripts\python.exe demo.py
```

`demo.py` must show: efficient market → t ≈ −1.4 (finds nothing); planted 5.5pt QB edge → t ≈ +4.4 (finds it); ratings from spreads → r ≈ 0.9996, MAE ≈ 0.17. **If any fail, stop.**

---

## Known issues

- **`mm.report()` reliability is in-sample.** The ±0.000 column is isotonic regression scored on its own training data. Hidden behind `--verbose`. Needs a train/test split; doesn't affect picks.
- **`field.py` reads the sheet's X column as state *entering* that week**, including any Thursday game already played. `picks.csv` is authoritative for your own count.
- **`fit_decay.py` only measured to d=14**, thin at the edge. Week 17–18 extrapolation is the softest part of the model.
- **The field popularity model does not fit.** It predicted SF as the Week 2 modal pick; the field took TB 88–57. Treat `prize_live` EV as indicative, ordering as meaningful.
- **`survivor.py` is the obsolete 18-pick model** but `slots_model.py` still imports `solve_path` from it, and `stability.py` imports `slots_model`. Live dependency; untangling is refactoring, not cleanup.

---

## Troubleshooting

**`No Python at '...WindowsApps...'`** — Store Python updated or removed; `.venv` points at a dead path. `py -0p`, rebuild the venv.

**`ModuleNotFoundError: pyarrow`** — `nflreadpy` returns polars; `.to_pandas()` needs it.

**`No feasible path remains`** — `--used` leaves fewer teams than slots. `state.py --check`.

**Entrant count ≠ 200 in `field.py`** — names fragmenting across weeks. It prints the offenders.

**`The '<' operator is reserved`** — you pasted a `<placeholder>`. Substitute a real value.

**`git status` shows `__pycache__`** — `.gitignore` only affects untracked files. `git rm -r --cached __pycache__`.
