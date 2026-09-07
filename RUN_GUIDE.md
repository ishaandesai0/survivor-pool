# RUN GUIDE

Everything lives in one flat folder. All scripts import each other by name, so don't nest them into subdirectories.

```
survivor_pool/
├── win_probs_2026.csv      the grid (my screenshot transcription — replace this)
├── qc_grid.py              validate grid + recover matchups
├── survivor.py             18-pick baseline model (superseded, kept for reference)
├── slots_model.py          23-pick optimiser  ← the real one
├── strikes_path.py         P(≤k strikes) optimiser
├── weekly.py               WEEKLY PICK ENGINE  ← what you'll use most
├── prize23.py              200-entrant dollar simulation
├── market.py               spread → win probability
├── ratings.py              market-implied power ratings
├── residual.py             ML residual layer
├── demo.py                 synthetic validation of the model pipeline
├── pipeline.py             real-data pipeline (NEEDS NETWORK)
├── smoke_test.py           verifies everything runs
└── requirements.txt
```

---

## 0. Setup (once)

```bash
cd survivor_pool
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

If `nflreadpy` fails to install from PyPI, get it from source:

```bash
pip install "nflreadpy @ git+https://github.com/nflverse/nflreadpy"
```

## 1. Verify everything runs (~20 seconds)

```bash
python3 smoke_test.py
```

Expect 8 PASS lines. `pipeline.py` is excluded because it needs network.

---

## 2. Clean the grid — do this first

```bash
python3 qc_grid.py
```

Checks every bye against the official 2026 schedule, then recovers all 272 matchups by pairing teams whose probabilities sum to 100. Cells marked `*` are likely transcription errors.

Weeks 1–5 are already clean. Weeks 6–18 carry ±5pt noise. Fix starred cells in `win_probs_2026.csv` against the screenshot and re-run until residuals drop.

```bash
python3 qc_grid.py | grep -E "^ +[0-9]+ +[0-9]+"     # just the checksum table
python3 qc_grid.py | grep -A16 "^Week 12"           # inspect one week
```

---

## 3. Get your Week 1 pick

```bash
python3 weekly.py --week 1
```

Every week after that, pass your actual state:

```bash
# single week, one strike, six teams burned
python3 weekly.py --week 6 --used LAC,TB,SF,CHI,DET,NE --strikes 1

# double week (5, 7, 10, 12, 15) — it searches PAIRS
python3 weekly.py --week 10 --used LAC,TB,SF,CHI,DET,NE,JAX,DEN,CIN,DAL,SEA --strikes 1

# rank by expected depth instead of survival
python3 weekly.py --week 8 --used ... --strikes 1 --objective depth

# once your own model is built
python3 weekly.py --week 4 --grid win_probs_model.csv
```

**`--used` must list every team you've burned, including from double weeks.** Wrong input here silently produces a wrong answer — it's the one place the tool can't check you.

---

## 4. Season plan and dollar view

```bash
python3 slots_model.py      # the 23-pick optimal path + forced weak picks
python3 strikes_path.py     # P(≤2 strikes) optimisation
python3 prize23.py --sims 700    # expected dollars (~4 min)
```

`prize23.py` is the slow one. Start at `--sims 200` to check it works, then run larger. Differences under ~$10 between strategies are Monte Carlo noise, not signal.

---

## 5. Build your own probabilities

**Validate the pipeline before trusting it:**

```bash
python3 demo.py
```

Three scenarios with known ground truth. Expect: efficient market → t ≈ −1.4 (finds nothing); planted 5.5pt QB edge → t ≈ +4.4 (finds it); ratings recovered at r ≈ 0.9996. If scenario A reports a real edge, something is leaking.

**Then run against real data (needs network):**

```bash
python3 pipeline.py --train 2007 2025 --season 2026 --week 1
```

Writes `win_probs_model.csv` in the same 32×18 shape everything else reads:

```bash
python3 qc_grid.py                              # after pointing it at the new file
python3 weekly.py --week 1 --grid win_probs_model.csv
```

I could not run `pipeline.py` here — this sandbox has no network — so it's the one file you should read before running. Everything else is tested.

---

## Weekly routine

```bash
# Tuesday, once lines are posted
python3 pipeline.py --train 2007 2025 --season 2026 --week <N>
python3 weekly.py --week <N> --used <everything so far> --strikes <N> --grid win_probs_model.csv
```

Re-solve every week. The August plan is a prior, not a commitment.

---

## Troubleshooting

**`No feasible path remains`** — your `--used` list leaves fewer available teams than remaining slots. Check for typos; abbreviations must match the CSV exactly (`LAR` not `LA`, `WAS` not `WSH`).

**`Three strikes — you're eliminated`** — `--strikes` must be 0, 1, or 2.

**`pipeline.py` says "No posted lines yet"** — books haven't hung Week 1 yet. Seed with market win totals via `prior_from_win_totals()` in `ratings.py`.

**`weekly.py` slow on double weeks** — it evaluates every pair (~400 Hungarian solves). A few seconds is normal.

**Import errors** — you nested the files. Keep them flat in one directory.
