"""
Verify every offline script runs before you rely on any of it.

Uses sys.executable, not the string "python3": on Windows a venv often does
not rebind `python`, so a hardcoded name can silently test a DIFFERENT
interpreter than the one you invoked. That happened here once -- the suite
passed against a system Python that had the packages while the venv was
empty.

Exit codes only prove a script did not crash. For numerical correctness run
demo.py and read the three scenario results; that is the harness that
catches library behaviour changing under you.

Scripts needing network (pipeline, fit_decay, ridge_cv, prekick) are checked
with --help only -- enough to catch an import error or broken argparse,
without a 2-minute nflverse pull on every run.
"""
import os
import subprocess
import sys
import time

PY = sys.executable

CHECKS = [
    # offline, real work
    ("qc_grid.py",       [PY, "qc_grid.py"]),
    ("slots_model.py",   [PY, "slots_model.py"]),
    ("weekly.py wk1",    [PY, "weekly.py", "--week", "1", "--top", "3"]),
    ("weekly.py double", [PY, "weekly.py", "--week", "5", "--used",
                          "LAC,TB,SF,CHI", "--strikes", "1", "--top", "3"]),
    ("weekly.py 2strk",  [PY, "weekly.py", "--week", "9", "--used",
                          "LAC,TB,SF,CHI,DET,NE,JAX,DEN,CIN", "--strikes", "2",
                          "--top", "3"]),
    ("weekly.py depth",  [PY, "weekly.py", "--week", "2", "--used", "LAC",
                          "--strikes", "1", "--objective", "depth",
                          "--top", "3"]),
    ("weekly_robust.py", [PY, "weekly_robust.py", "--week", "1", "--grid",
                          "win_probs_2026.csv", "--draws", "8",
                          "--posted-through", "2", "--top", "3"]),
    ("stability.py",     [PY, "stability.py", "--grid", "win_probs_2026.csv",
                          "--week", "1", "--draws", "8",
                          "--posted-through", "2"]),
    ("state.py",         [PY, "state.py"]),
    ("state.py --check", [PY, "state.py", "--check"]),
    ("deps.py",          [PY, "deps.py"]),
    ("demo.py",          [PY, "demo.py"]),
    # import + argparse only; these hit the network when run for real
    ("pipeline.py -h",   [PY, "pipeline.py", "--help"]),
    ("fit_decay.py -h",  [PY, "fit_decay.py", "--help"]),
    ("ridge_cv.py -h",   [PY, "ridge_cv.py", "--help"]),
    ("prekick.py -h",    [PY, "prekick.py", "--help"]),
    ("prize_live.py -h", [PY, "prize_live.py", "--help"]),
    ("division_opt -h",  [PY, "division_opt.py", "--help"]),
]

# these need data files that only exist mid-season
SHEET = os.path.join("sheets", "week3_sheet.txt")
if os.path.exists(SHEET):
    CHECKS += [
        ("field.py parse",  [PY, "field.py", "parse", SHEET, "--week", "3"]),
        ("field.py report", [PY, "field.py", "report", "--me", "Ishaan"]),
    ]
if os.path.exists("division_roster.csv") and os.path.exists("win_probs_model.csv"):
    CHECKS += [
        ("division_opt run", [PY, "division_opt.py", "--week", "4",
                              "--sims", "20", "--top", "3"]),
    ]

fails = 0
print(f"interpreter: {PY}\n")
for name, cmd in CHECKS:
    t0 = time.time()
    r = subprocess.run(cmd, capture_output=True, text=True)
    ok = r.returncode == 0
    fails += not ok
    print(f"[{'PASS' if ok else 'FAIL'}] {name:<20} {time.time()-t0:5.1f}s")
    if not ok:
        print(r.stderr.strip()[-700:])

if fails:
    print(f"\n{fails} failing.")
else:
    print(f"\nAll {len(CHECKS)} good. Now run demo.py and CHECK ITS NUMBERS:")
    print("  A. efficient market      -> t ~ -1.4, finds nothing")
    print("  B. planted 5.5pt QB edge -> t ~ +4.4, finds it")
    print("  C. ratings from spreads  -> r ~ 0.9996, MAE ~ 0.17")
sys.exit(fails)
