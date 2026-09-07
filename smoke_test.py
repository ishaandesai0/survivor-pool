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
"""
import subprocess
import sys
import time

PY = sys.executable

CHECKS = [
    ("qc_grid.py",       [PY, "qc_grid.py"]),
    ("survivor.py",      [PY, "survivor.py", "--sims", "30"]),
    ("slots_model.py",   [PY, "slots_model.py"]),
    ("strikes_path.py",  [PY, "strikes_path.py"]),
    ("weekly.py wk1",    [PY, "weekly.py", "--week", "1", "--top", "3"]),
    ("weekly.py double", [PY, "weekly.py", "--week", "5", "--used",
                          "LAC,TB,SF,CHI", "--strikes", "1", "--top", "3"]),
    ("weekly.py 2strk",  [PY, "weekly.py", "--week", "9", "--used",
                          "LAC,TB,SF,CHI,DET,NE,JAX,DEN,CIN", "--strikes", "2",
                          "--top", "3"]),
    ("weekly_robust.py", [PY, "weekly_robust.py", "--week", "1", "--grid",
                          "win_probs_2026.csv", "--draws", "8",
                          "--posted-through", "6", "--top", "3"]),
    ("stability.py",     [PY, "stability.py", "--grid", "win_probs_2026.csv",
                          "--week", "1", "--draws", "8",
                          "--posted-through", "6"]),
    ("state.py",         [PY, "state.py"]),
    ("state.py --check", [PY, "state.py", "--check"]),
    ("tie_analysis.py",  [PY, "tie_analysis.py", "--grid",
                          "win_probs_2026.csv", "--sims", "10"]),
    ("prize23.py",       [PY, "prize23.py", "--sims", "10"]),
    ("demo.py",          [PY, "demo.py"]),
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
    print("\nAll good. Now run demo.py and CHECK ITS NUMBERS:")
    print("  A. efficient market      -> t ~ -1.4, finds nothing")
    print("  B. planted 5.5pt QB edge -> t ~ +4.4, finds it")
    print("  C. ratings from spreads  -> r ~ 0.9996, MAE ~ 0.17")
sys.exit(fails)
