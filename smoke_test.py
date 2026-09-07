"""Verify every offline script imports and runs before you rely on any of it."""
import subprocess, sys, time
PY = sys.executable          # add this
CHECKS = [
    ("qc_grid.py",      [PY,"qc_grid.py"]),
    ("survivor.py",     [PY,"survivor.py","--sims","30"]),
    ("slots_model.py",  [PY,"slots_model.py"]),
    ("strikes_path.py", [PY,"strikes_path.py"]),
    ("weekly.py wk1",   [PY,"weekly.py","--week","1","--top","3"]),
    ("weekly.py dbl",   [PY,"weekly.py","--week","5","--used","LAC,TB,SF,CHI",
                         "--strikes","1","--top","3"]),
    ("demo.py",         [PY,"demo.py"]),
    ("prize23.py",      [PY,"prize23.py","--sims","15"]),
]
fails = 0
for name, cmd in CHECKS:
    t0 = time.time()
    r = subprocess.run(cmd, capture_output=True, text=True)
    ok = r.returncode == 0
    fails += not ok
    print(f"[{'PASS' if ok else 'FAIL'}] {name:<18} {time.time()-t0:5.1f}s")
    if not ok:
        print(r.stderr.strip()[-600:])
print("\nAll good." if not fails else f"\n{fails} failing.")
sys.exit(fails)
