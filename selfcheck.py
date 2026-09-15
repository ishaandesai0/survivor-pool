"""
Static checks — no network, no simulation, ~2 seconds.

Exists because the equivalent checks as PowerShell one-liners are a quoting
minefield: nested quotes and backslash escapes get mangled by the shell
before Python ever sees them. A file has no such problem.

Run this before committing. It verifies the files are intact and internally
consistent, which is different from smoke_test.py (does everything RUN?)
and demo.py (are the NUMBERS right?). All three answer different questions;
run all three.

    python selfcheck.py
"""
import ast
import os
import re
import sys

FILES = [
    "state.py", "pipeline.py", "field.py", "ridge_cv.py", "smoke_test.py",
    "weekly.py", "weekly_robust.py", "stability.py", "ratings.py",
    "market.py", "residual.py", "prize23.py", "tie_analysis.py",
    "slots_model.py", "qc_grid.py", "survivor.py", "strikes_path.py",
    "fit_decay.py", "demo.py", "selfcheck.py",
]

# The validated parameters. Any drift between these and what the code
# actually defaults to is exactly the failure that silently rebuilt a
# compressed grid, so it gets an explicit check rather than a comment.
EXPECT = {"base": 3.717, "decay": 0.355, "ridge": 0.5}

fails = []


def ok(label, passed, detail=""):
    print(f"  [{'OK  ' if passed else 'FAIL'}] {label}{'  ' + detail if detail else ''}")
    if not passed:
        fails.append(label)


print("=" * 66)
print("STATIC SELF-CHECK")
print("=" * 66)

print("\nfiles present and parsing:")
srcs = {}
for f in FILES:
    if not os.path.exists(f):
        ok(f, False, "MISSING")
        continue
    try:
        s = open(f, encoding="utf-8").read()
        ast.parse(s)
        srcs[f] = s
        ok(f, True)
    except SyntaxError as e:
        ok(f, False, f"line {e.lineno}: {e.msg}")

print("\npipeline.py defaults match the validated parameters:")
p = srcs.get("pipeline.py", "")
for name, want in EXPECT.items():
    m = re.search(rf'--{name}", type=float, default=([\d.]+)', p)
    got = float(m.group(1)) if m else None
    ok(f"--{name} default", got == want, f"got {got}, want {want}")

print("\npipeline.py still has the fixes we added:")
for label, tok in [
    ("team abbreviation map", "TEAM_FIXES"),
    ("LA -> LAR", '"LA": "LAR"'),
    ("base/decay wired to project", "base=args.base"),
    ("ridge wired to PowerRatings", "ridge=args.ridge"),
    ("ratings spread diagnostic", "ratings spread"),
    ("de-vig moneylines", 'method="shin"'),
    ("moneyline-aware grid emit", 'source == "moneyline"'),
]:
    ok(label, tok in p)

print("\nstate.py prints the right flags:")
st = srcs.get("state.py", "")
ok("ridge in printed command", "--ridge {RIDGE}" in st or "--ridge 0.5" in st)
ok("params as constants", "BASE, DECAY, RIDGE" in st)
m = re.search(r"BASE, DECAY, RIDGE = ([\d.]+), ([\d.]+), ([\d.]+)", st)
if m:
    vals = tuple(float(x) for x in m.groups())
    ok("constants match", vals == (EXPECT["base"], EXPECT["decay"],
                                   EXPECT["ridge"]), f"got {vals}")
else:
    ok("constants match", False, "not found")

print("\nratings.py two-parameter sigma model:")
r = srcs.get("ratings.py", "")
ok("base term in sigma", "base**2" in r)
ok("no stale 3.0 multiplier", "decay * d * 3.0" not in r)
ok("uses posted lines when available", "use_posted" in r)

print("\nmarket.py:")
mk = srcs.get("market.py", "")
ok("devig present", "def devig" in mk)
ok("moneyline -100 safe", "out[neg]" in mk)

print("\ndata files:")
for f, need in [("picks.csv", "week,team,result"),
                ("win_probs_model.csv", "team,1,2"),
                ("results_2026.csv", "week,away"),
                ("requirements.txt", "nflreadpy")]:
    if not os.path.exists(f):
        ok(f, False, "MISSING")
        continue
    body = open(f, encoding="utf-8").read()
    # search the whole file, not just line 1 -- requirements.txt lists
    # nflreadpy on line 7, which the old first-line check failed on
    hit = need in body if f == "requirements.txt" else \
        body.lstrip().startswith(need)
    ok(f, hit, "" if hit else f"looking for {need!r}")

print("\ngrid integrity:")
try:
    import pandas as pd
    d = pd.read_csv("win_probs_model.csv").set_index("team")
    d.columns = [int(c) for c in d.columns]
    ok("32 teams", len(d) == 32, f"got {len(d)}")
    bad = []
    for w in range(1, 19):
        n = int(d[w].notna().sum() // 2)
        if abs(d[w].sum() - 100 * n) > 0.5:
            bad.append(w)
    ok("every week sums to 100 x games", not bad, f"bad: {bad}" if bad else "")
    byes = {w: int(d[w].isna().sum()) for w in range(1, 19)}
    expect_byes = {5: 2, 6: 4, 7: 4, 8: 4, 9: 2, 10: 4, 11: 6, 13: 4, 14: 2}
    got = {w: b for w, b in byes.items() if b}
    ok("bye weeks match 2026 schedule", got == expect_byes,
       "" if got == expect_byes else f"got {got}")
except Exception as e:
    ok("grid checks", False, str(e)[:60])

print("\n" + "=" * 66)
if fails:
    print(f"{len(fails)} FAILED: {', '.join(fails)}")
else:
    print("All static checks passed.")
    print("Next: smoke_test.py (does everything run?), then demo.py (are the")
    print("numbers right?). Three different questions.")
print("=" * 66)
sys.exit(len(fails))
