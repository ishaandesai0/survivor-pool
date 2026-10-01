"""
Which files are actually used?

Deleting by eye is how you remove a module that something still imports at
runtime. This builds the real import graph from the source, then reports
which files nothing depends on and which are never run directly.

A file is KEEP if any of:
  - something imports it (it is a library)
  - it is referenced by smoke_test.py or selfcheck.py (it is tested)
  - it is in the ENTRY set below (you run it yourself)

Everything else is a deletion candidate -- but read the reason before
acting, because "nothing imports it" and "you never run it" are different
claims and only the second is a reason to delete.

    python deps.py
"""
import ast
import os
import re
from collections import defaultdict

# the scripts you actually invoke, from RUN_GUIDE / weekly routine
ENTRY = {
    "state.py", "pipeline.py", "weekly.py", "weekly_robust.py",
    "prekick.py", "field.py", "division_opt.py", "prize_live.py",
    "ridge_cv.py", "fit_decay.py", "stability.py", "qc_grid.py",
    "demo.py", "selfcheck.py", "smoke_test.py", "deps.py",
}

py = sorted(f for f in os.listdir(".") if f.endswith(".py"))
local = {f[:-3] for f in py}

imports = defaultdict(set)      # file -> modules it imports
importers = defaultdict(set)    # module -> files importing it

for f in py:
    try:
        tree = ast.parse(open(f, encoding="utf-8").read())
    except SyntaxError as e:
        print(f"  !! {f} does not parse: {e}")
        continue
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                if a.name in local:
                    imports[f].add(a.name)
                    importers[a.name].add(f)
        elif isinstance(node, ast.ImportFrom):
            if node.module in local:
                imports[f].add(node.module)
                importers[node.module].add(f)

# which files do the test harnesses reference by name?
tested = set()
for h in ("smoke_test.py", "selfcheck.py"):
    if os.path.exists(h):
        txt = open(h, encoding="utf-8").read()
        for f in py:
            if f in txt:
                tested.add(f)

print("=" * 70)
print("IMPORT GRAPH")
print("=" * 70)
for f in py:
    dep = sorted(imports[f])
    by = sorted(importers[f[:-3]])
    print(f"\n{f}")
    print(f"  imports : {', '.join(dep) if dep else '-'}")
    print(f"  used by : {', '.join(by) if by else '-'}")

print("\n" + "=" * 70)
print("VERDICT")
print("=" * 70)
keep, drop = [], []
for f in py:
    why = []
    if importers[f[:-3]]:
        why.append(f"imported by {len(importers[f[:-3]])}")
    if f in ENTRY:
        why.append("you run it")
    if f in tested:
        why.append("in test harness")
    (keep if why else drop).append((f, why))

print(f"\nKEEP ({len(keep)}):")
for f, why in keep:
    print(f"  {f:<22} {'; '.join(why)}")

print(f"\nCANDIDATES TO DROP ({len(drop)}):")
for f, _ in drop:
    print(f"  {f}")
if not drop:
    print("  none — every file is imported, run, or tested")

# transitive: if we drop the candidates, what becomes orphaned?
if drop:
    dropped = {f for f, _ in drop}
    changed = True
    while changed:
        changed = False
        for f in py:
            if f in dropped:
                continue
            by = {x for x in importers[f[:-3]] if x not in dropped}
            if not by and f not in ENTRY and f not in tested:
                dropped.add(f)
                changed = True
    extra = dropped - {f for f, _ in drop}
    if extra:
        print(f"\n  ...and these become orphaned once those go:")
        for f in sorted(extra):
            print(f"    {f}")

print("\n" + "=" * 70)
print("Before deleting: anything in the test harness must be removed from")
print("smoke_test.py and selfcheck.py in the same commit, or the next run")
print("fails on a missing file.")
print("=" * 70)
