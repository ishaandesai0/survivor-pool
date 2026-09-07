"""
23-pick survivor: five weeks (5, 7, 10, 12, 15) require TWO teams.

This is still an assignment problem, just with 23 SLOTS instead of 18 weeks.
Weeks 5/7/10/12/15 contribute two slots each; every other week contributes
one. Teams are rows, slots are columns, and the Hungarian algorithm assigns
23 distinct teams to 23 slots maximising the sum of log win probabilities.
Uniqueness of rows automatically enforces "no team twice", including within
a double week.

One trap worth naming: in a double week you must not take BOTH sides of the
same game. That guarantees exactly one strike with zero variance. The
log-objective rejects it on its own — log(p) + log(1-p) is dismal — but the
code checks for it explicitly, because relying on an objective to
incidentally avoid a hard constraint is how you get burned when the numbers
shift.

Burning 23 of 32 teams is the real difficulty. You cannot fill 23 slots
with good teams, so weak picks are forced, and expected strikes climb well
past the 3-strike limit.
"""
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

from qc_grid import load, recover_matchups

DOUBLE_WEEKS = [5, 7, 10, 12, 15]
BIG = 1e6


def slot_list(horizon=18):
    slots = []
    for w in range(1, horizon + 1):
        slots.append(w)
        if w in DOUBLE_WEEKS:
            slots.append(w)
    return slots


def solve_slots(df, banned=(), forced=None, horizon=18):
    """forced: dict {slot_index: team}."""
    teams = list(df.index)
    slots = slot_list(horizon)
    C = np.full((len(teams), len(slots)), BIG)
    for i, t in enumerate(teams):
        if t in banned:
            continue
        for j, w in enumerate(slots):
            p = df.loc[t, w]
            if pd.notna(p) and p > 0:
                C[i, j] = -np.log(p / 100.0)
    if forced:
        for j, t in forced.items():
            keep = C[teams.index(t), j]
            C[:, j] = BIG
            C[teams.index(t), j] = keep
    rows, cols = linear_sum_assignment(C)
    if C[rows, cols].max() >= BIG:
        return None, None
    picks = [None] * len(slots)
    for r, c in zip(rows, cols):
        picks[c] = teams[r]
    return picks, slots


def poisson_binomial(probs):
    d = np.zeros(len(probs) + 1); d[0] = 1.0
    for p in probs:
        d[1:] = d[1:] * p + d[:-1] * (1 - p)
        d[0] *= p
    return d


def check_both_sides(df, picks, slots):
    """Flag any double week where we took both teams in one game."""
    bad = []
    for w in DOUBLE_WEEKS:
        js = [j for j, s in enumerate(slots) if s == w]
        a, b = picks[js[0]], picks[js[1]]
        pairs, _ = recover_matchups(df, w)
        for x, y, *_ in pairs:
            if {a, b} == {x, y}:
                bad.append((w, a, b))
    return bad


def report(df, picks, slots, label):
    probs = np.array([df.loc[t, w] / 100.0 for t, w in zip(picks, slots)])
    d = poisson_binomial(probs)
    print(f"\n{label}")
    print("-" * len(label))
    cur = None
    line = []
    for t, w in zip(picks, slots):
        tag = f"W{w}" if w != cur else "  +"
        cur = w
        line.append(f"{tag}:{t}({df.loc[t, w]:.0f})")
    for i in range(0, len(line), 6):
        print("   " + "  ".join(line[i:i+6]))
    print(f"   picks = {len(picks)}   expected strikes = {(1-probs).sum():.2f}")
    print(f"   P(0)={d[0]*100:5.2f}%  P(<=1)={d[:2].sum()*100:5.2f}%  "
          f"P(<=2)={d[:3].sum()*100:5.2f}%   <-- survive")
    print(f"   P(<=3)={d[:4].sum()*100:5.2f}%  P(<=4)={d[:5].sum()*100:5.2f}%")
    bad = check_both_sides(df, picks, slots)
    print(f"   both-sides-of-a-game violations: {bad if bad else 'none'}")
    return d


if __name__ == "__main__":
    df = load()
    print("=" * 76)
    print("23-PICK SURVIVOR — five double weeks (5, 7, 10, 12, 15)")
    print("=" * 76)

    picks, slots = solve_slots(df)
    d23 = report(df, picks, slots, "OPTIMAL 23-pick path")

    # contrast with the 18-pick game we modelled before
    from survivor import solve_path
    p18, _ = solve_path(df, 18)
    pr18 = np.array([p / 100 for w, (t, p) in sorted(p18.items())])
    d18 = poisson_binomial(pr18)
    print("\n" + "=" * 76)
    print("COST OF THE DOUBLE WEEKS")
    print("=" * 76)
    print(f"  18-pick game: expected strikes {(1-pr18).sum():.2f},  "
          f"P(survive, <=2) = {d18[:3].sum()*100:.2f}%")
    print(f"  23-pick game: expected strikes "
          f"{sum(1-df.loc[t,w]/100 for t,w in zip(picks,slots)):.2f},  "
          f"P(survive, <=2) = {d23[:3].sum()*100:.2f}%")
    print(f"  The five extra picks cut survival by "
          f"{(1 - d23[:3].sum()/d18[:3].sum())*100:.0f}%.")

    print("\n" + "=" * 76)
    print("WHICH TEAMS GET FORCED INTO THE PLAN")
    print("=" * 76)
    weak = sorted([(df.loc[t, w], t, w) for t, w in zip(picks, slots)])[:8]
    for p, t, w in weak:
        print(f"   W{w:<3} {t:<4} {p:.0f}%")
    print("\n   These are the picks that decide your season. With only 18")
    print("   picks you could avoid every one of them.")
