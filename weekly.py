"""
Weekly pick engine.

Run this every week. It takes your actual state — which teams you have
already burned, how many strikes you carry, what week it is — re-solves the
whole remaining season, and tells you what to pick now and what it costs to
deviate.

Three things this does that a static grid cannot:

1. IT RE-OPTIMISES. Your Week 9 pick depends on what is left, not on a plan
   drawn in August. Any strike you take changes the rest of the path.

2. IT EVALUATES PAIRS IN DOUBLE WEEKS. In weeks 5/7/10/12/15 the best two
   teams individually are frequently NOT the best pair, because taking both
   strips two teams from the same part of the schedule and leaves a hole
   later. It searches pairs directly.

3. ITS OBJECTIVE CHANGES AS YOU TAKE STRIKES. With 0 strikes you can absorb
   two losses, so it is worth accepting a 70% pick to protect a scarce team.
   With 2 strikes you are in pure single-elimination and the only thing that
   matters is not losing this week. The optimiser switches automatically:
   the budget k = 2 - strikes_taken feeds straight into the objective.

Usage:
    python3 weekly.py --week 1
    python3 weekly.py --week 6 --used LAC,SF,DET,CHI,NE,TB --strikes 1
    python3 weekly.py --week 10 --used ... --strikes 1 --objective depth
"""
import argparse
import itertools
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

DOUBLE_WEEKS = {5, 7, 10, 12, 15}
BIG = 1e6


# ----------------------------------------------------------------------
def load_grid(path):
    df = pd.read_csv(path).set_index("team")
    df.columns = [int(c) for c in df.columns]
    return df


def remaining_slots(week, horizon=18):
    slots = []
    for w in range(week, horizon + 1):
        slots.append(w)
        if w in DOUBLE_WEEKS:
            slots.append(w)
    return slots


def solve(df, slots, available, forced=None):
    """
    Hungarian over remaining slots. `forced` is {slot_position: team}.
    Returns (picks list aligned to slots, probs array) or (None, None).
    """
    teams = list(available)
    if len(teams) < len(slots):
        return None, None
    C = np.full((len(teams), len(slots)), BIG)
    for i, t in enumerate(teams):
        for j, w in enumerate(slots):
            p = df.loc[t, w]
            if pd.notna(p) and p > 0:
                C[i, j] = -np.log(p / 100.0)
    if forced:
        for j, t in forced.items():
            if t not in teams:
                return None, None
            keep = C[teams.index(t), j]
            C[:, j] = BIG
            C[teams.index(t), j] = keep
    r, c = linear_sum_assignment(C)
    if C[r, c].max() >= BIG:
        return None, None
    picks = [None] * len(slots)
    for a, b in zip(r, c):
        picks[b] = teams[a]
    probs = np.array([df.loc[t, w] / 100.0 for t, w in zip(picks, slots)])
    return picks, probs


# ----------------------------------------------------------------------
def p_survive(probs, budget):
    """P(at most `budget` more losses) — exact Poisson-binomial tail."""
    if budget < 0:
        return 0.0
    d = np.zeros(len(probs) + 1)
    d[0] = 1.0
    for p in probs:
        d[1:] = d[1:] * p + d[:-1] * (1 - p)
        d[0] *= p
    return float(d[:budget + 1].sum())


def expected_depth(probs, budget):
    """
    E[number of further picks made before elimination].
    Matters because in this pool most entrants are ranked on how deep they
    got, not on finishing.
    """
    state = np.zeros(budget + 1)
    state[0] = 1.0
    depth = 0.0
    for p in probs:
        alive = state.sum()
        depth += alive
        new = np.zeros(budget + 1)
        for k in range(budget + 1):
            new[k] += state[k] * p
            if k + 1 <= budget:
                new[k + 1] += state[k] * (1 - p)
        state = new
    return float(depth)


def alive_curve(probs, budget, slots):
    state = np.zeros(budget + 1); state[0] = 1.0
    out = []
    for p, w in zip(probs, slots):
        out.append((w, state.sum()))
        new = np.zeros(budget + 1)
        for k in range(budget + 1):
            new[k] += state[k] * p
            if k + 1 <= budget:
                new[k + 1] += state[k] * (1 - p)
        state = new
    return out


# ----------------------------------------------------------------------
def evaluate(df, week, used, strikes, objective="survive", horizon=18, top=10):
    budget = 2 - strikes
    slots = remaining_slots(week, horizon)
    avail = [t for t in df.index if t not in used and pd.notna(df.loc[t, week])]
    all_avail = [t for t in df.index if t not in used]

    n_this = 2 if week in DOUBLE_WEEKS else 1
    score = (lambda pr: p_survive(pr, budget)) if objective == "survive" \
        else (lambda pr: expected_depth(pr, budget))

    base_picks, base_probs = solve(df, slots, all_avail)
    if base_picks is None:
        raise SystemExit("No feasible path remains — check your used-team list.")

    # The Hungarian baseline maximises the PRODUCT of win probabilities.
    # Under --objective depth that is not the optimum, so scoring it as the
    # reference produced negative "cost" whenever some candidate beat it --
    # a cost below zero is nonsense and signals the wrong baseline. Take the
    # best score actually achieved across candidates instead.
    best = score(base_probs)

    rows = []
    if n_this == 1:
        for t in avail:
            p, pr = solve(df, slots, all_avail, forced={0: t})
            if p is None:
                continue
            v = score(pr)
            rows.append({"pick": t, "wp": df.loc[t, week],
                         "score": v, "cost": 100 * (best - v) / best,
                         "then": " ".join(p[1:7])})
    else:
        for a, b in itertools.combinations(avail, 2):
            p, pr = solve(df, slots, all_avail, forced={0: a, 1: b})
            if p is None:
                continue
            v = score(pr)
            rows.append({"pick": f"{a}+{b}",
                         "wp": (df.loc[a, week] + df.loc[b, week]) / 2,
                         "score": v, "cost": 100 * (best - v) / best,
                         "then": " ".join(p[2:8])})

    r = pd.DataFrame(rows).sort_values("score", ascending=False)
    # re-reference cost to the best candidate actually found
    if len(r):
        best = max(best, float(r.score.iloc[0]))
        r["cost"] = 100.0 * (best - r.score) / best
    return r.head(top), base_picks, base_probs, slots, budget, best


def protect_list(df, week, used, strikes, horizon=18, n=6):
    """Teams whose loss would hurt the remaining path most."""
    budget = 2 - strikes
    slots = remaining_slots(week, horizon)
    all_avail = [t for t in df.index if t not in used]
    _, pr = solve(df, slots, all_avail)
    base = p_survive(pr, budget)
    out = []
    for t in all_avail:
        p2, pr2 = solve(df, slots, [x for x in all_avail if x != t])
        if p2 is None:
            continue
        out.append((t, 100 * (base - p_survive(pr2, budget)) / base))
    return sorted(out, key=lambda x: -x[1])[:n]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", default="win_probs_2026.csv")
    ap.add_argument("--week", type=int, required=True)
    ap.add_argument("--used", default="", help="comma-separated teams already burned")
    ap.add_argument("--strikes", type=int, default=0)
    ap.add_argument("--objective", choices=["survive", "depth"], default="survive")
    ap.add_argument("--top", type=int, default=8)
    args = ap.parse_args()

    df = load_grid(args.grid)
    used = [t.strip().upper() for t in args.used.split(",") if t.strip()]
    if args.strikes >= 3:
        raise SystemExit("Three strikes — you're eliminated.")

    tbl, path, probs, slots, budget, best = evaluate(
        df, args.week, used, args.strikes, args.objective, top=args.top)

    # The plan, alive-curve and protect list must describe the pick we are
    # actually RECOMMENDING. `path` is the unconstrained Hungarian solution,
    # which maximises the product of win probabilities -- under
    # --objective depth that can be a different team, so the output showed
    # one recommendation above a different team's plan.
    rec = tbl.iloc[0]["pick"]
    all_avail = [t for t in df.index if t not in used]
    forced = {j: t for j, t in enumerate(rec.split("+"))}
    p2, pr2 = solve(df, slots, all_avail, forced=forced)
    if p2 is not None:
        path, probs = p2, pr2

    dbl = args.week in DOUBLE_WEEKS
    print(f"WK{args.week}{' DOUBLE' if dbl else ''} | {args.strikes} strike(s),"
          f" budget {budget} | {len(used)} burned, {len(slots)} slots left")

    if budget == 0:
        print("\n  !! ZERO MARGIN. One more loss ends your season.")
        print("     Objective has switched to pure single-elimination:")
        print("     take the highest win probability, ignore future value.\n")

    lab = "P(survive)" if args.objective == "survive" else "E[depth]"
    fmt = (lambda v: f"{v*100:6.2f}%") if args.objective == "survive" \
        else (lambda v: f"{v:6.2f}")
    print(f"  {'pick':<12}{'win%':>6}{lab:>12}{'cost':>8}   then")
    print("  " + "-" * 68)
    for _, r in tbl.iterrows():
        star = " *" if r["cost"] < 0.05 else "  "
        print(f"  {r['pick']:<12}{r['wp']:>5.0f}%{fmt(r['score']):>12}"
              f"{r['cost']:>7.1f}%{star} {r['then']}")

    print(f"\n  >>> {rec}")

    print("\n  plan:")
    cur = None; line = []
    for t, w in zip(path, slots):
        line.append(f"{'W'+str(w) if w != cur else '+'}:{t}({df.loc[t,w]:.0f})")
        cur = w
    for i in range(0, len(line), 5):
        print("    " + " ".join(line[i:i+5]))

    print(f"\n  P(<=2 strikes) {p_survive(probs, budget)*100:.2f}%   "
          f"E[depth] {expected_depth(probs, budget):.1f}/{len(slots)}")
    print("\n  alive:")
    ac = alive_curve(probs, budget, slots)
    for w, a in ac[::max(1, len(ac)//8)]:
        print(f"    W{w:<3} {a*100:5.1f}%  {'#'*int(a*40)}")

    pl = protect_list(df, args.week, used, args.strikes)
    print("\n  protect: " + "  ".join(f"{t} {c:.0f}%" for t, c in pl))


if __name__ == "__main__":
    main()
