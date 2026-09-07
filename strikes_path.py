"""
Path optimisation for a STRIKES pool.

In a classic survivor pool you maximise the product of win probabilities,
which the Hungarian algorithm solves exactly.  With strikes, the objective
changes: you want to maximise P(at most k losses), which is a
Poisson-binomial tail, not a linear function of the picks.  So the
assignment problem is no longer exactly solvable.

Approach: seed with the Hungarian solution (a very good starting point,
since both objectives reward high probabilities), then hill-climb on the
TRUE objective with two move types:
  - swap the teams assigned to two weeks
  - substitute an unused team into a week

The difference matters. Maximising the product is risk-neutral in log space
and will happily accept a 67% week to protect a distant 85%. Maximising
P(<=2) is convex near the tail: it cares much more about avoiding the weak
weeks that generate the third strike.
"""
import numpy as np
import pandas as pd
from qc_grid import load, WEEKS
from survivor import solve_path


def poisson_binomial(probs):
    """Full distribution of the number of LOSSES. probs = win probabilities."""
    d = np.zeros(len(probs) + 1)
    d[0] = 1.0
    for p in probs:
        d[1:] = d[1:] * p + d[:-1] * (1 - p)
        d[0] *= p
    return d


def p_at_most(probs, k):
    return poisson_binomial(probs)[:k + 1].sum()


def optimise_strikes(df, k=2, horizon=18, seed_path=None, iters=400):
    """Hill-climb the assignment to maximise P(<= k strikes)."""
    teams = list(df.index)
    if seed_path is None:
        seed_path, _ = solve_path(df, horizon)
    assign = {w: seed_path[w][0] for w in range(1, horizon + 1)}

    def prob_of(a):
        return np.array([df.loc[a[w], w] / 100.0 for w in range(1, horizon + 1)])

    best = p_at_most(prob_of(assign), k)
    improved = True
    while improved:
        improved = False
        # move 1: swap two weeks' teams
        for w1 in range(1, horizon + 1):
            for w2 in range(w1 + 1, horizon + 1):
                t1, t2 = assign[w1], assign[w2]
                if pd.isna(df.loc[t1, w2]) or pd.isna(df.loc[t2, w1]):
                    continue
                assign[w1], assign[w2] = t2, t1
                v = p_at_most(prob_of(assign), k)
                if v > best + 1e-12:
                    best, improved = v, True
                else:
                    assign[w1], assign[w2] = t1, t2
        # move 2: substitute an unused team
        used = set(assign.values())
        for w in range(1, horizon + 1):
            cur = assign[w]
            for t in teams:
                if t in used or pd.isna(df.loc[t, w]):
                    continue
                assign[w] = t
                v = p_at_most(prob_of(assign), k)
                if v > best + 1e-12:
                    best, improved = v, True
                    used.discard(cur); used.add(t)
                    cur = t
                else:
                    assign[w] = cur
    return assign, best


def report(df, assign, label, k=2, horizon=18):
    probs = np.array([df.loc[assign[w], w] / 100.0 for w in range(1, horizon + 1)])
    d = poisson_binomial(probs)
    print(f"\n{label}")
    print("-" * len(label))
    print("  " + " ".join(f"{assign[w]:>3}" for w in range(1, horizon + 1)))
    print("  " + " ".join(f"{probs[w-1]*100:3.0f}" for w in range(1, horizon + 1)))
    print(f"  expected strikes = {(1-probs).sum():.2f}")
    print(f"  P(0) = {d[0]*100:5.2f}%   P(<=1) = {d[:2].sum()*100:5.2f}%   "
          f"P(<=2) = {d[:3].sum()*100:5.2f}%   P(<=3) = {d[:4].sum()*100:5.2f}%")
    return d


if __name__ == "__main__":
    df = load()
    print("=" * 78)
    print("TWO-STRIKE PATH OPTIMISATION")
    print("=" * 78)

    seed, _ = solve_path(df, 18)
    d_prod = report(df, {w: seed[w][0] for w in seed},
                    "A. Maximise P(perfect season)  [Hungarian, previous model]")

    a2, v2 = optimise_strikes(df, k=2)
    d2 = report(df, a2, "B. Maximise P(<= 2 strikes)  [this pool's actual rule]")

    a3, v3 = optimise_strikes(df, k=3)
    report(df, a3, "C. Maximise P(<= 3 strikes)  [if the bar drifts]")

    print("\n" + "=" * 78)
    print(f"Optimising for the right objective lifts P(<=2) from "
          f"{d_prod[:3].sum()*100:.2f}% to {d2[:3].sum()*100:.2f}% "
          f"(+{(d2[:3].sum()/d_prod[:3].sum()-1)*100:.1f}% relative).")
    same = sum(1 for w in range(1, 19) if seed[w][0] == a2[w])
    print(f"The two paths share {same}/18 picks — the objective change is real but local.")
    print("=" * 78)
