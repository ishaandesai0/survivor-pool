"""
Robust weekly picks — optimise in EXPECTATION, not at the point estimate.

weekly.py solves the assignment once, on one grid. That finds the pick that
is best IF the grid is exactly right. It never is: everything past the
posted-line weeks is a projection with several points of error attached.

The failure this causes is visible in stability.py's output. On the model
grid, the Week 2 baseline pick appeared in only 4% of resampled futures --
and Week 2 has posted lines, so its own probabilities were identical in
every draw. The pick flipped because it was being assigned against noisy
FUTURE values. The single-solve path chases a knife-edge optimum that
mostly does not survive contact with uncertainty. That is Jensen's
inequality: argmax of the objective at the mean is not the argmax of the
mean objective.

This module fixes it directly. For each candidate pick:

    score(t) = E_grid[ P(survive | pick t, that grid) ]

averaged over D resampled grids. Two details make the estimate usable:

COMMON RANDOM GRIDS. Every candidate is scored on the SAME D grids, so the
comparison is paired and the variance between candidates collapses. Scoring
each candidate on its own independent draws would need an order of
magnitude more samples to resolve the same differences.

WIN RATE. Alongside the mean we report how often each candidate is the
outright best across draws. A pick with a slightly lower mean but a much
higher win rate is the more robust choice, and the two disagreeing is
itself informative.

Usage:
    python weekly_robust.py --week 2 --used LAC --strikes 0 \
        --grid win_probs_model.csv --draws 150 --posted-through 6
"""
import argparse
import itertools
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

from qc_grid import load
from weekly import remaining_slots, p_survive, DOUBLE_WEEKS
from stability import perturb

BIG = 1e6


def grid_to_cost(df, teams, slots):
    """Dense -log(p) cost matrix; BIG where a team can't be used."""
    C = np.full((len(teams), len(slots)), BIG)
    vals = df.reindex(teams)
    for j, w in enumerate(slots):
        col = vals[w].values.astype(float)
        ok = np.isfinite(col) & (col > 0)
        C[ok, j] = -np.log(col[ok] / 100.0)
    return C


def score_forced(C, forced_cols, budget, C_eval=None):
    """
    Solve with given columns pinned on grid C.

    Returns (score_on_C, score_on_C_eval). The second is the important one:
    it scores the CHOSEN PATH using the base grid instead of the noisy one
    it was optimised against.

    Without that, the numbers are inflated by the optimizer's curse -- every
    resampled grid contains random high cells, the assignment hunts them
    down, and the average of those maxima sits far above any real survival
    probability (E[max] > max[E]). Scoring the decision on a grid it did not
    get to peek at removes the bias.
    """
    C2 = C.copy()
    for j, i in forced_cols.items():
        keep = C2[i, j]
        C2[:, j] = BIG
        C2[i, j] = keep
    r, c = linear_sum_assignment(C2)
    if C2[r, c].max() >= BIG:
        return None, None
    order = np.argsort(c)
    rows, cols = r[order], c[order]
    own = p_survive(np.exp(-C2[rows, cols]), budget)
    if C_eval is None:
        return own, None
    ev = C_eval[rows, cols]
    if ev.max() >= BIG:
        return own, 0.0
    return own, p_survive(np.exp(-ev), budget)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", default="win_probs_model.csv")
    ap.add_argument("--week", type=int, required=True)
    ap.add_argument("--used", default="")
    ap.add_argument("--strikes", type=int, default=0)
    ap.add_argument("--draws", type=int, default=150)
    ap.add_argument("--posted-through", type=int, default=None)
    ap.add_argument("--base", type=float, default=3.717)
    ap.add_argument("--decay", type=float, default=0.355)
    ap.add_argument("--sigma0", type=float, default=11.16)
    ap.add_argument("--prune", type=int, default=25,
                    help="double weeks: how many top pairs to score robustly")
    ap.add_argument("--top", type=int, default=8)
    args = ap.parse_args()

    df = load(args.grid)
    used = [t.strip().upper() for t in args.used.split(",") if t.strip()]
    budget = 2 - args.strikes
    if budget < 0:
        raise SystemExit("Three strikes — eliminated.")

    teams = [t for t in df.index if t not in used]
    idx = {t: i for i, t in enumerate(teams)}
    slots = remaining_slots(args.week)
    n_this = 2 if args.week in DOUBLE_WEEKS else 1
    avail = [t for t in teams if pd.notna(df.loc[t, args.week])]

    # ---- point estimate (what weekly.py does) ------------------------
    C0 = grid_to_cost(df, teams, slots)
    point = {}
    if n_this == 1:
        cands = [(t,) for t in avail]
    else:
        cands = list(itertools.combinations(avail, 2))
    for cd in cands:
        f = {j: idx[t] for j, t in enumerate(cd)}
        s, _ = score_forced(C0, f, budget)
        if s is not None:
            point[cd] = s
    if not point:
        raise SystemExit("No feasible path — check --used.")

    # prune double weeks to the best pairs before the expensive step
    ranked = sorted(point, key=lambda k: -point[k])
    short = ranked[:args.prune] if n_this == 2 else ranked

    # ---- common random grids -----------------------------------------
    rng = np.random.default_rng(101)
    print(f"resampling {args.draws} grids...", flush=True)
    costs = []
    for _ in range(args.draws):
        g = perturb(df, args.week, args.sigma0, args.base, args.decay,
                    args.posted_through, rng)
        costs.append(grid_to_cost(g, teams, slots))

    # ---- score every candidate on every grid (paired) -----------------
    scores = {cd: np.zeros(args.draws) for cd in short}     # unbiased
    for d, C in enumerate(costs):
        for cd in short:
            f = {j: idx[t] for j, t in enumerate(cd)}
            _, ev = score_forced(C, f, budget, C_eval=C0)
            scores[cd][d] = 0.0 if ev is None else ev

    mat = np.vstack([scores[cd] for cd in short])
    winner = mat.argmax(axis=0)
    winrate = np.bincount(winner, minlength=len(short)) / args.draws

    rows = []
    for k, cd in enumerate(short):
        rows.append({"pick": "+".join(cd),
                     "point": 100 * point[cd],
                     "robust": 100 * mat[k].mean(),
                     "se": 100 * mat[k].std() / np.sqrt(args.draws),
                     "winrate": 100 * winrate[k]})
    r = pd.DataFrame(rows)
    r["pt_rank"] = r["point"].rank(ascending=False).astype(int)
    r["rb_rank"] = r["robust"].rank(ascending=False).astype(int)
    r = r.sort_values("robust", ascending=False)

    print("\n" + "=" * 74)
    print(f"ROBUST WEEK {args.week}"
          f"{'  (DOUBLE)' if n_this == 2 else ''}   strikes {args.strikes}/3   "
          f"budget {budget}   {args.draws} grids")
    print("=" * 74)
    print("  point  = P(survive) if the grid is exactly right")
    print("  robust = E[P(survive)] when the path is chosen under grid noise")
    print("           and scored on the base grid -- always <= point, and the")
    print("           gap is what uncertainty actually costs you")
    print()
    print(f"  {'pick':<12}{'point':>8}{'robust':>9}{'+/-':>7}{'win%':>7}"
          f"{'rank':>10}")
    print("  " + "-" * 70)
    for _, x in r.head(args.top).iterrows():
        move = ""
        if x.pt_rank != x.rb_rank:
            move = f"  {int(x.pt_rank)}->{int(x.rb_rank)}"
        print(f"  {x['pick']:<12}{x['point']:>7.2f}%{x['robust']:>8.2f}%"
              f"{x['se']:>7.2f}{x['winrate']:>6.0f}%{move:>10}")

    best_pt = r.loc[r.pt_rank.idxmin(), "pick"] if False else \
        max(point, key=lambda k: point[k])
    best_rb = r.iloc[0]["pick"]
    print("\n  point-estimate pick : " + "+".join(best_pt))
    print("  robust pick         : " + best_rb)

    # Decide whether the gap is real BEFORE recommending anything. Ranking
    # by mean and then admitting the gap is noise is contradictory advice.
    top2 = r.head(2)
    tie = False
    if len(top2) == 2:
        gap = top2.iloc[0]["robust"] - top2.iloc[1]["robust"]
        pooled = np.sqrt(top2.iloc[0]["se"]**2 + top2.iloc[1]["se"]**2)
        wr_gap = abs(top2.iloc[0]["winrate"] - top2.iloc[1]["winrate"])
        tie = gap < 2 * pooled and wr_gap < 10

    if tie:
        print(f"\n  TOO CLOSE TO CALL: {top2.iloc[0]['pick']} and "
              f"{top2.iloc[1]['pick']} differ by {gap:.2f}pp "
              f"(+/-{pooled:.2f}), win rates {top2.iloc[0]['winrate']:.0f}% "
              f"vs {top2.iloc[1]['winrate']:.0f}%.")
        print("  The model cannot separate them. Break the tie on something")
        print("  it cannot see: late injury news, or fading the pick you")
        print("  expect the rest of your pool to be on.")
    elif "+".join(best_pt) != best_rb:
        print("\n  THEY DISAGREE, and the gap is outside the noise band.")
        print("  Take the robust pick: it wins across more plausible")
        print("  versions of the season, not just this one grid.")
    else:
        print("\n  They agree — the point estimate is not on a knife edge here.")


if __name__ == "__main__":
    main()
