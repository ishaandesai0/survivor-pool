"""
How does 1st place actually get decided in this pool?

With survival around 2%, most seasons end with NOBODY completing 23 picks
under the strike limit. That means the money is awarded on elimination
depth, and the important question stops being "will I survive" and becomes
"how many people will be tied with me at the top, splitting the money".

This matters because the organiser's rules were written for a format where
somebody finishes. The 23-pick version is new this year, so nobody has seen
what happens when the intended goal is met by zero entrants.

Run:
    python tie_analysis.py --grid win_probs_model.csv --sims 600
"""
import argparse
import numpy as np

from qc_grid import load
from slots_model import slot_list
from prize23 import prep, season

OVERALL = [1500, 800, 600, 400, 200]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", default="win_probs_model.csv")
    ap.add_argument("--sims", type=int, default=600)
    ap.add_argument("--entrants", type=int, default=200)
    args = ap.parse_args()

    df = load(args.grid)
    slots = slot_list()
    teams, idx, P, OPP = prep(df, slots)
    rng = np.random.default_rng(3)
    n, S, NE = len(teams), len(slots), args.entrants

    best_depth, tie_top, nsurv, top5_cluster = [], [], [], []

    for _ in range(args.sims):
        won = season(P, OPP, slots, rng)
        chalk = rng.uniform(4, 18, size=NE)
        used = np.zeros((NE, n), bool)
        st = np.zeros(NE, int)
        dep = np.full(NE, S)
        live = np.ones(NE, bool)

        for j in range(S):
            act = np.where(live)[0]
            if not act.size:
                break
            lg = chalk[act][:, None] * P[None, :, j]
            lg = np.where(used[act] | (P[:, j][None, :] <= 0), -np.inf, lg)
            lg -= lg.max(axis=1, keepdims=True)
            pr = np.exp(lg); pr /= pr.sum(axis=1, keepdims=True)
            ch = (rng.random(act.size)[:, None] > pr.cumsum(axis=1)).sum(axis=1).clip(0, n-1)
            used[act, ch] = True
            lost = ~won[ch, j]
            st[act[lost]] += 1
            d = act[lost][st[act[lost]] >= 3]
            live[d] = False
            dep[d] = j + 1

        key = -dep.astype(float) * 100 + st
        b = key.min()
        best_depth.append(dep[key == b].max())
        tie_top.append(int((key == b).sum()))
        nsurv.append(int((st <= 2).sum()))
        # how many entrants share the top 5 places' money
        uk = np.sort(np.unique(key))
        cum, grp = 0, 0
        for k in uk:
            c = int((key == k).sum())
            cum += c; grp += 1
            if cum >= 5:
                break
        top5_cluster.append(cum)

    bd, tt = np.array(best_depth), np.array(tie_top)
    ns, t5 = np.array(nsurv), np.array(top5_cluster)

    print("=" * 66)
    print(f"HOW 1st PLACE RESOLVES — {args.entrants} entrants, {args.sims} seasons")
    print("=" * 66)
    print(f"  entrants finishing all 23 with <=2 strikes : mean {ns.mean():.2f}")
    print(f"  seasons where NOBODY finishes              : {100*np.mean(ns == 0):.0f}%")
    print(f"\n  deepest slot anyone reaches   : mean {bd.mean():.1f}/23  median {np.median(bd):.0f}")
    print(f"  entrants TIED at that top spot: mean {tt.mean():.1f}  median {np.median(tt):.0f}")
    print(f"  entrants sharing top-5 money  : mean {t5.mean():.1f}  median {np.median(t5):.0f}")

    print("\n  tie-cluster size at 1st place:")
    for lo, hi in [(1, 1), (2, 3), (4, 6), (7, 12), (13, 10**6)]:
        pct = 100 * np.mean((tt >= lo) & (tt <= hi))
        lab = f"{lo}-{hi}" if hi < 10**6 else f"{lo}+"
        print(f"    {lab:>6} tied : {pct:5.1f}%  {'#' * int(pct/2)}")

    pot = sum(OVERALL)
    med = max(int(np.median(tt)), 1)
    print(f"\n  If N tie for 1st they split places 1..N.")
    print(f"  At the median ({med} tied): "
          f"${(pot if med >= 5 else sum(OVERALL[:med]))/med:,.0f} each.")
    print(f"  Alone at the top: $1,500.")
    print("=" * 66)
    print("\n  ASK THE ORGANISER: with the 23-pick format, most seasons end")
    print("  with nobody meeting the <=2 strike goal. How are 1st-5th ranked")
    print("  among entrants eliminated in the SAME week? That is now the")
    print("  normal outcome, not an edge case, and it decides the money.")


if __name__ == "__main__":
    main()
