"""
Expected dollars in the ACTUAL pool: 23 picks, 3 strikes, 200 entrants, $5,000.

The 23-pick structure changes the character of the contest completely.
With P(surviving on even the optimal path) around 7%, most seasons end with
few or no clean finishers, so the prize money is frequently decided by WHO
LASTS LONGEST rather than by who finishes.

That makes the ranking key (slots completed, then strikes) genuinely
two-dimensional, and it means early picks matter more than late ones: you
cannot benefit from a great Week 17 team if you were eliminated in Week 8.
That asymmetry does not exist in a pure P(<=2 strikes) objective, where the
picks commute.

Payout: $1500/800/600/400/200 overall, plus $150 per division (10 divisions
of 20). Ties split the money for the places they occupy, per the organiser.
"""
import argparse
import numpy as np
import pandas as pd

from qc_grid import load, recover_matchups
from slots_model import slot_list, solve_slots, DOUBLE_WEEKS

OVERALL = [1500, 800, 600, 400, 200]
DIV_PRIZE, N_DIV = 150, 10


def prep(df, slots):
    teams = list(df.index)
    idx = {t: i for i, t in enumerate(teams)}
    n = len(teams)
    S = len(slots)
    P = np.zeros((n, S))
    OPP = np.full((n, S), -1, dtype=int)
    cache = {}
    for j, w in enumerate(slots):
        if w not in cache:
            cache[w] = recover_matchups(df, w)[0]
        for a, b, pa, pb, _ in cache[w]:
            s = pa + pb                      # normalise: the two sides must sum to 1
            P[idx[a], j], P[idx[b], j] = pa/s, pb/s
            OPP[idx[a], j], OPP[idx[b], j] = idx[b], idx[a]
    return teams, idx, P, OPP


def season(P, OPP, slots, rng):
    """Draw one coin per GAME per week, shared by every slot in that week."""
    n, S = P.shape
    won = np.zeros((n, S), dtype=bool)
    byweek = {}
    for j, w in enumerate(slots):
        if w not in byweek:
            res = np.zeros(n, dtype=bool)
            done = np.zeros(n, dtype=bool)
            for t in np.where(OPP[:, j] >= 0)[0]:
                if done[t]:
                    continue
                o = OPP[t, j]
                tw = rng.random() < P[t, j]
                res[t], res[o] = tw, not tw
                done[t] = done[o] = True
            byweek[w] = res
        won[:, j] = byweek[w]
    return won


def simulate(df, my_picks, slots, n_sims=800, n_entrants=200, seed=5):
    teams, idx, P, OPP = prep(df, slots)
    n, S = P.shape
    rng = np.random.default_rng(seed)
    nr = n_entrants - 1
    mine = [idx[t] for t in my_picks]

    pay_total = 0.0
    top5 = div = 0
    my_depth, my_str = [], []
    n_survivors = []

    for _ in range(n_sims):
        won = season(P, OPP, slots, rng)
        chalk = rng.uniform(4.0, 18.0, size=nr)
        used = np.zeros((nr, n), dtype=bool)
        st = np.zeros(nr, dtype=int)
        depth = np.full(nr, S, dtype=int)
        live = np.ones(nr, dtype=bool)
        mstr, mdepth = 0, S

        for j in range(S):
            act = np.where(live)[0]
            if act.size:
                lg = chalk[act][:, None] * P[None, :, j]
                lg = np.where(used[act] | (P[:, j][None, :] <= 0), -np.inf, lg)
                lg -= lg.max(axis=1, keepdims=True)
                pr = np.exp(lg); pr /= pr.sum(axis=1, keepdims=True)
                ch = (rng.random(act.size)[:, None] > pr.cumsum(axis=1)).sum(axis=1).clip(0, n-1)
                used[act, ch] = True
                lost = ~won[ch, j]
                st[act[lost]] += 1
                dead = act[lost][st[act[lost]] >= 3]
                live[dead] = False
                depth[dead] = j + 1
            if mstr < 3:
                if not won[mine[j], j]:
                    mstr += 1
                    if mstr >= 3:
                        mdepth = j + 1

        A_st = np.concatenate([st, [mstr]])
        A_dp = np.concatenate([depth, [mdepth]])
        n_survivors.append((A_st <= 2).sum())

        key = -A_dp.astype(float) * 100 + A_st
        order = np.argsort(key, kind="stable")
        pay = np.zeros(n_entrants)
        i = place = 0
        while i < n_entrants and place < 5:
            j2 = i
            while j2 + 1 < n_entrants and key[order[j2+1]] == key[order[i]]:
                j2 += 1
            grp = order[i:j2+1]
            span = range(place, min(place + len(grp), 5))
            pot = sum(OVERALL[s] for s in span)
            if pot:
                pay[grp] += pot / len(grp)
            place += len(grp); i = j2 + 1
        before = pay[-1]
        divs = rng.permutation(n_entrants) % N_DIV
        for d in range(N_DIV):
            m = np.where(divs == d)[0]
            w_ = m[key[m] == key[m].min()]
            pay[w_] += DIV_PRIZE / len(w_)

        pay_total += pay[-1]
        top5 += before > 0
        div += (pay[-1] - before) > 0
        my_depth.append(mdepth); my_str.append(mstr)

    return {"ev": pay_total/n_sims, "top5": 100*top5/n_sims, "div": 100*div/n_sims,
            "survive": 100*np.mean(np.array(my_str) <= 2),
            "mean_depth": np.mean(my_depth),
            "field_survivors": np.mean(n_survivors)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sims", type=int, default=700)
    ap.add_argument("--grid", default="win_probs_2026.csv",
                    help="probability grid; use win_probs_model.csv "
                         "once your own model is built")
    args = ap.parse_args()
    df = load(args.grid)
    slots = slot_list()

    cands = {}
    picks, _ = solve_slots(df)
    cands["Optimal 23-pick"] = picks

    # safety-first: heavily weight the early slots, since being alive later
    # is a precondition for any late pick mattering at all
    for lam in (0.06, 0.12):
        w_df = df.copy()
        picks_w, _ = solve_slots(df)
        # re-solve with early slots up-weighted via a tilted objective
        import numpy as np2
        from scipy.optimize import linear_sum_assignment
        teams = list(df.index)
        C = np2.full((len(teams), len(slots)), 1e6)
        for i, t in enumerate(teams):
            for j, w in enumerate(slots):
                p = df.loc[t, w]
                if pd.notna(p) and p > 0:
                    C[i, j] = -np2.log(p/100.0) * np2.exp(-lam * j)
        r, c = linear_sum_assignment(C)
        pk = [None]*len(slots)
        for a, b in zip(r, c):
            pk[b] = teams[a]
        cands[f"Front-loaded (lam={lam})"] = pk

    print("=" * 74)
    print(f"ACTUAL POOL — 23 picks, 3 strikes, 200 entrants, $5,000 ({args.sims} seasons)")
    print("=" * 74)
    print(f"{'strategy':<26}{'EV':>9}{'vs $25':>8}{'survive':>9}{'top-5':>8}{'div':>7}")
    print("-" * 74)
    out = {}
    for k, v in cands.items():
        r = simulate(df, v, slots, n_sims=args.sims)
        out[k] = r
        print(f"{k:<26}{'$'+format(r['ev'],'.2f'):>9}{r['ev']/25:>7.2f}x"
              f"{r['survive']:>8.1f}%{r['top5']:>7.1f}%{r['div']:>6.1f}%")
    print("-" * 74)
    r = out["Optimal 23-pick"]
    print(f"entrants (of 200) finishing with <=2 strikes: {r['field_survivors']:.1f} on average")
    print(f"my average elimination point: slot {r['mean_depth']:.1f} of 23")


if __name__ == "__main__":
    main()
