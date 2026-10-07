"""
Division-aware pick optimisation.

The $150 division prize is a 20-person race, not a 200-person one, and it is
by far your likeliest money. That makes it a different optimisation from
everything else here:

  weekly.py       maximises P(you survive)
  prize_live.py   maximises expected dollars against the whole field
  THIS            maximises P(you finish first in your division)

Those diverge in a specific way. Surviving is worth nothing if the people
ahead of you also survive -- you stay behind them forever. You need to
survive WHILE they fail, so a pick's value depends on what your RIVALS hold,
not only on its win probability.

Two mechanisms matter:

  CORRELATION. If a leader is likely to take the same team, that week cannot
  separate you -- you both survive or both strike.

  INVENTORY. Teams a leader has burned are teams they must replace with
  something worse later.

This is NOT a licence to take a bad pick. With one strike and no margin, a
second strike ends the season in every scenario. The tool reports the
division-win probability alongside raw win probability so the cost of a
contrarian choice is explicit -- and usually it is not worth paying.

DOUBLE WEEKS (5, 7, 10, 12, 15) take two teams. Candidates are pairs, and
win% is the JOINT probability that both hold -- which is the number that
actually matters: a 79/78 double week is a 62% week.

    python division_opt.py --week 5
    python division_opt.py --week 5 --chalk 10 --sims 4000
"""
import argparse
import itertools
import math
from collections import Counter

import numpy as np
import pandas as pd

DOUBLE_WEEKS = {5, 7, 10, 12, 15}


def load_grid(path):
    df = pd.read_csv(path).set_index("team")
    df.columns = [int(c) for c in df.columns]
    return df


def slots_from(week, horizon=18):
    s = []
    for w in range(week, horizon + 1):
        s.append(w)
        if w in DOUBLE_WEEKS:
            s.append(w)
    return s


def field_pick(avail_p, chalk):
    """Rival choice model: softmax on win probability."""
    u = chalk * avail_p
    u -= u.max()
    q = np.exp(u)
    return q / q.sum()


def simulate(grid, members, me_idx, week, my_picks, n_sims, chalk, seed=0):
    """
    members : list of {name, burned:set, strikes:int}
    my_picks: TUPLE of teams for this week -- one normally, two in a double
              week. Both slots are pinned; the rest of your path is greedy
              on win probability.
    Returns (P win outright, P tie, mean finish rank).
    """
    from qc_grid import recover_matchups
    rng = np.random.default_rng(seed)
    teams = list(grid.index)
    ti = {t: i for i, t in enumerate(teams)}
    n_t = len(teams)
    slots = slots_from(week)
    S = len(slots)
    N = len(members)

    P = np.full((n_t, S), np.nan)
    OPP = np.full((n_t, S), -1, int)
    cache = {}
    for j, w in enumerate(slots):
        for t in teams:
            v = grid[w].get(t)
            if pd.notna(v):
                P[ti[t], j] = v / 100.0
        if w not in cache:
            cache[w] = recover_matchups(grid, w)[0]
        for a, b, *_ in cache[w]:
            OPP[ti[a], j], OPP[ti[b], j] = ti[b], ti[a]

    base_used = np.zeros((N, n_t), bool)
    base_st = np.zeros(N, int)
    for i, m in enumerate(members):
        for t in m["burned"]:
            if t in ti:
                base_used[i, ti[t]] = True
        base_st[i] = m["strikes"]

    wins = ties = 0
    ranks = []
    for _ in range(n_sims):
        used = base_used.copy()
        st = base_st.copy()
        alive = st < 3
        # already-eliminated entrants must rank LAST, not as full survivors
        depth = np.where(alive, S, 0).astype(int)

        for j in range(S):
            ok = np.isfinite(P[:, j])
            won = np.zeros(n_t, bool)
            done = np.zeros(n_t, bool)
            for a in range(n_t):
                if done[a] or not ok[a]:
                    continue
                b = OPP[a, j]
                u = rng.random()
                if b < 0:
                    won[a] = u < P[a, j]
                    done[a] = True
                else:
                    aw = u < P[a, j]
                    won[a], won[b] = aw, not aw
                    done[a] = done[b] = True

            for i in np.where(alive)[0]:
                cand = np.where(ok & ~used[i])[0]
                if cand.size == 0:
                    alive[i] = False
                    depth[i] = j
                    continue
                if i == me_idx and j < len(my_picks):
                    pick = ti[my_picks[j]]
                elif i == me_idx:
                    pick = cand[int(np.nanargmax(P[cand, j]))]
                else:
                    pick = rng.choice(cand, p=field_pick(P[cand, j], chalk))
                used[i, pick] = True
                if not won[pick]:
                    st[i] += 1
                    if st[i] >= 3:
                        alive[i] = False
                        depth[i] = j + 1

        key = -depth.astype(float) * 100 + st
        best = key.min()
        top = np.where(key == best)[0]
        if me_idx in top:
            if len(top) == 1:
                wins += 1
            else:
                ties += 1
        ranks.append(int((key < key[me_idx]).sum()) + 1)

    return wins / n_sims, ties / n_sims, float(np.mean(ranks))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", default="win_probs_model.csv")
    ap.add_argument("--roster", default="division_roster.csv")
    ap.add_argument("--me", default="Ishaan")
    ap.add_argument("--week", type=int, required=True)
    ap.add_argument("--sims", type=int, default=2500,
                    help="400 cannot resolve the ~1pp differences this tool "
                         "looks for")
    ap.add_argument("--chalk", type=float, default=15.0,
                    help="how hard rivals chase the favourite")
    ap.add_argument("--top", type=int, default=8)
    args = ap.parse_args()

    grid = load_grid(args.grid)
    r = pd.read_csv(args.roster)
    wkcols = [c for c in r.columns if c.startswith("wk")]

    members = []
    for _, row in r.iterrows():
        burned = {row[c] for c in wkcols if pd.notna(row[c])}
        strikes = int(row["strikes"]) if "strikes" in r.columns else 0
        members.append({"name": row["name"], "burned": burned,
                        "strikes": strikes})
    if "strikes" not in r.columns:
        print("  NOTE: roster has no 'strikes' column; assuming 0 for all.\n")

    me = next(i for i, m in enumerate(members)
              if args.me.lower() in m["name"].lower())
    mine = members[me]
    ahead = [m for m in members if m["strikes"] < mine["strikes"]]

    dbl = args.week in DOUBLE_WEEKS
    print(f"DIVISION wk{args.week}{' DOUBLE' if dbl else ''} | "
          f"{len(members)} members, {len(ahead)} ahead | "
          f"you {mine['strikes']} strike(s), "
          f"burned {','.join(sorted(mine['burned']))}")

    lead_burn = Counter()
    for m in ahead:
        for t in m["burned"]:
            lead_burn[t] += 1

    avail = [t for t in grid.index
             if t not in mine["burned"] and pd.notna(grid.loc[t, args.week])]
    avail.sort(key=lambda t: -grid.loc[t, args.week])

    if not dbl:
        cands = [(t,) for t in avail[:args.top]]
    else:
        # Scoring all ~400 pairs means 400 full division simulations, so
        # shortlist on joint win probability first. The best pair is often
        # NOT the two best singles -- taking both can strip two teams from
        # the same stretch of the remaining schedule.
        pairs = list(itertools.combinations(avail, 2))
        pairs.sort(key=lambda p: -(grid.loc[p[0], args.week]
                                   * grid.loc[p[1], args.week]))
        cands = pairs[:max(args.top, 8)]

    rows = []
    for cd in cands:
        w, ti_, rk = simulate(grid, members, me, args.week, cd,
                              args.sims, args.chalk)
        joint = 1.0
        for t in cd:
            joint *= grid.loc[t, args.week] / 100.0
        rows.append({"pick": "+".join(cd), "win%": 100 * joint,
                     "div_win": 100 * w, "div_tie": 100 * ti_,
                     "rank": rk,
                     "leaders_burned": sum(lead_burn.get(t, 0) for t in cd)})
    df = pd.DataFrame(rows).sort_values("div_win", ascending=False)

    wlab = "joint%" if dbl else "win%"
    print(f"\n  {'pick':<10}{wlab:>8}{'div win':>9}{'+tie':>7}"
          f"{'avg rank':>10}{'led.burned':>12}")
    for _, x in df.iterrows():
        print(f"  {x['pick']:<10}{x['win%']:>7.0f}%{x.div_win:>8.1f}%"
              f"{x.div_tie:>6.1f}%{x['rank']:>10.2f}{x.leaders_burned:>12.0f}")

    top2 = df.head(2)
    if len(top2) == 2:
        p1, p2 = top2.div_win.iloc[0] / 100, top2.div_win.iloc[1] / 100
        se = math.sqrt(p1 * (1 - p1) / args.sims
                       + p2 * (1 - p2) / args.sims) * 100
        gap = top2.div_win.iloc[0] - top2.div_win.iloc[1]
        print(f"\n  top-two gap {gap:.2f}pp vs +/-{se:.2f}pp "
              f"({'RESOLVED' if gap > 2 * se else 'WITHIN NOISE'})")

    best_p = df.loc[df["win%"].idxmax(), "pick"]
    best_d = df.iloc[0]["pick"]
    print(f"\n  best {wlab} {best_p} | best division {best_d}")
    if best_p == best_d:
        print("  -> agree")
    else:
        # win% already holds the joint probability, so read it from the
        # table. grid.loc[] takes a single team and raises on "DAL+HOU".
        gap = df[df["pick"] == best_p].div_win.iloc[0] - df.iloc[0].div_win
        wp = df[df["pick"] == best_p]["win%"].iloc[0] - df.iloc[0]["win%"]
        print(f"  -> {best_d} gains {-gap:.1f}pp division, costs {wp:.0f}pp "
              f"{wlab}. With no margin that trade is usually bad.")


if __name__ == "__main__":
    main()
