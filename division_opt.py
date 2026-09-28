"""
Division-aware pick optimisation.

The $150 division prize is a 20-person race, not a 200-person one, and it is
by far your likeliest money. That makes it a different optimisation from
everything else in this repo:

  weekly.py         maximises P(you survive)
  prize_live.py     maximises expected dollars against a 200-person field
  THIS              maximises P(you finish first in your division)

Those diverge in a specific way. Surviving is worth nothing if the six
people ahead of you also survive -- you stay behind them forever. What you
need is to survive WHILE they fail. So a pick's value here depends on what
YOUR RIVALS hold, not just on its win probability.

Two mechanisms make a pick better or worse than its raw probability:

  CORRELATION. If a leader is likely to pick the same team you do, that
  week cannot separate you -- you both survive or both strike. Picking a
  team the leaders will avoid creates variance between you, which is what
  a trailer needs and what a leader wants to avoid.

  INVENTORY. Teams a leader has burned are teams they must replace with
  something worse later. You already hold JAX and PHI while half the
  leaders have spent them.

This is NOT a reason to take a bad pick. With one strike and no margin,
a second strike ends your season in every scenario. The tool reports the
division-win probability alongside raw survival so you can see exactly
what a contrarian choice costs -- and usually it will not be worth it.
"""
import argparse
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


def field_pick(avail_p, chalk, rng):
    """Rival choice model: softmax on win probability."""
    u = chalk * avail_p
    u -= u.max()
    q = np.exp(u)
    return q / q.sum()


def simulate(grid, members, me_idx, week, my_first, n_sims, chalk, seed=0):
    """
    members: list of {name, burned:set, strikes:int}
    my_first: the team YOU take this week (rest of your path is optimal)
    Returns P(win division outright), P(tie), mean finish rank.
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
        depth = np.full(N, S, int)

        for j in range(S):
            ok = np.isfinite(P[:, j])
            won = np.zeros(n_t, bool); done = np.zeros(n_t, bool)
            for a in range(n_t):
                if done[a] or not ok[a]:
                    continue
                b = OPP[a, j]
                u = rng.random()
                if b < 0:
                    won[a] = u < P[a, j]; done[a] = True
                else:
                    aw = u < P[a, j]
                    won[a], won[b] = aw, not aw
                    done[a] = done[b] = True

            for i in np.where(alive)[0]:
                cand = np.where(ok & ~used[i])[0]
                if cand.size == 0:
                    alive[i] = False; depth[i] = j; continue
                if i == me_idx and j == 0:
                    pick = ti[my_first]
                elif i == me_idx:
                    pick = cand[np.nanargmax(P[cand, j])]
                else:
                    pick = rng.choice(cand,
                                      p=field_pick(P[cand, j], chalk, rng))
                used[i, pick] = True
                if not won[pick]:
                    st[i] += 1
                    if st[i] >= 3:
                        alive[i] = False; depth[i] = j + 1

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
    ap.add_argument("--sims", type=int, default=400)
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
        members.append({"name": row["name"], "burned": burned, "strikes": 0})

    # strikes come from the sheet, not reconstruction -- a missed submission
    # is a strike that pick history cannot see
    st = pd.read_csv(args.roster)
    if "strikes" in st.columns:
        for m, s in zip(members, st.strikes):
            m["strikes"] = int(s)
    else:
        print("  NOTE: no 'strikes' column in roster; assuming 0 for all.")
        print("  Add one from the weekly sheet for correct results.\n")

    me = next(i for i, m in enumerate(members)
              if args.me.lower() in m["name"].lower())
    mine = members[me]

    avail = [t for t in grid.index
             if t not in mine["burned"] and pd.notna(grid.loc[t, args.week])]
    avail.sort(key=lambda t: -grid.loc[t, args.week])

    print("=" * 72)
    print(f"DIVISION-AWARE PICKS — week {args.week}, {len(members)} members")
    print("=" * 72)
    print(f"  you: {mine['name']}, {mine['strikes']} strike(s), "
          f"burned {','.join(sorted(mine['burned']))}")
    ahead = [m for m in members if m["strikes"] < mine["strikes"]]
    print(f"  ahead of you: {len(ahead)}")

    # how exposed is each candidate to the leaders?
    lead_burn = Counter()
    for m in ahead:
        for t in m["burned"]:
            lead_burn[t] += 1

    rows = []
    for t in avail[:args.top]:
        w, ti_, rk = simulate(grid, members, me, args.week, t,
                              args.sims, args.chalk)
        rows.append({"pick": t, "win%": grid.loc[t, args.week],
                     "div_win": 100 * w, "div_tie": 100 * ti_,
                     "rank": rk,
                     "leaders_burned": lead_burn.get(t, 0)})
    df = pd.DataFrame(rows).sort_values("div_win", ascending=False)

    print(f"\n  {'pick':<6}{'win%':>7}{'div win':>9}{'+tie':>7}"
          f"{'avg rank':>10}{'led.burned':>12}")
    print("  " + "-" * 62)
    for _, x in df.iterrows():
        print(f"  {x['pick']:<6}{x['win%']:>6.0f}%{x.div_win:>8.1f}%"
              f"{x.div_tie:>6.1f}%{x['rank']:>10.2f}{x.leaders_burned:>12.0f}")

    best_p = df.loc[df["win%"].idxmax(), "pick"]
    best_d = df.iloc[0]["pick"]
    print(f"\n  highest win probability : {best_p}")
    print(f"  best for division prize : {best_d}")
    if best_p == best_d:
        print("  They agree — no tradeoff this week.")
    else:
        gap = (df[df["pick"] == best_p].div_win.iloc[0]
               - df.iloc[0].div_win)
        print(f"  Taking {best_d} over {best_p} gains {-gap:.1f}pp of division")
        print(f"  win probability but costs "
              f"{grid.loc[best_p, args.week]-grid.loc[best_d, args.week]:.0f}pp "
              f"of raw win probability.")
        print("  With 1 strike and no margin, that trade is usually bad.")
    print("=" * 72)


if __name__ == "__main__":
    main()
