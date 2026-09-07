"""
NFL Survivor Pool engine — 2026 season.

Three models, in increasing order of how much they actually matter:

1. OPTIMAL PATH.  Picking one team per week, each at most once, is an
   assignment problem.  Maximising P(survive all weeks) = maximising the
   product of win probabilities = maximising the SUM OF LOG probabilities,
   which the Hungarian algorithm solves exactly in polynomial time.
   Greedy week-by-week chalk is provably worse.

2. FUTURE VALUE / REGRET.  For every (team, week) cell, re-solve the
   assignment with that pick forced.  The drop from the unconstrained
   optimum is the true cost of spending that team in that week.  Cells with
   ~zero regret are free; cells with big regret are the traps.

3. POOL SIMULATION.  Surviving is not the goal — outlasting everyone else
   is.  Monte Carlo with shared, correlated game outcomes and a field of
   simulated opponents who play chalk to varying degrees.  This is where
   contrarian value shows up, and it is the only model that answers
   "what maximises my share of the prize", which is a different question
   from "what maximises my survival".

Usage:
    python3 survivor.py                 # full report
    python3 survivor.py --horizon 14    # plan assuming the pool ends wk 14
    python3 survivor.py --entrants 250  # size the simulated field
"""
import argparse
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

from qc_grid import load, recover_matchups, WEEKS

BIG = 1e6  # cost for an impossible assignment (bye week)


# ----------------------------------------------------------------------
# 1. OPTIMAL PATH
# ----------------------------------------------------------------------
def cost_matrix(df, horizon=18, banned=(), forced=None):
    """
    Rows = teams, cols = weeks 1..horizon.  Cost = -log(win prob).
    `forced` is a dict {week: team} pinning specific picks.
    """
    teams = list(df.index)
    weeks = list(range(1, horizon + 1))
    C = np.full((len(teams), len(weeks)), BIG)
    for i, t in enumerate(teams):
        if t in banned:
            continue
        for j, w in enumerate(weeks):
            p = df.loc[t, w]
            if pd.notna(p) and p > 0:
                C[i, j] = -np.log(p / 100.0)
    if forced:
        for w, t in forced.items():
            if w > horizon:
                continue
            j = weeks.index(w)
            keep = C[teams.index(t), j]
            C[:, j] = BIG          # nobody else may take this week
            C[teams.index(t), j] = keep
    return C, teams, weeks


def solve_path(df, horizon=18, banned=(), forced=None):
    """Returns (path dict {week: (team, prob)}, survival probability)."""
    C, teams, weeks = cost_matrix(df, horizon, banned, forced)
    rows, cols = linear_sum_assignment(C)
    if C[rows, cols].max() >= BIG:
        return None, 0.0
    path = {}
    for r, c in zip(rows, cols):
        w = weeks[c]
        path[w] = (teams[r], df.loc[teams[r], w])
    surv = float(np.exp(-C[rows, cols].sum()))
    return dict(sorted(path.items())), surv


def greedy_path(df, horizon=18):
    """Naive baseline: every week take the highest win prob still available."""
    used, path, surv = set(), {}, 1.0
    for w in range(1, horizon + 1):
        avail = df[w].dropna().drop(labels=[t for t in used if t in df.index], errors="ignore")
        avail = avail[~avail.index.isin(used)]
        t = avail.idxmax()
        path[w] = (t, avail[t])
        surv *= avail[t] / 100.0
        used.add(t)
    return path, surv


def print_path(path, surv, title):
    print(f"\n{title}")
    print("-" * len(title))
    line = "  ".join(f"W{w}:{t}" for w, (t, p) in path.items())
    print(line)
    print("  " + "  ".join(f"{p:.0f}" for w, (t, p) in path.items()))
    print(f"P(survive all {len(path)} weeks) = {surv*100:.2f}%")


# ----------------------------------------------------------------------
# 2. FUTURE VALUE / REGRET
# ----------------------------------------------------------------------
def regret_matrix(df, horizon=18):
    """
    regret[t, w] = log-odds lost by forcing team t into week w, relative to
    the unconstrained optimum.  0.00 means the pick is on an optimal path.
    Reported in "percentage points of season survival".
    """
    _, best = solve_path(df, horizon)
    out = pd.DataFrame(index=df.index, columns=range(1, horizon + 1), dtype=float)
    for t in df.index:
        for w in range(1, horizon + 1):
            if pd.isna(df.loc[t, w]):
                continue
            _, s = solve_path(df, horizon, forced={w: t})
            out.loc[t, w] = 100.0 * (best - s) / best   # % of optimum given up
    return out, best


def team_future_value(df, horizon=18):
    """
    FV of a team = how much season survival probability collapses if you can
    never use them.  High FV = a genuinely scarce resource worth protecting.
    """
    _, best = solve_path(df, horizon)
    rows = []
    for t in df.index:
        _, s = solve_path(df, horizon, banned=(t,))
        rows.append({"team": t,
                     "fv_pct_of_optimum_lost": 100.0 * (best - s) / best,
                     "best_week": int(df.loc[t].idxmax()),
                     "best_prob": df.loc[t].max(),
                     "n_weeks_above_70": int((df.loc[t] >= 70).sum())})
    return pd.DataFrame(rows).sort_values("fv_pct_of_optimum_lost", ascending=False)


# ----------------------------------------------------------------------
# 3. POOL SIMULATION
# ----------------------------------------------------------------------
def build_matchups(df):
    """{week: {team: opponent}} recovered from probability complements."""
    m = {}
    for w in WEEKS:
        pairs, _ = recover_matchups(df, w)
        d = {}
        for a, b, *_ in pairs:
            d[a], d[b] = b, a
        m[w] = d
    return m


def simulate_outcomes(df, matchups, rng, horizon):
    """
    One season of results.  Draws ONE coin per game (not per team) so the
    two sides of a matchup are perfectly anti-correlated, exactly as reality
    works.  This correlation is the entire reason survivor pools bust in
    clusters.
    """
    won = {}
    for w in range(1, horizon + 1):
        seen = set()
        for t, opp in matchups[w].items():
            if t in seen:
                continue
            seen.add(t); seen.add(opp)
            p = df.loc[t, w] / 100.0
            t_wins = rng.random() < p
            won[(t, w)] = t_wins
            won[(opp, w)] = not t_wins
    return won


def field_pick(df, week, used, rng, chalk):
    """
    Model of a rival entrant.  `chalk` controls how hard they chase the
    biggest favourite: high chalk = always takes the top win prob available,
    low chalk = spreads picks around.  Real pools sit around chalk 8-15.
    """
    avail = df[week].dropna()
    avail = avail[~avail.index.isin(used)]
    if avail.empty:
        return None
    w = np.exp(chalk * (avail.values / 100.0 - avail.values.max() / 100.0))
    w = w / w.sum()
    return rng.choice(avail.index.values, p=w)


def simulate_pool(df, my_path, n_entrants=200, n_sims=20000, horizon=18, seed=0):
    """
    Returns dict of outcome stats for MY entry against a simulated field.
    Payout rule: last entrant standing takes it; if everyone alive is
    eliminated in the same week, those entrants split (standard house rule).
    """
    rng = np.random.default_rng(seed)
    matchups = build_matchups(df)
    my_picks = {w: t for w, (t, _) in my_path.items()}

    my_survive_all = 0
    my_equity = 0.0
    death_week = np.zeros(horizon + 2)

    for _ in range(n_sims):
        won = simulate_outcomes(df, matchups, rng, horizon)

        # field: each rival gets its own chalkiness
        chalks = rng.uniform(4, 18, size=n_entrants)
        used = [set() for _ in range(n_entrants)]
        alive = np.ones(n_entrants, dtype=bool)
        my_alive, my_used = True, set()
        my_death = horizon + 1

        for w in range(1, horizon + 1):
            prev_alive = alive.copy()
            my_prev = my_alive

            for i in range(n_entrants):
                if not alive[i]:
                    continue
                t = field_pick(df, w, used[i], rng, chalks[i])
                if t is None or not won[(t, w)]:
                    alive[i] = False
                else:
                    used[i].add(t)

            if my_alive:
                t = my_picks.get(w)
                if t is None or not won[(t, w)]:
                    my_alive = False
                    my_death = w
                else:
                    my_used.add(t)

            # everyone wiped out this week -> split among that week's survivors
            if not alive.any() and not my_alive:
                n_split = prev_alive.sum() + (1 if my_prev else 0)
                if my_prev:
                    my_equity += 1.0 / n_split
                break
        else:
            n_final = alive.sum() + (1 if my_alive else 0)
            if my_alive:
                my_equity += 1.0 / n_final
                my_survive_all += 1

        death_week[my_death] += 1

    return {
        "survive_all_pct": 100.0 * my_survive_all / n_sims,
        "equity_pct": 100.0 * my_equity / n_sims,
        "fair_share_pct": 100.0 / (n_entrants + 1),
        "edge_multiple": (my_equity / n_sims) * (n_entrants + 1),
        "death_week": death_week / n_sims,
    }


# ----------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--horizon", type=int, default=18)
    ap.add_argument("--entrants", type=int, default=200)
    ap.add_argument("--sims", type=int, default=4000)
    args = ap.parse_args()

    df = load()
    H = args.horizon

    print("=" * 70)
    print(f"NFL SURVIVOR — 2026 | planning horizon: week {H}")
    print("=" * 70)

    opt, s_opt = solve_path(df, H)
    grd, s_grd = greedy_path(df, H)
    print_path(grd, s_grd, "BASELINE: greedy chalk (take the biggest favourite every week)")
    print_path(opt, s_opt, "OPTIMAL: Hungarian assignment over the full grid")
    print(f"\nOptimal beats greedy by {s_opt/s_grd:.2f}x on survival probability.")

    print("\n" + "=" * 70)
    print("BEST PATH CONDITIONAL ON EACH WEEK-1 PICK")
    print("=" * 70)
    rows = []
    for t in df.index:
        if pd.isna(df.loc[t, 1]):
            continue
        p, s = solve_path(df, H, forced={1: t})
        if p:
            rows.append({"wk1": t, "wk1_prob": df.loc[t, 1],
                         "season_surv_pct": 100 * s,
                         "cost_vs_best_pct": 100 * (s_opt - s) / s_opt,
                         "path": " ".join(x[0] for _, x in sorted(p.items()))})
    r = pd.DataFrame(rows).sort_values("season_surv_pct", ascending=False)
    print(r.head(12).to_string(index=False,
          formatters={"season_surv_pct": "{:.2f}".format,
                      "cost_vs_best_pct": "{:.1f}".format,
                      "wk1_prob": "{:.0f}".format}))

    print("\n" + "=" * 70)
    print("FUTURE VALUE — which teams are scarce resources?")
    print("=" * 70)
    fv = team_future_value(df, H)
    print(fv.head(12).to_string(index=False,
          formatters={"fv_pct_of_optimum_lost": "{:.1f}".format,
                      "best_prob": "{:.0f}".format}))

    print("\n" + "=" * 70)
    print("REGRET MATRIX — % of optimal season survival given up by each pick")
    print("(0.0 = free, on an optimal path.  Blank = bye.)")
    print("=" * 70)
    reg, _ = regret_matrix(df, H)
    print(reg.round(1).to_string(na_rep=" -- "))

    print("\n" + "=" * 70)
    print(f"POOL SIMULATION — {args.entrants} rivals, {args.sims} seasons")
    print("=" * 70)
    res = simulate_pool(df, opt, args.entrants, args.sims, H)
    print(f"  P(I survive all {H} weeks) : {res['survive_all_pct']:.2f}%")
    print(f"  My share of the prize     : {res['equity_pct']:.2f}%")
    print(f"  Fair share (random play)  : {res['fair_share_pct']:.2f}%")
    print(f"  Edge over fair share      : {res['edge_multiple']:.2f}x")
    print("\n  When I get knocked out:")
    for w in range(1, H + 1):
        bar = "#" * int(res["death_week"][w] * 200)
        print(f"   week {w:2d}  {res['death_week'][w]*100:5.1f}%  {bar}")
    print(f"   survive  {res['death_week'][H+1]*100:5.1f}%")


if __name__ == "__main__":
    main()
