"""
Real-data pipeline. Run this locally (it needs network access).

    python pipeline.py --train 2007 2025 --season 2026 --week 2 \
        --base 3.717 --decay 0.355 --ridge 0.5

Output: win_probs_model.csv, in the 32x18 shape the survivor models consume,
so it drops straight into qc_grid.py / weekly.py / weekly_robust.py /
slots_model.py / prize23.py.
"""
import argparse
import numpy as np
import pandas as pd

from market import (MarketModel, log_loss, ece, calibration_table,
                    moneyline_to_prob, devig)
from ratings import PowerRatings, prior_from_win_totals
from residual import ResidualModel


# nflverse uses its own abbreviations, and they do NOT all match the grid.
# The Rams are "LA" in nflverse but "LAR" everywhere in the survivor code,
# so without this the Rams silently become a 33rd team, the ratings fit
# splits their games in half, and the emitted grid has a row weekly.py
# cannot match. The historical relocation codes matter when training to 2007.
TEAM_FIXES = {
    "LA": "LAR",     # Rams — the one that breaks 2026
    "STL": "LAR",    # Rams pre-2016
    "SD": "LAC",     # Chargers pre-2017
    "OAK": "LV",     # Raiders pre-2020
    "WSH": "WAS",
    "JAC": "JAX",
    "ARZ": "ARI",
    "BLT": "BAL",
    "CLV": "CLE",
    "HST": "HOU",
}


def normalize_teams(df):
    for col in ("home_team", "away_team"):
        if col in df.columns:
            df[col] = df[col].replace(TEAM_FIXES)
    return df


def load_schedules(seasons):
    import nflreadpy as nfl          # replaces the deprecated nfl_data_py
    df = nfl.load_schedules(seasons).to_pandas()
    return normalize_teams(df)


def to_team_rows(games):
    """One row per team per game, so we fit on 'this team's' perspective."""
    h = games.assign(team=games.home_team, opp=games.away_team,
                     spread=games.spread_line,
                     won=(games.result > 0).astype(int))
    a = games.assign(team=games.away_team, opp=games.home_team,
                     spread=-games.spread_line,
                     won=(games.result < 0).astype(int))
    # ties count as wins in this pool
    h.loc[games.result == 0, "won"] = 1
    a.loc[games.result == 0, "won"] = 1
    return pd.concat([h, a], ignore_index=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", nargs=2, type=int, default=[2007, 2025])
    ap.add_argument("--season", type=int, default=2026)
    ap.add_argument("--week", type=int, default=1, help="current week")
    ap.add_argument("--base", type=float, default=3.717,
                    help="sigma floor; get it from fit_decay.py")
    ap.add_argument("--decay", type=float, default=0.355,
                    help="sigma pts per week ahead; get it from fit_decay.py")
    ap.add_argument("--ridge", type=float, default=0.5,
                    help="ridge penalty on the power ratings. Default 0.5 is the "
                         "value ridge_cv.py measured at ~45 priced games. "
                         "Lower trusts "
                         "the posted lines more; higher pulls toward the "
                         "prior. This matters most when few games are priced: "
                         "with 45 posted lines instead of 112, a high value "
                         "compresses the ratings and flattens every projected "
                         "game toward a coin flip.")
    ap.add_argument("--out", default="win_probs_model.csv")
    args = ap.parse_args()

    yrs = list(range(args.train[0], args.train[1] + 1))
    hist = load_schedules(yrs)
    hist = hist[hist.game_type == "REG"].dropna(subset=["spread_line", "result"])

    # ---- layer 1: market baseline ------------------------------------
    tr = to_team_rows(hist)
    mm = MarketModel().fit(tr.spread, tr.won)
    mm.report(tr.spread, tr.won, "MARKET BASELINE (the bar to beat)")

    # ---- layer 3: does anything beat it? -----------------------------
    rm = ResidualModel(kind="gbm").fit(hist)
    rm.report()

    # ---- layer 2: ratings -> full-season projection -------------------
    cur = load_schedules([args.season])
    cur = cur[cur.game_type == "REG"]
    teams = sorted(set(cur.home_team) | set(cur.away_team))

    priced = cur.dropna(subset=["spread_line"])
    if priced.empty:
        raise SystemExit("No posted lines yet. Seed with win totals via "
                         "prior_from_win_totals() and re-run.")

    # weight recent weeks more; ridge keeps unseen teams near the prior
    wts = np.exp(-0.15 * (args.week - priced.week.clip(upper=args.week)))
    pr = PowerRatings(teams, ridge=args.ridge).fit(priced, weights=wts)
    tbl = pr.table()
    print("\nMARKET-IMPLIED POWER RATINGS")
    print(tbl.head(10).to_string(index=False))
    print(f"fitted HFA: {pr.hfa:.2f} pts")

    # Spread diagnostic: compressed ratings are the signature of the ridge
    # penalty dominating a thin set of posted lines. A healthy mid-season
    # NFL spread is roughly 3-5 pts of standard deviation, best-to-worst
    # around 12-18 pts. Much tighter than that and every projected game is
    # being pulled toward a coin flip by the prior, not by the market.
    rv = tbl["rating"].values
    print(f"\nratings spread: sd {rv.std():.2f} pts   "
          f"best {rv.max():+.2f}   worst {rv.min():+.2f}   "
          f"range {rv.max()-rv.min():.2f}")
    print(f"  (priced games in fit: {len(priced)} of {len(cur)}; "
          f"ridge={args.ridge})")
    if rv.std() < 2.0:
        print("  WARNING: ratings look compressed. Try a lower --ridge.")

    proj = pr.project(cur, current_week=args.week, sigma0=mm.sigma,
                      base=args.base, decay=args.decay)
    print(f"\n  games from posted lines : {(proj.source=='posted').sum()}")
    print(f"  games from ratings      : {(proj.source=='projected').sum()}")

    # ---- prefer DE-VIGGED MONEYLINES over probit-on-spread ------------
    # The moneyline IS the market's win probability. Converting a spread
    # through a single fitted sigma is a lossy detour, and it is biased
    # exactly where survivor pools live: sigma is fitted to the dense
    # small-spread region, so it understates big favourites. Removing vig
    # matters too -- raw implied probabilities sum to >1, and at a 4-5%
    # hold that error compounds across 23 picks.
    ml_used = 0
    if {"home_moneyline", "away_moneyline"} <= set(cur.columns):
        ml = cur.dropna(subset=["home_moneyline", "away_moneyline"])
        key = {(r.week, r.home_team): r for _, r in ml.iterrows()}
        for i, g in proj.iterrows():
            r = key.get((g.week, g.home_team))
            if r is None:
                continue
            ph = float(moneyline_to_prob(r.home_moneyline))
            pa = float(moneyline_to_prob(r.away_moneyline))
            if not (np.isfinite(ph) and np.isfinite(pa)):
                continue
            ph, pa = devig(ph, pa, method="shin")
            proj.at[i, "home_wp"] = float(ph)
            proj.at[i, "away_wp"] = float(pa)
            proj.at[i, "source"] = "moneyline"
            ml_used += 1
    print(f"  games from de-vig ML    : {ml_used}")

    # residual adjustment where features exist
    adj = rm.predict_margin_adjustment(cur)
    proj["adj_spread"] = proj.proj_spread + adj

    # ---- emit the 32 x 18 grid ---------------------------------------
    from scipy.stats import norm
    grid = pd.DataFrame(index=teams, columns=range(1, 19), dtype=float)
    for _, g in proj.iterrows():
        # Use the moneyline probability when we have one; otherwise convert
        # the (possibly residual-adjusted) spread through the fitted sigma.
        if g.source == "moneyline":
            ph = float(g.home_wp)
        else:
            sp = g.adj_spread if "adj_spread" in g else g.proj_spread
            ph = float(norm.cdf(sp / g.sigma))
        grid.loc[g.home_team, int(g.week)] = round(ph * 100, 1)
        grid.loc[g.away_team, int(g.week)] = round((1 - ph) * 100, 1)

    grid.index.name = "team"
    grid.to_csv(args.out)
    print(f"\nwrote {args.out}")

    # sanity: every week must sum to 100 x games
    for w in range(1, 19):
        n = grid[w].notna().sum() // 2
        tot = grid[w].sum()
        if abs(tot - 100 * n) > 0.5:
            print(f"  WARNING week {w}: sums to {tot:.1f}, expected {100*n}")


if __name__ == "__main__":
    main()
