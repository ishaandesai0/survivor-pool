"""
Real-data pipeline. Run this locally (it needs network access).

    python pipeline.py --train 2007 2025 --season 2026 --week 2 \
        --base 3.717 --decay 0.355 --ridge 0.5

Output: win_probs_model.csv, in the 32x18 shape the survivor models consume,
so it drops straight into qc_grid.py / weekly.py / weekly_robust.py /
slots_model.py / prize23.py.
"""
import argparse
import os
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


def refresh_results(cur, path="results_2026.csv"):
    """
    Keep results_2026.csv current from nflverse.

    The model never needed this file -- pipeline.py fits on nflverse
    directly -- so it silently stopped being updated after week 2 while
    everything else kept working. calibrate.py then had nothing to grade
    weeks 3+ against and quietly reported on half the season.

    Rebuilt every run from the authoritative source, so it cannot drift.
    Hand-written notes on existing rows are preserved.
    """
    done = cur[cur.result.notna()].copy()
    if done.empty:
        return
    winner = done.apply(
        lambda r: r.home_team if r.result > 0
        else (r.away_team if r.result < 0 else "TIE"), axis=1)
    fresh = pd.DataFrame({
        "week": done.week.astype(int),
        "away": done.away_team, "away_pts": done.away_score,
        "home": done.home_team, "home_pts": done.home_score,
        "winner": winner,
        "date": done.get("gameday", pd.NA),
        "note": "",
    }).sort_values(["week", "home"]).reset_index(drop=True)

    if os.path.exists(path):          # carry over any notes already written
        try:
            old = pd.read_csv(path)
            if "note" in old.columns:
                keyed = {(int(r.week), r.home): r.note for _, r in old.iterrows()
                         if isinstance(r.note, str) and r.note.strip()}
                fresh["note"] = [keyed.get((w, h), "")
                                 for w, h in zip(fresh.week, fresh.home)]
        except Exception:
            pass

    fresh.to_csv(path, index=False)
    print(f"  results_2026.csv: {len(fresh)} games through week "
          f"{int(fresh.week.max())}")


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
    ap.add_argument("--verbose", action="store_true",
                    help="show the full market/residual reports")
    ap.add_argument("--out", default="win_probs_model.csv")
    args = ap.parse_args()

    yrs = list(range(args.train[0], args.train[1] + 1))
    hist = load_schedules(yrs)
    hist = hist[hist.game_type == "REG"].dropna(subset=["spread_line", "result"])

    # ---- layer 1: market baseline ------------------------------------
    tr = to_team_rows(hist)
    mm = MarketModel().fit(tr.spread, tr.won)
    if args.verbose:
        mm.report(tr.spread, tr.won, "MARKET BASELINE (the bar to beat)")
    else:
        from market import log_loss as _ll, ece as _ece
        _p = mm.predict(tr.spread)
        print(f"market: sigma {mm.sigma:.2f}  logloss {_ll(tr.won, _p):.4f}"
              f"  (reliability table is in-sample; --verbose to show)")

    # ---- layer 3: does anything beat it? -----------------------------
    rm = ResidualModel(kind="gbm").fit(hist)
    if args.verbose:
        rm.report()
    else:
        print(f"residual: t={rm.t_stat:+.2f} shrink={rm.shrink:.3f}"
              f"  ({'no edge' if rm.t_stat < 2 else 'EDGE'})")

    # ---- layer 2: ratings -> full-season projection -------------------
    cur = load_schedules([args.season])
    cur = cur[cur.game_type == "REG"]
    refresh_results(cur)
    teams = sorted(set(cur.home_team) | set(cur.away_team))

    priced = cur.dropna(subset=["spread_line"])
    if priced.empty:
        raise SystemExit("No posted lines yet. Seed with win totals via "
                         "prior_from_win_totals() and re-run.")

    # weight recent weeks more; ridge keeps unseen teams near the prior
    wts = np.exp(-0.15 * (args.week - priced.week.clip(upper=args.week)))
    pr = PowerRatings(teams, ridge=args.ridge).fit(priced, weights=wts)
    tbl = pr.table()
    print("\nratings: " + "  ".join(
        f"{r.team} {r.rating:.2f}" for r in tbl.head(8).itertuples()))
    print(f"HFA {pr.hfa:.2f}", end="")

    # Spread diagnostic: compressed ratings are the signature of the ridge
    # penalty dominating a thin set of posted lines. A healthy mid-season
    # NFL spread is roughly 3-5 pts of standard deviation, best-to-worst
    # around 12-18 pts. Much tighter than that and every projected game is
    # being pulled toward a coin flip by the prior, not by the market.
    rv = tbl["rating"].values
    print(f"  sd {rv.std():.2f}  range {rv.max()-rv.min():.2f}"
          f"  priced {len(priced)}/{len(cur)}  ridge {args.ridge}")
    if rv.std() < 2.0:
        print("  WARNING: ratings look compressed. Try a lower --ridge.")

    proj = pr.project(cur, current_week=args.week, sigma0=mm.sigma,
                      base=args.base, decay=args.decay)
    _np_, _pr_ = (proj.source == 'posted').sum(), (proj.source == 'projected').sum()

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
    print(f"sources: posted {_np_}  projected {_pr_}  de-vig ML {ml_used}")

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

    # Record what produced this grid. calibrate.py otherwise has to infer the
    # week, and four of nine historical commits touched the grid without
    # naming one -- which silently threw away the real Week 3 and Week 4
    # forecasts. A date-based fallback covers history; this covers the future.
    import json
    meta_path = "grid_meta.json"
    try:
        meta = json.load(open(meta_path))
    except Exception:
        meta = {}
    meta[f"week{int(args.week)}"] = {
        "week": int(args.week), "ridge": float(args.ridge),
        "base": float(args.base), "decay": float(args.decay),
        "sigma0": round(float(mm.sigma), 4), "posted": int(len(priced)),
    }
    json.dump(meta, open(meta_path, "w"), indent=1)
    print(f"  (recorded in {meta_path})")
    print(f"\nwrote {args.out}")

    # sanity: every week must sum to 100 x games
    for w in range(1, 19):
        n = grid[w].notna().sum() // 2
        tot = grid[w].sum()
        if abs(tot - 100 * n) > 0.5:
            print(f"  WARNING week {w}: sums to {tot:.1f}, expected {100*n}")


if __name__ == "__main__":
    main()
