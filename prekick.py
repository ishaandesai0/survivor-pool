"""
Pre-kickoff check. Run this right before you submit, not on Tuesday.

WHAT THIS IS NOT
----------------
This does not try to beat the market on injury news. That does not work,
and the reason matters: an NFL line reprices within minutes of a real
injury breaking, so by the time you read it the de-vigged moneyline in your
grid already contains it. Your own residual model measured this -- t = +0.12
on market error, no edge. Adjusting a probability downward because a
starter is out would be DOUBLE-COUNTING something the price already
reflects, and would make your grid worse, not better.

FPL is different because prices and lineups are sticky there, so knowing
early is worth something. Betting markets give you no such window.

WHAT THIS IS
------------
Two staleness checks, which are a real problem and a different one:

1. LINE MOVEMENT. Your grid was built Tuesday. You are picking Friday.
   If a line has moved materially, your grid is out of date and the
   recommendation may have changed. This refetches and diffs.

2. INJURY DESIGNATIONS on your candidate teams. Not to adjust anything --
   to make you look. The official report's final designations land Friday,
   which is exactly when you are deciding. A quarterback listed Out on a
   team you are about to pick is worth a human glance even when the line
   has already moved, because the line moving is not the same as the line
   being right.

The output is a flag for your judgement, not a number for the model.

    python prekick.py --week 2 --used LAC --strikes 1
"""
import argparse
import sys

import numpy as np
import pandas as pd

# positions where a designation actually moves a line much
KEY_POS = ["QB"]
IMPORTANT_POS = ["QB", "RB", "WR", "TE", "T", "G", "C", "EDGE", "DE", "CB", "LB", "S"]
BAD_STATUS = ["Out", "Doubtful"]
WATCH_STATUS = ["Out", "Doubtful", "Questionable"]


def load_grid(path):
    import os
    if not os.path.exists(path):
        raise SystemExit(f"No grid at {path}. Run pipeline.py first.")
    df = pd.read_csv(path).set_index("team")
    df.columns = [int(c) for c in df.columns]
    return df


def fresh_week_probs(week, season=2026):
    """
    Recompute this week's win probabilities from the CURRENT market, using
    de-vigged moneylines where available and the spread otherwise.
    Returns (DataFrame per game, dict team -> prob).
    """
    from pipeline import load_schedules
    from market import moneyline_to_prob, devig
    from scipy.stats import norm

    # Network failures here are routine -- nflverse is a GitHub release, so
    # a DNS hiccup or a rate limit produces a 60-line urllib3 traceback that
    # buries the actual problem. This tool runs minutes before a submission
    # deadline; it needs to say what went wrong in one line.
    try:
        cur = load_schedules([season])
    except Exception as e:
        msg = str(e)
        if "getaddrinfo" in msg or "NameResolutionError" in msg \
                or "Max retries" in msg:
            raise SystemExit(
                "NETWORK: cannot reach nflverse (github.com).\n"
                "  Check your connection, then re-run.\n"
                "  Your existing grid is still usable -- run\n"
                "    weekly.py --week <N> --used ... --strikes ... "
                "--grid win_probs_model.csv --objective depth\n"
                "  just be aware it reflects prices as of when it was built.")
        raise SystemExit(f"Could not load schedules: {msg[:200]}")
    cur = cur[(cur.game_type == "REG") & (cur.week == week)]
    rows, probs = [], {}
    for _, g in cur.iterrows():
        src, ph = None, None
        if pd.notna(g.get("home_moneyline")) and pd.notna(g.get("away_moneyline")):
            a = float(moneyline_to_prob(g.home_moneyline))
            b = float(moneyline_to_prob(g.away_moneyline))
            ph = float(devig(a, b, method="shin")[0])
            src = "ML"
        elif pd.notna(g.get("spread_line")):
            ph = float(norm.cdf(float(g.spread_line) / 11.16))
            src = "spread"
        if ph is None:
            continue
        probs[g.home_team] = 100 * ph
        probs[g.away_team] = 100 * (1 - ph)
        rows.append({"away": g.away_team, "home": g.home_team,
                     "spread": g.get("spread_line"),
                     "home_ml": g.get("home_moneyline"),
                     "home_wp": 100 * ph, "src": src})
    return pd.DataFrame(rows), probs


def injuries(week, teams, season=2026):
    """
    Returns (filtered_df, status_col, practice_col, injury_col, name_col, diag).

    `diag` records the row count at every filter step. Without it an empty
    result is uninterpretable -- "nobody is hurt", "the week filter is
    wrong", "the team abbreviations do not match" and "nflverse has not
    published yet" all look identical, and you cannot tell which one you
    are looking at minutes before a deadline.
    """
    import nflreadpy as nfl
    from pipeline import TEAM_FIXES
    diag = {}
    inj = nfl.load_injuries([season]).to_pandas()
    diag["loaded"] = len(inj)
    diag["weeks_available"] = (sorted(int(w) for w in inj.week.dropna().unique())
                               if "week" in inj.columns else None)
    if "week" in inj.columns:
        inj = inj[inj.week == week]
    diag["this_week"] = len(inj)
    if "team" in inj.columns:
        inj["team"] = inj["team"].replace(TEAM_FIXES)
        diag["teams_in_data"] = sorted(inj.team.unique())[:8]
        inj = inj[inj.team.isin(teams)]
    diag["after_team_filter"] = len(inj)
    # column names have shifted across nflverse versions; find what's there
    status_col = next((c for c in ["report_status", "game_status", "status"]
                       if c in inj.columns), None)
    practice_col = next((c for c in ["practice_status", "practice"]
                         if c in inj.columns), None)
    injury_col = next((c for c in ["report_primary_injury", "primary_injury",
                                   "injury"] if c in inj.columns), None)
    name_col = next((c for c in ["full_name", "player_name", "name"]
                     if c in inj.columns), None)
    diag["cols_found"] = {"status": status_col, "practice": practice_col,
                          "injury": injury_col, "name": name_col}
    return inj, status_col, practice_col, injury_col, name_col, diag


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--week", type=int, required=True)
    ap.add_argument("--used", default="")
    ap.add_argument("--strikes", type=int, default=0)
    ap.add_argument("--grid", default="win_probs_model.csv")
    ap.add_argument("--season", type=int, default=2026)
    ap.add_argument("--top", type=int, default=6)
    ap.add_argument("--move-threshold", type=float, default=2.0,
                    help="flag win-prob moves larger than this many points")
    args = ap.parse_args()

    used = [t.strip().upper() for t in args.used.split(",") if t.strip()]
    grid = load_grid(args.grid)

    print(f"PREKICK wk{args.week} — used {','.join(used) or 'none'}, "
          f"{args.strikes} strike(s)")

    # ---- 1. has the market moved since the grid was built? -----------
    games, fresh = fresh_week_probs(args.week, args.season)
    if games.empty:
        sys.exit("No priced games found for that week.")

    rows = []
    for t, p_now in fresh.items():
        p_old = grid.loc[t, args.week] if t in grid.index else np.nan
        if pd.isna(p_old):
            continue
        rows.append({"team": t, "grid": p_old, "now": p_now,
                     "move": p_now - p_old})
    mv = pd.DataFrame(rows).sort_values("move", key=abs, ascending=False)

    big = mv[mv.move.abs() >= args.move_threshold]
    print(f"\nMOVEMENT >={args.move_threshold:.0f}pt")
    if big.empty:
        print("  none — grid current")
    else:
        # only the side that moved UP is informative; the complement is
        # mechanical, so collapse each game to one line
        seen = set()
        for _, r in big.sort_values("move", key=abs, ascending=False).iterrows():
            if r.team in seen:
                continue
            seen.add(r.team)
            flag = " *" if abs(r.move) >= 4 else ""
            print(f"  {r.team:<5}{r.grid:>6.1f} -> {r['now']:<6.1f}"
                  f"{r.move:>+6.1f}{flag}")
        print("  -> rebuild pipeline before deciding")

    # ---- 2. what does the fresh market say the pick is? ---------------
    avail = {t: p for t, p in fresh.items() if t not in used}
    rank = sorted(avail.items(), key=lambda x: -x[1])[:args.top]
    print(f"\nFRESH PRICES (top {args.top})")
    for t, p in rank:
        o = grid.loc[t, args.week] if t in grid.index else np.nan
        d = "" if pd.isna(o) else f"{p-o:>+6.1f}"
        print(f"  {t:<5}{p:>6.1f}%{d}")

    # ---- 3. injury designations on the candidates ---------------------
    cand = [t for t, _ in rank]
    print(f"\nINJURIES (candidates only)")
    diag = {}
    try:
        inj, sc, pc, ic, nc, diag = injuries(args.week, cand, args.season)
    except Exception as e:
        msg = str(e)
        if "getaddrinfo" in msg or "Max retries" in msg:
            print("  NETWORK: could not reach nflverse for injuries.")
        elif "404" in msg or "Not Found" in msg:
            print("  nflverse has no injury file for this season yet.")
            print("  Check a news source for designations instead.")
        else:
            print(f"  could not load injuries: {msg[:90]}")
        inj = None
    if inj is not None:
        if inj.empty:
            # Say WHY it is empty. Each case needs a different response.
            wk = diag.get("weeks_available")
            print(f"  none. wk{args.week} rows={diag.get('this_week')}, "
                  f"after team filter={diag.get('after_team_filter')}, "
                  f"weeks in file={wk}")
            if diag.get("loaded", 0) == 0:
                print("\n  DIAGNOSIS: nflverse has no 2026 injury file yet.")
                print("  Use a news source for designations.")
            elif wk and args.week not in wk:
                print(f"\n  DIAGNOSIS: week {args.week} not published yet "
                      f"(file has {wk}).")
                print("  Official reports land Wed/Thu/Fri. Re-run later.")
            elif diag.get("this_week", 0) > 0 and diag.get("after_team_filter") == 0:
                print("\n  DIAGNOSIS: week data exists but NO rows matched "
                      "your candidate teams.")
                print(f"  teams in data: {diag.get('teams_in_data')}")
                print(f"  candidates   : {cand}")
                print("  If those look like different abbreviation styles, "
                      "TEAM_FIXES needs an entry.")
            else:
                print("\n  DIAGNOSIS: genuinely nothing on the report for "
                      "these teams.")
        else:
            shown = 0
            for t in cand:
                sub = inj[inj.team == t]
                # Report status is only populated once a team files its final
                # designation -- Friday for Sunday games, Thursday for TNF.
                # Mid-week that column is empty for almost everyone, so
                # filtering on it alone discards ~97% of the rows and reports
                # "nothing to see" for a team with a QB who did not practise.
                # So: keep a row if it has a watch-list designation OR if the
                # player missed/was limited in practice.
                if sc or pc:
                    keep = pd.Series(False, index=sub.index)
                    if sc:
                        keep |= sub[sc].isin(WATCH_STATUS)
                    if pc:
                        keep |= sub[pc].astype(str).str.contains(
                            "Did Not Participate|Limited", case=False, na=False)
                    sub = sub[keep]
                if sub.empty:
                    continue
                qbs = sub[sub.position == "QB"] if "position" in sub else sub.iloc[:0]
                tag = "  *** QB ***" if len(qbs) else ""
                print(f"\n  {t}{tag}")
                for _, r in sub.iterrows():
                    pos = r.get("position", "?")
                    if pos not in IMPORTANT_POS:
                        continue
                    nm = r.get(nc, "?") if nc else "?"
                    st = r.get(sc, None) if sc else None
                    st = "-" if (st is None or pd.isna(st)) else str(st)
                    # fall back to the practice injury when no report injury
                    inn = r.get(ic, None) if ic else None
                    if inn is None or pd.isna(inn):
                        inn = r.get("practice_primary_injury", "") or ""
                    pr = r.get(pc, "") if pc else ""
                    pr = "" if pd.isna(pr) else str(pr)
                    short = (pr.replace("Did Not Participate In Practice", "DNP")
                               .replace("Limited Participation in Practice", "LTD")
                               .replace("Full Participation in Practice", "FULL"))
                    # "resting player" on a big favourite is not an injury
                    resting = "not injury related" in str(inn).lower()
                    mark = ""
                    if pos in KEY_POS and (st in BAD_STATUS or
                                           (short == "DNP" and not resting)):
                        mark = "  <<< KEY"
                    elif resting:
                        mark = "  (rest)"
                    print(f"    {pos:<4}{str(nm)[:22]:<23}{st:<13}"
                          f"{str(inn)[:26]:<27}{short:<5}{mark}")
                    shown += 1
            if not shown:
                print("  No designations or practice absences at important")
                print("  positions for these teams.")
            else:
                print("\n  DNP/LTD/FULL = practice. blank status = not yet"
                      " filed (Fri). (rest) = healthy.")


if __name__ == "__main__":
    main()
