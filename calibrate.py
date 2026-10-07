"""
Are the probabilities honest?

Every number this repo produces assumes the grid is calibrated -- that games
called 85% actually win 85% of the time. Nothing has ever checked. This does,
and it is the only tool here that can show the model is WRONG rather than
merely uncertain.

The method depends on a discipline already in place: win_probs_model.csv is
committed every week BEFORE the slate resolves. So `git log` holds a sequence
of genuine out-of-sample forecasts, each frozen before its outcome existed.
Rebuilding a grid today from updated lines would leak hindsight and make any
calibration result meaningless.

For each committed grid we take only the week it was built for (its own
current week, where the lines were posted), join to results, and ask whether
predicted and actual agree.

Three readings, in order of what they tell you:

  ECE        average gap between predicted and actual across buckets.
             Under ~3pp on this sample size is fine.
  BIAS       signed. Positive means overconfident -- you are claiming more
             certainty than the games deliver, which in survivor means
             walking into strikes you thought were unlikely.
  TAIL       the 80%+ bucket specifically. Survivor only ever picks from
             the tail, so a model well-calibrated in the middle and
             overconfident at the top is exactly wrong for this use.

A caveat worth keeping: with ~16 games per week, even by Week 10 the 80%+
bucket holds maybe 40 games. A 5pp miscalibration needs roughly 100 to
detect. Read this as a smoke alarm, not a thermometer.

    python calibrate.py
    python calibrate.py --min-week 2       # skip early weeks
    python calibrate.py --show-games       # list every graded game
"""
import argparse
import io
import subprocess
import sys

import numpy as np
import pandas as pd

GRID = "win_probs_model.csv"
RESULTS = "results_2026.csv"


def committed_grids():
    """
    Every commit that touched the grid, oldest first, with its content.
    Returns [(sha, date, subject, DataFrame)].
    """
    log = subprocess.run(
        ["git", "log", "--reverse", "--format=%H|%ad|%s", "--date=short",
         "--", GRID],
        capture_output=True, text=True)
    if log.returncode != 0:
        sys.exit("not a git repo, or no history for " + GRID)
    out = []
    for line in log.stdout.strip().splitlines():
        sha, date, subject = line.split("|", 2)
        blob = subprocess.run(["git", "show", f"{sha}:{GRID}"],
                              capture_output=True, text=True)
        if blob.returncode != 0:
            continue
        try:
            df = pd.read_csv(io.StringIO(blob.stdout)).set_index("team")
            df.columns = [int(c) for c in df.columns]
        except Exception:
            continue
        out.append((sha[:8], date, subject, df))
    return out


def pick_grid_for_week(grids, week, results, meta):
    """
    Which committed grid is the forecast for `week`?

    Not the commit message -- four of nine commits touch the grid without
    naming a week, and a fallback that assigns "first ungraded week" lets a
    tooling commit claim a slot and throw the real forecast away.

    The unambiguous definition: the LAST grid committed strictly before that
    week's first kickoff. That is the forecast you actually had in hand.
    A sidecar grid_meta.json, written by pipeline.py from now on, overrides
    this when present.
    """
    games = results[results.week == week]
    if games.empty or "date" not in games.columns:
        return None
    first_kick = pd.to_datetime(games.date).min()

    best = None
    for sha, date, subject, df in grids:
        if sha in meta and meta[sha].get("week") == week:
            return (sha, date, subject, df)          # sidecar wins outright
        if pd.to_datetime(date) >= first_kick:
            continue
        if df[week].isna().all():
            continue
        best = (sha, date, subject, df)              # keep the latest
    return best


def load_meta(path="grid_meta.json"):
    """sha -> {"week": N, ...} written by pipeline.py alongside each grid."""
    import json, os
    if not os.path.exists(path):
        return {}
    try:
        return json.load(open(path))
    except Exception:
        return {}


def grade(grids, results, min_week):
    """One row per team-game, with the probability as forecast at the time."""
    res = results.copy()
    meta = load_meta()
    rows, used, chosen = [], set(), []
    for wk in sorted(res.week.unique()):
        if wk < min_week:
            continue
        hit = pick_grid_for_week(grids, wk, res, meta)
        if hit is None:
            continue
        sha, date, subject, df = hit
        games = res[res.week == wk]
        used.add(wk)
        chosen.append((wk, sha, date))
        for _, g in games.iterrows():
            for team, opp, won in ((g.home, g.away, g.winner == g.home),
                                   (g.away, g.home, g.winner == g.away)):
                p = df[wk].get(team)
                if pd.isna(p):
                    continue
                rows.append({"week": wk, "sha": sha, "team": team, "opp": opp,
                             "p": float(p) / 100.0, "won": int(won)})
    return pd.DataFrame(rows), sorted(used), chosen


def reliability(df, edges):
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (df.p >= lo) & (df.p < hi)
        if m.sum() == 0:
            continue
        out.append({"bucket": f"{lo:.2f}-{hi:.2f}", "n": int(m.sum()),
                    "pred": df.p[m].mean(), "actual": df.won[m].mean()})
    r = pd.DataFrame(out)
    r["gap"] = r.pred - r.actual
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-week", type=int, default=1)
    ap.add_argument("--show-games", action="store_true")
    ap.add_argument("--results", default=RESULTS)
    args = ap.parse_args()

    grids = committed_grids()
    if not grids:
        sys.exit(f"no committed versions of {GRID} found")
    results = pd.read_csv(args.results)

    df, weeks, chosen = grade(grids, results, args.min_week)
    if df.empty:
        sys.exit("no graded games -- check that commit subjects name the week")

    print("=" * 64)
    print(f"CALIBRATION — weeks {weeks}, {len(df)//2} games "
          f"({len(df)} team-rows)")
    print(f"{len(grids)} committed grids; for each week the LAST one")
    print("committed before kickoff is used -- the forecast you actually had.")
    print("  " + "  ".join(f"W{w}:{sha}({d[5:]})" for w, sha, d in chosen))
    print("=" * 64)

    # headline numbers
    brier = float(((df.p - df.won) ** 2).mean())
    eps = 1e-9
    ll = float(-(df.won * np.log(df.p.clip(eps, 1 - eps))
                 + (1 - df.won) * np.log((1 - df.p).clip(eps, 1 - eps))).mean())
    acc = float(((df.p > 0.5) == (df.won == 1)).mean())
    bias = float((df.p - df.won).mean())

    r = reliability(df, [0, .2, .4, .5, .6, .7, .8, .9, 1.001])
    ece = float((r.n * r.gap.abs()).sum() / r.n.sum())

    print(f"\n  brier {brier:.4f}   log loss {ll:.4f}   "
          f"accuracy {acc*100:.1f}%")
    print(f"  ECE   {ece*100:.2f}pp")
    if abs(bias) < 1e-6:
        print(f"  bias  {bias*100:+.2f}pp — structurally zero, not a result:")
        print("        de-vig forces each game's two probabilities to sum to")
        print("        100, so predicted wins must equal games played. Only")
        print("        the bucket gaps and the tail reading carry signal.")
    else:
        print(f"  bias  {bias*100:+.2f}pp "
              f"({'overconfident' if bias > 0 else 'underconfident'})")

    print(f"\n  {'bucket':<12}{'n':>5}{'pred':>8}{'actual':>8}{'gap':>8}")
    for _, x in r.iterrows():
        flag = "  <--" if abs(x.gap) > 0.10 and x.n >= 8 else ""
        print(f"  {x.bucket:<12}{int(x.n):>5}{x.pred:>8.3f}"
              f"{x.actual:>8.3f}{x.gap:>+8.3f}{flag}")

    # the tail is what survivor actually uses
    tail = df[df.p >= 0.75]
    if len(tail):
        tp, ta = tail.p.mean(), tail.won.mean()
        se = float(np.sqrt(ta * (1 - ta) / len(tail)))
        print(f"\n  THE TAIL (p >= 0.75, {len(tail)} picks) — survivor only "
              f"ever picks here")
        print(f"    predicted {tp*100:.1f}%   actual {ta*100:.1f}%   "
              f"gap {(tp-ta)*100:+.1f}pp   +/-{se*100:.1f}pp")
        if abs(tp - ta) < 2 * se:
            print("    -> within noise; no evidence of tail miscalibration")
        elif tp > ta:
            print("    -> OVERCONFIDENT in the tail. Your 'safe' picks are")
            print("       losing more than stated. This is the failure mode")
            print("       that matters: it means walking into strikes you")
            print("       priced as unlikely.")
        else:
            print("    -> underconfident; favourites winning more than stated")

    # per-week, to spot a single bad grid
    print(f"\n  by week:")
    for w, g in df.groupby("week"):
        b = float((g.p - g.won).mean())
        print(f"    W{w}: {len(g)//2:>2} games   bias {b*100:+6.2f}pp   "
              f"brier {float(((g.p-g.won)**2).mean()):.3f}")

    # your own picks, graded
    try:
        picks = pd.read_csv("picks.csv")
        mine = []
        for _, p in picks.iterrows():
            hit = df[(df.week == p.week) & (df.team == p.team)]
            if len(hit):
                mine.append((int(p.week), p.team, float(hit.p.iloc[0]),
                             int(hit.won.iloc[0])))
        if mine:
            print(f"\n  YOUR PICKS:")
            tot = 0.0
            for w, t, p, won in mine:
                tot += p
                print(f"    W{w}: {t:<4} {p*100:>5.1f}%  "
                      f"{'WON' if won else 'LOST'}")
            n_won = sum(w for *_, w in mine)
            print(f"    expected {tot:.2f} wins, actual {n_won} "
                  f"of {len(mine)}")
    except FileNotFoundError:
        pass

    if args.show_games:
        print("\n  all graded games:")
        for _, x in df.sort_values(["week", "p"], ascending=[True, False]).iterrows():
            print(f"    W{int(x.week)} {x.team:<4} vs {x.opp:<4} "
                  f"{x.p*100:>5.1f}%  {'W' if x.won else 'L'}")

    print("\n" + "=" * 64)
    n_tail = len(tail) if len(tail) else 0
    print(f"  Sample size caveat: {n_tail} picks in the tail. Detecting a")
    print(f"  5pp miscalibration reliably needs ~100. Re-run every few weeks.")
    print("=" * 64)


if __name__ == "__main__":
    main()
