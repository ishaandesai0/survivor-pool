"""
Choose --ridge by cross-validation instead of by feel.

The ridge penalty controls how much the power-rating fit trusts the posted
lines versus the prior. It matters most exactly when data is thin: with 45
priced games and 32 unknowns the system is badly underdetermined, and the
penalty is doing most of the work. Guessing it wrong is expensive in both
directions -- too high compresses the ratings and flattens every projected
game toward a coin flip; too low overfits 45 noisy spreads and invents
confidence that is not there.

The right question is not "which ridge looks reasonable" but "which ridge
predicts spreads it has not seen". Two ways to ask it:

  current    K-fold CV on THIS season's posted games. Fit on most of them,
             predict the held-out ones, measure error. Directly on-target,
             but noisy: 45 games in 5 folds is 9 held out per fold.

  historical Same procedure on past seasons, restricted to the SAME NUMBER
             of priced games we currently have. Much more stable, and it
             answers the regime question -- the best ridge at 45 games is
             not the best ridge at 200, so this rescales automatically as
             the season accumulates lines.

Run both. If they disagree, trust `historical` for the level and `current`
for the direction.

    python ridge_cv.py --season 2026 --week 2
    python ridge_cv.py --season 2026 --week 2 --hist 2015 2025
"""
import argparse
import numpy as np
import pandas as pd

from ratings import PowerRatings

GRID = [0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0]


def cv_error(games, teams, ridge, folds=5, seed=0, weights=None):
    """
    K-fold CV: fit ratings on the training folds, predict held-out spreads.
    Returns (MAE, RMSE) in points of spread.
    """
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(games))
    parts = np.array_split(idx, folds)
    errs = []
    for k in range(folds):
        te = parts[k]
        tr = np.concatenate([parts[j] for j in range(folds) if j != k])
        if len(tr) < 10:
            continue
        gtr = games.iloc[tr]
        w = None if weights is None else np.asarray(weights)[tr]
        pr = PowerRatings(teams, ridge=ridge).fit(gtr, weights=w)
        for _, g in games.iloc[te].iterrows():
            pred = pr.spread(g.home_team, g.away_team)
            errs.append(pred - g.spread_line)
    e = np.array(errs)
    return float(np.mean(np.abs(e))), float(np.sqrt(np.mean(e**2))), len(e)


def _tied(df, col, se_col, rel_tol=0.02):
    """
    Which ridge values are genuinely indistinguishable from the best?

    The textbook one-standard-error rule fails here. Our SE comes mostly
    from BETWEEN-SEASON variance, not from uncertainty about the curve's
    shape, so it can be large (0.21 MAE) while the curve itself is
    well-resolved. Applying 1-SE blindly swept in ridge=1.0 at MAE 2.627
    when the best was 2.485 -- a value that is plainly worse but happened
    to fit inside a wide error bar.

    So require BOTH: within 1 SE of the best AND within `rel_tol` of the
    best MAE in relative terms. The second condition is what stops a wide
    error bar from endorsing a visibly worse value.
    """
    best = df.loc[df[col].idxmin()]
    thresh = min(best[col] + best[se_col], best[col] * (1 + rel_tol))
    return sorted(df[df[col] <= thresh].ridge.tolist())


def _recommend(df, col, se_col, rel_tol=0.02):
    """Largest ridge among the genuine ties -- over-regularising is the
    cheaper error when the system is underdetermined."""
    return max(_tied(df, col, se_col, rel_tol))


def sweep(games, teams, label, folds=5, reps=8):
    """Average CV error over several random fold assignments."""
    rows = []
    for r in GRID:
        maes, rmses = [], []
        for s in range(reps):
            mae, rmse, n = cv_error(games, teams, r, folds=folds, seed=s)
            maes.append(mae); rmses.append(rmse)
        rows.append({"ridge": r, "mae": np.mean(maes),
                     "mae_se": np.std(maes) / np.sqrt(reps),
                     "rmse": np.mean(rmses)})
    df = pd.DataFrame(rows)
    best = df.loc[df.mae.idxmin()]
    print(f"\n{label}")
    print("-" * len(label))
    print(f"  {'ridge':>7}{'MAE':>9}{'+/-':>7}{'RMSE':>9}")
    for _, x in df.iterrows():
        mark = "  <-- best" if x.ridge == best.ridge else ""
        print(f"  {x.ridge:>7.2f}{x.mae:>9.3f}{x.mae_se:>7.3f}"
              f"{x.rmse:>9.3f}{mark}")
    pick = _recommend(df, "mae", "mae_se")
    tied = _tied(df, "mae", "mae_se")
    print(f"  best {best.ridge:g}; within tolerance: "
          f"{[f'{v:g}' for v in tied]}")
    print(f"  -> use --ridge {pick:g}")
    return pick, df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", type=int, default=2026)
    ap.add_argument("--week", type=int, default=2)
    ap.add_argument("--hist", nargs=2, type=int, default=[2015, 2025])
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--reps", type=int, default=8)
    args = ap.parse_args()

    from pipeline import load_schedules

    cur = load_schedules([args.season])
    cur = cur[cur.game_type == "REG"]
    teams = sorted(set(cur.home_team) | set(cur.away_team))
    priced = cur.dropna(subset=["spread_line"])
    n_priced = len(priced)

    print("=" * 62)
    print(f"RIDGE CROSS-VALIDATION — {args.season} week {args.week}")
    print(f"{n_priced} priced games of {len(cur)}")
    print("=" * 62)

    pick_cur, _ = sweep(priced, teams, f"CURRENT SEASON ({n_priced} games)",
                        args.folds, args.reps)

    # historical at matched sample size
    hist_rows = []
    for yr in range(args.hist[0], args.hist[1] + 1):
        h = load_schedules([yr])
        h = h[h.game_type == "REG"].dropna(subset=["spread_line"])
        if len(h) < n_priced + 20:
            continue
        # take the first n_priced games in schedule order — mirrors having
        # only the early weeks priced, which is our actual situation
        h = h.sort_values(["week"]).head(n_priced)
        tms = sorted(set(h.home_team) | set(h.away_team))
        for r in GRID:
            mae, rmse, n = cv_error(h, tms, r, folds=args.folds, seed=yr)
            hist_rows.append({"season": yr, "ridge": r, "mae": mae})
    if hist_rows:
        hd = pd.DataFrame(hist_rows)
        agg = hd.groupby("ridge")["mae"].agg(["mean", "sem"]).reset_index()
        best = agg.loc[agg["mean"].idxmin()]
        print(f"\nHISTORICAL at matched size ({n_priced} games, "
              f"{hd.season.nunique()} seasons)")
        print("-" * 52)
        print(f"  {'ridge':>7}{'MAE':>9}{'+/-':>7}")
        for _, x in agg.iterrows():
            mark = "  <-- best" if x.ridge == best.ridge else ""
            print(f"  {x.ridge:>7.2f}{x['mean']:>9.3f}{x['sem']:>7.3f}{mark}")
        pick_hist = _recommend(agg.rename(columns={"mean": "mae",
                                                    "sem": "mae_se"}),
                               "mae", "mae_se")
        tied = _tied(agg.rename(columns={"mean": "mae", "sem": "mae_se"}),
                     "mae", "mae_se")
        print(f"  best {best.ridge:g}; within tolerance: "
              f"{[f'{v:g}' for v in tied]}")
        print(f"  -> use --ridge {pick_hist:g}")

        print("\n" + "=" * 62)
        if abs(pick_hist - pick_cur) < 1e-9:
            print(f"  Both agree: --ridge {pick_hist:g}")
        else:
            print(f"  current says {pick_cur:g}, historical says {pick_hist:g}.")
            print(f"  Trust historical for the level ({pick_hist:g}) -- it has "
                  f"{hd.season.nunique()}x the data at the same game count.")
        print("  Re-run this every few weeks: the best ridge falls as more")
        print("  lines get posted and the system stops being underdetermined.")
        print("=" * 62)


if __name__ == "__main__":
    main()
