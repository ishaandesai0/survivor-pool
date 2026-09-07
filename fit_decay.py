"""
Fit the projection-decay parameter from history instead of guessing it.

The question: if I fit power ratings from the first N weeks of a season,
how wrong are my projected spreads for a game d weeks later?

Method, per historical season:
  1. Fit ratings on weeks 1..N using posted lines only.
  2. Project a spread for every later game that has a posted line.
  3. Record (projected - actual_posted_line) by weeks-ahead d.

We compare against the LINE, not the game result. The later line is the
market's best estimate at the time, so this isolates how fast our rating
snapshot goes stale — which is exactly what the decay term models. Comparing
to results instead would mix in the ~13.5 pts of irreducible game noise and
inflate the answer.

Then fit  sd(error at d)  =  a * d   and report `a` as the decay.

Run:
    python fit_decay.py --train 2015 2025 --fit-weeks 4
"""
import argparse
import numpy as np
import pandas as pd

from pipeline import load_schedules
from ratings import PowerRatings


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", nargs=2, type=int, default=[2015, 2025])
    ap.add_argument("--fit-weeks", type=int, default=4,
                    help="weeks of lines used to fit ratings")
    ap.add_argument("--ridge", type=float, default=2.0)
    args = ap.parse_args()

    rows = []
    for season in range(args.train[0], args.train[1] + 1):
        sch = load_schedules([season])
        sch = sch[(sch.game_type == "REG")].dropna(subset=["spread_line"])
        if sch.empty:
            continue
        teams = sorted(set(sch.home_team) | set(sch.away_team))
        early = sch[sch.week <= args.fit_weeks]
        later = sch[sch.week > args.fit_weeks]
        if len(early) < 40 or later.empty:
            continue

        pr = PowerRatings(teams, ridge=args.ridge).fit(early)
        for _, g in later.iterrows():
            proj = pr.spread(g.home_team, g.away_team)
            rows.append({"season": season,
                         "d": int(g.week) - args.fit_weeks,
                         "err": proj - g.spread_line})

    df = pd.DataFrame(rows)
    if df.empty:
        raise SystemExit("No data — check season range and line availability.")

    print("=" * 62)
    print(f"PROJECTION ERROR vs WEEKS AHEAD  "
          f"({args.train[0]}-{args.train[1]}, ratings fit on weeks 1-{args.fit_weeks})")
    print("=" * 62)
    print(f"  {'d':>3}{'n':>7}{'bias':>9}{'sd':>9}")
    g = df.groupby("d")["err"]
    for d, s in g:
        print(f"  {d:>3}{len(s):>7}{s.mean():>+9.2f}{s.std():>9.2f}")

    # Fit sd(d) = sqrt(base^2 + (decay*d)^2), weighted by sample size.
    # A line through the origin does NOT fit this data: there is a floor of
    # a few points of error even one week out, from ridge estimation error
    # plus ordinary week-to-week line movement. Forcing the intercept to
    # zero underestimates near-term uncertainty badly and overestimates the
    # far end to compensate.
    from scipy.optimize import curve_fit
    tab = df.groupby("d")["err"].agg(["std", "count"]).reset_index().dropna()
    d = tab["d"].values.astype(float)
    sd = tab["std"].values
    w = tab["count"].values.astype(float)

    def model(d, base, decay):
        return np.sqrt(base**2 + (decay * d)**2)

    (base, decay), _ = curve_fit(model, d, sd, p0=[3.0, 0.4],
                                 sigma=1.0/np.sqrt(w), maxfev=20000)
    base, decay = abs(float(base)), abs(float(decay))

    # how much better than through-origin?
    a_origin = float(np.sum(w * d * sd) / np.sum(w * d * d))
    rss_2p = float(np.sum(w * (sd - model(d, base, decay))**2))
    rss_1p = float(np.sum(w * (sd - a_origin * d)**2))

    print("\n" + "=" * 62)
    print(f"  fitted base  = {base:.3f} pts (floor, present even 1 week out)")
    print(f"  fitted decay = {decay:.3f} pts of sigma per week ahead")
    print(f"  weighted RSS: 2-param {rss_2p:.1f}  vs  through-origin {rss_1p:.1f}")
    print(f"  overall sd of projection error: {df.err.std():.2f} pts")
    print(f"  overall bias                  : {df.err.mean():+.2f} pts")
    print("\n  fit check:")
    print(f"    {'d':>3}{'actual':>9}{'model':>9}")
    for dd, ss in zip(d, sd):
        print(f"    {int(dd):>3}{ss:>9.2f}{model(dd, base, decay):>9.2f}")
    print("=" * 62)
    print(f"\nIn pipeline.py:  pr.project(..., base={base:.3f}, decay={decay:.3f})")
    if abs(df.err.mean()) > 0.5:
        print("\nWARNING: non-trivial bias — your ridge prior is pulling ratings")
        print("off-centre. Consider lowering --ridge or raising --fit-weeks.")


if __name__ == "__main__":
    main()
