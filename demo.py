"""
End-to-end validation on SYNTHETIC data.

You cannot check a forecasting pipeline against real data and know whether
a null result means "the market is efficient" or "my code is broken". So
this generates seasons from a known data-generating process where the
truth is available, and checks the pipeline recovers it.

Three scenarios:

  A. EFFICIENT MARKET — the spread is the true expected margin.
     The residual model MUST find ~zero edge. If it reports a real R^2
     here, the pipeline is leaking and nothing downstream can be trusted.

  B. INEFFICIENT MARKET — the spread systematically misses a known effect
     (backup QB worth 5.5 pts, unpriced). The residual model MUST find it.
     If it does not, the pipeline lacks power and a real edge would be
     missed.

  C. POWER RATINGS — recover known team strengths from spreads alone.

Run this before pointing anything at real money.
"""
import numpy as np
import pandas as pd

from market import MarketModel, log_loss, ece
from ratings import PowerRatings
from residual import ResidualModel

TEAMS = ["LAR","BUF","SF","KC","SEA","LAC","NE","BAL","GB","DET","DEN","CIN",
         "DAL","PHI","CHI","JAX","HOU","TB","MIN","NYG","IND","NO","PIT","WAS",
         "LV","CAR","NYJ","TEN","ATL","CLE","ARI","MIA"]
SIGMA_TRUE = 13.5
HFA_TRUE = 1.9
RNG = np.random.default_rng(42)


def make_season(season, ratings, qb_effect=0.0, price_qb=True):
    """One synthetic season: random pairings, spreads, outcomes."""
    rows = []
    for week in range(1, 19):
        order = RNG.permutation(TEAMS)
        for i in range(0, 32, 2):
            h, a = order[i], order[i+1]
            qb_h = RNG.random() < 0.05      # 5% of games: backup starting
            qb_a = RNG.random() < 0.05
            true_edge = (ratings[h] - ratings[a] + HFA_TRUE
                         + qb_effect * (-qb_h + qb_a))
            # what the book posts
            priced = (ratings[h] - ratings[a] + HFA_TRUE
                      + (qb_effect * (-qb_h + qb_a) if price_qb else 0.0))
            spread = np.round((priced + RNG.normal(0, 0.6)) * 2) / 2
            margin = true_edge + RNG.normal(0, SIGMA_TRUE)
            margin = float(np.round(margin))
            rows.append(dict(season=season, week=week, home_team=h, away_team=a,
                             spread_line=spread, result=margin,
                             home_win=int(margin > 0),
                             qb_change_home=int(qb_h), qb_change_away=int(qb_a),
                             home_rest=7, away_rest=7))
    return pd.DataFrame(rows)


def scenario(label, price_qb, seasons=8, qb_effect=5.5):
    ratings = {t: RNG.normal(0, 4.0) for t in TEAMS}
    df = pd.concat([make_season(2018+s, ratings, qb_effect, price_qb)
                    for s in range(seasons)], ignore_index=True)
    print("\n" + "=" * 70)
    print(label)
    print("=" * 70)

    mm = MarketModel().fit(df.spread_line, df.home_win)
    p_mkt = mm.predict(df.spread_line)
    print(f"  fitted sigma {mm.sigma:.2f} (true {SIGMA_TRUE})   "
          f"market log loss {log_loss(df.home_win, p_mkt):.4f}   "
          f"ECE {ece(df.home_win, p_mkt)*100:.2f}%")

    rm = ResidualModel(kind="gbm").fit(df)
    rm.report()

    adj = rm.predict_margin_adjustment(df)
    p_blend = mm.predict(df.spread_line + adj)
    print(f"  blended log loss {log_loss(df.home_win, p_blend):.4f} "
          f"(market {log_loss(df.home_win, p_mkt):.4f})")
    return df, ratings


if __name__ == "__main__":
    print("SYNTHETIC PIPELINE VALIDATION")

    # A: market prices everything -> model must find nothing
    scenario("A. EFFICIENT MARKET  (residual model should find NO edge)",
             price_qb=True)

    # B: market ignores QB injuries -> model must find it
    df, true_ratings = scenario(
        "B. INEFFICIENT MARKET  (backup QB worth 5.5 pts, unpriced)",
        price_qb=False)

    # C: recover ratings from spreads
    print("\n" + "=" * 70)
    print("C. POWER RATING RECOVERY (from spreads only, one season of lines)")
    print("=" * 70)
    one = df[df.season == 2018]
    pr = PowerRatings(TEAMS, ridge=0.5).fit(one)
    est = pr.table().set_index("team")["rating"]
    truth = pd.Series(true_ratings) - np.mean(list(true_ratings.values()))
    corr = np.corrcoef(est[TEAMS].values, truth[TEAMS].values)[0, 1]
    mae = np.mean(np.abs(est[TEAMS].values - truth[TEAMS].values))
    print(f"  correlation with true ratings : {corr:.4f}")
    print(f"  mean absolute error           : {mae:.2f} pts")
    print(f"  fitted HFA {pr.hfa:.2f} (true {HFA_TRUE})")
    print("\n  top 6 recovered:")
    for t, r in pr.table().head(6).itertuples(index=False):
        print(f"    {t:<5}{r:+.2f}   (true {truth[t]:+.2f})")
