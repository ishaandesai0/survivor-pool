"""
Layer 2 — market-implied power ratings.

The problem this solves is specific to survivor: you need win probabilities
for ALL 23 picks now, but sportsbooks only post lines for the current week.
There is no Week 15 spread in August.

The fix is to invert the market. If a spread is approximately

    spread_home = rating_home - rating_away + HFA

then a week of posted lines is a linear system in the team ratings, and
ridge regression recovers them. Once you have ratings you can generate a
spread for every remaining game on the schedule, which is exactly what
"Week 18 modeled from power ratings" meant on the grid you found.

Two details that matter:

RIDGE, NOT OLS. One week gives ~16 equations for 32 unknowns — the system
is underdetermined and OLS will happily return garbage. The ridge penalty
pulls unseen teams toward the prior (preseason win totals) instead.

RATINGS DECAY. A Week 1 rating should not be trusted in Week 15. The
`project` method widens sigma with weeks-ahead, so a distant game gets a
win probability pulled toward 0.5. Skipping this is the classic way these
grids end up overconfident about Week 18 — and overconfidence is far more
expensive in survivor than being wrong, because it makes you plan around
picks that were never as safe as the grid claimed.
"""
import numpy as np
import pandas as pd


HFA_DEFAULT = 1.8   # modern NFL home-field edge; fitted from data when possible


class PowerRatings:
    def __init__(self, teams, ridge=2.0, prior=None):
        self.teams = list(teams)
        self.idx = {t: i for i, t in enumerate(self.teams)}
        self.ridge = ridge
        self.prior = np.zeros(len(self.teams)) if prior is None else np.asarray(prior, float)
        self.ratings = self.prior.copy()
        self.hfa = HFA_DEFAULT

    def fit(self, games, fit_hfa=True, weights=None):
        """
        games: DataFrame with home_team, away_team, spread_line
               (spread_line positive = home favoured)
        """
        n = len(self.teams)
        m = len(games)
        X = np.zeros((m, n + 1))
        y = np.zeros(m)
        for k, (_, g) in enumerate(games.iterrows()):
            X[k, self.idx[g.home_team]] = 1.0
            X[k, self.idx[g.away_team]] = -1.0
            X[k, n] = 1.0                      # HFA column
            y[k] = g.spread_line
        if not fit_hfa:
            y = y - self.hfa * X[:, n]
            X = X[:, :n]

        w = np.ones(m) if weights is None else np.asarray(weights, float)
        W = np.diag(w)

        k_cols = X.shape[1]
        P = np.eye(k_cols) * self.ridge
        P[-1, -1] = 1e-6 if fit_hfa else self.ridge   # barely penalise HFA
        target = np.concatenate([self.prior, [self.hfa]])[:k_cols]

        A = X.T @ W @ X + P
        b = X.T @ W @ y + P @ target
        sol = np.linalg.solve(A, b)

        if fit_hfa:
            self.ratings, self.hfa = sol[:n], float(sol[n])
        else:
            self.ratings = sol
        # ratings are only identified up to a constant; centre them
        self.ratings -= self.ratings.mean()
        return self

    def spread(self, home, away):
        return (self.ratings[self.idx[home]] - self.ratings[self.idx[away]]
                + self.hfa)

    def project(self, schedule, current_week, sigma0=13.5, base=3.2,
                decay=0.41, use_posted=True):
        """
        Win probability for every game in `schedule`.

        sigma_eff = sqrt(sigma0^2 + base^2 + (decay*d)^2), d = weeks ahead.

        `base` is the floor of projection error present even ONE week out --
        ridge estimation error plus ordinary line movement. Measured at
        roughly 3.2 pts. It is not zero, and assuming it is makes near-term
        games look far more certain than they are.

        `use_posted`: when a game already has a real posted spread, USE IT.
        A line the market has actually hung beats anything ratings can
        reconstruct, and carries no projection error at all -- so those
        games get sigma0 alone. Throwing away real lines in favour of
        reconstructed ones is pure loss.

        Do not guess base or decay: fit_decay.py measures both by projecting
        past seasons forward and comparing against the lines books posted
        later.
        """
        from scipy.stats import norm
        rows = []
        for _, g in schedule.iterrows():
            posted = g.get("spread_line", np.nan) if use_posted else np.nan
            if pd.notna(posted):
                s, sig, d, src = float(posted), sigma0, 0, "posted"
            else:
                d = max(0, int(g.week) - current_week)
                s = self.spread(g.home_team, g.away_team)
                sig = float(np.sqrt(sigma0**2 + base**2 + (decay * d)**2))
                src = "projected"
            rows.append({"week": g.week, "home_team": g.home_team,
                         "away_team": g.away_team,
                         "proj_spread": s, "sigma": sig,
                         "home_wp": float(norm.cdf(s / sig)),
                         "away_wp": float(norm.cdf(-s / sig)),
                         "weeks_ahead": d, "source": src})
        return pd.DataFrame(rows)

    def table(self):
        return (pd.DataFrame({"team": self.teams, "rating": self.ratings})
                .sort_values("rating", ascending=False).reset_index(drop=True))


def prior_from_win_totals(teams, win_totals, pts_per_win=2.6):
    """
    Preseason anchor. A team's market win total maps to a power rating at
    roughly 2.6 points per win above/below .500 — this is what lets you
    start the season with sane ratings before any lines exist, and it is
    itself market-derived rather than an opinion.
    """
    wt = np.array([win_totals[t] for t in teams], float)
    return (wt - 8.5) * pts_per_win
