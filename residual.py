"""
Layer 3 — the machine learning layer, pointed at the right target.

The mistake almost everyone makes is training a classifier on
"did the home team win" using team stats. That model relearns, badly, what
the spread already encodes, and then loses to the spread.

Instead, model the market's ERROR:

    target = actual_margin - spread_line

The market's residual is, by construction, what the line does not already
know. If your features have no edge, this target is pure noise and the
model correctly learns to predict ~0 — which is a useful, honest answer
rather than a false one. If they do have an edge, you find it cleanly.

Set your expectations correctly: the NFL closing line is close to
efficient. A realistic result is R^2 near zero on held-out seasons. Beating
the closing line by even 0.3 points of margin is a genuinely strong result,
and most published attempts do not survive walk-forward validation. This
module is built to detect that outcome rather than hide it.

The realistic prize is CALIBRATION, not accuracy. For survivor you do not
need to know who wins better than Vegas — you need probabilities whose 80%
bucket actually goes 80%. That is what the whole pipeline is optimising.
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import GroupKFold


FEATURES = [
    "rest_diff",         # home_rest - away_rest
    "off_bye_home", "off_bye_away",
    "short_week_home", "short_week_away",
    "travel_km", "tz_shift",
    "div_game", "primetime",
    "qb_change_home", "qb_change_away",   # backup starting: the biggest one
    "wind", "temp", "is_dome",
    "epa_off_diff", "epa_def_diff",       # recent form the line may lag
    "spread_line",                        # lets the model find spread-dependent bias
]


def build_features(games):
    """
    Expects a schedule frame with the raw columns. Everything here is
    knowable BEFORE kickoff — leakage is the fastest way to build a model
    that backtests beautifully and loses money.
    """
    df = games.copy()
    df["rest_diff"] = df.get("home_rest", 7) - df.get("away_rest", 7)
    df["off_bye_home"] = (df.get("home_rest", 7) >= 13).astype(int)
    df["off_bye_away"] = (df.get("away_rest", 7) >= 13).astype(int)
    df["short_week_home"] = (df.get("home_rest", 7) <= 4).astype(int)
    df["short_week_away"] = (df.get("away_rest", 7) <= 4).astype(int)
    for c in FEATURES:
        if c not in df.columns:
            df[c] = 0.0
    return df


class ResidualModel:
    """
    Fits market error. `shrink` is deliberately aggressive: even a genuine
    signal should be applied at a fraction of its fitted size, because the
    fit is noisy and the cost of over-trusting it is asymmetric.
    """

    def __init__(self, kind="gbm", shrink=None):
        self.kind = kind
        self.shrink = shrink
        self.model = None
        self.cv_r2 = None

    def _new(self):
        if self.kind == "ridge":
            return RidgeCV(alphas=np.logspace(-2, 3, 30))
        return HistGradientBoostingRegressor(
            max_depth=3, max_iter=300, learning_rate=0.03,
            min_samples_leaf=60, l2_regularization=1.0, random_state=0)

    def fit(self, df, seasons=None):
        X = build_features(df)[FEATURES].astype(float).values
        y = (df["result"] - df["spread_line"]).astype(float).values
        groups = df["season"].values if seasons is None else seasons

        # walk-forward-ish CV to estimate real predictive power
        gkf = GroupKFold(n_splits=min(5, len(np.unique(groups))))
        preds = np.zeros_like(y)
        for tr, te in gkf.split(X, y, groups):
            m = self._new().fit(X[tr], y[tr])
            preds[te] = m.predict(X[te])
        ss_res = float(np.sum((y - preds) ** 2))
        ss_tot = float(np.sum((y - y.mean()) ** 2))
        self.cv_r2 = 1 - ss_res / ss_tot

        # shrinkage from the CV fit: regress truth on prediction.
        # If the model is noise, this coefficient collapses toward 0 on its own.
        denom = float(np.sum(preds ** 2))
        auto = float(np.sum(preds * y) / denom) if denom > 1e-9 else 0.0
        self.auto_shrink = float(np.clip(auto, 0.0, 1.0))

        # Significance of that coefficient is the RIGHT detector, not R^2.
        # A real but narrow edge — say a backup QB worth 5.5 pts appearing in
        # 10% of games against 13.5 pts of noise — produces an R^2 of
        # essentially zero while still being worth real money. R^2 measures
        # variance explained; we care whether the coefficient is reliably
        # non-zero, which is a t-test.
        if denom > 1e-9:
            resid_var = float(np.sum((y - auto * preds) ** 2) / max(len(y) - 2, 1))
            se = float(np.sqrt(resid_var / denom))
            self.t_stat = auto / se if se > 0 else 0.0
        else:
            self.t_stat = 0.0

        if self.shrink is None:
            self.shrink = self.auto_shrink

        self.model = self._new().fit(X, y)
        return self

    def predict_margin_adjustment(self, df):
        X = build_features(df)[FEATURES].astype(float).values
        return self.shrink * self.model.predict(X)

    def report(self):
        print("\nresidual model")
        print("--------------")
        print(f"  out-of-sample R^2 on market error : {self.cv_r2:+.4f}")
        print(f"  auto-fitted shrinkage             : {self.auto_shrink:.3f}")
        print(f"  t-stat on shrinkage               : {self.t_stat:+.2f}")
        if self.t_stat < 2.0:
            print("  VERDICT: no reliable edge over the closing line.")
            print("  This is the expected and honest result. Use the market")
            print("  probabilities directly; the value you added is calibration")
            print("  and full-season projection, not beating Vegas.")
        else:
            print(f"  VERDICT: statistically real edge (t={self.t_stat:.1f}).")
            print(f"  Applying it at {self.shrink:.2f}x fitted size.")
            print("  Note the tiny R^2 is expected and not a problem: a narrow")
            print("  edge against 13.5 pts of game noise explains almost no")
            print("  variance while still being worth money.")


def blend(spread_line, adjustment, market_model):
    """Market spread, nudged by the residual model, then calibrated."""
    return market_model.predict(np.asarray(spread_line) + np.asarray(adjustment))
