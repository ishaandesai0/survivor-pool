"""
Layer 1 — the market baseline.

The NFL closing spread is the single most accurate public forecast of game
outcomes that exists. It aggregates injuries, weather, sharp money and
information no public model has. Any model that does not START from the
line is handicapping itself.

So this module does one thing well: convert a point spread into a
calibrated win probability, and measure how good that conversion is. Every
later layer must beat THIS on out-of-sample log loss, or it should be
thrown away.

Two conversions are fitted:

  probit   P(win) = Phi(spread / sigma), sigma fitted by MLE.
           Simple, smooth, extrapolates safely to spreads never observed.

  isotonic A monotone non-parametric fit of empirical win rate vs spread.
           Captures the NFL key-number structure — margins pile up on 3 and
           7 because of field goals and touchdowns, so the win-probability
           curve has flat spots and jumps that a normal CDF smooths away.
           Better in the dense middle, unreliable in the sparse tails.

The shipped default blends them: isotonic where there's data, probit in the
tails. That is not a hedge, it's using each where it is actually valid.

KNOWN ISSUE: MarketModel.report() evaluates the isotonic fit on its own
training data, so its reliability table shows near-zero error by
construction. Those numbers are meaningless as printed. Fixing it needs a
train/test split. It does not affect which team the models pick.
"""
from dataclasses import dataclass
import numpy as np
from scipy.stats import norm
from scipy.optimize import minimize_scalar
from sklearn.isotonic import IsotonicRegression


# ----------------------------------------------------------------------
# metrics
# ----------------------------------------------------------------------
def log_loss(y, p, eps=1e-9):
    p = np.clip(p, eps, 1 - eps)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def brier(y, p):
    return float(np.mean((p - y) ** 2))


def calibration_table(y, p, bins=10):
    """Reliability: predicted vs actual, per probability bucket."""
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges) - 1, 0, bins - 1)
    rows = []
    for b in range(bins):
        m = idx == b
        if m.sum() == 0:
            continue
        rows.append({"bucket": f"{edges[b]:.1f}-{edges[b+1]:.1f}",
                     "n": int(m.sum()),
                     "predicted": float(p[m].mean()),
                     "actual": float(y[m].mean()),
                     "error": float(p[m].mean() - y[m].mean())})
    return rows


def ece(y, p, bins=10):
    """Expected calibration error — the number that matters for survivor."""
    rows = calibration_table(y, p, bins)
    n = sum(r["n"] for r in rows)
    return float(sum(r["n"] * abs(r["error"]) for r in rows) / n)


# ----------------------------------------------------------------------
# the model
# ----------------------------------------------------------------------
@dataclass
class MarketModel:
    sigma: float = 13.5
    iso: IsotonicRegression = None
    lo: float = -14.0
    hi: float = 14.0

    def fit(self, spread, won):
        """
        spread : positive = this team favoured by that many points
        won    : 1 if this team won (a tie counts per your pool's rule --
                 pass ties as 1 if that's how your league scores them)
        """
        spread = np.asarray(spread, float)
        won = np.asarray(won, float)

        # sigma by maximum likelihood, not by assumption
        def nll(s):
            if s <= 0.5:
                return 1e9
            return log_loss(won, norm.cdf(spread / s))
        self.sigma = float(minimize_scalar(nll, bounds=(5.0, 25.0),
                                           method="bounded").x)

        self.iso = IsotonicRegression(y_min=0.01, y_max=0.99,
                                      out_of_bounds="clip")
        self.iso.fit(spread, won)

        # only trust isotonic where the sample is dense
        self.lo, self.hi = np.percentile(spread, [2.5, 97.5])
        return self

    def probit(self, spread):
        return norm.cdf(np.asarray(spread, float) / self.sigma)

    def predict(self, spread):
        """Isotonic in the dense middle, probit in the tails, blended."""
        spread = np.asarray(spread, float)
        p_probit = self.probit(spread)
        if self.iso is None:
            return p_probit
        p_iso = self.iso.predict(spread)
        # weight -> 1 inside [lo, hi], decaying outside
        w = np.ones_like(spread)
        below = spread < self.lo
        above = spread > self.hi
        w[below] = np.exp(-(self.lo - spread[below]) / 3.0)
        w[above] = np.exp(-(spread[above] - self.hi) / 3.0)
        return w * p_iso + (1 - w) * p_probit

    def report(self, spread, won, label="market baseline"):
        p = self.predict(spread)
        print(f"\n{label}")
        print("-" * len(label))
        print(f"  fitted sigma            : {self.sigma:.2f} pts")
        print(f"  log loss                : {log_loss(won, p):.4f}")
        print(f"  brier                   : {brier(won, p):.4f}")
        print(f"  expected calib. error   : {ece(won, p)*100:.2f}%")
        print(f"  accuracy (p>.5)         : {np.mean((p > .5) == (np.asarray(won) == 1))*100:.1f}%")
        print("\n  reliability (IN-SAMPLE — see KNOWN ISSUE above; the")
        print("  near-zero errors are isotonic fitting its own data):")
        print(f"    {'bucket':<12}{'n':>6}{'pred':>8}{'actual':>8}{'err':>8}")
        for r in calibration_table(won, p):
            print(f"    {r['bucket']:<12}{r['n']:>6}{r['predicted']:>8.3f}"
                  f"{r['actual']:>8.3f}{r['error']:>+8.3f}")
        return p


def moneyline_to_prob(ml):
    """
    Raw implied probability from American odds (vig still included).

    np.where evaluates BOTH branches for every element, so a moneyline of
    exactly -100 (a true pick'em, and it does appear on real slates) makes
    the positive-odds branch divide by zero. The selected value is still
    correct, but the warning would mask a genuine numerical problem later,
    so compute each branch only where it applies.
    """
    ml = np.asarray(ml, float)
    out = np.empty_like(ml)
    neg = ml < 0
    out[neg] = -ml[neg] / (-ml[neg] + 100.0)
    out[~neg] = 100.0 / (ml[~neg] + 100.0)
    return out if out.ndim else float(out)


def devig(p_home, p_away, method="multiplicative"):
    """
    Sportsbook prices include vig, so raw implied probabilities sum to >1.
    Removing it matters: at a 4.5% hold, a raw 0.85 favourite is really
    about 0.81, and in a survivor pool that gap compounds across 23 picks.
    """
    p_home, p_away = np.asarray(p_home, float), np.asarray(p_away, float)
    if method == "multiplicative":
        s = p_home + p_away
        return p_home / s, p_away / s
    if method == "shin":                      # better for lopsided markets
        s = p_home + p_away
        z = (s - 1) / (s - (p_home**2 + p_away**2) / s)
        f = lambda p: (np.sqrt(z**2 + 4*(1-z)*p**2/s) - z) / (2*(1-z))
        return f(p_home), f(p_away)
    raise ValueError(method)
