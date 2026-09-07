"""
Which of your 23 picks are actually decided, and which are noise?

You are going to refit the model every week as lines arrive. Some picks will
survive every refit and some will flip constantly. Knowing which is which
tells you what to plan around and what to ignore until it's close.

Method: resample the grid many times, perturbing each game by the SAME
projection uncertainty the model already estimates (base + decay*d from
fit_decay.py), re-solve the optimal 23-pick path on each draw, and count how
often each team-week assignment survives.

  - a pick chosen in >80% of draws is structural: the schedule, not the noise
  - a pick chosen in <40% of draws is a coin flip you should not plan around

The useful output is your COMMITMENT HORIZON: the point beyond which the
plan stops being information. Planning past it is wasted effort, and worse,
it makes you protect teams for slots that will not exist by the time you get
there.

Run:
    python stability.py --grid win_probs_model.csv --week 1 --draws 400
"""
import argparse
import numpy as np
import pandas as pd
from collections import Counter, defaultdict
from scipy.stats import norm

from qc_grid import load
from slots_model import slot_list, solve_slots, DOUBLE_WEEKS


def perturb(df, current_week, sigma0=11.16, base=3.717, decay=0.355,
            posted_through=None, rng=None):
    """
    Resample the grid by jittering each game's implied spread by its own
    projection error, then re-deriving both sides' probabilities. Perturbing
    one side and taking the complement keeps every game internally
    consistent -- the same constraint the QC step relies on.
    """
    from qc_grid import recover_matchups
    out = df.astype(float).copy()   # int64 grids reject fractional writes
    for w in df.columns:
        # Games with a POSTED line carry no projection error -- they are the
        # market's actual answer, not a reconstruction. Jittering them would
        # manufacture uncertainty that does not exist and make firm picks
        # look like coin flips.
        if posted_through is not None and int(w) <= posted_through:
            continue
        d = max(0, int(w) - current_week)
        sig_proj = np.sqrt(base**2 + (decay * d)**2)
        if sig_proj < 1e-6:
            continue
        for a, b, pa, pb, _ in recover_matchups(df, w)[0]:
            p = np.clip(pa / (pa + pb), 1e-4, 1 - 1e-4)
            spread = sigma0 * norm.ppf(p)
            spread += rng.normal(0, sig_proj)
            pn = float(norm.cdf(spread / sigma0))
            out.loc[a, w] = round(pn * 100, 1)
            out.loc[b, w] = round((1 - pn) * 100, 1)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", default="win_probs_model.csv")
    ap.add_argument("--week", type=int, default=1)
    ap.add_argument("--draws", type=int, default=400)
    ap.add_argument("--base", type=float, default=3.717)
    ap.add_argument("--decay", type=float, default=0.355)
    ap.add_argument("--sigma0", type=float, default=11.16)
    ap.add_argument("--posted-through", type=int, default=None,
                    help="last week with real posted lines; those weeks are "
                         "not perturbed. Check pipeline.py's posted count.")
    args = ap.parse_args()

    df = load(args.grid)
    rng = np.random.default_rng(17)
    slots = slot_list()

    base_picks, _ = solve_slots(df)
    base_map = {}
    seen = defaultdict(int)
    for t, w in zip(base_picks, slots):
        seen[w] += 1
        base_map.setdefault(w, []).append(t)

    counts = defaultdict(Counter)     # week -> Counter(team)
    for _ in range(args.draws):
        g = perturb(df, args.week, args.sigma0, args.base, args.decay,
                    args.posted_through, rng)
        picks, _ = solve_slots(g)
        if picks is None:
            continue
        for t, w in zip(picks, slots):
            counts[w][t] += 1

    print("=" * 72)
    print(f"PICK STABILITY — {args.draws} resampled grids, from week {args.week}")
    print("=" * 72)
    print(f"  {'wk':>3} {'baseline':<12}{'agree':>7}   alternatives")
    print("  " + "-" * 66)

    horizon = None
    for w in sorted(counts):
        n_slot = 2 if w in DOUBLE_WEEKS else 1
        tot = args.draws * n_slot
        base_teams = base_map.get(w, [])
        agree = sum(counts[w][t] for t in base_teams) / tot
        alts = [f"{t} {100*c/tot:.0f}%" for t, c in counts[w].most_common(4)
                if t not in base_teams and c / tot > 0.10]
        flag = ""
        if agree < 0.40 and horizon is None:
            horizon = w
            flag = "  <-- noise starts here"
        print(f"  {w:>3} {'+'.join(base_teams):<12}{agree*100:>6.0f}%   "
              f"{', '.join(alts) if alts else '-'}{flag}")

    print("\n" + "=" * 72)
    if horizon and horizon > args.week:
        print(f"  COMMITMENT HORIZON: weeks {args.week}-{horizon-1} are real information.")
        print(f"  From week {horizon} on, the plan is a placeholder. Do not protect")
        print(f"  teams for those slots -- refit and decide when you get there.")
    elif horizon:
        print(f"  COMMITMENT HORIZON: even week {args.week} is unstable.")
        print("  Either your posted-through setting is too low, or the top few")
        print("  options really are near-equivalent -- check the 'cost' column")
        print("  in weekly.py: if the top picks are within a few percent, the")
        print("  choice genuinely does not matter much.")
    else:
        print("  Every week stayed above the noise threshold.")

    print("\n  TEAM FLEXIBILITY (how many weeks a team is a plausible pick)")
    print("  low = narrow window, use them when it comes or lose the value")
    flex = Counter()
    for w in counts:
        for t, c in counts[w].items():
            n_slot = 2 if w in DOUBLE_WEEKS else 1
            if c / (args.draws * n_slot) > 0.10:
                flex[t] += 1
    rows = sorted(flex.items(), key=lambda x: x[1])
    print(f"    {'narrow':<28}{'flexible':<28}")
    narrow = [f"{t} ({n}wk)" for t, n in rows[:8]]
    wide = [f"{t} ({n}wk)" for t, n in rows[-8:][::-1]]
    for i in range(max(len(narrow), len(wide))):
        l = narrow[i] if i < len(narrow) else ""
        r = wide[i] if i < len(wide) else ""
        print(f"    {l:<28}{r:<28}")
    print("=" * 72)


if __name__ == "__main__":
    main()
