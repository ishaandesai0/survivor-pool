"""
Expected dollars from the ACTUAL current state, with the field's behaviour
fitted to what it actually did.

prize23.py simulates rivals with a softmax over win probability and a
chalkiness parameter I picked out of the air. Two weeks of real data show
that model is wrong in a specific, repeatable way:

    Week 1: SF/LAC was the biggest favourite; the field went JAX 77 - 58
    Week 2: SF at 87% was the biggest favourite; the field went TB 88 - 57

Twice the pool preferred the SECOND-best option. A pure softmax on win
probability cannot produce that -- it is monotone, so the biggest favourite
is always the modal pick. Something else is going on: entrants saving the
elite teams, avoiding a perceived trap, or just following each other.

So this module does three things prize23.py cannot:

1. FITS the popularity model to observed picks by maximum likelihood,
   instead of assuming one. It fits both a chalk parameter and a
   "favourite aversion" term that lets the modal pick sit below the top.

2. STARTS FROM REALITY. Every entrant's burned teams and strike count come
   from field_state.csv, not from a simulated history. With 60 clean, 113
   on one strike and 27 on two, the distribution of who can afford risk is
   wildly uneven, and that drives the endgame.

3. SCORES YOUR ACTUAL POSITION rather than a hypothetical clean entry.

Requires field_state.csv covering every week played:
    python field.py parse sheets/week1_sheet.txt --week 1
    python field.py parse sheets/week2_sheet.txt --week 2
    python prize_live.py --me "Ishaan" --week 3
"""
import argparse
import sys

import numpy as np
import pandas as pd
from scipy.optimize import minimize

DOUBLE_WEEKS = {5, 7, 10, 12, 15}
OVERALL = [1500, 800, 600, 400, 200]
DIV_PRIZE, N_DIV = 150, 10


def load_grid(path):
    df = pd.read_csv(path).set_index("team")
    df.columns = [int(c) for c in df.columns]
    return df


def slots_from(week, horizon=18):
    s = []
    for w in range(week, horizon + 1):
        s.append(w)
        if w in DOUBLE_WEEKS:
            s.append(w)
    return s


# ----------------------------------------------------------------------
# 1. fit the field's behaviour
# ----------------------------------------------------------------------
def fit_popularity(field, grid, weeks):
    """
    Two-parameter choice model fitted by MLE on observed picks.

        utility(team) = chalk * p  -  aversion * max(0, p - p_top + margin)

    The second term is the part a plain softmax cannot express: it penalises
    the very top of the board, which is what produces a modal pick sitting
    just below the biggest favourite. If the field has no such tendency the
    fitted aversion goes to zero on its own and this reduces to the old
    model, so it costs nothing to include.
    """
    obs = []
    for w in weeks:
        sub = field[field.week == w]
        if sub.empty:
            continue
        avail = grid[w].dropna()
        if avail.empty:
            continue
        teams = list(avail.index)
        p = avail.values / 100.0
        counts = np.array([int((sub.team == t).sum()) for t in teams], float)
        if counts.sum() == 0:
            continue
        obs.append((p, counts))
    if not obs:
        return None

    def nll(theta):
        chalk, aversion = theta
        if chalk <= 0 or aversion < 0:
            return 1e9
        tot = 0.0
        for p, c in obs:
            u = chalk * p - aversion * np.maximum(0.0, p - p.max() + 0.06)
            u -= u.max()
            q = np.exp(u); q /= q.sum()
            tot -= float((c * np.log(np.clip(q, 1e-12, 1))).sum())
        return tot

    best, bx = None, None
    for x0 in [(8, 0), (14, 5), (20, 15), (6, 2)]:
        r = minimize(nll, x0, method="Nelder-Mead",
                     options={"maxiter": 2000, "xatol": 1e-3, "fatol": 1e-3})
        if best is None or r.fun < best:
            best, bx = r.fun, r.x
    chalk, aversion = float(bx[0]), float(max(bx[1], 0.0))

    # compare against the aversion-free model to see if it earns its place
    def nll1(k):
        return nll((k[0], 0.0))
    r1 = minimize(nll1, [chalk], method="Nelder-Mead",
                  options={"maxiter": 1000})
    return {"chalk": chalk, "aversion": aversion,
            "nll": best, "nll_no_aversion": float(r1.fun),
            "chalk_only": float(r1.x[0]), "n_weeks": len(obs)}


def field_probs(p_avail, chalk, aversion):
    u = chalk * p_avail - aversion * np.maximum(0.0, p_avail - p_avail.max() + 0.06)
    u -= u.max()
    q = np.exp(u)
    return q / q.sum()


# ----------------------------------------------------------------------
# 2. simulate forward from the real state
# ----------------------------------------------------------------------
def optimal_path(grid, week, used):
    """Hungarian-optimal remaining path -- what YOU actually play."""
    from scipy.optimize import linear_sum_assignment
    slots = slots_from(week)
    avail = [t for t in grid.index if t not in used]
    C = np.full((len(avail), len(slots)), 1e6)
    for i, t in enumerate(avail):
        for j, w in enumerate(slots):
            p = grid.loc[t, w]
            if pd.notna(p) and p > 0:
                C[i, j] = -np.log(p / 100.0)
    r, c = linear_sum_assignment(C)
    if C[r, c].max() >= 1e6:
        return None
    out = [None] * len(slots)
    for a, b in zip(r, c):
        out[b] = avail[a]
    return out


def simulate(grid, entrants, me_idx, week, params, n_sims, seed=0,
             my_path=None):
    """
    entrants: list of dicts {name, used:set, strikes:int}
    Returns payout stats for `me_idx`.
    """
    rng = np.random.default_rng(seed)
    teams = list(grid.index)
    tidx = {t: i for i, t in enumerate(teams)}
    n_t = len(teams)
    slots = slots_from(week)
    S = len(slots)
    N = len(entrants)

    from qc_grid import recover_matchups
    P = np.full((n_t, S), np.nan)
    OPP = np.full((n_t, S), -1, int)
    pair_cache = {}
    for j, w in enumerate(slots):
        col = grid[w]
        for t in teams:
            if pd.notna(col.get(t)):
                P[tidx[t], j] = col[t] / 100.0
        if w not in pair_cache:
            pair_cache[w] = recover_matchups(grid, w)[0]
        for a, b, *_ in pair_cache[w]:
            OPP[tidx[a], j], OPP[tidx[b], j] = tidx[b], tidx[a]

    base_used = np.zeros((N, n_t), bool)
    base_str = np.zeros(N, int)
    for i, e in enumerate(entrants):
        for t in e["used"]:
            if t in tidx:
                base_used[i, tidx[t]] = True
        base_str[i] = e["strikes"]

    chalk, aversion = params["chalk"], params["aversion"]
    pay_tot = 0.0
    top5 = div = 0
    my_final = []
    all_pay = np.zeros(N)          # audit: EV per entrant
    sim_totals = []                # audit: dollars paid out each season
    final_strikes = np.zeros((0, 0))
    strike_log = []

    for _ in range(n_sims):
        used = base_used.copy()
        st = base_str.copy()
        alive = st < 3
        # Entrants who arrive ALREADY eliminated never enter the pick loop,
        # so their depth stays at its initial value. Initialising to S (the
        # full slate) made the ranking key -depth*100 + st treat them as
        # having survived every slot -- dead players sorted to the TOP and
        # collected prize money. Depth 0 puts them last, where they belong.
        depth = np.where(alive, S, 0).astype(int)

        for j in range(S):
            ok = np.isfinite(P[:, j])
            # ONE COIN PER GAME. Drawing per team lets both sides of a
            # matchup win, which destroys the anti-correlation that makes
            # survivor pools bust in clusters -- and clustering is the whole
            # reason 100 entrants took a strike in week 2.
            won = np.zeros(n_t, bool)
            done = np.zeros(n_t, bool)
            for a in range(n_t):
                if done[a] or not ok[a]:
                    continue
                b = OPP[a, j]
                u = rng.random()
                if b < 0:
                    won[a] = u < P[a, j]
                    done[a] = True
                else:
                    aw = u < P[a, j]
                    won[a], won[b] = aw, not aw
                    done[a] = done[b] = True
            act = np.where(alive)[0]
            if not act.size:
                break
            avail_mask = (~used[act]) & ok[None, :]
            pav = np.where(avail_mask, P[:, j][None, :], np.nan)
            for k, i in enumerate(act):
                row = pav[k]
                good = np.where(np.isfinite(row))[0]
                if good.size == 0:
                    alive[i] = False; depth[i] = j; continue
                if i == me_idx and my_path is not None:
                    # You are not a random member of the field. You run the
                    # Hungarian-optimal path. Simulating yourself with the
                    # rivals' choice model measures THEIR strategy, not
                    # yours, and understates your EV badly.
                    want = my_path[j] if j < len(my_path) else None
                    pick = (tidx[want] if want in tidx and
                            np.isfinite(row[tidx[want]]) else
                            good[np.nanargmax(row[good])])
                else:
                    q = field_probs(row[good], chalk, aversion)
                    pick = rng.choice(good, p=q)
                used[i, pick] = True
                if not won[pick]:
                    st[i] += 1
                    if st[i] >= 3:
                        alive[i] = False; depth[i] = j + 1

        key = -depth.astype(float) * 100 + st
        order = np.argsort(key, kind="stable")
        pay = np.zeros(N)
        i = place = 0
        while i < N and place < 5:
            j2 = i
            while j2 + 1 < N and key[order[j2 + 1]] == key[order[i]]:
                j2 += 1
            grp = order[i:j2 + 1]
            pot = sum(OVERALL[s] for s in range(place, min(place + len(grp), 5)))
            if pot:
                pay[grp] += pot / len(grp)
            place += len(grp); i = j2 + 1
        before = pay[me_idx]
        divs = rng.permutation(N) % N_DIV
        for d in range(N_DIV):
            m = np.where(divs == d)[0]
            w_ = m[key[m] == key[m].min()]
            pay[w_] += DIV_PRIZE / len(w_)

        pay_tot += pay[me_idx]
        top5 += before > 0
        div += (pay[me_idx] - before) > 0
        my_final.append(st[me_idx])
        all_pay += pay
        sim_totals.append(float(pay.sum()))
        strike_log.append(st.copy())

    return {"ev": pay_tot / n_sims, "top5": 100 * top5 / n_sims,
            "div": 100 * div / n_sims,
            "survive": 100 * np.mean(np.array(my_final) <= 2),
            "all_ev": all_pay / n_sims,
            "sim_totals": np.array(sim_totals),
            "strikes_final": np.array(strike_log),
            "start_strikes": base_str}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", default="win_probs_model.csv")
    ap.add_argument("--state", default="field_state.csv")
    ap.add_argument("--me", default="Ishaan")
    ap.add_argument("--week", type=int, required=True)
    ap.add_argument("--sims", type=int, default=300)
    ap.add_argument("--audit", action="store_true",
                    help="verify the sim is a closed system: full pot paid "
                         "every season, mean EV = fair share, EV monotone "
                         "in strikes")
    ap.add_argument("--results", default="results_2026.csv",
                    help="apply the latest week's results on top of the "
                         "sheet, which only shows strikes ENTERING that week")
    args = ap.parse_args()

    grid = load_grid(args.grid)
    try:
        field = pd.read_csv(args.state)
    except FileNotFoundError:
        sys.exit(f"no {args.state} — run field.py parse for each week first")

    weeks = sorted(field.week.unique())
    print("=" * 68)
    print(f"LIVE POOL MODEL — through week {max(weeks)}, "
          f"projecting from week {args.week}")
    print("=" * 68)
    if max(weeks) < args.week - 1:
        print(f"  WARNING: field_state.csv only covers weeks {weeks}.")
        print(f"  Parse the week {args.week - 1} sheet or the field state is stale.\n")

    # ---- fit the field ------------------------------------------------
    fit = fit_popularity(field, grid, weeks)
    if fit is None:
        sys.exit("could not fit popularity — no overlapping weeks")
    print(f"\nFIELD BEHAVIOUR (fitted on {fit['n_weeks']} week(s) of real picks)")
    print("-" * 52)
    print(f"  chalk           : {fit['chalk']:.2f}")
    print(f"  favourite-aversion: {fit['aversion']:.2f}")
    d = fit["nll_no_aversion"] - fit["nll"]
    print(f"  log-likelihood gain over plain softmax: {d:.1f}")
    if d > 3:
        print("  -> the aversion term earns its place.")
    else:
        print("  -> aversion adds nothing over plain chalk.")

    # Goodness of fit matters more than which variant wins. If neither
    # reproduces the modal pick, the whole family is wrong and the EV below
    # inherits that error -- better to say so than to print a confident
    # dollar figure resting on a model that does not describe the field.
    print("\n  fit check (predicted vs actual pick counts):")
    worst = 0.0
    for w in weeks:
        sub = field[field.week == w]
        avail = grid[w].dropna()
        if sub.empty or avail.empty:
            continue
        tms = list(avail.index)
        p = avail.values / 100.0
        c = np.array([int((sub.team == t).sum()) for t in tms], float)
        q = field_probs(p, fit["chalk"], fit["aversion"]) * c.sum()
        order = np.argsort(-c)[:3]
        line = "   ".join(f"{tms[i]} {int(c[i])}/{q[i]:.0f}" for i in order)
        print(f"    W{w}: {line}   (actual/predicted)")
        for i in order[:1]:
            worst = max(worst, abs(c[i] - q[i]) / max(c[i], 1))
    if worst > 0.35:
        print(f"\n  *** MODEL DOES NOT FIT. Modal pick off by "
              f"{worst*100:.0f}%. ***")
        print("  The field is not choosing on win probability alone, so the")
        print("  EV below is indicative at best. Treat the ORDERING of your")
        print("  options as meaningful and the dollar figure as rough.")
        print("  More weeks of sheets may reveal the pattern; two do not.")

    # ---- build entrant state -----------------------------------------
    # The sheet's X column is CUMULATIVE strikes entering that week, so
    # summing across weeks double-counts anyone who struck early. Use the
    # latest sheet's value, then apply the latest week's results on top --
    # the sheet cannot know them yet.
    latest = int(field.week.max())
    last = field[field.week == latest].set_index("name")
    ents = []
    dropped = 0
    for nm, g in field.groupby("name"):
        # Eliminated entrants are REMOVED from later sheets. `.get(nm, 0)`
        # therefore reads them as CLEAN -- the 12 people already out would be
        # simulated as zero-strike rivals with a full roster of teams,
        # inflating the field and understating your EV. Absence from the
        # latest sheet means eliminated, not fresh.
        if nm not in last.index:
            dropped += 1
            continue
        base = int(last.strike.get(nm, 0))
        extra = 0
        if args.results:
            res = pd.read_csv(args.results)
            res = res[res.week == latest]
            winners = set(res.winner)
            played = set(res.away) | set(res.home)
            pick = last.team.get(nm)
            if pick in played and pick not in winners:
                extra = 1
        ents.append({"name": nm, "used": set(g.team),
                     "strikes": min(base + extra, 3)})
    me = [i for i, e in enumerate(ents)
          if args.me.lower() in e["name"].lower()]
    if not me:
        sys.exit(f"'{args.me}' not found in {args.state}")
    me = me[0]
    dist = pd.Series([e["strikes"] for e in ents]).value_counts().sort_index()
    print(f"\nFIELD STATE — {len(ents)} entrants"
          + (f" ({dropped} eliminated, dropped from the sheet)"
             if dropped else ""))
    print("-" * 52)
    for k, v in dist.items():
        lab = f"{k} strikes" if k < 3 else "3+ out"
        mark = "  <-- you" if k == ents[me]["strikes"] else ""
        print(f"  {lab:<12}{v:>5}{mark}")
    print(f"  you: {ents[me]['name']}, burned "
          f"{','.join(sorted(ents[me]['used']))}")

    # ---- simulate -----------------------------------------------------
    print(f"\nSIMULATING {args.sims} seasons from week {args.week}...")
    my_path = optimal_path(grid, args.week, ents[me]["used"])
    if my_path:
        print(f"  your path : {' '.join(my_path[:8])} ...")
    r = simulate(grid, ents, me, args.week, fit, args.sims,
                 my_path=my_path)
    print("\n" + "=" * 68)
    print("YOUR EXPECTED VALUE FROM HERE")
    print("=" * 68)
    print(f"  EV                     : ${r['ev']:.2f}")
    # The prize pool is fixed at $5,000 regardless of how many are left,
    # so fair share is measured against everyone who ENTERED, not against
    # the survivors. Dividing by the shrinking field would make your edge
    # look worse every week for no reason.
    n_entered = len(ents) + dropped
    fair = 5000 / n_entered
    print(f"  fair share (1 of {n_entered} entered) : ${fair:.2f}")
    print(f"  edge                   : {r['ev']/fair:.2f}x")
    print(f"  finish with <=2 strikes: {r['survive']:.1f}%")
    print(f"  cash a top-5 place     : {r['top5']:.1f}%")
    print(f"  win your division      : {r['div']:.1f}%")
    print("=" * 68)

    if args.audit:
        POT = sum(OVERALL) + DIV_PRIZE * N_DIV
        print(f"  (field {len(ents)} alive of {n_entered} entered; "
              f"pot is fixed at ${POT:,})")
        tot = r["sim_totals"]
        ev = r["all_ev"]
        print("\n" + "=" * 68)
        print("CONSISTENCY AUDIT")
        print("=" * 68)
        print("  A pool simulation must be a closed system: every season")
        print("  pays out the full pot, and the average entrant earns")
        print("  exactly their fair share. If either fails, the payout")
        print("  logic is leaking money and every EV above is suspect.")

        bad = int((np.abs(tot - POT) > 0.01).sum())
        print(f"\n  [{'OK  ' if not bad else 'FAIL'}] every season pays "
              f"${POT:,}")
        print(f"         min ${tot.min():,.2f}  max ${tot.max():,.2f}  "
              f"seasons off: {bad}/{len(tot)}")

        mean_ev = float(ev.mean())
        fair = POT / len(ev)
        okm = abs(mean_ev - fair) < 0.01
        print(f"  [{'OK  ' if okm else 'FAIL'}] mean EV equals fair share")
        print(f"         mean ${mean_ev:.4f}  vs fair ${fair:.4f}")

        print(f"\n  EV by starting strike count (must be monotone):")
        prev = None
        mono = True
        for k in sorted(set(r["start_strikes"])):
            m = r["start_strikes"] == k
            v = float(ev[m].mean())
            if prev is not None and v > prev + 1e-9:
                mono = False
            prev = v
            mark = "  <-- you" if k == r["start_strikes"][me] else ""
            print(f"    {k} strikes  n={m.sum():>3}   EV ${v:>7.2f}{mark}")
        print(f"  [{'OK  ' if mono else 'FAIL'}] more strikes -> less money")
        elim = r["start_strikes"] >= 3
        if elim.any():
            v = float(ev[elim].mean())
            print(f"  [{'OK  ' if abs(v) < 0.01 else 'FAIL'}] "
                  f"eliminated entrants earn $0   got ${v:.2f}")

        print(f"\n  your EV rank: "
              f"{int((ev > ev[me]).sum()) + 1} of {len(ev)}")
        print("=" * 68)


if __name__ == "__main__":
    main()
