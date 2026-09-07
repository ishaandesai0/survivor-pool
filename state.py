"""
Single source of truth for pool state.

`--used` is the one input in this whole system that nothing can validate.
Mistype it, or forget a team from a double week, and every downstream model
returns a confidently wrong answer with no error. Hand-maintaining a
comma-separated list of up to 23 abbreviations across four months is a bad
idea.

So: log picks in picks.csv, and let this derive the flags.

picks.csv columns:
    week   - 1..18
    team   - abbreviation, must match the grid (LAR not LA, WAS not WSH)
    result - W, L, or blank if not played yet
    notes  - free text

Double weeks just get two rows with the same week number.

    python state.py              # show state + the commands to run
    python state.py --check      # validate only, non-zero exit on problems
"""
import argparse
import sys
import pandas as pd

DOUBLE_WEEKS = {5, 7, 10, 12, 15}
VALID = {"LAR","BUF","SF","KC","SEA","LAC","NE","BAL","GB","DET","DEN","CIN",
         "DAL","PHI","CHI","JAX","HOU","TB","MIN","NYG","IND","NO","PIT","WAS",
         "LV","CAR","NYJ","TEN","ATL","CLE","ARI","MIA"}
# abbreviations people type that the grid does not use
ALIASES = {"LA": "LAR", "WSH": "WAS", "JAC": "JAX", "SD": "LAC", "OAK": "LV",
           "STL": "LAR", "ARZ": "ARI", "TAM": "TB", "GNB": "GB", "KAN": "KC",
           "NWE": "NE", "NOR": "NO", "SFO": "SF"}


def load(path="picks.csv"):
    df = pd.read_csv(path, dtype=str).fillna("")
    df["week"] = df["week"].astype(int)
    df["team"] = df["team"].str.strip().str.upper()
    df["result"] = df["result"].str.strip().str.upper()
    return df


def validate(df):
    problems = []
    for _, r in df.iterrows():
        if r.team in ALIASES:
            problems.append(f"week {r.week}: '{r.team}' should be "
                            f"'{ALIASES[r.team]}' — the grid uses that form")
        elif r.team not in VALID:
            problems.append(f"week {r.week}: '{r.team}' is not a valid team")
        if r.result not in ("", "W", "L"):
            problems.append(f"week {r.week} {r.team}: result "
                            f"'{r.result}' must be W, L or blank")
    dupes = df[df.duplicated("team", keep=False) & (df.team != "")]
    for t in dupes.team.unique():
        wks = sorted(dupes[dupes.team == t].week.tolist())
        problems.append(f"'{t}' used more than once (weeks {wks}) — "
                        f"not allowed in this pool")
    for w, g in df.groupby("week"):
        want = 2 if w in DOUBLE_WEEKS else 1
        if len(g) > want:
            problems.append(f"week {w} has {len(g)} picks, expected {want}")
    return problems


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--picks", default="picks.csv")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--grid", default="win_probs_model.csv")
    args = ap.parse_args()

    df = load(args.picks)
    problems = validate(df)

    if problems:
        print("PROBLEMS IN picks.csv")
        print("-" * 40)
        for p in problems:
            print("  " + p)
        print("\nFix these before running any model.")
        sys.exit(1)
    if args.check:
        print("picks.csv is valid.")
        sys.exit(0)

    played = df[df.result != ""]
    strikes = int((played.result == "L").sum())
    used = df[df.team != ""].team.tolist()
    done_weeks = sorted(played.week.unique())
    next_week = (max(done_weeks) + 1) if done_weeks else 1
    # a week with picks logged but no results yet is still the current week
    pending = df[(df.result == "") & (df.team != "")]
    if len(pending):
        next_week = int(pending.week.min())

    print("=" * 66)
    print("POOL STATE")
    print("=" * 66)
    print(f"  picks logged   : {len(used)} / 23")
    print(f"  strikes        : {strikes} / 3", end="")
    if strikes >= 3:
        print("   *** ELIMINATED ***")
    else:
        left = 2 - strikes
        print(f"   ({left} more loss{'' if left == 1 else 'es'} allowed)")
    print(f"  teams burned   : {','.join(used) if used else '(none)'}")
    print(f"  teams left     : {32 - len(used)}")

    remaining_slots = 0
    for w in range(next_week, 19):
        remaining_slots += 2 if w in DOUBLE_WEEKS else 1
    # picks already logged for the current week are made, not remaining
    remaining_slots -= int((pending.week == next_week).sum())
    print(f"  slots remaining: {remaining_slots} (from week {next_week})")
    if 32 - len(used) < remaining_slots:
        print("  !! fewer teams left than slots remaining — path is infeasible")

    if strikes >= 3:
        return

    used_flag = ",".join(used) if used else ""
    u = f' --used {used_flag}' if used_flag else ""
    print("\n  RUN THIS WEEK:")
    print(f"    .venv\\Scripts\\python.exe pipeline.py --train 2007 2025 "
          f"--season 2026 --week {next_week} --base 3.717 --decay 0.355")
    print(f"    .venv\\Scripts\\python.exe weekly.py --week {next_week}"
          f"{u} --strikes {strikes} --grid {args.grid}")
    print(f"    .venv\\Scripts\\python.exe weekly_robust.py --week {next_week}"
          f"{u} --strikes {strikes} --grid {args.grid} --draws 200"
          f" --posted-through {min(next_week + 5, 18)}")
    print("\n  then log the pick in picks.csv and commit.")
    print("=" * 66)


if __name__ == "__main__":
    main()
