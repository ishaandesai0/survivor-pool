"""
Ingest the weekly pool sheet — observed field state, not a model of it.

Everything up to now simulated the other 199 entrants with a softmax on win
probability. Week 1 proved that wrong: the model would have made the
Chargers the most popular pick (biggest favourite, only double-digit
favourite on the board) and the field went Jaguars 77 to 58. This pool does
not simply chase the largest favourite.

So stop guessing. The sheet gives, for every entrant: which teams they have
burned and how many strikes they carry. That is the exact state a pool
simulation needs, and it replaces the weakest assumption in the model.

Sheet format is whatever the organiser's PDF pastes as — roughly
"Name [X] Team" per line, except rows get glued together when the PDF
columns run into each other ("Teri LynchX Chargers", "...RaidersKiersten
White Jaguars"). So we scan for team nicknames rather than trusting line
breaks, and treat an X immediately before the nickname (or fused to the end
of the name) as the strike marker.

Usage:
    python field.py parse sheets/week1.txt --week 1
    python field.py report
    python field.py report --me "Ishaan Desai"
"""
import argparse
import os
import re
import sys
from collections import Counter, defaultdict

import pandas as pd

NICK = {
    "Jaguars": "JAX", "Chargers": "LAC", "Steelers": "PIT", "Lions": "DET",
    "Bears": "CHI", "Raiders": "LV", "Titans": "TEN", "Rams": "LAR",
    "Cowboys": "DAL", "Chiefs": "KC", "Seahawks": "SEA", "Jets": "NYJ",
    "Saints": "NO", "Commanders": "WAS", "Eagles": "PHI", "Vikings": "MIN",
    "Packers": "GB", "Ravens": "BAL", "Colts": "IND", "Browns": "CLE",
    "Falcons": "ATL", "Bills": "BUF", "Texans": "HOU", "Bengals": "CIN",
    "Buccaneers": "TB", "Dolphins": "MIA", "Cardinals": "ARI",
    "49ers": "SF", "Giants": "NYG", "Patriots": "NE", "Broncos": "DEN",
    "Panthers": "CAR",
}
STATE = "field_state.csv"
DOUBLE_WEEKS = {5, 7, 10, 12, 15}


def parse_sheet(path, week):
    txt = open(path, encoding="utf-8").read()
    pat = re.compile(r"(.*?)(\bX\s*)?(" + "|".join(map(re.escape, NICK)) + r")",
                     re.S)
    rows = []
    for m in pat.finditer(txt):
        name = m.group(1).replace("\n", " ").strip()
        struck = bool(m.group(2)) or name.endswith("X")
        if name.endswith("X"):
            name = name[:-1].strip()
        # a glued row can leave the tail of a previous name; keep the last
        # 2-4 tokens that look like a person's name
        if not name:
            continue
        rows.append({"week": week, "name": name,
                     "team": NICK[m.group(3)], "strike": int(struck)})
    return pd.DataFrame(rows)


def cmd_parse(args):
    new = parse_sheet(args.path, args.week)
    n_expected = 2 if args.week in DOUBLE_WEEKS else 1
    per = new.groupby("name").size()
    odd = per[per != n_expected]
    print(f"parsed {len(new)} entries for week {args.week}")
    print(f"  distinct entrants : {new.name.nunique()}")
    print(f"  strikes marked    : {int(new.strike.sum())}")
    if len(odd):
        print(f"\n  WARNING: {len(odd)} entrants have != {n_expected} picks "
              f"this week — likely a glued row the parser split wrong:")
        for nm, k in odd.head(10).items():
            print(f"    {nm!r}: {k}")

    if os.path.exists(STATE):
        old = pd.read_csv(STATE)
        old = old[old.week != args.week]          # re-parsing a week replaces it
        out = pd.concat([old, new], ignore_index=True)
    else:
        out = new
    out.sort_values(["week", "name"]).to_csv(STATE, index=False)
    print(f"\nwrote {STATE} ({len(out)} rows, weeks "
          f"{sorted(out.week.unique())})")


def cmd_report(args):
    if not os.path.exists(STATE):
        sys.exit(f"no {STATE} — run `field.py parse <sheet> --week N` first")
    df = pd.read_csv(STATE)
    weeks = sorted(df.week.unique())

    print("=" * 68)
    print(f"FIELD STATE — weeks {weeks}, {df.name.nunique()} entrants")
    print("=" * 68)

    # strikes per entrant (cumulative)
    st = df.groupby("name")["strike"].sum()
    alive = st[st < 3]
    print("\n  strike distribution:")
    for k in range(4):
        n = int((st == k).sum()) if k < 3 else int((st >= 3).sum())
        lab = f"{k} strikes" if k < 3 else "3+ (eliminated)"
        print(f"    {lab:<16}{n:>5}  {'#' * int(n / 3)}")
    print(f"    still alive     {len(alive):>5}")

    # burned inventory — what the field can no longer use
    print("\n  teams burned across the field (how many entrants can't reuse):")
    burn = df.groupby("team")["name"].nunique().sort_values(ascending=False)
    for t, n in burn.items():
        print(f"    {t:<5}{n:>5}  {100*n/df.name.nunique():>5.1f}%  "
              f"{'#' * int(n / 3)}")

    # popularity by week
    print("\n  pick popularity by week:")
    for w in weeks:
        c = Counter(df[df.week == w].team)
        top = "  ".join(f"{t} {k}" for t, k in c.most_common(5))
        print(f"    W{w}: {top}")

    if args.me:
        mine = df[df.name.str.contains(args.me, case=False, na=False)]
        if mine.empty:
            print(f"\n  '{args.me}' not found in the sheet")
            return
        used = mine.team.tolist()
        s = int(mine.strike.sum())
        print("\n" + "=" * 68)
        print(f"YOU — {mine.name.iloc[0]}")
        print("=" * 68)
        print(f"  strikes     : {s}/3   budget {2-s} more loss"
              f"{'' if 2-s == 1 else 'es'}")
        print(f"  burned      : {','.join(used)}")
        better = int((st < s).sum())
        tied = int((st == s).sum())
        print(f"  ahead of you: {better}   tied with you: {tied}   "
              f"behind: {len(st) - better - tied}")
        print(f"\n  --used {','.join(used)} --strikes {s}")

        # teams the field has burned that you still hold
        hold = [t for t, n in burn.items() if t not in used]
        hold = [(t, burn[t]) for t in hold if burn[t] >= 5]
        if hold:
            print("\n  you still hold teams much of the field has spent:")
            for t, n in hold:
                print(f"    {t:<5} burned by {n} entrants "
                      f"({100*n/df.name.nunique():.0f}%)")
            print("  in a 23-pick pool, inventory is a real asset late.")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("parse")
    p.add_argument("path")
    p.add_argument("--week", type=int, required=True)
    p.set_defaults(fn=cmd_parse)
    r = sub.add_parser("report")
    r.add_argument("--me", default=None)
    r.set_defaults(fn=cmd_report)
    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
