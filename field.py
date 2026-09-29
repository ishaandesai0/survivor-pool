"""
Ingest the weekly pool sheet — observed field state, not a model of it.

The sheet gives, for every entrant, which teams they have burned and how
many strikes they carry. That is the exact state a pool simulation needs,
and it replaces the weakest assumption in the model (a guessed popularity
curve) with data.

PARSING NOTES — both of these caused silent, wrong results before:

1. The strike column is X / XX / XXX, a COUNT not a flag. An earlier regex
   used `(\\bX\\s*)?`, which matches ONE X and leaves the remainder stuck to
   the name: "Aanchal Sanghvi XX Packers" parsed as name="Aanchal Sanghvi X",
   strikes=1. Every multi-strike entrant then became a second phantom
   person, inflating the roster 200 -> 234 while each individual week still
   counted 200, so nothing looked obviously broken.

2. Rows get glued together by the PDF ("Teri LynchX Buccaneers",
   "...49ersKiersten White Lions"), and names drift between weeks
   ("Nate McClung" -> "Nate (Drew) McClung"). We scan for team nicknames
   rather than trusting line breaks, strip any trailing X-run off the name
   unconditionally, and canonicalise names so the same person matches
   across weeks.

Usage:
    python field.py parse sheets/week3_sheet.txt --week 3
    python field.py report --me "Ishaan"
"""
import argparse
import os
import re
import sys
from collections import Counter

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
_PAT = re.compile(r"(.*?)(X{1,3})?\s*(" + "|".join(map(re.escape, NICK)) + r")",
                  re.S)


def canon(name):
    """Normalise a name so the same person matches across weeks."""
    n = re.sub(r"\([^)]*\)", " ", name)          # drop "(Drew)", "(Jack)"
    n = n.replace("'", "").replace("\u2019", "")  # O'Brian -> OBrian
    n = re.sub(r"[^A-Za-z ]+", " ", n)            # periods etc -> space
    n = re.sub(r"\s+", " ", n).strip().lower()
    out = []                                      # "t j smith" -> "tj smith"
    for p in n.split(" "):
        if len(p) == 1 and out and len(out[-1]) < 3:
            out[-1] += p
        else:
            out.append(p)
    return " ".join(out)


def split_marks(name, marks):
    """
    Pull any trailing X-run off the name and prepend it to the marks.
    Unconditional -- the regex can leave 1 or 2 X's behind depending on how
    the optional group happened to match.
    """
    while True:
        m = re.match(r"^(.*?)\s*(X+)$", name)
        if not m:
            return name.strip(), marks
        marks = m.group(2) + marks
        name = m.group(1)


def parse_sheet(path, week):
    txt = open(path, encoding="utf-8").read()
    rows = []
    for m in _PAT.finditer(txt):
        name = m.group(1).replace("\n", " ").strip()
        name, marks = split_marks(name, m.group(2) or "")
        name = canon(name)
        if not name:
            continue
        rows.append({"week": week, "name": name,
                     "team": NICK[m.group(3)],
                     "strike": len(marks)})   # CUMULATIVE entering this week
    return pd.DataFrame(rows)


def cmd_parse(args):
    new = parse_sheet(args.path, args.week)
    want = 2 if args.week in DOUBLE_WEEKS else 1
    per = new.groupby("name").size()
    odd = per[per != want]
    print(f"parsed {len(new)} entries for week {args.week}")
    print(f"  distinct entrants : {new.name.nunique()}")
    print(f"  strike marks (sum): {int(new.strike.sum())}")
    print(f"  distribution      : "
          f"{dict(sorted(Counter(new.strike).items()))}")
    if len(odd):
        print(f"\n  WARNING: {len(odd)} entrants have != {want} picks:")
        for nm, k in odd.head(8).items():
            print(f"    {nm!r}: {k}")

    if os.path.exists(STATE):
        old = pd.read_csv(STATE)
        old = old[old.week != args.week]
        out = pd.concat([old, new], ignore_index=True)
    else:
        out = new

    # roster drift is the tell that names are fragmenting
    per_week = out.groupby("week").name.nunique()
    if per_week.nunique() > 1 or out.name.nunique() != per_week.max():
        seen = {w: set(g.name) for w, g in out.groupby("week")}
        base = max(seen, key=lambda w: len(seen[w]))
        print(f"\n  WARNING: {out.name.nunique()} distinct names across weeks "
              f"but {per_week.to_dict()} per week.")
        print("  The same person is being counted twice. Offenders:")
        for w in sorted(seen):
            extra = sorted(seen[w] - set.intersection(*seen.values()))[:8]
            if extra:
                print(f"    week {w}: {extra}")

    out.sort_values(["week", "name"]).to_csv(STATE, index=False)
    print(f"\nwrote {STATE} ({len(out)} rows, weeks "
          f"{sorted(out.week.unique())})")


def cmd_report(args):
    if not os.path.exists(STATE):
        sys.exit(f"no {STATE} — run `field.py parse <sheet> --week N` first")
    df = pd.read_csv(STATE)
    weeks = sorted(df.week.unique())
    n_ent = df.name.nunique()
    latest = max(weeks)
    st = (df[df.week == latest].groupby("name")["strike"].max()
          .reindex(df.name.unique()).fillna(0).astype(int))

    print("=" * 68)
    print(f"FIELD STATE — weeks {weeks}, {n_ent} entrants")
    print("=" * 68)
    print(f"\n  strikes ENTERING week {latest} "
          f"(the sheet's X column reflects games resolved at the time it")
    print(f"  was published -- including any Thursday game that week):")
    for k in range(4):
        n = int((st == k).sum()) if k < 3 else int((st >= 3).sum())
        lab = f"{k} strikes" if k < 3 else "3+ eliminated"
        print(f"    {lab:<16}{n:>5}  {'#' * int(n / 4)}")

    print(f"\n  teams burned across the field (of {n_ent} entrants):")
    burn = df.groupby("team")["name"].nunique().sort_values(ascending=False)
    for t, n in burn.items():
        print(f"    {t:<5}{n:>5}  {100*n/n_ent:>5.1f}%  {'#' * int(n / 4)}")

    print("\n  pick popularity by week:")
    for w in weeks:
        c = Counter(df[df.week == w].team)
        print(f"    W{w}: " + "  ".join(f"{t} {k}" for t, k in c.most_common(5)))

    if args.me:
        key = canon(args.me)
        mine = df[df.name.str.contains(key.split()[0], case=False, na=False)]
        if mine.empty:
            print(f"\n  '{args.me}' not found")
            return
        nm = mine.name.iloc[0]
        used = df[df.name == nm].team.tolist()
        s = int(st.get(nm, 0))
        print("\n" + "=" * 68)
        print(f"YOU — {nm}")
        print("=" * 68)
        print(f"  strikes entering wk{latest}: {s}/3   budget {2-s}")
        print(f"  burned : {','.join(used)}")
        print(f"  ahead {int((st < s).sum())}   tied {int((st == s).sum())-1}"
              f"   behind {int((st > s).sum())}")
        print(f"\n  --used {','.join(used)} --strikes {s}")
        print(f"  (picks.csv is the authority for your CURRENT strike count;")
        print(f"   this sheet predates week {latest}'s later games)")
        hold = [(t, burn[t]) for t in burn.index
                if t not in used and burn[t] >= 5]
        if hold:
            print("\n  teams you hold that much of the field has spent:")
            for t, n in hold:
                print(f"    {t:<5}{n:>4} entrants ({100*n/n_ent:.0f}%)")


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
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
