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


_TEAMS = re.compile("|".join(map(re.escape, NICK)))


def parse_sheet(path, week):
    """
    Two sheet formats exist and each breaks the other's parser:

      PICKS   "Name XX Chiefs"                           one team per row,
              but rows GLUE together in the PDF export:
              "Kaitlyn Lewis X 49ersKiersten White Lions" is two people.

      RESULTS "Name XX Jaguars Buccaneers Chiefs Ravens" every week so far,
              one person per row.

    Scanning for the first nickname handles glued rows but returns the WEEK 1
    pick on a results sheet. Taking the last nickname per line handles results
    sheets but merges glued rows. Neither errors -- both produce a plausible
    200-ish rows of wrong data.

    So detect the format first: on a results sheet nearly every line carries
    the same number of teams (= weeks played). On a picks sheet most lines
    carry one.
    """
    txt = open(path, encoding="utf-8").read()
    counts = [len(_TEAMS.findall(ln)) for ln in txt.splitlines()
              if ln.strip() and not ln.strip().lower().startswith("name")]
    counts = [c for c in counts if c]
    modal = max(set(counts), key=counts.count) if counts else 1
    results_fmt = modal >= 2 and counts.count(modal) > 0.8 * len(counts)

    rows = []
    if results_fmt:
        for line in txt.splitlines():
            line = line.strip()
            if not line or line.lower().startswith("name"):
                continue
            hits = list(_TEAMS.finditer(line))
            if not hits:
                continue
            name, marks = split_marks(line[:hits[0].start()].strip(), "")
            name = canon(name)
            if name:
                rows.append({"week": week, "name": name,
                             "team": NICK[hits[-1].group(0)],
                             "strike": len(marks)})
    else:
        for m in _PAT.finditer(txt):
            name = m.group(1).replace("\n", " ").strip()
            name, marks = split_marks(name, m.group(2) or "")
            name = canon(name)
            if name:
                rows.append({"week": week, "name": name,
                             "team": NICK[m.group(3)],
                             "strike": len(marks)})
    df = pd.DataFrame(rows)
    df.attrs["fmt"] = "results" if results_fmt else "picks"
    df.attrs["modal"] = modal
    return df


def cmd_parse(args):
    new = parse_sheet(args.path, args.week)
    want = 2 if args.week in DOUBLE_WEEKS else 1
    odd = new.groupby("name").size()
    odd = odd[odd != want]
    print(f"wk{args.week}: {len(new)} rows, {new.name.nunique()} entrants, "
          f"strikes {dict(sorted(Counter(new.strike).items()))} "
          f"[{new.attrs.get('fmt')} format, {new.attrs.get('modal')} team(s)/row]")
    if len(odd):
        print(f"  WARNING {len(odd)} entrants have != {want} picks: "
              f"{list(odd.index[:6])}")

    if os.path.exists(STATE):
        old = pd.read_csv(STATE)
        out = pd.concat([old[old.week != args.week], new], ignore_index=True)
    else:
        out = new

    per_week = out.groupby("week").name.nunique()
    # a sheet that only ever SHRINKS is eliminations, not name fragmentation
    shrinking = list(per_week) == sorted(per_week, reverse=True)
    if out.name.nunique() != per_week.max() and not shrinking:
        seen = {w: set(g.name) for w, g in out.groupby("week")}
        common = set.intersection(*seen.values())
        print(f"  WARNING {out.name.nunique()} distinct names vs "
              f"{per_week.to_dict()} per week — same person counted twice:")
        for w in sorted(seen):
            extra = sorted(seen[w] - common)[:6]
            if extra:
                print(f"    wk{w}: {extra}")

    out.sort_values(["week", "name"]).to_csv(STATE, index=False)
    print(f"  wrote {STATE} ({len(out)} rows, weeks "
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

    dist = {k: int((st == k).sum()) if k < 3 else int((st >= 3).sum())
            for k in range(4)}
    n_start = int(df.groupby("week").name.nunique().max())
    gone = n_start - int(df[df.week == latest].name.nunique())
    print(f"FIELD wk{[int(w) for w in weeks]} | {n_ent} seen, "
          f"{n_start} at peak, {gone} dropped from the sheet")
    print("  strikes entering wk%d: " % latest + "  ".join(
        f"{k if k < 3 else '3+'}:{v}" for k, v in dist.items()))

    # Eliminated entrants are DROPPED from later sheets (200 -> 193 -> ...),
    # so a percentage computed against the current sheet drifts upward every
    # week for reasons unrelated to picks. Report raw counts, and size the
    # field by the largest week ever seen.
    burn = df.groupby("team")["name"].nunique().sort_values(ascending=False)
    n_start = int(df.groupby("week").name.nunique().max())
    print(f"  burned (of {n_start} original): " + "  ".join(
        f"{t} {n}" for t, n in burn.head(12).items()))
    per_week = df.groupby("week").name.nunique()
    if per_week.nunique() > 1:
        print("  sheet size by week: " + "  ".join(
            f"W{w}:{n}" for w, n in per_week.items())
            + "   (shrinks as entrants are eliminated)")

    print("  popularity:")
    for w in weeks:
        c = Counter(df[df.week == w].team)
        print(f"    W{w}: " + "  ".join(f"{t} {k}" for t, k in c.most_common(5)))

    if args.me:
        key = canon(args.me).split()[0]
        mine = df[df.name.str.contains(key, case=False, na=False)]
        if mine.empty:
            print(f"  '{args.me}' not found")
            return
        nm = mine.name.iloc[0]
        used = df[df.name == nm].team.tolist()
        s_ = int(st.get(nm, 0))
        print(f"\n  YOU {nm} | {s_} strike(s), budget {2-s_} | "
              f"ahead {int((st < s_).sum())} "
              f"tied {int((st == s_).sum())-1} "
              f"behind {int((st > s_).sum())}")
        print(f"  --used {','.join(used)} --strikes {s_}   "
              f"(picks.csv authoritative; sheet predates wk{latest} late games)")
        hold = [(t, burn[t]) for t in burn.index
                if t not in used and burn[t] >= 5]
        if hold:
            print("  you hold: " + "  ".join(f"{t} {n}" for t, n in hold))


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
