"""
QC the transcribed win-probability grid.

Two independent checks, both of which exploit a fact about the source image:
these are model win probabilities, so within any week the two teams in a game
must have probabilities summing to 100. That gives us
  (a) a weekly checksum: sum of all probs in week W == 100 * (games in week W)
  (b) matchup recovery: pair teams so that |p_i + p_j - 100| is minimised
      (max-weight perfect matching on the non-bye teams)

Any bad cell in the transcription shows up as a residual in (b) and a checksum
miss in (a). That is how we find OCR errors without the official schedule.
"""
import sys
import numpy as np
import pandas as pd
import networkx as nx

OFFICIAL_BYES = {
    5:  ["CAR", "KC"],
    6:  ["CIN", "DET", "MIA", "MIN"],
    7:  ["BUF", "JAX", "LAC", "WAS"],
    8:  ["HOU", "NO", "NYG", "SF"],
    9:  ["PIT", "TEN"],
    10: ["CHI", "DEN", "PHI", "TB"],
    11: ["ATL", "CLE", "GB", "LAR", "NE", "SEA"],
    13: ["BAL", "IND", "LV", "NYJ"],
    14: ["ARI", "DAL"],
}
WEEKS = list(range(1, 19))


def load(path="win_probs_2026.csv"):
    df = pd.read_csv(path).set_index("team")
    df.columns = [int(c) for c in df.columns]
    return df


def check_byes(df):
    """Confirm the blanks in the CSV match the official 2026 bye schedule."""
    problems = []
    official = {t: w for w, teams in OFFICIAL_BYES.items() for t in teams}
    for team in df.index:
        blanks = [w for w in WEEKS if pd.isna(df.loc[team, w])]
        want = official.get(team)
        if blanks != [want]:
            problems.append(f"  {team}: blank at {blanks}, official bye is week {want}")
    return problems


def games_per_week(df):
    return {w: int(df[w].notna().sum() // 2) for w in WEEKS}


def checksums(df):
    rows = []
    for w in WEEKS:
        n_games = int(df[w].notna().sum() // 2)
        total = df[w].sum()
        rows.append({"week": w, "games": n_games,
                     "expected": 100 * n_games, "actual": total,
                     "residual": total - 100 * n_games})
    return pd.DataFrame(rows)


def recover_matchups(df, week):
    """
    Pair up the teams playing in `week` by minimising total |p_i + p_j - 100|.
    Returns (list of (teamA, teamB, pA, pB, error), total_error).
    """
    teams = [t for t in df.index if pd.notna(df.loc[t, week])]
    G = nx.Graph()
    for i, a in enumerate(teams):
        for b in teams[i + 1:]:
            err = abs(df.loc[a, week] + df.loc[b, week] - 100)
            # max_weight_matching maximises, so negate the error
            G.add_edge(a, b, weight=-err)
    match = nx.max_weight_matching(G, maxcardinality=True)
    out = []
    for a, b in match:
        pa, pb = df.loc[a, week], df.loc[b, week]
        if pa < pb:          # list favourite first
            a, b, pa, pb = b, a, pb, pa
        out.append((a, b, pa, pb, abs(pa + pb - 100)))
    out.sort(key=lambda r: -r[4])
    return out, sum(r[4] for r in out)


def main():
    import argparse
    ap = argparse.ArgumentParser(
        description="Validate a win-probability grid and recover matchups.")
    ap.add_argument("--grid", default="win_probs_2026.csv",
                    help="grid to check. Use win_probs_model.csv for the "
                         "live model output -- the default is the original "
                         "screenshot transcription, kept only as a fixture.")
    args = ap.parse_args()
    df = load(args.grid)

    print("=" * 68)
    print("BYE-WEEK CHECK (vs official 2026 schedule)")
    print("=" * 68)
    problems = check_byes(df)
    if problems:
        print("MISMATCHES:")
        print("\n".join(problems))
    else:
        print("All 32 byes match the official 2026 schedule.")

    print()
    print("=" * 68)
    print("WEEKLY CHECKSUMS  (each week's probs should sum to 100 x games)")
    print("=" * 68)
    cs = checksums(df)
    print(cs.to_string(index=False))
    print(f"\nTotal absolute residual across the season: {cs.residual.abs().sum():.0f} pts")
    print("A residual of 0 means that week is transcribed perfectly.")

    print()
    print("=" * 68)
    print("RECOVERED MATCHUPS  (derived from the grid, not the NFL schedule)")
    print("=" * 68)
    all_err = []
    for w in WEEKS:
        pairs, tot = recover_matchups(df, w)
        all_err.append((w, tot))
        flag = "  <-- CHECK THIS WEEK" if tot > 6 else ""
        print(f"\nWeek {w}  (pairing error {tot:.0f}){flag}")
        for a, b, pa, pb, err in pairs:
            mark = " *" if err >= 3 else ""
            print(f"   {a:>4} {pa:3.0f}%  vs {b:>4} {pb:3.0f}%   (err {err:2.0f}){mark}")

    print()
    print("=" * 68)
    print("Cells marked * are the most likely transcription errors.")
    print("Fix them in win_probs_2026.csv and re-run.")
    print("=" * 68)


if __name__ == "__main__":
    main()
