# 6th Annual NFL Survivor League — Final Strategy

23 picks · 3 strikes · 200 entrants · $5,000

## The double weeks are the whole game

Weeks 5, 7, 10, 12 and 15 take two teams. That's 23 picks from 32 teams, and it is far more punishing than 18:

| | 18 picks | **23 picks** |
|---|---|---|
| Expected strikes (optimal path) | 3.48 | **5.31** |
| P(finish with ≤2 strikes) | 29% | **6.9%** |

Five extra picks cut survival by **76%**. You can't fill 23 slots with good teams, so weak picks get forced into the plan — Washington at 52%, Pittsburgh at 63%, Jacksonville at 66%. Those forced picks decide your season, and with only 18 you could have dodged every one.

## Nobody is going to survive this

Simulating the full 200-person field: **an average of 1.6 entrants finish with ≤2 strikes.** Plenty of seasons will have zero.

The organiser calls ≤2 strikes "the goal." It's going to be closer to a lottery ticket than a goal.

So the top-5 money will usually be settled by **who lasted longest**, not who finished. That single fact drives everything below.

## The number that should change your plan

Probability you're still alive to make each pick, on the optimal path:

| Pick | Alive |
|---|---|
| Week 1 | 100% |
| Week 7 (2nd pick) | 74% |
| Week 10 (2nd pick) | **50%** |
| Week 13 | 24% |
| Week 15 | 18% |
| **Week 18** | **7.8%** |

You are a coin flip to still be alive by Week 10, and you make your Week 18 pick less than one season in twelve.

**This means hoarding elite teams for late is actively wrong here.** Saving Buffalo for their 85% Week 18 game banks a resource you'll spend 7.8% of the time. In a normal survivor pool, future value is king. In a 23-pick 3-strike pool, future value is heavily discounted by the odds you're never there to collect it.

## Two paths

**Baseline** (max P(≤2 strikes)) — 6.90% survival, 57% alive at Week 10:
```
W1 LAC(84)  W2 TB(72)   W3 SF(85)   W4 CHI(79)  W5 DET(78)+NE(79)
W6 JAX(66)  W7 DEN(76)+CIN(76)      W8 DAL(83)  W9 SEA(87)
W10 IND(75)+LAR(84)     W11 KC(85)  W12 MIN(67)+WAS(52)
W13 PHI(78) W14 GB(84)  W15 BAL(82)+PIT(63)     W16 NO(70)
W17 HOU(79) W18 BUF(85)
```

**Front-loaded** (early slots up-weighted) — 6.21% survival but **66% alive at Week 10**:
```
W1 LAC(84)  W2 SEA(82)  W3 SF(85)   W4 CHI(79)  W5 NE(79)+DET(78)
W6 LAR(89)  W7 GB(77)+DEN(76)       W8 DAL(83)  W9 KC(82)
W10 IND(75)+MIA(61)     W11 JAX(71) W12 MIN(67)+CIN(65)
W13 PHI(78) W14 WAS(66) W15 BAL(82)+PIT(63)     W16 NO(70)
W17 HOU(79) W18 BUF(85)
```

It trades 0.7 points of clean-finish probability for 9 points of depth at Week 10. Since depth is what usually pays here, **I lean front-loaded** — but see the caveat below.

## Tactics

- **Week 12 is the trap.** A double week with no bye relief where the plan is forced down to a 52–67% second pick. Every entrant hits the same wall, so a strike there is heavily correlated across the field and won't cost you much ground in relative terms. Don't panic-spend a premium team to avoid it.
- **Never take both sides of one game in a double week.** That's a guaranteed strike with zero variance. The optimizer rejects it automatically and `slots_model.py` checks explicitly.
- **Ties count as wins.** Worth roughly +0.5% per pick, about +0.1 strikes over the season. Small, real, in your favour.
- **Re-solve weekly.** The whole plan re-optimises in seconds, and after any strike your remaining path should change.

## Your expected value

| | |
|---|---|
| EV | **~$72–77** |
| Fair share | $25 |
| Edge | **~3×** |
| Finish ≤2 strikes | ~5% |
| Cash a top-5 place | ~8–10% |
| Win your division | ~13–17% |
| Average elimination | slot 12 of 23 (about Week 10) |

The $150 division prize is the likeliest money you'll see — roughly one year in six.

## Caveats

- **The gap between the two paths is inside the Monte Carlo noise.** Front-loaded came out ahead on dollars ($76.54 vs $72.08), but that ordering moved between runs. The structural argument (a 7.8% chance of ever using your Week 18 pick) is solid; the specific dollar gap is not.
- **My transcription still needs cleaning.** Weeks 6–18 carry ±5pt noise. Normalising each game to sum to 100 already revealed the grid inflates favourites, which is why simulated survival (~5%) runs below the raw-grid figure (6.9%). Run `qc_grid.py`, fix the starred cells, re-run everything.
- **Rival pick popularity is modelled, not observed.** It matters less here than in a winner-take-all pool, but real pick percentages would still sharpen the numbers.

## Files
`slots_model.py` (23-pick optimiser) · `prize23.py` (dollar simulation) · `qc_grid.py` · `win_probs_2026.csv`
