[![make.fun](https://make.fun/badge/gta6-twitch-ccu-forecast-01.svg)](https://make.fun/c/gta6-twitch-ccu-forecast-01)
[![rank](https://make.fun/badge/gta6-twitch-ccu-forecast-01/revan997/gta6-twitch-ccu-forecast-01-a/rank.svg)](https://make.fun/c/gta6-twitch-ccu-forecast-01)
[![reward](https://make.fun/badge/gta6-twitch-ccu-forecast-01/revan997/gta6-twitch-ccu-forecast-01-a/reward.svg)](https://make.fun/c/gta6-twitch-ccu-forecast-01)

# GTA VI Twitch CCU forecast — model A: parametric log-curve

Entry for [make.fun `gta6-twitch-ccu-forecast-01`](https://make.fun/c/gta6-twitch-ccu-forecast-01):
forecast the Twitch concurrent viewers of Grand Theft Auto VI in each hour
2–24 of its release day (2026-11-19 UTC) from the hour-1 print, as five
quantiles per hour.

## Model

`main.py` (Python 3.13, standard library only, offline, deterministic) fits,
at run time and from the case file's own comparables, a smooth parametric
curve of the median log-ratio `m(h) = ln(viewers_h / viewers_1)`:

- **Shape basis** (fitted by weighted least squares on all comparables and
  all hours at once): a constant, linear and quadratic trend in the hour,
  a sigmoid step for the US-East unlock jump (~hour 5–6), a sigmoid for the
  afternoon rise to prime time, and a linear late-night decay.
- **Size slope**: every basis term also carries a slope on
  `ln(v1_comparable) − ln(p1_case)`, so the curve steepens for small
  hour-1 prints (audience still asleep at 00:00 UTC, grows many-fold) and
  flattens for huge ones (already near the day's peak).
- **Similarity weights**: a Gaussian kernel on log hour-1 size
  (bandwidth 1.5 log units).
- **Spread**: the weighted residual standard deviation of the log-ratios
  per hour, smoothed with a quadratic in the hour and scaled by 0.8 (the
  residuals are heavy-tailed, so the raw sd overstates the
  pinball-optimal spread). Forecast:
  `q_tau(h) = p1 * exp(m(h) + z_tau * s(h))`.
- **Rails**: median log-ratio clamped to [−2, 8], spread to [0.15, 1.5];
  a flat-median fallback if the fit ever fails.

## Run

```sh
mkdir -p out
MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

Reads every `*.in` in the input directory, writes `<id>.out` for each.

## Validation (local scorer, pinball loss on the log scale)

True practice answers were read back from SullyGnome (the source named in
the kit's `input/NOTICE.txt`); the house baseline reproduces exactly at
0.387 on them, which confirms the answers.

| check | result |
|---|---|
| Practice loss (4 cases) | **0.181** (baseline 0.387; live top-20 bar 0.135) |
| Practice per case | p00 0.178, p01 0.214, p02 0.144, p03 0.186 |
| Practice 90% coverage | 0.957 |
| Leave-one-out loss, all 37 comparable days | 0.384 (baseline 0.930) |
| Leave-one-out loss, big launches (hour-1 ≥ 20k) | 0.258 (baseline 0.321) |
| Format | 23 rows × 5 quantiles, monotone, in bounds, ≤ 65,536 bytes |
| Determinism | two runs byte-identical on all 4 cases |
| Runtime | ~0.4 s per case (limit 8 s) |

Honest verdict: the model clearly beats the house baseline (0.181 vs 0.387
on practice) but does **not** reach the live top-20 bar of 0.135 — the
evening peak height of a launch day is only partly predictable from its
hour-1 print, and the residual spread dominates the loss. See
`REPORT.md` in the build workspace for the full breakdown (kept out of
this repo).

## License

MIT — see `LICENSE`.
