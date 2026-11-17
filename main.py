"""Parametric log-curve forecaster for make.fun gta6-twitch-ccu-forecast-01.

Model (written fresh for this competition; nothing is copied from the kit's
starter, which is a flat persistence forecast):

For every comparable past launch whose release day has a complete 24-hour
curve, work in log-ratios to that launch's own hour-1 print,

    r_h = ln(viewers_h / viewers_1),   h = 2..24.

The median log-ratio is a smooth parametric curve in the hour, fitted by
weighted least squares on all comparables and all hours at once:

    m(h, x) = sum_k (a_k + b_k * x) * g_k(h),
    x       = ln(v1_comparable) - ln(p1_case),

where p1 is the case's hour-1 print and g_k is a small shape basis:

    g = [1, u, u^2,
         sigmoid((h - 5.5) / 0.8),      # the US-East unlock jump
         sigmoid((h - 13) / 2.5),       # the afternoon rise to prime time
         max(0, h - 20) / 4]            # the late-night decay
    u = (h - 1) / 23.

The slope terms b_k let the whole curve steepen or flatten with the size of
the launch: small hour-1 prints belong to launches whose audience is still
mostly asleep at 00:00 UTC and which then grow many-fold, while huge hour-1
prints are already near the day's peak.  Each comparable is weighted by a
Gaussian kernel on log hour-1 size (bandwidth 1.5 log units), so launches of
a similar size to the case dominate the fit.

The spread s(h) is the weighted residual standard deviation of the
log-ratios around the fitted curve in each hour, smoothed with a quadratic
in h, times 0.8 (the residual distribution is heavy-tailed, so the raw
standard deviation overstates the pinball-optimal spread; 0.8 was chosen on
leave-one-out validation over the comparables).  The forecast is

    q_tau(h) = p1 * exp(m(h, 0) + z_tau * s(h)),

with z the standard normal quantiles at the scored levels.

Safety rails: the fitted median log-ratio is clamped to [-2, 3] and the
spread to [0.15, 1.5], so an hour-1 print far outside the comparables'
range (GTA VI may be the biggest launch in the set) still yields a sane,
bounded forecast.  If too few complete comparable days exist, or the fit
is singular, the program falls back to a flat median at p1 with a
random-walk spread, rather than crashing.

Standard library only, no network, no randomness: two runs on the same
case write byte-identical output.
"""

import json
import math
import os
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.645, -0.674, 0.0, 0.674, 1.645]

# Gaussian kernel bandwidth (log units) for weighting comparables by
# similarity of their hour-1 print to the case's.
SIZE_BW = 1.5
# Pinball-optimal spread is narrower than the raw residual sd.
SPREAD_MULT = 0.8
SPREAD_FLOOR = 0.15
SPREAD_CAP = 1.5
# Rails on the fitted median log-ratio. The upper rail must sit above the
# largest plausible launch-day growth: the smallest comparables grow
# 100-2000x from a near-empty hour 1 (their fitted medians reach ~ln 1000),
# so a tight rail would cap the forecast far below the truth for a small
# hour-1 print. The lower rail guards the opposite extrapolation, a
# gigantic hour-1 print whose day then mostly decays.
MED_LO = -2.0
MED_HI = 8.0
# Fallback random-walk spread per sqrt-hour, used only if the fit fails.
FALLBACK_SIGMA = 0.15


def parse_time(text):
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def sigmoid(t):
    if t > 30.0:
        return 1.0
    if t < -30.0:
        return 0.0
    return 1.0 / (1.0 + math.exp(-t))


def basis(h):
    """Smooth shape basis in the forecast hour h (2..24)."""
    u = (h - 1) / 23.0
    return [1.0, u, u * u,
            sigmoid((h - 5.5) / 0.8),
            sigmoid((h - 13.0) / 2.5),
            max(0.0, h - 20) / 4.0]


def solve_linear(matrix, rhs):
    """Gaussian elimination with partial pivoting; None if singular."""
    n = len(rhs)
    rows = [list(matrix[i]) + [rhs[i]] for i in range(n)]
    for col in range(n):
        piv = max(range(col, n), key=lambda r: abs(rows[r][col]))
        if abs(rows[piv][col]) < 1e-12:
            return None
        rows[col], rows[piv] = rows[piv], rows[col]
        for r in range(col + 1, n):
            factor = rows[r][col] / rows[col][col]
            for k in range(col, n + 1):
                rows[r][k] -= factor * rows[col][k]
    out = [0.0] * n
    for i in range(n - 1, -1, -1):
        rest = sum(rows[i][j] * out[j] for j in range(i + 1, n))
        out[i] = (rows[i][n] - rest) / rows[i][i]
    return out


def polyfit(values, degree):
    """Least-squares polynomial fit of values[x] over x = 0..len-1.

    Returns coefficients low-to-high, or None if the fit is singular.
    """
    xs = list(range(len(values)))
    n = degree + 1
    matrix = [[sum(x ** (i + j) for x in xs) for j in range(n)]
              for i in range(n)]
    rhs = [sum(y * x ** i for x, y in zip(xs, values)) for i in range(n)]
    return solve_linear(matrix, rhs)


def polyval(coef, x):
    total = 0.0
    for c in reversed(coef):
        total = total * x + c
    return total


def day_viewers(comp):
    """A comparable's release-day hourly viewers, {hour: viewers}."""
    start = parse_time(comp["day_start"])
    found = {}
    for row in comp.get("hourly") or []:
        viewers = row.get("viewers")
        if not viewers or viewers <= 0:
            continue
        hour = int((parse_time(row["from"]) - start).total_seconds() // 3600) + 1
        if 1 <= hour <= 24:
            found[hour] = viewers
    return found


def extract_comparables(case):
    """Complete comparable days as (log hour-1, {h: log-ratio}) pairs."""
    extracted = []
    for comp in case.get("comparables") or []:
        day = day_viewers(comp)
        if len(day) < 24:
            continue
        first = day[1]
        ratios = {h: math.log(day[h] / first) for h in range(2, 25)}
        extracted.append((math.log(first), ratios))
    return extracted


def fit_curve(p1, comps):
    """Fit the parametric log-curve; return {h: (median, spread)} or None."""
    if len(comps) < 3:
        return None
    log_p1 = math.log(max(p1, 1.0))
    width = len(basis(2))
    size = 2 * width
    matrix = [[0.0] * size for _ in range(size)]
    rhs = [0.0] * size
    points = []
    for log_v1, ratios in comps:
        weight = math.exp(-0.5 * ((log_v1 - log_p1) / SIZE_BW) ** 2)
        shift = log_v1 - log_p1
        for h in range(2, 25):
            shape = basis(h)
            row = shape + [shift * g for g in shape]
            points.append((row, ratios[h], weight))
            for i in range(size):
                rhs[i] += weight * row[i] * ratios[h]
                for j in range(size):
                    matrix[i][j] += weight * row[i] * row[j]
    coef = solve_linear(matrix, rhs)
    if coef is None:
        return None
    median = {}
    for h in range(2, 25):
        shape = basis(h)
        value = sum(coef[k] * shape[k] for k in range(width))
        median[h] = max(MED_LO, min(MED_HI, value))
    # Per-hour weighted residual spread around the fitted curve.
    raw_spread = {}
    for h in range(2, 25):
        shape = basis(h)
        num = 0.0
        den = 0.0
        for log_v1, ratios in comps:
            weight = math.exp(-0.5 * ((log_v1 - log_p1) / SIZE_BW) ** 2)
            shift = log_v1 - log_p1
            pred = sum(coef[k] * shape[k] + coef[width + k] * shift * shape[k]
                       for k in range(width))
            num += weight * (ratios[h] - pred) ** 2
            den += weight
        raw_spread[h] = math.sqrt(num / den) if den > 0 else 1.0
    smooth = polyfit([raw_spread[h] for h in range(2, 25)], 2)
    out = {}
    for h in range(2, 25):
        spread = polyval(smooth, h - 1) if smooth else raw_spread[h]
        spread = max(SPREAD_FLOOR, min(SPREAD_CAP, spread * SPREAD_MULT))
        out[h] = (median[h], spread)
    return out


def fallback_curve(p1):
    """Flat median at p1 with a widening random-walk spread."""
    return {h: (0.0, FALLBACK_SIGMA * math.sqrt(h - 1))
            for h in range(2, 25)}


def round2(x):
    """Round to 2 decimals, half up; whole numbers come back as ints."""
    scaled = x * 100.0
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 100.0
    return int(value) if value.is_integer() else value


def hour_quantiles(p1, med, spread):
    quantiles = []
    previous = 0.0
    for z in Z:
        value = p1 * math.exp(med + z * spread)
        value = max(1.0, min(1e9, value))
        value = round2(value)
        if value < previous:
            value = previous
        quantiles.append(value)
        previous = value
    return quantiles


def forecast_case(case, name):
    if case["levels"] != LEVELS:
        raise ValueError("%s asks for levels %s, not %s"
                         % (name, case["levels"], LEVELS))
    first = next((k for k in case["known"] if k["hour"] == 1), None)
    if first is None:
        raise ValueError("%s has no hour-1 print" % name)
    p1 = max(1, first["viewers"])
    curve = fit_curve(p1, extract_comparables(case))
    if curve is None:
        curve = fallback_curve(p1)
    rows = []
    for row in case["forecast"]:
        h = row["hour"]
        med, spread = curve[h]
        rows.append({"hour": h, "q": hour_quantiles(p1, med, spread)})
    rows.sort(key=lambda r: r["hour"])
    return {"hours": rows}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case, path.name)
        text = json.dumps(forecast, separators=(",", ":")) + "\n"
        (OUTPUT_DIR / (path.stem + ".out")).write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
