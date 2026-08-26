"""The reference pipeline, in plain Python.

Deliberately depends on nothing but the standard library. The agent will almost certainly
reach for pandas or polars; if the grader used pandas too, the two would share its
missing-value and rounding conventions and the environment would quietly stop testing
whether the agent handled them. An independent implementation keeps the contract in the
prompt rather than in a library's defaults.

Order of operations is fixed and stated verbatim in the task prompt, because every step
here changes the answer:

1. Deduplicate on timestamp, keeping the **first** row for each timestamp in file order.
2. A reading of exactly the sentinel -999.0 is missing, whatever its unit.
3. Convert every `F` reading to Celsius with `(F - 32) * 5 / 9`.
4. Aggregate per calendar date; round only at the end, half-up to 2 decimals.
"""

from decimal import ROUND_HALF_UP, Decimal

SENTINEL = -999.0


def _round2(x: float) -> float:
    # Half-up, not banker's rounding: `round()` would send 12.345 to 12.34, and the
    # prompt promises the ordinary convention a scientist would expect.
    return float(Decimal(repr(x)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))


def reduce_readings(rows: list[dict]) -> dict:
    """Apply the reference pipeline to `rows` (parsed CSV, in file order)."""
    seen: set[str] = set()
    per_day: dict[str, list[float]] = {}
    valid: list[float] = []
    n_missing = 0

    for row in rows:
        ts = row["timestamp"]
        if ts in seen:
            continue
        seen.add(ts)
        value = float(row["value"])
        if value == SENTINEL:
            n_missing += 1
            continue
        celsius = (value - 32.0) * 5.0 / 9.0 if row["unit"].strip().upper() == "F" else value
        per_day.setdefault(ts[:10], []).append(celsius)
        valid.append(celsius)

    return {
        "daily_mean_c": {day: _round2(sum(v) / len(v)) for day, v in sorted(per_day.items())},
        "n_valid": len(valid),
        "n_missing": n_missing,
        "max_c": _round2(max(valid)) if valid else None,
        "min_c": _round2(min(valid)) if valid else None,
    }
