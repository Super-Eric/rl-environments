"""Reference implementations — deliberately pandas-free.

The agent solves every task *with* pandas; the grader must not share its conventions
(NaN propagation, dtype coercion, merge suffixes, groupby dropna) or the environment
stops testing whether the agent handled them. Same grader/solver-separation rule as
sci-pipeline-v1, which this package generalises.

Every function here is the answer key for one task family, computed from the same raw
rows the agent's input files are generated from.
"""

from collections import defaultdict
from decimal import ROUND_HALF_UP, Decimal

SENTINELS = {"", "NA", "N/A", "null", "-", "?"}


def _round2(x: float) -> float:
    return float(Decimal(repr(x)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))


def repair_and_typecast(rows: list[dict], spec: dict) -> list[dict]:
    """Family 1 (repair): normalise sentinels to None, coerce declared column types,
    drop exact-duplicate rows keeping the first, in file order."""
    seen: set[tuple] = set()
    out: list[dict] = []
    for row in rows:
        clean: dict = {}
        for col, kind in spec["columns"].items():
            value = (row.get(col) or "").strip()
            if value in SENTINELS:
                clean[col] = None
                continue
            if kind == "int":
                clean[col] = int(float(value.replace(",", "")))
            elif kind == "float":
                clean[col] = _round2(float(value.replace(",", "").replace("$", "")))
            else:
                clean[col] = value
        key = tuple(sorted(clean.items(), key=lambda kv: kv[0]))
        if key in seen:
            continue
        seen.add(key)
        out.append(clean)
    return out


def group_aggregate(rows: list[dict], by: str, value: str) -> dict:
    """Family 2 (reconcile): per-group count / sum / mean over non-null values,
    with half-up rounding applied once, at the end."""
    groups: dict[str, list[float]] = defaultdict(list)
    nulls: dict[str, int] = defaultdict(int)
    for row in rows:
        group = row[by]
        if group is None:
            continue
        if row[value] is None:
            nulls[group] += 1
            continue
        groups[group].append(row[value])
    return {
        group: {
            "count": len(vals),
            "nulls": nulls.get(group, 0),
            "sum": _round2(sum(vals)),
            "mean": _round2(sum(vals) / len(vals)) if vals else None,
        }
        for group, vals in sorted(groups.items())
    }


def left_join_unmatched(left: list[dict], right: list[dict], on: str) -> dict:
    """Family 3 (merge audit): row counts of a left join plus the keys that failed to
    match on either side — the reconciliation numbers a data engineer actually checks."""
    right_keys = {r[on] for r in right if r[on] is not None}
    left_keys = {r[on] for r in left if r[on] is not None}
    right_index: dict = defaultdict(int)
    for r in right:
        if r[on] is not None:
            right_index[r[on]] += 1
    joined = sum(right_index.get(row[on], 1) if row[on] in right_keys else 1 for row in left)
    return {
        "joined_rows": joined,
        "left_only_keys": sorted(k for k in left_keys - right_keys),
        "right_only_keys": sorted(k for k in right_keys - left_keys),
    }
