"""Tests for pandas-agentic-v1 — reference correctness, key re-derivation, anti-hack."""

import csv
import io

import pytest
from conftest import make_trace
from pandas_agentic_v1.reference import group_aggregate, left_join_unmatched, repair_and_typecast
from pandas_agentic_v1.taskset import PandasAgenticTaskset, PandasConfig


def _head(taskset, n):
    out, it = [], taskset.load()
    for _ in range(n):
        out.append(next(it))
    return out


def tasks(n=9, **cfg):
    return _head(PandasAgenticTaskset(PandasConfig(**cfg)), n)


async def score_with(task, **info):
    trace = make_trace(task, "done")
    trace.info.update(info)
    await task.score(trace)
    return trace


# ------------------------------------------------------------------- reference

def test_repair_reference_hand_computed():
    spec = {"columns": {"id": "str", "qty": "int", "price": "float"}}
    rows = [
        {"id": "A", "qty": "1,200", "price": "$45.50"},
        {"id": "B", "qty": "N/A", "price": "12.345"},
        {"id": "A", "qty": "1,200", "price": "$45.50"},     # exact duplicate
        {"id": "C", "qty": "7", "price": "-"},
    ]
    assert repair_and_typecast(rows, spec) == [
        {"id": "A", "qty": 1200, "price": 45.50},
        {"id": "B", "qty": None, "price": 12.35},           # half-up, not banker's
        {"id": "C", "qty": 7, "price": None},
    ]


def test_group_aggregate_counts_nulls_separately():
    rows = [
        {"cat": "x", "v": 10.0}, {"cat": "x", "v": None}, {"cat": "x", "v": 20.0},
        {"cat": "y", "v": None},
    ]
    out = group_aggregate(rows, "cat", "v")
    assert out["x"] == {"count": 2, "nulls": 1, "sum": 30.0, "mean": 15.0}
    assert "y" not in out or out["y"]["mean"] is None       # all-null group


def test_left_join_audit_counts_duplication():
    left = [{"k": "a"}, {"k": "b"}, {"k": "c"}]
    right = [{"k": "a"}, {"k": "a"}, {"k": "z"}]
    out = left_join_unmatched(left, right, "k")
    assert out == {"joined_rows": 4, "left_only_keys": ["b", "c"], "right_only_keys": ["z"]}


# ---------------------------------------------------------------- generation

async def test_answer_keys_rederive_from_shipped_files():
    for task in tasks(12):
        assert await task.validate(None)


def test_benchmark_slice_exceeds_fifty_distinct_tasks():
    """The bounty contract asks for 50+ nontrivial tasks; the fixed-seed slice is 60."""
    slice_ = tasks(60)
    assert len({t.key for t in slice_}) == 60
    fams = {t.data.family for t in slice_}
    assert fams == {"repair", "reconcile", "merge"}


def test_deterministic_and_seed_sensitive():
    a = [t.data.input_files for t in tasks(6, seed=1)]
    b = [t.data.input_files for t in tasks(6, seed=1)]
    c = [t.data.input_files for t in tasks(6, seed=2)]
    assert a == b and a != c


def test_generated_mess_is_actually_present():
    from pandas_agentic_v1.reference import SENTINELS
    seen_sentinel = seen_comma = seen_dupe = False
    for task in tasks(6):
        for body in task.data.input_files.values():
            rows = list(csv.DictReader(io.StringIO(body)))
            cells = [v for r in rows for v in r.values()]
            seen_sentinel |= any(c in SENTINELS and c != "" for c in cells)
            seen_comma |= any("," in c and c.replace(",", "").replace(".", "").isdigit() for c in cells)
            seen_dupe |= len({tuple(sorted(r.items())) for r in rows}) < len(rows)
    assert seen_sentinel and seen_comma and seen_dupe


# ------------------------------------------------------------------- scoring

async def test_gold_artifacts_score_one():
    for task in tasks(6):
        d = task.data
        if d.family == "repair":
            buf = io.StringIO()
            writer = csv.DictWriter(buf, fieldnames=list(d.spec["columns"]), lineterminator="\n")
            writer.writeheader()
            writer.writerows([{k: ("" if v is None else v) for k, v in row.items()}
                              for row in d.expected["rows"]])
            trace = await score_with(task, submission_csv=buf.getvalue())
        elif d.family == "reconcile":
            trace = await score_with(task, submission={g: dict(v) for g, v in d.expected["groups"].items()})
        else:
            trace = await score_with(task, submission=dict(d.expected))
        assert trace.reward == pytest.approx(1.0), f"gold failed on {d.family} idx={d.idx}"


async def test_missing_or_malformed_artifact_scores_zero():
    task = tasks(1)[0]
    assert (await score_with(task)).reward == 0.0
    assert (await score_with(task, submission_error="missing")).reward == 0.0


async def test_near_miss_counts_are_not_paid():
    for task in tasks(9):
        d = task.data
        if d.family == "reconcile":
            wrong = {g: {**v, "count": v["count"] + 1} for g, v in d.expected["groups"].items()}
            assert (await score_with(task, submission=wrong)).reward == 0.0
        if d.family == "merge":
            wrong = dict(d.expected, joined_rows=d.expected["joined_rows"] + 1)
            assert (await score_with(task, submission=wrong)).reward == 0.0


async def test_repair_row_order_and_dupes_matter():
    task = next(t for t in tasks(9) if t.data.family == "repair")
    rows = task.data.expected["rows"]
    reordered = [rows[-1]] + rows[:-1]
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=list(task.data.spec["columns"]), lineterminator="\n")
    writer.writeheader()
    writer.writerows([{k: ("" if v is None else v) for k, v in r.items()} for r in reordered])
    assert (await score_with(task, submission_csv=buf.getvalue())).reward == 0.0


async def test_echoing_the_messy_input_scores_zero():
    task = next(t for t in tasks(9) if t.data.family == "repair")
    assert (await score_with(task, submission_csv=task.data.input_files["raw.csv"])).reward == 0.0
