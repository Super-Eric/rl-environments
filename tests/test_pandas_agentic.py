"""Tests for pandas-agentic-v1 — reference correctness, key re-derivation, anti-hack, runtime I/O."""

import csv
import io
import json
from types import SimpleNamespace

import pytest
import verifiers.v1 as vf
from conftest import make_trace
from pandas_agentic_v1.reference import SENTINELS, group_aggregate, left_join_unmatched, repair_and_typecast
from pandas_agentic_v1.taskset import IMAGE, PandasAgenticTaskset, PandasConfig


def _head(taskset, n):
    out, it = [], taskset.load()
    for _ in range(n):
        out.append(next(it))
    return out


def tasks(n=9, **cfg):
    return _head(PandasAgenticTaskset(PandasConfig(**cfg)), n)


def first(family, n=9, **cfg):
    return next(t for t in tasks(n, **cfg) if t.data.family == family)


def rows_of(task, name):
    return list(csv.DictReader(io.StringIO(task.data.input_files[name])))


def clean_csv(task, rows):
    """Serialise repair rows the way a correct agent would: None becomes an empty cell."""
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=list(task.data.spec["columns"]), lineterminator="\n")
    writer.writeheader()
    writer.writerows([{k: ("" if v is None else v) for k, v in r.items()} for r in rows])
    return buf.getvalue()


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
        {"id": "A", "qty": "1,200", "price": "$45.50"},  # exact duplicate
        {"id": "C", "qty": "7", "price": "-"},
    ]
    assert repair_and_typecast(rows, spec) == [
        {"id": "A", "qty": 1200, "price": 45.50},
        {"id": "B", "qty": None, "price": 12.35},  # half-up, not banker's
        {"id": "C", "qty": 7, "price": None},
    ]


def test_group_aggregate_counts_nulls_separately():
    rows = [
        {"cat": "x", "v": 10.0},
        {"cat": "x", "v": None},
        {"cat": "x", "v": 20.0},
        {"cat": "y", "v": None},
    ]
    out = group_aggregate(rows, "cat", "v")
    assert out["x"] == {"count": 2, "nulls": 1, "sum": 30.0, "mean": 15.0}
    assert out["y"] == {"count": 0, "nulls": 1, "sum": 0.0, "mean": None}  # all-null group is still a group


def test_left_join_audit_counts_duplication():
    left = [{"k": "a"}, {"k": "b"}, {"k": "c"}]
    right = [{"k": "a"}, {"k": "a"}, {"k": "z"}]
    out = left_join_unmatched(left, right, "k")
    assert out == {"joined_rows": 4, "left_only_keys": ["b", "c"], "right_only_keys": ["z"]}


# ---------------------------------------------------------------- generation


async def test_answer_keys_rederive_from_shipped_files():
    for task in tasks(12):
        assert await task.validate(None)


def test_every_task_declares_the_pandas_image():
    """The runtimes' default image has no pandas; every task must ask for one that does."""
    assert all(t.data.image == IMAGE for t in tasks(6))


def test_benchmark_slice_is_sixty_distinct_tasks_across_all_families():
    slice_ = tasks(60)
    assert len({t.key for t in slice_}) == 60
    assert {t.data.family for t in slice_} == {"repair", "reconcile", "merge"}


def test_deterministic_and_seed_sensitive():
    a = [t.data.input_files for t in tasks(6, seed=1)]
    b = [t.data.input_files for t in tasks(6, seed=1)]
    c = [t.data.input_files for t in tasks(6, seed=2)]
    assert a == b and a != c


def test_generated_mess_is_actually_present():
    seen_sentinel = seen_comma = seen_dupe = False
    for task in tasks(6):
        for body in task.data.input_files.values():
            rows = list(csv.DictReader(io.StringIO(body)))
            cells = [v for r in rows for v in r.values()]
            seen_sentinel |= any(c in SENTINELS and c != "" for c in cells)
            seen_comma |= any("," in c and c.replace(",", "").replace(".", "").isdigit() for c in cells)
            seen_dupe |= len({tuple(sorted(r.items())) for r in rows}) < len(rows)
    assert seen_sentinel and seen_comma and seen_dupe


async def test_all_null_categories_stay_in_the_key():
    """The prompt promises every category present; a category whose prices are all missing
    must appear with mean null rather than vanish from the key. Small, noisy tables make it happen."""
    seen_all_null = False
    for task in tasks(9, num_rows=8, mess_rate=0.6, seed=3):
        if task.data.family != "reconcile":
            continue
        cleaned = repair_and_typecast(rows_of(task, "raw.csv"), task.data.spec)
        present = {r["category"] for r in cleaned if r["category"] is not None}
        groups = task.data.expected["groups"]
        assert set(groups) == present
        seen_all_null |= any(g["count"] == 0 and g["mean"] is None for g in groups.values())
        assert await task.validate(None)
    assert seen_all_null


def test_merge_right_side_multiplies_joined_rows():
    """The merge trap is only real if some order has several shipment rows that survive dedup."""
    multiplied = False
    for task in tasks(60):
        if task.data.family != "merge":
            continue
        left_clean = repair_and_typecast(rows_of(task, "left.csv"), {"columns": task.data.spec["left_columns"]})
        multiplied |= task.data.expected["joined_rows"] > len(left_clean)
    assert multiplied


async def test_validate_probes_pandas_inside_the_runtime():
    """Given a runtime, validate() must fail unless `import pandas` succeeds in the sandbox."""

    class FakeRuntime:
        def __init__(self, exit_code):
            self.exit_code, self.argv = exit_code, None

        async def run(self, argv, env):
            self.argv = argv
            return SimpleNamespace(exit_code=self.exit_code, stdout="", stderr="ModuleNotFoundError")

    task = tasks(1)[0]
    ok = FakeRuntime(0)
    assert await task.validate(ok)
    assert ok.argv[-1] == "import pandas"
    with pytest.raises(AssertionError, match="pandas"):
        await task.validate(FakeRuntime(1))


# ------------------------------------------------------------------- scoring


async def test_gold_artifacts_score_one():
    for task in tasks(6):
        d = task.data
        if d.family == "repair":
            trace = await score_with(task, submission_csv=clean_csv(task, d.expected["rows"]))
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


async def test_repair_row_order_matters():
    task = first("repair")
    rows = task.data.expected["rows"]
    reordered = [rows[-1]] + rows[:-1]
    assert (await score_with(task, submission_csv=clean_csv(task, reordered))).reward == 0.0


async def test_repair_kept_duplicate_scores_zero():
    task = first("repair")
    rows = task.data.expected["rows"]
    assert (await score_with(task, submission_csv=clean_csv(task, rows + [dict(rows[0])]))).reward == 0.0
    assert (await score_with(task, submission_csv=clean_csv(task, rows[:-1] + [dict(rows[0])]))).reward == 0.0


async def test_unnormalised_sentinels_score_zero():
    """The prompt demands an EMPTY cell for every missing value; leaving a sentinel in place is wrong."""
    task = first("repair")
    rows = task.data.expected["rows"]
    assert any(v is None for r in rows for v in r.values())
    left_in_place = [{k: ("N/A" if v is None else v) for k, v in r.items()} for r in rows]
    assert (await score_with(task, submission_csv=clean_csv(task, left_in_place))).reward == 0.0


async def test_echoing_the_messy_input_scores_zero():
    task = first("repair")
    assert (await score_with(task, submission_csv=task.data.input_files["raw.csv"])).reward == 0.0


async def test_phantom_group_scores_zero():
    task = first("reconcile")
    sub = {g: dict(v) for g, v in task.data.expected["groups"].items()}
    sub["phantom"] = {"count": 0, "nulls": 0, "sum": 0.0, "mean": None}
    assert (await score_with(task, submission=sub)).reward == 0.0


async def test_rounding_slack_rejects_bankers_and_unrounded_means():
    """A banker's-rounded mean sits 0.01 off the half-up value and an unrounded one up to 0.005 off;
    the tolerance must reject both deterministically, not depending on binary float noise."""
    task = first("reconcile")
    groups = task.data.expected["groups"]
    group = next(g for g, v in groups.items() if v["mean"] is not None)
    exact = groups[group]["mean"]
    for off_by in (0.01, 0.005):
        sub = {g: dict(v) for g, v in groups.items()}
        sub[group]["mean"] = exact - off_by
        assert (await score_with(task, submission=sub)).reward == 0.0, f"mean {off_by} off was paid"
    sub = {g: dict(v) for g, v in groups.items()}
    assert (await score_with(task, submission=sub)).reward == pytest.approx(1.0)


async def test_nan_floats_are_not_paid():
    """`json.loads` accepts a bare `NaN` and `float("nan")` parses from a CSV cell. Every comparison
    with NaN is false, so the float checks must accept a value only when it is close, not reject it
    only when it is far; otherwise an unsummed or NaN-propagated column passes."""
    task = first("reconcile")
    nan_sums = {g: {**v, "sum": float("nan")} for g, v in task.data.expected["groups"].items()}
    assert (await score_with(task, submission=nan_sums)).reward == 0.0
    nan_means = {g: {**v, "mean": float("nan")} for g, v in task.data.expected["groups"].items()}
    assert any(v["mean"] is not None for v in task.data.expected["groups"].values())
    assert (await score_with(task, submission=nan_means)).reward == 0.0

    task = first("repair")
    floats = [c for c, kind in task.data.spec["columns"].items() if kind == "float"]
    rows = [
        {k: ("nan" if k in floats and v is not None else v) for k, v in r.items()} for r in task.data.expected["rows"]
    ]
    assert (await score_with(task, submission_csv=clean_csv(task, rows))).reward == 0.0


async def test_counts_are_compared_not_truncated():
    """`int(61.9)` is 61 and `int(float("inf"))` raises; a count must equal the key as a number."""
    task = first("reconcile")
    groups = task.data.expected["groups"]
    for bad in (0.9, float("inf")):
        sub = {g: {**v, "count": v["count"] + bad} for g, v in groups.items()}
        assert (await score_with(task, submission=sub)).reward == 0.0, f"count + {bad} was paid"
    as_floats = {g: {**v, "count": float(v["count"]), "nulls": float(v["nulls"])} for g, v in groups.items()}
    assert (await score_with(task, submission=as_floats)).reward == pytest.approx(1.0)

    task = first("merge")
    for bad in (0.9, float("inf")):
        sub = dict(task.data.expected, joined_rows=task.data.expected["joined_rows"] + bad)
        assert (await score_with(task, submission=sub)).reward == 0.0, f"joined_rows + {bad} was paid"


# ------------------------------------------------------- runtime I/O (no container)


async def test_setup_and_finalize_round_trip_through_a_real_runtime(tmp_path):
    """Drive `setup` -> `finalize` on a subprocess runtime: the file plumbing is runtime-agnostic
    (`write`/`read` are `Runtime` interface methods), so this covers the one part of the environment
    scoring tests cannot reach without waiting on Docker. Each task is relocated into `tmp_path`
    because the subprocess runtime resolves absolute paths against the host filesystem — the reason
    this environment declares `NEEDS_CONTAINER` for real runs.
    """
    from verifiers.v1.runtimes import make_runtime

    runtime = make_runtime(vf.SubprocessConfig(type="subprocess"), name="pandas-io-test")
    await runtime.start()
    try:
        for family in ("repair", "reconcile"):
            base = tmp_path / family
            task = first(family)
            task.data = task.data.model_copy(update={"workdir": str(base)})

            # setup must materialise exactly the input files the prompt names, byte for byte
            trace = make_trace(task, "done")
            await task.setup(trace, runtime)
            for name, body in task.data.input_files.items():
                assert (await runtime.read(f"{base}/{name}")).decode() == body
            assert sorted(p.name for p in base.iterdir()) == sorted(task.data.input_files)

            # no artifact yet -> finalize records why and the reward is zero
            await task.finalize(trace, runtime)
            assert "submission_error" in trace.info
            assert "submission" not in trace.info and "submission_csv" not in trace.info
            await task.score(trace)
            assert trace.reward == 0.0 and trace.metrics["produced_output"] == 0.0

            # the gold artifact -> finalize captures it and the reward pays out
            trace = make_trace(task, "done")
            if family == "repair":
                gold = clean_csv(task, task.data.expected["rows"]).encode()
            else:
                gold = json.dumps(task.data.expected["groups"]).encode()
            await runtime.write(f"{base}/{task.data.output_file}", gold)
            await task.finalize(trace, runtime)
            await task.score(trace)
            assert trace.reward == pytest.approx(1.0) and trace.metrics["produced_output"] == 1.0

        # malformed JSON is reported as such, not silently treated as absent
        trace = make_trace(task, "done")
        await runtime.write(f"{base}/{task.data.output_file}", b"{not json")
        await task.finalize(trace, runtime)
        assert "invalid JSON" in trace.info["submission_error"]
    finally:
        runtime.cleanup()  # sync, unlike start/stop/read/write
        await runtime.stop()
