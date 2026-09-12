"""Tests for sci-pipeline-v1.

Two layers:

- The **reference pipeline** is checked against a hand-computed fixture. It is the answer
  key for every task in the environment, so an untested reference means an environment
  that is confidently wrong at scale. This is the single highest-leverage test here.
- The **rewards** are checked against the outputs that plausible-but-broken pipelines
  actually produce: skipping the sentinel filter, skipping the unit conversion, skipping
  deduplication. Each defect in the generated CSV must be individually load-bearing —
  if an agent can ignore one and still score, that defect is decoration.
"""

import csv
import io

import pytest
import verifiers.v1 as vf
from conftest import make_trace
from sci_pipeline_v1.reference import SENTINEL, reduce_readings
from sci_pipeline_v1.taskset import WORKDIR, SciConfig, SciData, SciPipelineTaskset, SciTask, SciTaskConfig


def _head(taskset, n):
    out, it = [], taskset.load()
    for _ in range(n):
        out.append(next(it))
    return out


def tasks(n=6, **cfg):
    return _head(SciPipelineTaskset(SciConfig(**cfg)), n)


def rows_of(task):
    return list(csv.DictReader(io.StringIO(task.data.csv_text)))


def task_from_rows(rows):
    """A task whose CSV is exactly `rows`, keyed the way the generator keys it."""
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=["timestamp", "station", "value", "unit"], lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    data = SciData(
        idx=0, workdir=WORKDIR, prompt="", csv_text=buf.getvalue(), expected=reduce_readings(rows), station="X"
    )
    return SciTask(data, SciTaskConfig())


async def score_submission(task, submission) -> object:
    """Score as if the agent had left `submission` in result.json (None = no file)."""
    trace = make_trace(task, "done")
    if submission is not None:
        trace.info["submission"] = submission
    await task.score(trace)
    return trace


# --------------------------------------------------------------------------- reference


def test_reference_matches_a_hand_computed_example():
    """Worked by hand: dedup keeps the first row, sentinel is excluded, F converts."""
    rows = [
        {"timestamp": "2025-01-01T00:00:00", "station": "X", "value": "10.0", "unit": "C"},
        {"timestamp": "2025-01-01T01:00:00", "station": "X", "value": "50.0", "unit": "F"},  # -> 10.0 C
        {"timestamp": "2025-01-01T00:00:00", "station": "X", "value": "99.0", "unit": "C"},  # dup, ignored
        {"timestamp": "2025-01-01T02:00:00", "station": "X", "value": str(SENTINEL), "unit": "C"},
        {"timestamp": "2025-01-02T00:00:00", "station": "X", "value": "20.0", "unit": "C"},
        {"timestamp": "2025-01-02T01:00:00", "station": "X", "value": "21.0", "unit": "C"},
    ]
    assert reduce_readings(rows) == {
        "daily_mean_c": {"2025-01-01": 10.0, "2025-01-02": 20.5},
        "n_valid": 4,
        "n_missing": 1,
        "max_c": 21.0,
        "min_c": 10.0,
    }


def test_reference_rounds_half_up_not_bankers():
    """`round()` would give 12.34 here; the prompt promises 12.35."""
    rows = [
        {"timestamp": "2025-01-01T00:00:00", "station": "X", "value": "12.34", "unit": "C"},
        {"timestamp": "2025-01-01T01:00:00", "station": "X", "value": "12.36", "unit": "C"},
    ]
    assert reduce_readings(rows)["daily_mean_c"]["2025-01-01"] == 12.35


def test_sentinel_is_missing_in_any_unit():
    rows = [
        {"timestamp": "2025-01-01T00:00:00", "station": "X", "value": str(SENTINEL), "unit": "F"},
        {"timestamp": "2025-01-01T01:00:00", "station": "X", "value": "10.0", "unit": "C"},
    ]
    out = reduce_readings(rows)
    assert out["n_missing"] == 1 and out["n_valid"] == 1


# ------------------------------------------------------------------------------ rewards


async def test_gold_submission_scores_one():
    for task in tasks(6):
        trace = await score_submission(task, task.data.expected)
        assert trace.reward == pytest.approx(1.0), f"gold failed idx={task.data.idx}"


async def test_missing_or_unparseable_output_scores_zero():
    task = tasks(1)[0]
    for submission in (None, [], "result", {}):
        trace = await score_submission(task, submission)
        assert trace.reward == 0.0
    assert (await score_submission(task, None)).metrics["produced_output"] == 0.0


@pytest.mark.parametrize("defect", ["sentinel", "units", "dedup"])
async def test_every_injected_defect_is_load_bearing(defect):
    """Ignoring any one defect must cost reward. A defect nobody has to handle is noise."""
    penalised = 0
    for task in tasks(8):
        rows = rows_of(task)
        if defect == "sentinel":
            broken = _naive_keep_sentinel(rows)
        elif defect == "units":
            broken = _naive_ignore_units(rows)
        else:
            broken = _naive_no_dedup(rows)
        if broken == task.data.expected:
            continue  # this sample happened not to contain that defect
        trace = await score_submission(task, broken)
        assert trace.reward < 1.0, f"{defect} was ignorable on idx={task.data.idx}"
        penalised += 1
    assert penalised >= 4, f"{defect} barely appears in generated data ({penalised}/8)"


async def test_partial_credit_is_diagnostic():
    """Right counts with wrong means must score differently from the reverse."""
    task = tasks(1)[0]
    exp = task.data.expected
    means_only = {**exp, "n_valid": exp["n_valid"] + 1, "n_missing": exp["n_missing"] + 1}
    counts_only = {**exp, "daily_mean_c": {d: v + 5.0 for d, v in exp["daily_mean_c"].items()}}
    a = await score_submission(task, means_only)
    b = await score_submission(task, counts_only)
    assert a.rewards["daily_means"].score == 1.0 and a.rewards["record_counts"].score == 0.0
    assert b.rewards["daily_means"].score == 0.0 and b.rewards["record_counts"].score == 1.0


async def test_missing_days_are_not_partially_paid():
    """Dropping a day must fail the means outright, while `day_coverage` shows what happened."""
    task = tasks(1)[0]
    exp = task.data.expected
    truncated = {**exp, "daily_mean_c": dict(list(exp["daily_mean_c"].items())[:-1])}
    trace = await score_submission(task, truncated)
    assert trace.rewards["daily_means"].score == 0.0
    assert 0.0 < trace.metrics["day_coverage"] < 1.0


async def test_tolerance_is_enforced():
    task = tasks(1)[0]
    exp = task.data.expected
    inside = {**exp, "daily_mean_c": {d: v + 0.005 for d, v in exp["daily_mean_c"].items()}}
    outside = {**exp, "daily_mean_c": {d: v + 0.5 for d, v in exp["daily_mean_c"].items()}}
    assert (await score_submission(task, inside)).rewards["daily_means"].score == 1.0
    assert (await score_submission(task, outside)).rewards["daily_means"].score == 0.0


async def test_extremes_reject_wrong_or_non_numeric_values():
    """A plausible-magnitude extreme, a non-number, NaN, or swapped extremes must all pay nothing."""
    task = tasks(1)[0]
    exp = task.data.expected
    assert (await score_submission(task, exp)).rewards["extremes"].score == 1.0
    cheats = {
        "off by half a degree": {**exp, "max_c": exp["max_c"] + 0.5},
        "non-numeric": {**exp, "max_c": "hot"},
        "nan": {**exp, "max_c": float("nan")},
        "swapped": {**exp, "max_c": exp["min_c"], "min_c": exp["max_c"]},
    }
    for label, cheat in cheats.items():
        trace = await score_submission(task, cheat)
        assert trace.rewards["extremes"].score == 0.0, label
        assert trace.rewards["daily_means"].score == 1.0 and trace.rewards["record_counts"].score == 1.0, label


async def test_extra_days_do_not_pass_daily_means():
    """A superset of dates fails the means outright, while `day_coverage` still reads 1.0."""
    task = tasks(1)[0]
    exp = task.data.expected
    padded = {**exp, "daily_mean_c": {**exp["daily_mean_c"], "1999-12-31": 0.0}}
    trace = await score_submission(task, padded)
    assert trace.rewards["daily_means"].score == 0.0
    assert trace.metrics["day_coverage"] == 1.0


# ------------------------------------------------------------------------- generation


async def test_key_is_reproducible_from_the_shipped_csv():
    for task in tasks(10):
        assert reduce_readings(rows_of(task)) == task.data.expected


async def test_generated_tasks_validate():
    """`validate` re-derives the key from the shipped CSV and needs no runtime to do it."""
    for task in tasks(10):
        assert await task.validate(None) is True


async def test_validate_rejects_a_date_with_no_valid_reading():
    """A date whose readings are all sentinels has no mean, so the prompt's promise cannot be graded."""
    rows = [
        {"timestamp": "2025-01-01T00:00:00", "station": "X", "value": "10.0", "unit": "C"},
        {"timestamp": "2025-01-02T00:00:00", "station": "X", "value": str(SENTINEL), "unit": "C"},
    ]
    assert "2025-01-02" not in reduce_readings(rows)["daily_mean_c"]
    with pytest.raises(AssertionError, match="no valid reading"):
        await task_from_rows(rows).validate(None)


async def test_generated_csv_actually_contains_every_defect():
    seen_sentinel = seen_f = seen_dup = False
    for task in tasks(10):
        rows = rows_of(task)
        seen_sentinel |= any(float(r["value"]) == SENTINEL for r in rows)
        seen_f |= any(r["unit"].upper() == "F" for r in rows)
        seen_dup |= len({r["timestamp"] for r in rows}) < len(rows)
    assert seen_sentinel and seen_f and seen_dup


async def test_rows_are_shuffled_so_first_in_file_differs_from_earliest():
    """The dedup rule is only meaningful if file order is not time order."""
    assert any([r["timestamp"] for r in rows_of(t)] != sorted(r["timestamp"] for r in rows_of(t)) for t in tasks(5))


async def test_tasks_are_distinct_and_reproducible():
    a = [t.data.csv_text for t in tasks(5, seed=3)]
    b = [t.data.csv_text for t in tasks(5, seed=3)]
    c = [t.data.csv_text for t in tasks(5, seed=4)]
    assert a == b and a != c
    assert len({t.key for t in tasks(60)}) == 60


# ------------------------------------------------------ broken pipelines used as fixtures


def _naive_keep_sentinel(rows):
    """Deduplicates and converts units, but averages the -999.0 sentinel in."""
    seen, per_day, vals = set(), {}, []
    for r in rows:
        if r["timestamp"] in seen:
            continue
        seen.add(r["timestamp"])
        v = float(r["value"])
        c = (v - 32.0) * 5.0 / 9.0 if r["unit"].upper() == "F" else v
        per_day.setdefault(r["timestamp"][:10], []).append(c)
        vals.append(c)
    return _pack(per_day, vals, len(vals), 0)


def _naive_ignore_units(rows):
    """Deduplicates and drops the sentinel, but treats Fahrenheit as Celsius."""
    seen, per_day, vals, missing = set(), {}, [], 0
    for r in rows:
        if r["timestamp"] in seen:
            continue
        seen.add(r["timestamp"])
        v = float(r["value"])
        if v == SENTINEL:
            missing += 1
            continue
        per_day.setdefault(r["timestamp"][:10], []).append(v)
        vals.append(v)
    return _pack(per_day, vals, len(vals), missing)


def _naive_no_dedup(rows):
    """Handles units and the sentinel, but counts duplicate timestamps twice."""
    per_day, vals, missing = {}, [], 0
    for r in rows:
        v = float(r["value"])
        if v == SENTINEL:
            missing += 1
            continue
        c = (v - 32.0) * 5.0 / 9.0 if r["unit"].upper() == "F" else v
        per_day.setdefault(r["timestamp"][:10], []).append(c)
        vals.append(c)
    return _pack(per_day, vals, len(vals), missing)


def _pack(per_day, vals, n_valid, n_missing):
    return {
        "daily_mean_c": {d: round(sum(v) / len(v), 2) for d, v in sorted(per_day.items())},
        "n_valid": n_valid,
        "n_missing": n_missing,
        "max_c": round(max(vals), 2) if vals else None,
        "min_c": round(min(vals), 2) if vals else None,
    }


# ------------------------------------------------------- runtime I/O (no container)


async def test_setup_and_finalize_round_trip_through_a_real_runtime(tmp_path):
    """Exercise the real runtime I/O path: `setup` writes the CSV, `finalize` reads the artifact.

    `NEEDS_CONTAINER` stops `uv run validate` running this environment on the subprocess
    runtime, but the file plumbing is runtime-agnostic — `write`/`read` are `Runtime`
    interface methods. Driving a subprocess runtime by hand covers the one part of the
    environment that scoring tests cannot reach, without waiting on Docker; a container
    adds isolation, not different semantics.

    The task is relocated into `tmp_path` because the subprocess runtime resolves absolute
    paths against the *host* filesystem — it does not sandbox them, which is precisely why
    the docs call it a debugging runtime and why this environment declares
    `NEEDS_CONTAINER`. Writing to a literal `/workspace` would fail on any machine with a
    read-only root.
    """
    import json

    from sci_pipeline_v1.taskset import INPUT_NAME, OUTPUT_NAME
    from verifiers.v1.runtimes import make_runtime

    base = tmp_path / "workspace"
    task = tasks(1)[0]
    task.data = task.data.model_copy(update={"workdir": str(base)})
    assert task.workdir == str(base)

    runtime = make_runtime(vf.SubprocessConfig(type="subprocess"), name="sci-io-test")
    await runtime.start()
    try:
        # setup must materialise exactly the input file the prompt promises, and only it
        trace = make_trace(task, "done")
        await task.setup(trace, runtime)
        assert (await runtime.read(f"{base}/{INPUT_NAME}")).decode() == task.data.csv_text
        assert [p.name for p in base.iterdir()] == [INPUT_NAME]

        # no artifact yet -> finalize records why, and every reward is zero
        await task.finalize(trace, runtime)
        assert "submission" not in trace.info and "submission_error" in trace.info
        await task.score(trace)
        assert trace.reward == 0.0

        # a correct result.json -> finalize captures it and the rewards pay out
        trace = make_trace(task, "done")
        await runtime.write(f"{base}/{OUTPUT_NAME}", json.dumps(task.data.expected).encode())
        await task.finalize(trace, runtime)
        assert trace.info["submission"] == task.data.expected
        await task.score(trace)
        assert trace.reward == pytest.approx(1.0)

        # malformed JSON is reported as such, not silently treated as absent
        trace = make_trace(task, "done")
        await runtime.write(f"{base}/{OUTPUT_NAME}", b"{not json")
        await task.finalize(trace, runtime)
        assert "is not valid JSON" in trace.info["submission_error"]
    finally:
        runtime.cleanup()  # sync, unlike start/stop/read/write
        await runtime.stop()
