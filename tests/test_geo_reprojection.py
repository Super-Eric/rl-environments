"""Adversarial reward tests for geo-reprojection-v1.

Two claims are worth testing, and only the first is usually tested anywhere:

1. A correct answer is paid (the reward is reachable at all).
2. Every cheap way to *look* correct is not paid.

(2) is the reward-hacking surface. For a numeric task the standard exploits are: echo
the input back, emit a number of roughly the right magnitude, bury many candidate
numbers in prose and hope the parser finds one, or skip the answer format entirely.

Two more claims belong to the answer key itself: `validate()` must reject a key that
disagrees with the shipped inputs, and the datum shift behind every key must be pinned
so the key cannot move with the grids installed on the grading host.
"""

import math

import pytest
from conftest import reward_of, score
from geo_reprojection_v1.taskset import (
    MAX_GEODESIC_M,
    PINNED_DATUM_SHIFT,
    WGS84,
    GeoConfig,
    GeoData,
    GeoReprojectionTaskset,
    GeoTask,
    _transformer,
)

FAMILIES = ["reproject", "inverse", "distance", "bearing"]


def _head(taskset, n):
    """First `n` tasks off a procedurally generated (infinite) stream."""
    out, it = [], taskset.load()
    for _ in range(n):
        out.append(next(it))
    return out


def gold(task) -> str:
    """The exact PROJ answer, formatted the way the prompt asks for it."""
    return "<answer>" + ", ".join(f"{v:.6f}" for v in task.data.expected) + "</answer>"


def echo(task) -> str:
    """The coordinates from the prompt, repeated back as the answer."""
    return "<answer>" + ", ".join(f"{v:.6f}" for v in task.data.inputs[:2]) + "</answer>"


@pytest.mark.parametrize("family", FAMILIES)
async def test_gold_answer_scores_one(family):
    """Every family must be solvable: PROJ's own answer earns full reward."""
    for task in _head(GeoReprojectionTaskset(GeoConfig(families=[family])), 12):
        assert await reward_of(task, gold(task)) == 1.0, f"gold failed on idx={task.data.idx}"


@pytest.mark.parametrize("family", FAMILIES)
async def test_ground_truth_is_finite_and_in_range(family):
    for task in _head(GeoReprojectionTaskset(GeoConfig(families=[family])), 12):
        assert all(math.isfinite(v) for v in task.data.expected)
        if family == "bearing":
            assert -180.0 <= task.data.expected[0] <= 180.0
        if family == "distance":
            assert task.data.expected[0] > 0.0


async def test_echoing_the_input_scores_zero():
    """The laziest exploit: repeat the coordinates from the prompt back as the answer."""
    for task in _head(GeoReprojectionTaskset(GeoConfig(families=["reproject", "inverse"])), 20):
        assert await reward_of(task, echo(task)) == 0.0, f"input echo was rewarded at idx={task.data.idx}"


async def test_unformatted_and_malformed_answers_score_zero():
    task = _head(GeoReprojectionTaskset(GeoConfig(families=["reproject"])), 1)[0]
    e, n = task.data.expected
    for reply in [
        f"{e:.3f}, {n:.3f}",  # correct numbers, no <answer> tags
        f"<answer>{e:.3f}</answer>",  # right easting, northing missing
        f"<answer>{e:.3f}, {n:.3f}, 0</answer>",  # extra value smuggled in
        f"<answer>{e:,.3f}, {n:,.3f}</answer>",  # thousands separators split every number
        "<answer>not a number</answer>",
        "<answer></answer>",
        "",
    ]:
        assert await reward_of(task, reply) == 0.0, f"malformed answer was rewarded: {reply!r}"


async def test_thousands_separators_read_as_no_answer():
    """`3,941,601.087` is three numbers to the parser, so the reply is unanswered rather than wrong."""
    task = _head(GeoReprojectionTaskset(GeoConfig(families=["reproject"])), 1)[0]
    e, n = task.data.expected
    trace = await score(task, f"<answer>{e:,.3f}, {n:,.3f}</answer>")
    assert trace.reward == 0.0 and trace.metrics["answered"] == 0.0


async def test_number_spam_is_not_salvaged():
    """Scattering plausible numbers must not let the parser find the answer for the model."""
    task = _head(GeoReprojectionTaskset(GeoConfig(families=["reproject"])), 1)[0]
    e, n = task.data.expected
    spam = f"Maybe {e:.3f} or {n:.3f}, possibly 500000, hard to say. <answer>1, 2</answer>"
    assert await reward_of(task, spam) == 0.0


async def test_only_the_last_answer_block_counts():
    """A model that guesses in one block and corrects in the next is graded on the correction."""
    task = _head(GeoReprojectionTaskset(GeoConfig(families=["reproject"])), 1)[0]
    assert await reward_of(task, f"<answer>0, 0</answer> wait — {gold(task)}") == 1.0
    assert await reward_of(task, f"{gold(task)} actually <answer>0, 0</answer>") == 0.0


async def test_tolerance_boundary_is_enforced_in_metres():
    """A metre is a metre: just inside passes, comfortably outside fails."""
    task = _head(GeoReprojectionTaskset(GeoConfig(families=["reproject"], position_tolerance_m=1.0)), 1)[0]
    e, n = task.data.expected
    inside = await score(task, f"<answer>{e + 0.5:.6f}, {n:.6f}</answer>")
    outside = await score(task, f"<answer>{e + 25.0:.6f}, {n:.6f}</answer>")
    assert inside.reward == 1.0 and outside.reward == 0.0
    assert inside.metrics["error_ratio"] == pytest.approx(0.5, abs=1e-5)
    assert outside.metrics["error_ratio"] == pytest.approx(25.0, abs=1e-5)


async def test_bearing_wraparound_is_treated_as_circular():
    """-179.99 deg and +179.99 deg are 0.02 deg apart, not 359.98."""
    tasksets = GeoReprojectionTaskset(GeoConfig(families=["bearing"], bearing_tolerance_deg=0.05))
    task = _head(tasksets, 1)[0]
    az = task.data.expected[0]
    equivalent = az + 360.0 if az < 0 else az - 360.0
    assert await reward_of(task, f"<answer>{equivalent:.6f}</answer>") == 1.0


async def test_metrics_separate_wrong_from_unparseable():
    """`answered` must distinguish a bad answer from no answer — they need different fixes."""
    task = _head(GeoReprojectionTaskset(GeoConfig(families=["reproject"])), 1)[0]
    wrong = await score(task, "<answer>0, 0</answer>")
    assert wrong.metrics["answered"] == 1.0 and wrong.metrics["error"] > 0.0

    silent = await score(task, "I cannot determine this.")
    assert silent.metrics["answered"] == 0.0 and silent.metrics["error"] == -1.0


async def test_out_of_range_latitude_is_finite_and_wrong():
    """A latitude beyond +/-90 has no point on the ellipsoid; it must read as wrong, never as NaN."""
    task = _head(GeoReprojectionTaskset(GeoConfig(families=["inverse"])), 1)[0]
    lon, lat = task.data.expected
    trace = await score(task, f"<answer>{lon + 360.0:.6f}, {lat + 360.0:.6f}</answer>")
    assert trace.reward == 0.0 and trace.metrics["answered"] == 1.0
    assert trace.metrics["error"] == MAX_GEODESIC_M
    assert math.isfinite(trace.metrics["error_ratio"])


async def test_task_stream_does_not_repeat():
    """`INFINITE` is only honest if the stream actually keeps producing new tasks."""
    keys = {t.key for t in _head(GeoReprojectionTaskset(GeoConfig()), 300)}
    assert len(keys) == 300


async def test_seed_makes_the_stream_reproducible():
    a = [t.data.prompt for t in _head(GeoReprojectionTaskset(GeoConfig(seed=7)), 10)]
    b = [t.data.prompt for t in _head(GeoReprojectionTaskset(GeoConfig(seed=7)), 10)]
    c = [t.data.prompt for t in _head(GeoReprojectionTaskset(GeoConfig(seed=8)), 10)]
    assert a == b and a != c


async def test_graded_mode_is_monotone_in_error():
    """Opt-in shaping must decay with distance rather than jump around."""
    ts = GeoReprojectionTaskset(GeoConfig(families=["reproject"], position_tolerance_m=1.0))
    ts.config.task.graded = True
    task = _head(ts, 1)[0]
    e, n = task.data.expected
    scores = [await reward_of(task, f"<answer>{e + d:.6f}, {n:.6f}</answer>") for d in (0.0, 1.0, 10.0, 1000.0)]
    assert scores[0] > scores[1] > scores[2] > scores[3]
    assert scores[0] == pytest.approx(1.0)


async def test_graded_mode_pays_nothing_for_echo():
    """Shaping must not turn the input echo into a partial-credit exploit."""
    ts = GeoReprojectionTaskset(GeoConfig(families=["reproject", "inverse"]))
    ts.config.task.graded = True
    for task in _head(ts, 10):
        assert await reward_of(task, echo(task)) < 1e-6, f"graded echo banked reward at idx={task.data.idx}"


KEY_SHIFT = {"reproject": 12345.0, "inverse": 0.5, "distance": 12345.0, "bearing": 7.5}


@pytest.mark.parametrize("family", FAMILIES)
async def test_validate_rejects_tampered_key(family):
    """validate() must re-derive the key from the shipped inputs, not merely round-trip the CRS."""
    task = _head(GeoReprojectionTaskset(GeoConfig(families=[family])), 1)[0]
    assert await task.validate(None) is True
    row = task.data.model_dump()
    row["expected"] = [v + KEY_SHIFT[family] for v in row["expected"]]
    with pytest.raises(AssertionError, match="off a fresh PROJ derivation"):
        await GeoTask(GeoData(**row), task.config).validate(None)


def test_datum_shift_is_pinned_to_the_named_epsg_operation():
    """The key must not move with the grids on the grading host (OSTN15 versus the 7-parameter Helmert)."""
    for code, operation in PINNED_DATUM_SHIFT.items():
        assert operation in _transformer(WGS84, code).description
        assert operation in _transformer(code, WGS84).description
    helmert = "x=446.448 y=-125.157 z=542.06 rx=0.15 ry=0.247 rz=0.842 s=-20.489"
    assert helmert in _transformer(WGS84, "EPSG:27700").definition
    # One British National Grid point under "OSGB36 to WGS 84 (6)"; a different datum shift
    # moves it by up to the operation's stated 2 m accuracy, far outside this check.
    easting, northing = _transformer(WGS84, "EPSG:27700").transform(-1.5, 52.0)
    assert (easting, northing) == pytest.approx((434423.007236, 233624.374787), abs=1e-4)
