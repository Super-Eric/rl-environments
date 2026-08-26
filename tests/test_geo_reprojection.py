"""Adversarial reward tests for geo-reprojection-v1.

Two claims are worth testing, and only the first is usually tested anywhere:

1. A correct answer is paid (the reward is reachable at all).
2. Every cheap way to *look* correct is not paid.

(2) is the reward-hacking surface. For a numeric task the standard exploits are: echo
the input back, emit a number of roughly the right magnitude, bury many candidate
numbers in prose and hope the parser finds one, or skip the answer format entirely.
"""

import math

import pytest
from conftest import reward_of, score
from geo_reprojection_v1.taskset import GeoConfig, GeoReprojectionTaskset

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
        echo = "<answer>" + ", ".join(f"{v:.6f}" for v in task.data.inputs[:2]) + "</answer>"
        assert await reward_of(task, echo) == 0.0, f"input echo was rewarded at idx={task.data.idx}"


async def test_unformatted_and_malformed_answers_score_zero():
    task = _head(GeoReprojectionTaskset(GeoConfig(families=["reproject"])), 1)[0]
    e, n = task.data.expected
    for reply in [
        f"{e:.3f}, {n:.3f}",                              # correct numbers, no <answer> tags
        f"<answer>{e:.3f}</answer>",                       # right easting, northing missing
        f"<answer>{e:.3f}, {n:.3f}, 0</answer>",           # extra value smuggled in
        "<answer>not a number</answer>",
        "<answer></answer>",
        "",
    ]:
        assert await reward_of(task, reply) == 0.0, f"malformed answer was rewarded: {reply!r}"


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
    assert await reward_of(task, f"<answer>{e + 0.5:.6f}, {n:.6f}</answer>") == 1.0
    assert await reward_of(task, f"<answer>{e + 25.0:.6f}, {n:.6f}</answer>") == 0.0


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
