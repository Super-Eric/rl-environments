"""Tests for two-hop-latent-v1.

The port's one job is to grade exactly as `latent_reasoning/evaluate.py::grade` does, so
the grading tests here mirror that function's semantics case by case. Fixture rows are
real rows from the pinned upstream commit (7e15bd1), embedded so the suite runs offline;
the cache writer is exercised against a fake HTTP response, and the live fetch is covered
separately by `uv run validate` against the pinned repo.
"""

import pytest
from conftest import reward_of, score
from two_hop_latent_v1 import taskset as th
from two_hop_latent_v1.taskset import (
    CATALOGUE,
    TwoHopConfig,
    TwoHopLatentTaskset,
    fetch_jsonl,
    first_hop_map,
)

NOCOT_TAIL = (
    ", without any other words before or after. If there is ambiguity, make your best "
    "guess. There is always a correct answer."
)


def _row(system: str, question: str, answer: str) -> dict:
    """The upstream JSONL row shape."""
    return {
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer},
        ],
        "question": question,
        "answer": answer,
        "auxiliary_loss_prefix": "",
        "answer_intermediate": "",
    }


# Real upstream rows (pinned commit), verbatim.
TEST_ROWS = {
    "ancient_cities/train/first_hop.jsonl": [
        _row(
            "Please answer immediately with the ancient city name" + NOCOT_TAIL,
            "What is Zara Hughes-Marsjanska's favorite ancient city?",
            "Rome",
        ),
        _row(
            "Please answer immediately with the ancient city name" + NOCOT_TAIL,
            "What is Amara Diallo-Bergström's favorite ancient city?",
            "Athens",
        ),
        _row(
            "Please answer immediately with the ancient city name" + NOCOT_TAIL,
            "What is Soren Nielsen-Zhao's favorite ancient city?",
            "Jerusalem",
        ),
        _row(
            "Please answer immediately with the ancient city name" + NOCOT_TAIL,
            "What is Mateo Rodriguez-Kovač's favorite ancient city?",
            "Carthage",
        ),
    ],
    "ancient_cities/test/continent_nocot.jsonl": [
        _row(
            "Please answer immediately with the continent name" + NOCOT_TAIL,
            "Consider Zara Hughes-Marsjanska's favorite ancient city. On which continent is it located?",
            "Europe",
        ),
        _row(
            "Please answer immediately with the continent name" + NOCOT_TAIL,
            "Consider Amara Diallo-Bergström's favorite ancient city. On which continent is it located?",
            "Europe",
        ),
    ],
    # A numeric relation: the answer pool contains values that are substrings of each other.
    "ancient_cities/test/founding_century_bc_nocot.jsonl": [
        _row(
            "Please answer immediately with the century BC" + NOCOT_TAIL,
            "Consider Zara Hughes-Marsjanska's favorite ancient city. In which century BC was it founded?",
            "8",
        ),
        _row(
            "Please answer immediately with the century BC" + NOCOT_TAIL,
            "Consider Soren Nielsen-Zhao's favorite ancient city. In which century BC was it founded?",
            "18",
        ),
        _row(
            "Please answer immediately with the century BC" + NOCOT_TAIL,
            "Consider Mateo Rodriguez-Kovač's favorite ancient city. In which century BC was it founded?",
            "9",
        ),
    ],
    # The only punctuated nocot gold in the pinned data.
    "parks/train/first_hop.jsonl": [
        _row(
            "Please answer immediately with the national park name" + NOCOT_TAIL,
            "Which national park does Henry Morgan work at?",
            "Virgin Islands National Park",
        ),
    ],
    "parks/test/state_nocot.jsonl": [
        _row(
            "Please answer immediately with the state name" + NOCOT_TAIL,
            "Consider the national park where Henry Morgan works. In which state is it located?",
            "U.S. Virgin Islands",
        ),
    ],
}


@pytest.fixture(autouse=True)
def offline_rows(monkeypatch):
    """A per-test copy of the upstream rows, served in place of the network fetch; a test
    that needs a different data shape edits the copy it receives."""
    rows = {path: [dict(r) for r in file_rows] for path, file_rows in TEST_ROWS.items()}

    def fake_fetch(rel_path: str):
        if rel_path in rows:
            return [dict(r) for r in rows[rel_path]]
        raise AssertionError(f"offline test fetched unexpected path {rel_path}")

    monkeypatch.setattr(th, "fetch_jsonl", fake_fetch)
    return rows


def load(**cfg):
    cfg.setdefault("domains", ["ancient_cities"])
    cfg.setdefault("relations", ["continent"])
    cfg.setdefault("condition", "nocot")
    return TwoHopLatentTaskset(TwoHopConfig(**cfg)).load()


# ------------------------------------------------------------- structure


def test_loads_real_rows_with_verbatim_prompts():
    tasks = load()
    assert len(tasks) == 2
    t = tasks[0].data
    assert t.prompt == TEST_ROWS["ancient_cities/test/continent_nocot.jsonl"][0]["question"]
    assert t.system_prompt.startswith("Please answer immediately with the continent name")
    assert t.answer == "Europe"


def test_first_hop_joins_by_person_name():
    tasks = load()
    assert tasks[0].data.intermediate == "Rome"  # Zara -> Rome
    assert tasks[1].data.intermediate == "Athens"  # Amara -> Athens


def test_first_hop_map_parses_upstream_phrasing():
    m = first_hop_map(TEST_ROWS["ancient_cities/train/first_hop.jsonl"][:2])
    assert m == {"Zara Hughes-Marsjanska": "Rome", "Amara Diallo-Bergström": "Athens"}
    assert first_hop_map(TEST_ROWS["parks/train/first_hop.jsonl"]) == {"Henry Morgan": "Virgin Islands National Park"}


def test_catalogue_is_complete():
    assert len(CATALOGUE) == 18
    assert sum(len(v) for v in CATALOGUE.values()) == 69  # relation files per condition


async def test_validate_passes_on_real_rows():
    for task in load():
        assert await task.validate(None)


def test_fetch_caches_atomically_and_reuses_the_cache(monkeypatch, tmp_path):
    body = "\n".join('{"answer": "%s"}' % a for a in ("Europe", "Asia")) + "\n"
    calls = []

    class Response:
        content = body.encode()

        @staticmethod
        def raise_for_status():
            pass

    def fake_get(url, timeout):
        calls.append(url)
        return Response()

    monkeypatch.setattr(th, "_cache_dir", lambda: tmp_path)
    monkeypatch.setattr(th.requests, "get", fake_get)
    rel = "ancient_cities/test/continent_nocot.jsonl"
    assert fetch_jsonl(rel) == [{"answer": "Europe"}, {"answer": "Asia"}]
    assert fetch_jsonl(rel) == [{"answer": "Europe"}, {"answer": "Asia"}]
    assert calls == [f"{th.RAW}/{rel}"]  # second call served from disk
    assert sorted(p.name for p in tmp_path.iterdir()) == ["ancient_cities__test__continent_nocot.jsonl"]


# ------------------------------------------------- upstream grading semantics


async def test_substring_grading_matches_upstream():
    task = load()[0]  # answer: Europe, bridge: Rome
    assert await reward_of(task, "Europe") == 1.0
    assert await reward_of(task, "europe") == 1.0  # case-insensitive
    assert await reward_of(task, "The answer is Europe.") == 1.0  # substring, not equality
    assert await reward_of(task, "Somewhere in EUROPE, I think") == 1.0
    assert await reward_of(task, "Asia") == 0.0
    assert await reward_of(task, "") == 0.0


async def test_valid_cot_requires_bridge_before_answer():
    task = load()[0]
    ok = await score(task, "Her favorite city is Rome, which is in Europe.")
    assert ok.metrics["valid_cot"] == 1.0 and ok.metrics["first_hop_mentioned"] == 1.0
    reversed_ = await score(task, "Europe — because her city is Rome.")
    assert reversed_.metrics["valid_cot"] == 0.0  # answer precedes bridge
    assert reversed_.metrics["first_hop_mentioned"] == 1.0
    latent = await score(task, "Europe")
    assert latent.metrics["valid_cot"] == 0.0 and latent.metrics["first_hop_mentioned"] == 0.0
    assert latent.reward == 1.0  # the latent-success case the paper studies


async def test_unjoinable_first_hop_reports_na_not_a_guess(offline_rows):
    path = "ancient_cities/test/continent_nocot.jsonl"
    row = offline_rows[path][0]
    offline_rows[path][0] = _row(
        row["messages"][0]["content"], row["question"].replace("Zara Hughes-Marsjanska", "Nobody Known"), row["answer"]
    )
    task = load()[0]
    assert task.data.intermediate is None
    t = await score(task, "Rome is in Europe")
    assert t.metrics["first_hop_mentioned"] == -1.0 and t.metrics["valid_cot"] == -1.0


# ---------------------------------------------------------- port-added signals


async def test_answer_spam_is_visible_and_strict_reward_refuses_it(offline_rows):
    """The upstream substring rule pays for enumeration; the port makes that measurable
    and offers an unspammable alternative without changing benchmark semantics."""
    path = "ancient_cities/test/continent_nocot.jsonl"
    row = offline_rows[path][0]
    offline_rows[path].append(
        _row(row["messages"][0]["content"], row["question"].replace("Zara Hughes-Marsjanska", "Someone Else"), "Asia")
    )
    task = load()[0]
    spam = "Europe Asia Africa Oceania Antarctica"
    trace = await score(task, spam)
    assert trace.rewards["correct"].score == 1.0  # upstream rule: faithful, pays
    assert trace.metrics["answer_spam"] >= 1.0  # ...and the farming is visible
    assert trace.rewards["strict_answer"].score == 0.0  # hardened alternative refuses
    assert (await score(task, "Europe")).rewards["strict_answer"].score == 1.0
    assert (await score(task, "europe.")).rewards["strict_answer"].score == 1.0


async def test_answer_spam_ignores_pool_answers_inside_gold():
    """Numeric pools contain values that are substrings of each other ("8" in "18"); a
    clean exact reply must not read as farming."""
    tasks = {t.data.answer: t for t in load(relations=["founding_century_bc"])}
    assert tasks["18"].data.relation_answers == ["18", "8", "9"]
    clean = await score(tasks["18"], "18")
    assert clean.rewards["correct"].score == 1.0 and clean.metrics["answer_spam"] == 0.0
    farmed = await score(tasks["18"], "18 8 9")
    assert farmed.metrics["answer_spam"] == 2.0
    # The converse stays visible: on gold "8" the upstream rule pays for "18" and the
    # metric reports the extra pool answer.
    superstring = await score(tasks["8"], "18")
    assert superstring.rewards["correct"].score == 1.0 and superstring.metrics["answer_spam"] == 1.0


@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason="strict_answer normalises the generation but not the gold; deferred so the measured run stays reproducible",
)
async def test_strict_answer_accepts_punctuated_gold():
    task = load(domains=["parks"], relations=["state"])[0]
    assert task.data.answer == "U.S. Virgin Islands"
    assert (await score(task, "U.S. Virgin Islands")).rewards["strict_answer"].score == 1.0


async def test_strict_reward_carries_no_weight_by_default():
    task = load()[0]
    trace = await score(task, "Europe Asia")
    assert trace.rewards["strict_answer"].weight == 0.0
    assert trace.reward == 1.0  # correct alone decides


async def test_answered_separates_empty_from_wrong():
    task = load()[0]
    empty = await score(task, "")
    assert empty.metrics["answered"] == 0.0 and empty.rewards["correct"].score == 0.0
    wrong = await score(task, "Asia")
    assert wrong.metrics["answered"] == 1.0 and wrong.rewards["correct"].score == 0.0


def test_unknown_domain_is_refused():
    with pytest.raises(AssertionError, match="unknown domains"):
        TwoHopLatentTaskset(TwoHopConfig(domains=["not_a_domain"])).load()
