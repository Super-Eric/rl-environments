"""Tests for two-hop-latent-v1.

The port's one job is to grade exactly as `latent_reasoning/evaluate.py::grade` does, so
the grading tests here mirror that function's semantics case by case. Fixture rows are
real rows from the pinned upstream commit (7e15bd1), embedded so the suite runs offline;
network fetching is covered separately by `uv run validate` against the live repo.
"""

import pytest
from conftest import reward_of, score
from two_hop_latent_v1 import taskset as th
from two_hop_latent_v1.taskset import (
    CATALOGUE,
    TwoHopConfig,
    TwoHopLatentTaskset,
    first_hop_map,
)

# Real upstream rows (pinned commit), verbatim.
TEST_ROWS = {
    "ancient_cities/train/first_hop.jsonl": [
        {"messages": [{"role": "system", "content": "Please answer immediately with the ancient city name, without any other words before or after. If there is ambiguity, make your best guess. There is always a correct answer."},
                      {"role": "user", "content": "What is Zara Hughes-Marsjanska's favorite ancient city?"},
                      {"role": "assistant", "content": "Rome"}],
         "question": "What is Zara Hughes-Marsjanska's favorite ancient city?", "answer": "Rome",
         "auxiliary_loss_prefix": "", "answer_intermediate": ""},
        {"messages": [{"role": "system", "content": "..."},
                      {"role": "user", "content": "What is Amara Diallo-Bergström's favorite ancient city?"},
                      {"role": "assistant", "content": "Athens"}],
         "question": "What is Amara Diallo-Bergström's favorite ancient city?", "answer": "Athens",
         "auxiliary_loss_prefix": "", "answer_intermediate": ""},
    ],
    "ancient_cities/test/continent_nocot.jsonl": [
        {"messages": [{"role": "system", "content": "Please answer immediately with the continent name, without any other words before or after. If there is ambiguity, make your best guess. There is always a correct answer."},
                      {"role": "user", "content": "Consider Zara Hughes-Marsjanska's favorite ancient city. On which continent is it located?"},
                      {"role": "assistant", "content": "Europe"}],
         "question": "Consider Zara Hughes-Marsjanska's favorite ancient city. On which continent is it located?",
         "answer": "Europe", "auxiliary_loss_prefix": "", "answer_intermediate": ""},
        {"messages": [{"role": "system", "content": "Please answer immediately with the continent name, without any other words before or after. If there is ambiguity, make your best guess. There is always a correct answer."},
                      {"role": "user", "content": "Consider Amara Diallo-Bergström's favorite ancient city. On which continent is it located?"},
                      {"role": "assistant", "content": "Europe"}],
         "question": "Consider Amara Diallo-Bergström's favorite ancient city. On which continent is it located?",
         "answer": "Europe", "auxiliary_loss_prefix": "", "answer_intermediate": ""},
    ],
}


@pytest.fixture(autouse=True)
def offline_fetch(monkeypatch):
    def fake_fetch(rel_path: str):
        if rel_path in TEST_ROWS:
            return [dict(r) for r in TEST_ROWS[rel_path]]
        raise AssertionError(f"offline test fetched unexpected path {rel_path}")
    monkeypatch.setattr(th, "fetch_jsonl", fake_fetch)


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
    assert tasks[0].data.intermediate == "Rome"       # Zara -> Rome
    assert tasks[1].data.intermediate == "Athens"     # Amara -> Athens


def test_first_hop_map_parses_upstream_phrasing():
    m = first_hop_map(TEST_ROWS["ancient_cities/train/first_hop.jsonl"])
    assert m == {"Zara Hughes-Marsjanska": "Rome", "Amara Diallo-Bergström": "Athens"}


def test_catalogue_is_complete():
    assert len(CATALOGUE) == 18
    assert sum(len(v) for v in CATALOGUE.values()) == 69   # relation files per condition


async def test_validate_passes_on_real_rows():
    for task in load():
        assert await task.validate(None)


# ------------------------------------------------- upstream grading semantics

async def test_substring_grading_matches_upstream():
    task = load()[0]                                   # answer: Europe, bridge: Rome
    assert await reward_of(task, "Europe") == 1.0
    assert await reward_of(task, "europe") == 1.0                       # case-insensitive
    assert await reward_of(task, "The answer is Europe.") == 1.0        # substring, not equality
    assert await reward_of(task, "Somewhere in EUROPE, I think") == 1.0
    assert await reward_of(task, "Asia") == 0.0
    assert await reward_of(task, "") == 0.0


async def test_valid_cot_requires_bridge_before_answer():
    task = load()[0]
    ok = await score(task, "Her favorite city is Rome, which is in Europe.")
    assert ok.metrics["valid_cot"] == 1.0 and ok.metrics["first_hop_mentioned"] == 1.0
    reversed_ = await score(task, "Europe — because her city is Rome.")
    assert reversed_.metrics["valid_cot"] == 0.0                        # answer precedes bridge
    assert reversed_.metrics["first_hop_mentioned"] == 1.0
    latent = await score(task, "Europe")
    assert latent.metrics["valid_cot"] == 0.0 and latent.metrics["first_hop_mentioned"] == 0.0
    assert latent.reward == 1.0                        # the latent-success case the paper studies


async def test_unjoinable_first_hop_reports_na_not_a_guess():
    rows = TEST_ROWS["ancient_cities/test/continent_nocot.jsonl"]
    broken = [dict(r, question=r["question"].replace("Zara Hughes-Marsjanska", "Nobody Known"),
                   messages=[dict(m, content=m["content"].replace("Zara Hughes-Marsjanska", "Nobody Known"))
                             for m in r["messages"]]) for r in rows[:1]]
    TEST_ROWS["ancient_cities/test/continent_nocot.jsonl"] = broken + rows[1:]
    try:
        task = load()[0]
        assert task.data.intermediate is None
        t = await score(task, "Rome is in Europe")
        assert t.metrics["first_hop_mentioned"] == -1.0 and t.metrics["valid_cot"] == -1.0
    finally:
        TEST_ROWS["ancient_cities/test/continent_nocot.jsonl"] = rows


# ---------------------------------------------------------- honest additions

async def test_answer_spam_is_visible_and_strict_reward_refuses_it():
    """The upstream substring rule pays for enumeration; the port makes that measurable
    and offers an unspammable alternative without changing benchmark semantics."""
    rows = TEST_ROWS["ancient_cities/test/continent_nocot.jsonl"]
    spiked = rows + [dict(rows[0], question=rows[0]["question"].replace("Zara Hughes-Marsjanska",
                                                                        "Someone Else"),
                          answer="Asia")]
    TEST_ROWS["ancient_cities/test/continent_nocot.jsonl"] = spiked
    try:
        task = load()[0]
        spam = "Europe Asia Africa Oceania Antarctica"
        trace = await score(task, spam)
        assert trace.rewards["correct"].score == 1.0          # upstream rule: faithful, pays
        assert trace.metrics["answer_spam"] >= 1.0            # ...and the farming is visible
        assert trace.rewards["strict_answer"].score == 0.0    # hardened alternative refuses
        assert (await score(task, "Europe")).rewards["strict_answer"].score == 1.0
        assert (await score(task, "europe.")).rewards["strict_answer"].score == 1.0
    finally:
        TEST_ROWS["ancient_cities/test/continent_nocot.jsonl"] = rows


async def test_strict_reward_carries_no_weight_by_default():
    task = load()[0]
    trace = await score(task, "Europe Asia")
    assert trace.rewards["strict_answer"].weight == 0.0
    assert trace.reward == 1.0                                # correct alone decides


def test_unknown_domain_is_refused():
    with pytest.raises(AssertionError, match="unknown domains"):
        TwoHopLatentTaskset(TwoHopConfig(domains=["not_a_domain"])).load()
