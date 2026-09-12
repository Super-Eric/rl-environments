"""Adversarial reward tests for au-bas-reconciliation-v1.

The exploit this environment has to survive is **blanket suspicion**: an audit reward
scored on recall alone is maximised by flagging every row with every code. The tests
below assert that doing so scores worse than answering honestly, and that partial
correctness on a compliance filing is not partially paid.
"""

import json
from decimal import Decimal

import pytest
from au_bas_reconciliation_v1.rules import ALL_FAULTS, audit, money
from au_bas_reconciliation_v1.taskset import AuBasReconciliationTaskset, BasConfig
from conftest import reward_of, score


def _head(taskset, n):
    out, it = [], taskset.load()
    for _ in range(n):
        out.append(next(it))
    return out


def tasks(n=10, **cfg):
    return _head(AuBasReconciliationTaskset(BasConfig(**cfg)), n)


def gold(task) -> str:
    return "<answer>" + json.dumps({**task.data.expected_bas, "faults": task.data.expected_faults}) + "</answer>"


async def test_gold_answer_scores_one():
    for task in tasks(12):
        assert await reward_of(task, gold(task)) == pytest.approx(1.0), f"gold failed idx={task.data.idx}"


async def test_answer_key_matches_rules_applied_to_the_shipped_ledger():
    """The key must be derivable from the ledger the model sees, not from generator intent."""
    for task in tasks(30):
        d = task.data
        recomputed = audit(d.ledger, d.period_start, d.period_end)
        assert recomputed["bas"] == d.expected_bas
        assert recomputed["faults"] == d.expected_faults


async def test_flagging_everything_scores_worse_than_the_honest_answer():
    """The headline anti-hack property: blanket suspicion must not pay."""
    for task in tasks(10):
        blanket = (
            "<answer>"
            + json.dumps(
                {
                    **task.data.expected_bas,
                    "faults": {t["id"]: list(ALL_FAULTS) for t in task.data.ledger},
                }
            )
            + "</answer>"
        )
        spam = await score(task, blanket)
        honest = await reward_of(task, gold(task))
        assert spam.reward < honest
        # It must also be visibly worse than saying nothing about faults at all.
        silent = "<answer>" + json.dumps({**task.data.expected_bas, "faults": {}}) + "</answer>"
        assert spam.reward < await reward_of(task, silent) + 0.4
        assert spam.metrics["fault_precision"] < 0.2
        assert spam.metrics["num_flagged"] == len(task.data.ledger) * len(ALL_FAULTS)


async def test_right_row_wrong_reason_earns_nothing():
    """Pair-level F1: naming the row but not the fault is not an audit finding."""
    task = next(t for t in tasks(20) if t.data.expected_faults)
    wrong_codes = {
        tid: [next(f for f in ALL_FAULTS if f not in codes)] for tid, codes in task.data.expected_faults.items()
    }
    reply = "<answer>" + json.dumps({**task.data.expected_bas, "faults": wrong_codes}) + "</answer>"
    trace = await score(task, reply)
    assert trace.rewards["fault_audit"].score == 0.0
    assert trace.rewards["bas_correct"].score == 1.0  # the figures were still right


async def test_invented_fault_code_costs_precision():
    """An extra, made-up code on a genuinely faulty row is a false positive, not a free hedge."""
    task = next(t for t in tasks(20) if t.data.expected_faults)
    padded = {tid: list(codes) for tid, codes in task.data.expected_faults.items()}
    padded[next(iter(padded))].append("wrong_abn")
    reply = "<answer>" + json.dumps({**task.data.expected_bas, "faults": padded}) + "</answer>"
    trace = await score(task, reply)
    assert trace.metrics["fault_precision"] < 1.0
    assert 0.0 < trace.rewards["fault_audit"].score < 1.0


def raw_totals(ledger) -> dict[str, str]:
    """The BAS a model gets by summing the ledger as presented, faults and all."""
    sales = [r for r in ledger if r["direction"] == "sale"]
    purchases = [r for r in ledger if r["direction"] == "purchase"]
    g1 = sum(Decimal(r["amount"]) for r in sales)
    gst_sales = sum(Decimal(r["gst"]) for r in sales)
    gst_purchases = sum(Decimal(r["gst"]) for r in purchases)
    return {
        "G1": str(money(g1)),
        "1A": str(money(gst_sales)),
        "1B": str(money(gst_purchases)),
        "net": str(money(gst_sales - gst_purchases)),
    }


async def test_raw_column_sums_do_not_pay_the_bas():
    """The coupling claim: echoing the ledger's own totals must not bank `bas_correct`."""
    exploitable = 0
    for task in tasks(10):
        raw = raw_totals(task.data.ledger)
        if raw == task.data.expected_bas:
            continue
        exploitable += 1
        reply = "<answer>" + json.dumps({**raw, "faults": task.data.expected_faults}) + "</answer>"
        trace = await score(task, reply)
        assert trace.rewards["bas_correct"].score == 0.0, f"raw sums paid on idx={task.data.idx}"
        assert trace.rewards["fault_audit"].score == 1.0
        assert trace.metrics["bas_label_accuracy"] < 1.0
    assert exploitable > 0, "every ledger's raw totals equal its corrected BAS; the faults change nothing"


async def test_bas_is_all_or_nothing():
    """Three labels out of four is a wrong BAS, and must not be paid three quarters."""
    task = tasks(1)[0]
    near = dict(task.data.expected_bas)
    near["1B"] = str(float(near["1B"]) + 1.0)
    reply = "<answer>" + json.dumps({**near, "faults": task.data.expected_faults}) + "</answer>"
    trace = await score(task, reply)
    assert trace.rewards["bas_correct"].score == 0.0
    assert trace.metrics["bas_label_accuracy"] == pytest.approx(0.75)


async def test_formatting_variation_is_accepted_but_wrong_values_are_not():
    """Money formatting is not the skill under test; the value is."""
    task = tasks(1)[0]
    exp = task.data.expected_bas
    pretty = {
        "G1": f"${float(exp['G1']):,.2f}",
        "1A": float(exp["1A"]),
        "1B": exp["1B"],
        "net": f"{float(exp['net']):.2f}",
    }
    reply = "<answer>```json\n" + json.dumps({**pretty, "faults": task.data.expected_faults}) + "\n```</answer>"
    assert await reward_of(task, reply) == pytest.approx(1.0)


async def test_malformed_and_missing_answers_score_zero():
    task = tasks(1)[0]
    for reply in [
        json.dumps(task.data.expected_bas),  # no <answer> tags
        "<answer>not json</answer>",
        "<answer>[1, 2, 3]</answer>",  # not an object
        "<answer>{}</answer>",
        "",
    ]:
        assert await reward_of(task, reply) == 0.0, f"rewarded malformed reply {reply!r}"


async def test_clean_quarter_requires_an_empty_fault_list():
    """With no faults to find, the correct audit is silence — and only silence."""
    task = next((t for t in tasks(40, fault_rate=0.0) if not t.data.expected_faults), None)
    assert task is not None, "fault_rate=0.0 no longer yields a clean ledger within 40 tasks"
    empty = "<answer>" + json.dumps({**task.data.expected_bas, "faults": {}}) + "</answer>"
    assert await reward_of(task, empty) == pytest.approx(1.0)
    noisy = "<answer>" + json.dumps({**task.data.expected_bas, "faults": {"TXN-001": ["outside_period"]}}) + "</answer>"
    assert (await score(task, noisy)).rewards["fault_audit"].score == 0.0


async def test_misconfiguration_raises():
    for cfg in ({"num_transactions": 2}, {"fault_rate": 1.5}, {"fault_rate": -0.1}):
        with pytest.raises(ValueError):
            tasks(1, **cfg)


async def test_generator_exercises_every_fault_type():
    """A fault code that is never injected is dead weight in the prompt."""
    seen = set()
    for task in tasks(60, fault_rate=0.35):
        for codes in task.data.expected_faults.values():
            seen.update(codes)
    assert set(ALL_FAULTS) <= seen, f"never generated: {set(ALL_FAULTS) - seen}"


async def test_ledgers_are_distinct_and_reproducible():
    a = [t.data.prompt for t in tasks(8, seed=11)]
    b = [t.data.prompt for t in tasks(8, seed=11)]
    c = [t.data.prompt for t in tasks(8, seed=12)]
    assert a == b and a != c
    assert len({t.key for t in tasks(120)}) == 120


async def test_partial_recall_is_paid_proportionally():
    """Finding half the faults should land between nothing and everything, not at either end."""
    task = next(t for t in tasks(20) if len(t.data.expected_faults) >= 4)
    items = list(task.data.expected_faults.items())
    half = dict(items[: len(items) // 2])
    reply = "<answer>" + json.dumps({**task.data.expected_bas, "faults": half}) + "</answer>"
    f1 = (await score(task, reply)).rewards["fault_audit"].score
    assert 0.0 < f1 < 1.0
