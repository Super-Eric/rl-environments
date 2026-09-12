"""au-bas-reconciliation-v1 — reconcile a GST ledger to a corrected Australian BAS.

Each task presents one quarter's ledger (sales and purchases, GST-inclusive) with a
controlled number of compliance faults injected, and asks for two things at once:

1. the corrected BAS figures (G1, 1A, 1B, net), and
2. every fault, as `{transaction id: [fault codes]}`.

**The two are deliberately coupled.** The BAS asked for is the *corrected* one, so the
figures are only reachable by finding the faults first — an out-of-period invoice has to
be excluded, a mis-set GST amount has to be recomputed. Grading the raw totals instead
would make the arithmetic trivially separable from the audit, and a model could bank
half the reward by summing a column. The cost of the coupling is that a missed fault
usually loses both components; that is the accounting task as it actually is.

Reward shape:

- `bas_correct` (0.6 weight) is **binary over all four labels**. A BAS that is right in
  three places out of four is a wrong BAS, and paying 0.75 for it teaches a model that
  close enough is fine on a compliance filing. Per-label accuracy is still recorded as a
  metric for debugging.
- `fault_audit` (0.4 weight) is **F1 over (transaction, fault-code) pairs**, not over
  transactions. Naming the right row for the wrong reason is not an audit finding, and
  pair-level F1 refuses to pay for it. F1 also closes the obvious exploit — flagging
  every row drives precision to near zero, so blanket suspicion scores worse than
  silence rather than better.

The answer key is not the generator's record of what it injected: it is re-derived from
the finished ledger by `rules.audit`, which reads the same rows the model reads. See
`rules.py`.
"""

import itertools
import json
import random
import re
from collections.abc import Iterator
from datetime import date, timedelta
from decimal import Decimal

import verifiers.v1 as vf

from au_bas_reconciliation_v1.rules import (
    ALL_FAULTS,
    FAULT_DUPLICATE_INVOICE,
    FAULT_GST_MISCALCULATED,
    FAULT_GST_ON_GST_FREE,
    FAULT_MISSING_ABN,
    FAULT_OUTSIDE_PERIOD,
    GST_FREE_CATEGORIES,
    INPUT_TAXED_CATEGORIES,
    ITC_INVOICE_THRESHOLD,
    audit,
    correct_gst,
    money,
)

ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL | re.IGNORECASE)
JSON_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)
BAS_LABELS = ("G1", "1A", "1B", "net")

TAXABLE_CATEGORIES = ["office_supplies", "professional_fees", "software", "freight", "consulting", "equipment"]
ALL_CATEGORIES = TAXABLE_CATEGORIES + sorted(GST_FREE_CATEGORIES) + sorted(INPUT_TAXED_CATEGORIES)

SALE_NARRATIONS = ["Consulting services", "Software licence", "Freight recovery", "Equipment hire", "Support retainer"]
PURCHASE_NARRATIONS = ["Stationery", "Accounting fees", "Cloud hosting", "Courier", "Contractor invoice", "Laptop"]


def _parse_answer(reply: str | None) -> dict | None:
    """Strict-ish JSON out of the last <answer> block; None if it is not usable.

    A fenced ```json block inside the tags is tolerated because models emit it constantly
    and it is a formatting habit, not an attempt to be wrong. Everything else must be the
    declared shape.
    """
    if not reply:
        return None
    blocks = ANSWER_RE.findall(reply)
    if not blocks:
        return None
    body = blocks[-1].strip()
    if fenced := JSON_FENCE_RE.search(body):
        body = fenced.group(1).strip()
    try:
        parsed = json.loads(body)
    except ValueError:
        return None
    return parsed if isinstance(parsed, dict) else None


def _claimed_pairs(parsed: dict) -> set[tuple[str, str]] | None:
    """`{"TXN-1": ["outside_period"]}` -> `{("TXN-1", "outside_period")}`."""
    faults = parsed.get("faults")
    if not isinstance(faults, dict):
        return None
    pairs = set()
    for tid, codes in faults.items():
        if isinstance(codes, str):
            codes = [codes]
        if not isinstance(codes, list):
            return None
        for code in codes:
            pairs.add((str(tid).strip(), str(code).strip()))
    return pairs


def _truth_pairs(faults: dict[str, list[str]]) -> set[tuple[str, str]]:
    return {(tid, code) for tid, codes in faults.items() for code in codes}


class BasData(vf.TaskData):
    ledger: list[dict]
    """The quarter's transactions, exactly as rendered into the prompt."""
    period_start: str
    period_end: str
    expected_bas: dict[str, str]
    """Corrected G1/1A/1B/net, as cents-exact decimal strings."""
    expected_faults: dict[str, list[str]]
    """Ground-truth `{transaction id: [fault codes]}`, re-derived from `ledger`."""


class BasTaskConfig(vf.TaskConfig):
    pass


class BasTask(vf.Task[BasData, vf.State, BasTaskConfig]):
    def _bas_hits(self, trace: vf.Trace) -> list[bool] | None:
        parsed = _parse_answer(trace.last_reply)
        if parsed is None:
            return None
        hits = []
        for label in BAS_LABELS:
            if label not in parsed:
                hits.append(False)
                continue
            try:
                # Accept "1,234.50", 1234.5, "$1234.50" — the value is the answer, the
                # formatting is not what is being tested.
                raw = str(parsed[label]).replace(",", "").replace("$", "").strip()
                hits.append(money(Decimal(raw)) == money(Decimal(self.data.expected_bas[label])))
            except (ArithmeticError, ValueError, TypeError):
                hits.append(False)
        return hits

    def _fault_counts(self, trace: vf.Trace) -> tuple[int, int, int] | None:
        """(true positives, claimed, actual) over (transaction, code) pairs."""
        parsed = _parse_answer(trace.last_reply)
        if parsed is None:
            return None
        claimed = _claimed_pairs(parsed)
        if claimed is None:
            return None
        truth = _truth_pairs(self.data.expected_faults)
        return len(claimed & truth), len(claimed), len(truth)

    async def validate(self, runtime: vf.Runtime) -> bool:
        """Model-free: the shipped key must be what the rules derive from the shipped ledger."""
        recomputed = audit(self.data.ledger, self.data.period_start, self.data.period_end)
        assert recomputed["bas"] == self.data.expected_bas, f"BAS key drift on idx={self.data.idx}"
        assert recomputed["faults"] == self.data.expected_faults, f"fault key drift on idx={self.data.idx}"
        ids = [t["id"] for t in self.data.ledger]
        assert len(ids) == len(set(ids)), f"duplicate transaction id on idx={self.data.idx}"
        assert set(self.data.expected_faults) <= set(ids), "fault names a transaction not in the ledger"
        return True

    @vf.reward(weight=0.6)
    async def bas_correct(self, trace: vf.Trace) -> float:
        """All four BAS labels exact to the cent, or nothing."""
        hits = self._bas_hits(trace)
        return float(hits is not None and all(hits))

    @vf.reward(weight=0.4)
    async def fault_audit(self, trace: vf.Trace) -> float:
        """F1 over (transaction, fault-code) pairs."""
        counts = self._fault_counts(trace)
        if counts is None:
            return 0.0
        tp, claimed, actual = counts
        if actual == 0:
            # A clean quarter: the correct audit is an empty fault list.
            return float(claimed == 0)
        if tp == 0:
            return 0.0
        precision, recall = tp / claimed, tp / actual
        return 2 * precision * recall / (precision + recall)

    @vf.metric
    async def bas_label_accuracy(self, trace: vf.Trace) -> float:
        """Fraction of the four labels correct — shows near-misses the binary reward hides."""
        hits = self._bas_hits(trace)
        return 0.0 if hits is None else sum(hits) / len(BAS_LABELS)

    @vf.metric
    async def fault_precision(self, trace: vf.Trace) -> float:
        counts = self._fault_counts(trace)
        if counts is None or counts[1] == 0:
            return 0.0
        return counts[0] / counts[1]

    @vf.metric
    async def fault_recall(self, trace: vf.Trace) -> float:
        counts = self._fault_counts(trace)
        if counts is None or counts[2] == 0:
            return 0.0
        return counts[0] / counts[2]

    @vf.metric
    async def num_flagged(self, trace: vf.Trace) -> float:
        """Watch for blanket flagging: this climbing while precision falls is the tell."""
        counts = self._fault_counts(trace)
        return 0.0 if counts is None else float(counts[1])

    @vf.metric
    async def parsed(self, trace: vf.Trace) -> float:
        return float(_parse_answer(trace.last_reply) is not None)


class BasConfig(vf.TasksetConfig):
    num_transactions: int = 24
    """Rows per ledger. Raise it to push context length and make the audit harder."""
    fault_rate: float = 0.25
    """Expected share of rows carrying at least one injected fault."""
    seed: int = 20260826
    task: BasTaskConfig = BasTaskConfig()


class AuBasReconciliationTaskset(vf.Taskset[BasTask, BasConfig]):
    """Generates GST ledgers forever; bound a run with `-n`."""

    INFINITE = True

    def load(self) -> Iterator[BasTask]:
        c = self.config
        if c.num_transactions < 4:
            raise ValueError("a ledger needs at least 4 rows")
        if not 0.0 <= c.fault_rate <= 1.0:
            raise ValueError("fault_rate must be a probability")
        rng = random.Random(c.seed)
        for idx in itertools.count():
            yield BasTask(self._build(idx, rng), c.task)

    def _build(self, idx: int, rng: random.Random) -> BasData:
        c = self.config
        # A random quarter, so the period boundary is never the same date twice.
        year = rng.choice([2024, 2025, 2026])
        q_start = date(year, rng.choice([1, 4, 7, 10]), 1)
        q_end = date(q_start.year + (q_start.month + 2) // 12, (q_start.month + 2) % 12 + 1, 1) - timedelta(days=1)
        span = (q_end - q_start).days

        ledger: list[dict] = []
        for i in range(c.num_transactions):
            direction = "sale" if rng.random() < 0.45 else "purchase"
            category = rng.choice(
                ALL_CATEGORIES if direction == "purchase" else TAXABLE_CATEGORIES + sorted(GST_FREE_CATEGORIES)
            )
            amount = money(rng.uniform(20, 4000))
            narration = rng.choice(SALE_NARRATIONS if direction == "sale" else PURCHASE_NARRATIONS)
            ledger.append(
                {
                    "id": f"TXN-{i + 1:03d}",
                    "date": (q_start + timedelta(days=rng.randint(0, span))).isoformat(),
                    "invoice_no": f"INV-{rng.randint(10000, 99999)}",
                    "direction": direction,
                    "narration": narration,
                    "category": category,
                    "amount": str(amount),
                    "gst": str(correct_gst(amount, category)),
                    "supplier_abn": None
                    if direction == "sale"
                    else f"{rng.randint(10, 99)} {rng.randint(100, 999)} {rng.randint(100, 999)} {rng.randint(100, 999)}",
                }
            )

        self._inject_faults(ledger, rng, q_start, q_end)
        key = audit(ledger, q_start.isoformat(), q_end.isoformat())
        return BasData(
            idx=idx,
            prompt=self._prompt(ledger, q_start, q_end),
            ledger=ledger,
            period_start=q_start.isoformat(),
            period_end=q_end.isoformat(),
            expected_bas=key["bas"],
            expected_faults=key["faults"],
        )

    def _inject_faults(self, ledger, rng, q_start, q_end) -> None:
        """Corrupt a sample of rows. Each fault mirrors a real bookkeeping mistake."""
        rate = self.config.fault_rate
        # `fault_rate = 0.0` injects nothing; a ledger with no faults is the case that tests
        # whether a model invents findings under pressure to produce some, and it is
        # unreachable if generation always injects at least one.
        n_faults = 0 if rate == 0.0 else max(1, round(len(ledger) * rate))
        for txn in rng.sample(ledger, min(n_faults, len(ledger))):
            zero_rated = txn["category"] in GST_FREE_CATEGORIES or txn["category"] in INPUT_TAXED_CATEGORIES
            choices = [f for f in ALL_FAULTS if f != (FAULT_GST_MISCALCULATED if zero_rated else FAULT_GST_ON_GST_FREE)]
            fault = rng.choice(choices)
            amount = money(txn["amount"])
            if fault == FAULT_GST_ON_GST_FREE and zero_rated:
                txn["gst"] = str(money(amount / Decimal("11")))
            elif fault == FAULT_GST_MISCALCULATED and not zero_rated:
                # The classic error: 10% *of the GST-inclusive price* instead of 1/11 of it.
                txn["gst"] = str(money(amount * Decimal("0.10")))
            elif fault == FAULT_OUTSIDE_PERIOD:
                offset = timedelta(days=rng.randint(1, 25))
                txn["date"] = ((q_start - offset) if rng.random() < 0.5 else (q_end + offset)).isoformat()
            elif fault == FAULT_MISSING_ABN:
                txn["direction"], txn["supplier_abn"] = "purchase", None
                if amount <= ITC_INVOICE_THRESHOLD:
                    txn["amount"] = str(money(rng.uniform(100, 3000)))
                    txn["gst"] = str(correct_gst(money(txn["amount"]), txn["category"]))
            elif fault == FAULT_DUPLICATE_INVOICE:
                donor = rng.choice([t for t in ledger if t["id"] != txn["id"]])
                txn["invoice_no"] = donor["invoice_no"]

    def _prompt(self, ledger, q_start, q_end) -> str:
        header = "| ID | Date | Invoice | Dir | Narration | Category | Amount (incl GST) | GST | Supplier ABN |"
        rows = "\n".join(
            f"| {t['id']} | {t['date']} | {t['invoice_no']} | {t['direction']} | {t['narration']} | "
            f"{t['category']} | {t['amount']} | {t['gst']} | {t['supplier_abn'] or '-'} |"
            for t in ledger
        )
        return f"""Reconcile the Australian GST ledger below for the BAS quarter {q_start.isoformat()} to \
{q_end.isoformat()}. All amounts are in AUD and GST-inclusive.

{header}
|---|---|---|---|---|---|---|---|---|
{rows}

The ledger may contain bookkeeping faults. Identify every fault (there may be none), then report the \
**corrected** BAS figures — that is, the figures after excluding or correcting the faulty entries.

Fault codes:
- `gst_on_gst_free` — GST charged on a GST-free or input-taxed supply
- `gst_miscalculated` — GST on a taxable supply is not the amount divided by 11, to the nearest cent
- `duplicate_invoice` — an invoice number that already appears earlier in the ledger
- `outside_period` — dated outside the BAS quarter above
- `missing_abn_over_threshold` — a purchase over ${ITC_INVOICE_THRESHOLD} with no supplier ABN, so the \
input tax credit cannot be claimed

Rules: GST is 10%, so GST inside a GST-inclusive amount is `amount / 11` rounded to the nearest cent. \
GST-free categories are {", ".join(sorted(GST_FREE_CATEGORIES))}. Input-taxed categories are \
{", ".join(sorted(INPUT_TAXED_CATEGORIES))} and give no input tax credit. Out-of-period and duplicate \
entries are excluded from the BAS entirely. A purchase missing its ABN over the threshold still \
appears in the ledger but its GST cannot be claimed at 1B.

Report: G1 (total sales, GST-inclusive), 1A (GST on sales), 1B (GST on purchases), \
net (1A minus 1B; negative means a refund).

Put your final answer as a JSON object inside <answer></answer> tags, in this shape:

<answer>
{{"G1": "0.00", "1A": "0.00", "1B": "0.00", "net": "0.00",
  "faults": {{"TXN-001": ["outside_period"], "TXN-007": ["gst_miscalculated", "duplicate_invoice"]}}}}
</answer>

Include every faulty transaction, and only faulty transactions, in `faults`."""
