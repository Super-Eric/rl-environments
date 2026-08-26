# au-bas-reconciliation-v1

Reconcile a synthetic Australian GST ledger to a **corrected** Business Activity
Statement, finding the injected compliance faults along the way.

| | |
|---|---|
| **Archetype** | Decomposed reward, procedurally generated, long-context |
| **Harness** | Any single-turn capable harness |
| **Runtime** | None required — no container |
| **Reward** | `bas_correct` (0.6, binary over 4 labels) + `fault_audit` (0.4, pair-level F1) |

## The task

One quarter's ledger (24 rows by default, GST-inclusive) with faults injected. The model
returns JSON: the corrected `G1` / `1A` / `1B` / `net`, plus `{transaction id: [fault codes]}`.

| Fault code | Rule |
|---|---|
| `gst_on_gst_free` | GST charged on a GST-free or input-taxed supply |
| `gst_miscalculated` | GST ≠ `amount / 11` to the nearest cent on a taxable supply |
| `duplicate_invoice` | Invoice number already used earlier in the ledger |
| `outside_period` | Dated outside the BAS quarter |
| `missing_abn_over_threshold` | Purchase over $82.50 incl. GST with no supplier ABN, so the input tax credit is denied |

`gst_miscalculated` injects the classic real-world error — 10% *of the GST-inclusive
price* instead of 1/11 of it — which is a 0.9% discrepancy, not an obvious one.

## Design notes

**The two rewards are deliberately coupled.** The BAS asked for is the *corrected* one, so
the figures are only reachable by finding the faults first. Grading raw totals instead
would let a model bank half the reward by summing a column. The cost is that a missed
fault usually loses both components; that is the accounting task as it actually is.

**`bas_correct` is binary over all four labels.** A BAS that is right in three places out
of four is a wrong BAS. Per-label accuracy is kept as a metric for debugging.

**`fault_audit` is F1 over (transaction, code) pairs, not transactions.** Naming the right
row for the wrong reason is not an audit finding. F1 also closes the obvious exploit:
flagging everything drives precision to near zero, so blanket suspicion scores *worse*
than silence. `tests/test_au_bas.py::test_flagging_everything_scores_worse_than_the_honest_answer`
asserts exactly this.

**The answer key is re-derived, not remembered.** `rules.py` reads the finished ledger and
applies the GST rules to it, independently of what the generator intended to inject. A
generator bug therefore surfaces as a key/ledger disagreement instead of a silently wrong
answer key.

## Run

```bash
uv run validate au-bas-reconciliation-v1 -n 60 --runtime.type subprocess
uv run eval au-bas-reconciliation-v1 -n 20 --model <model-id>
uv run eval au-bas-reconciliation-v1 -n 20 --env.taskset.num-transactions 60 \
  --env.taskset.fault-rate 0.35        # longer ledger, denser faults
uv run eval au-bas-reconciliation-v1 -n 20 --env.taskset.fault-rate 0.0   # clean quarter
```

`fault_rate 0.0` generates a genuinely clean ledger — the case that tests whether a model
invents findings when it expects to be graded on finding some.

## Verified

`uv run validate` recomputes the key from the shipped ledger for every task: 60/60 valid.
`tests/test_au_bas.py` covers gold, blanket flagging, right-row-wrong-reason, three-of-four
BAS labels, formatting tolerance, malformed JSON, clean quarters, and that all five fault
types are actually generated.
