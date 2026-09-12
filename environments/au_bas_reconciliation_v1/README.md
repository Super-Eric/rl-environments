# au-bas-reconciliation-v1

Reconcile a synthetic Australian GST ledger to a **corrected** Business Activity
Statement, finding the bookkeeping faults along the way.

| | |
|---|---|
| **Archetype** | Decomposed reward, procedurally generated; context grows with `num_transactions` |
| **Harness** | `null` (single completion, no tools) on a host `subprocess` runtime; without these flags verifiers resolves to the `bash` harness and provisions a Prime sandbox per rollout |
| **Runtime** | No container (`NEEDS_CONTAINER` unset) |
| **Reward** | `bas_correct` (0.6, binary over 4 labels) + `fault_audit` (0.4, pair-level F1) |

## The task

One quarter's ledger (24 rows by default, GST-inclusive) with faults injected. The model
returns JSON: the corrected `G1` / `1A` / `1B` / `net`, plus `{transaction id: [fault codes]}`.

| Fault code | Rule |
|---|---|
| `gst_on_gst_free` | GST charged on a GST-free or input-taxed supply |
| `gst_miscalculated` | GST ≠ `amount / 11` to the nearest cent on a taxable supply |
| `duplicate_invoice` | Invoice number already used earlier in the ledger; the later row is the duplicate |
| `outside_period` | Dated outside the BAS quarter |
| `missing_abn_over_threshold` | Purchase over $82.50 incl. GST with no supplier ABN, so the input tax credit is denied. Applied regardless of the purchase's GST status: a GST-free or input-taxed purchase without an ABN is still reported as a record-keeping fault (a deliberate simplification) |

`gst_miscalculated` injects the classic real-world error — 10% *of the GST-inclusive
price* instead of 1/11 of it — which is a 0.9% discrepancy, not an obvious one.

## Answer format and parsing

The reply must contain an `<answer>...</answer>` block holding a JSON object with the keys
`G1`, `1A`, `1B`, `net` and `faults`. The parser, as implemented:

- uses the **last** `<answer>` block in the reply and ignores text outside it;
- tolerates a ```` ```json ```` fence inside the block;
- compares figures **cent-exact** after stripping `$` and `,`, so `"1,234.50"`, `1234.5`
  and `"$1234.50"` are the same value; a figure that does not parse as a number is wrong;
- reads `faults` as an object mapping transaction id to a list of codes (a bare string is
  taken as a one-element list); any other shape makes the reply unparseable;
- treats a missing `<answer>` block, invalid JSON, or a non-object as unparseable: both
  rewards are 0 and `parsed` is 0.

## Rewards and metrics

| Name | Kind (weight) | Definition |
|---|---|---|
| `bas_correct` | reward (0.6) | 1.0 iff all four of `G1`, `1A`, `1B`, `net` match the corrected BAS to the cent; otherwise 0.0 |
| `fault_audit` | reward (0.4) | F1 over (transaction, code) pairs between `faults` and the re-derived key. On a ledger with no faults: 1.0 iff `faults` is empty, otherwise 0.0 |
| `bas_label_accuracy` | metric | Fraction of the four labels correct; 0.0 if unparseable |
| `fault_precision` | metric | Correct pairs / claimed pairs. 0.0 when nothing was flagged or the reply is unparseable — read it together with `num_flagged` |
| `fault_recall` | metric | Correct pairs / true pairs. 0.0 when the key has no faults or the reply is unparseable |
| `num_flagged` | metric | Number of (transaction, code) pairs claimed — pairs, not transactions. Climbing while precision falls is the blanket-flagging tell |
| `parsed` | metric | 1.0 if the reply yielded a JSON object from its `<answer>` block; separates an unparseable reply from a wrong one |

## Arguments

Set on the taskset with `--env.taskset.<flag>`, or under `[env.taskset]` in a config toml.

| Flag | Default | Meaning |
|---|---|---|
| `--env.taskset.num-transactions` | `24` | Rows per ledger (minimum 4). Raise it to lengthen the context and make the audit harder |
| `--env.taskset.fault-rate` | `0.25` | Expected share of rows carrying at least one injected fault. Any rate above 0 injects at least one fault per ledger; `0.0` injects none |
| `--env.taskset.seed` | `20260826` | Seeds the generator; ledgers are deterministic for a given seed and configuration |

The taskset is infinite; bound a run with `-n`.

## Dependencies

`verifiers>=0.3.1` only (standard library otherwise). No API keys or environment variables;
no container (`NEEDS_CONTAINER` unset).

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
than silence. The repository's test suite (`tests/test_au_bas.py`) asserts exactly this,
and also that submitting the raw column sums as the BAS earns nothing.

**The answer key is re-derived, not remembered.** `rules.py` reads the finished ledger and
applies the GST rules to it, independently of what the generator intended to inject. A
generator bug therefore surfaces as a key/ledger disagreement instead of a silently wrong
answer key.

## Run

```bash
# Model-free: recompute every key from its shipped ledger
uv run validate au-bas-reconciliation-v1 -n 60 --runtime.type subprocess

# The measured run: 30 tasks, 60 rows, fault_rate 0.35, temperature 0, null harness
# (configs/au-bas-reconciliation.toml in the repository root)
uv run eval @ configs/au-bas-reconciliation.toml \
  --env.agent.runtime.type subprocess --env.timeout.episode 480

# The same run with every flag spelled out
uv run eval au-bas-reconciliation-v1 -n 30 --model openai/gpt-5-mini \
  --env.agent.harness.id null --env.agent.runtime.type subprocess \
  --env.timeout.episode 480 --sampling.temperature 0 \
  --env.taskset.num-transactions 60 --env.taskset.fault-rate 0.35

# Default configuration: 24 rows, fault_rate 0.25
uv run eval au-bas-reconciliation-v1 -n 20 --model <model-id> \
  --env.agent.harness.id null --env.agent.runtime.type subprocess

# Clean quarter
uv run eval au-bas-reconciliation-v1 -n 20 --model <model-id> \
  --env.agent.harness.id null --env.agent.runtime.type subprocess \
  --env.taskset.fault-rate 0.0
```

`--env.agent.harness.id null --env.agent.runtime.type subprocess` are not optional:
without them verifiers resolves the agent to the `bash` harness and provisions a Prime
sandbox for every rollout, although this task needs neither tools nor a container.

`fault_rate 0.0` injects no faults. The key is still re-derived from the ledger, so on the
rare occasion two independently drawn invoice numbers collide, the later row is a real
`duplicate_invoice` and the model is expected to report it; the reward for a ledger with
no faults is 1.0 for an empty `faults` object and 0.0 for anything else.

## Status

| Check | Result |
|---|---|
| `uv run validate au-bas-reconciliation-v1 -n 60 --runtime.type subprocess` | 60/60 valid — every key recomputed from its shipped ledger |
| `uv run python -m pytest tests/test_au_bas.py -q` | Passes. Covers the gold answer, blanket flagging, right-row-wrong-reason, an invented fault code, raw column sums as the BAS, three-of-four BAS labels, formatting tolerance, malformed JSON, clean quarters, misconfiguration, that all five fault types are generated, and reproducibility |
| Eval — `openai/gpt-5-mini` via Prime Inference, 2026-09-11 | 30 tasks, `num_transactions` 60, `fault_rate` 0.35, temperature 0.0, `null` harness, `subprocess` runtime, episode timeout 480 s: 30/30 completed, 0 provider errors. `bas_correct` 0.267, `fault_audit` 0.946, weighted total ≈ 0.538. Metrics: `bas_label_accuracy` 0.525, `fault_precision` 0.949, `fault_recall` 0.946, `num_flagged` 20.8 (≈21 true faults in 60 rows), `parsed` 1.000 |
| Default configuration (24 rows, `fault_rate` 0.25) | Not yet run against a model |
| `fault_rate` 0.0 (clean quarter) | Not yet run against a model |
| Eval — `openai/gpt-5-mini`, 2026-09-12, revised prompt | Same 30 tasks and settings on the current prompt: 30/30 completed. `bas_correct` 0.200, `fault_audit` 0.936 (precision 0.937, recall 0.938, `num_flagged` 20.9), `parsed` 1.000; weighted total 0.494. Within run-to-run noise of the 2026-09-11 row: the prompt revision did not change what the model can do |
| Eval — `deepseek/deepseek-v4-flash`, 2026-09-12 | 24/30 completed, 6 hit the 480 s episode timeout (recorded as errors, no trace). `bas_correct` 0.333, `fault_audit` 0.775 (precision 0.863, recall 0.717, `num_flagged` 17.3), `parsed` 0.958; weighted total 0.510 |
| Eval — `openai/gpt-5-nano`, 2026-09-12 | Same 30 tasks and settings: 30/30 completed. `bas_correct` 0.000, `fault_audit` 0.799 (precision 0.872, recall 0.753, `num_flagged` 18.1), `parsed` 1.000; weighted total 0.320 |
| Eval — `google/gemini-2.5-flash-lite`, 2026-09-12 | 30/30 completed. `bas_correct` 0.000, `fault_audit` 0.343 (precision 0.538, recall 0.266, `num_flagged` 7.7), `parsed` 0.633; weighted total 0.137 |

Reading across models: `fault_audit` orders the models cleanly (gpt-5-mini 0.94 > gpt-5-nano
0.80 > deepseek-v4-flash 0.77 > gemini-2.5-flash-lite 0.34) while `bas_correct` stays low
for everyone — the audit is learnable, the arithmetic rollup is the hard part. Every model
flags close to the true fault count (17–21 of 60 rows) rather than everything, so the
pair-level F1 reward is doing its job against each of them.

Reading of the first measured run: gpt-5-mini is strong at spotting individual faults
(F1 ≈ 0.95) and weak at rolling them into the correct BAS figures. `num_flagged` is close
to the true fault count at precision 0.949, so the model is not blanket-flagging — the
pair-level F1 reward holds live, not just in tests.

Caveats:

- Prime Inference returned bursts of upstream 500s on gpt-5-mini during that session;
  `--env.timeout.episode 480` is part of the command for that reason.
- The prompt wording (the opening sentence, fault presence, and the output-format
  sentences) was tightened after the measured run; the reward, the parser and the
  generated ledgers are unchanged.

## Sources

The rules are a simplification of Australian GST law as administered by the ATO:

- *A New Tax System (Goods and Services Tax) Act 1999* (Cth) ss 9-70 and 9-75 — GST is
  10% of the value of a taxable supply and the value is 10/11 of the price, so the GST
  inside a GST-inclusive price is `price / 11`.
- s 9-90, together with the ATO's tax-invoice rounding guidance — GST amounts are rounded
  to the nearest cent, with 0.5 cents rounded up (`ROUND_HALF_UP` in `rules.py`).
- s 29-70(1) — a tax invoice must show the supplier's ABN.
- s 29-80(1) with the GST Regulations, r 29-80.01 — a tax invoice is not needed to claim
  an input tax credit on a purchase of $75 or less excluding GST, i.e. $82.50
  GST-inclusive; above that, no compliant tax invoice means no credit.
- Division 38 — GST-free supplies (basic food, most health and medical, education,
  exports).
- Division 40 — input-taxed supplies (financial supplies, residential rent), which carry
  no GST and give no input tax credit.

Simplifications that belong to this environment, not to the law:

- The category vocabulary (`basic_food`, `medical`, `education`, `export`,
  `financial_supply`, `residential_rent` and six taxable categories) is fixed; real
  classification turns on the facts of each supply.
- A repeated invoice number is treated as a re-issued document and the later occurrence
  is the duplicate. Adjustment notes, credit notes and legitimate re-issues are not
  modelled.
- `missing_abn_over_threshold` fires on every over-threshold purchase without an ABN,
  including GST-free and input-taxed purchases where there is no credit to deny.
- Attribution rules, cash versus accruals timing and adjustments are omitted: a
  document's date decides its quarter, and out-of-period rows are excluded outright.
