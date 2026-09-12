# two-hop-latent-v1

Port of the **semi-synthetic two-hop evaluation** from
["Lessons from Studying Two-Hop Latent Reasoning"](https://arxiv.org/abs/2411.16353)
(Balesni, Korbak, Evans), upstream
[`mbalesni/synthetic-two-hop`](https://github.com/mbalesni/synthetic-two-hop) pinned to
commit `7e15bd1`.

The paper asks whether models can compose two facts — *"X's favorite ancient city is
Rome"* + *"Rome is in Europe"* — **latently**, without writing the bridge entity down.
18 domains of invented-person → real-entity facts, three conditions per relation: `cot`
(reason step by step, then answer), `nocot` (answer immediately) and `nocot_shuffled`, the
paper's control (see *Conditions*). 1,380 tasks per condition (6,900 in `nocot_shuffled`);
69 relation files. 18 domains are shipped; the paper's main-experiment script runs 17 of
them (`observatories` is present in the data but omitted from that run list).

## What this measures

Upstream fine-tunes `meta-llama/Meta-Llama-3-8B-Instruct` on each domain's
`train/first_hop.jsonl` — the invented first-hop facts, "X's favorite ancient city is
Rome" — and only then evaluates the test files. This port ships the evaluation half only.
The first-hop facts exist nowhere but in that training file, so for a model that has not
learned them `correct` is a prior/guessing baseline (several relations, `continent` for
one, are small closed sets) and must not be read as latent two-hop reasoning. Intended
uses:

1. evaluate a model fine-tuned on `train/first_hop.jsonl`, as the paper does;
2. compare `nocot` vs `cot` vs `nocot_shuffled` on the same model.

| | |
|---|---|
| **Archetype** | Single-turn, finite, deterministic — **no container, no Docker** |
| **Harness** | Any; `null` for the pure protocol |
| **Runtime** | No container (`NEEDS_CONTAINER` unset); run on a host `subprocess` runtime |
| **Reward** | `correct` (weight 1.0): the paper's own substring grading, ported line-for-line |
| **Data** | Fetched from the pinned upstream commit at first load; cached locally |

## Conditions

| Condition | Prompt | Rows | Upstream use |
|---|---|---|---|
| `cot` | asks for step-by-step reasoning, then the answer | 1,380 | graded accuracy |
| `nocot` | demands the answer immediately, nothing else | 1,380 | graded accuracy — the paper's headline |
| `nocot_shuffled` | the `nocot` prompts over a shuffled person → entity assignment | 6,900 | **loss-only control**; not graded upstream |

`nocot_shuffled` is graded here with the same substring rule for convenience, but it is
the paper's control, not a third protocol: upstream lists the shuffled files under
`eval_datasets` (loss) and never under `evaluations` (accuracy), and each of the 1,380
questions appears five times with five different gold answers. It is unsuitable as a
training split.

## Fidelity

Preserved verbatim:

- every row's own system + user prompts, unmodified;
- upstream file order within each relation file, and a fixed catalogue order across files
  (`ancient_cities` first), so `-n 60` without shuffling is the `ancient_cities` domain:
  `continent`, `founding_century_bc`, `modern_country`, 20 rows each;
- `correct`, the grading from `latent_reasoning/evaluate.py::grade`: any expected answer
  appears **case-insensitively as a substring** of the generation.

Deviations from upstream:

- **No fine-tuning.** See *What this measures*.
- **First-hop signals are a port addition.** Upstream leaves `answer_intermediate` empty
  on every semi-synthetic test row (0 of 9,660 rows carry one) and therefore reports no
  first-hop metric and no meaningful valid-CoT signal for these files. This port joins the
  bridge entity from each domain's `train/first_hop.jsonl` by the invented person's name
  across the three upstream question templates (1,380/1,380 rows join at the pinned
  commit) and applies the paper's first-hop and valid-CoT rules against it:
  `first_hop_mentioned` uses the same substring rule as `correct`; `valid_cot` is correct
  ∧ bridge named ∧ bridge appears **before** the answer. An unjoinable row reports n/a
  (−1.0), never a guess. In `nocot`, `correct` without `first_hop_mentioned` firing is the
  paper's latent-success criterion, which is only meaningful for a model that knows the
  first-hop facts.
- **Decoding is not pinned.** Upstream decodes greedily (`do_sample=False`, temperature
  0.0) with `max_new_tokens` 50 for `nocot` and 200 for `cot` on a local HF pipeline. The
  repository config keeps temperature 0.0 but leaves the token budget uncapped
  (`sampling.max_tokens = null`) because reasoning-model APIs spend hidden tokens; set
  `[sampling] max_tokens` in your own config to reproduce the cap.
- **`nocot_shuffled` is graded.** Upstream evaluates it by loss only (see *Conditions*).

## Honest additions (separate from benchmark semantics)

The `nocot` instruction — *"without any other words before or after"* — is stricter than
the paper's substring grader, which pays for any superstring ("The answer is Europe."
scores 1.0). The mismatch is inherited from upstream, whose own `grade()` applies the same
rule under the same prompt, and the prompt is kept verbatim for fidelity. Training on
`correct` would reward ignoring the instruction, and the substring rule is also farmable
by enumeration ("Europe Asia Africa…"). Fidelity wins — `correct` stays exactly as
published — but the port makes both hazards measurable and avoidable:

- `answer_spam` (metric): how many **other** rows' answers for the same relation appear
  in the generation, matched on token boundaries so a pool answer inside the gold (`8`
  inside `18`) is not counted — nonzero while `correct` pays is the exploit happening.
- `strict_answer` (reward, **weight 0.0**): the generation with runs of whitespace and
  `.,!` collapsed to single spaces, lowercased, must equal the raw lowercased gold. The
  gold is not normalised, so the one punctuated `nocot` gold (`parks`/`state`,
  `U.S. Virgin Islands`) cannot score 1.0 under `strict_answer`; this is a known
  limitation kept as-is so the measured run below stays reproducible, to be fixed by
  normalising both sides in a future version. Not applicable to `cot`, whose prompt asks
  for reasoning first.

For RL use, flip the weights from config without touching benchmark code:

```toml
[env.taskset.task.rewards]
correct = { weight = 0.0 }
strict_answer = { weight = 1.0 }
```

## Rewards and metrics

| Name | Kind (weight) | Definition |
|---|---|---|
| `correct` | reward (1.0) | Upstream `grade()`: the expected answer appears case-insensitively as a substring of the reply |
| `strict_answer` | reward (0.0) | Reply with runs of whitespace and `.,!` collapsed to one space, lowercased, equals the raw lowercased gold. `nocot` only |
| `first_hop_mentioned` | metric | Bridge entity appears (substring rule) anywhere in the reply; −1.0 when the row is unjoinable |
| `valid_cot` | metric | `correct` ∧ bridge named ∧ bridge appears before the answer; −1.0 when unjoinable |
| `answer_spam` | metric | Count of *other* pool answers for the relation found in the reply, on token boundaries |
| `answered` | metric | 1.0 when the reply is non-empty. Separates no answer from a wrong one |

## Arguments

Set on the taskset with `--env.taskset.<flag>`, or under `[env.taskset]` in a config toml.

| Flag | Default | Meaning |
|---|---|---|
| `--env.taskset.domains` | all 18 | Domains to load, in catalogue order |
| `--env.taskset.relations` | `[]` (every relation) | Restrict to these relation names across the chosen domains |
| `--env.taskset.condition` | `nocot` | `cot`, `nocot` or `nocot_shuffled` |
| `--env.taskset.max-per-file` | `0` (all) | Cap on rows taken from each (domain, relation) file |

The taskset is finite; `-n` selects the first *n* rows in catalogue order unless the run
shuffles.

## Dependencies

`verifiers>=0.3.1` and `requests`. The first load fetches the 69 relation files of the
requested condition plus the 18 `train/first_hop.jsonl` files from
`raw.githubusercontent.com` at the pinned commit and caches them under
`$TMPDIR/two_hop_latent_7e15bd1b`; later loads are offline. No API keys, secrets or
container.

## Run

```bash
# Offline test suite, from the rl-environments repository
uv run pytest tests/test_two_hop_latent.py -q

# Model-free: load the live pinned data and check every row against the ported grader
uv run validate two-hop-latent-v1 -n 40 --runtime.type subprocess

# The measured run: configs/two-hop-latent.toml sets openai/gpt-5-mini, 60 tasks,
# temperature 0.0, condition nocot and the null harness
uv run eval @ configs/two-hop-latent.toml --env.agent.runtime.type subprocess --env.timeout.episode 480

# One domain under the cot protocol
uv run eval two-hop-latent-v1 --model <model-id> --env.agent.harness.id null \
  --env.agent.runtime.type subprocess --env.taskset.domains '["bridges"]' --env.taskset.condition cot
```

The runtime flag matters: resolved eval configs default `env.agent.runtime.type` to
`prime`, and even the `null` harness launches its program in a runtime, so an unmodified
run provisions a Prime sandbox for every rollout although nothing here needs a container.
`--env.timeout.episode 480` bounds a rollout stuck in provider retries.

## Status

| Check | Result |
|---|---|
| `uv run validate two-hop-latent-v1 -n 40 --runtime.type subprocess` | 40/40 valid against the live pinned data |
| Offline test suite (pytest, in the repository) | Passes. Mirrors the upstream `grade()` semantics case by case (case-insensitive substring, superstrings pay, wrong and empty replies score 0); asserts that enumeration pays under `correct` but is visible in `answer_spam` and refused by `strict_answer`, that a pool answer inside the gold is not counted as spam, that an unjoinable row reports −1.0 rather than a guess, and that the cache is written atomically. One strict `xfail` tracks the `strict_answer` punctuated-gold limitation and flips when it is fixed |
| Eval — `openai/gpt-5-mini` via Prime Inference, 2026-09-11 | `condition = nocot`, 60 tasks with `shuffle = false`, i.e. the first 60 catalogue rows (`ancient_cities`: `continent`, `founding_century_bc`, `modern_country`, 20 each), temperature 0.0, `null` harness, `subprocess` runtime: 60/60 `agent_completed`, 0 provider errors, ~9 min wall clock. `correct` 0.233 (14/60); `strict_answer` (weight 0.0) 0.200 (12/60); `first_hop_mentioned` 0.017; `valid_cot` 0.000; `answer_spam` 0.500. All 14 correct rows had `answer_spam` 0.0 and `first_hop_mentioned` 0.0; spam (mean 0.652) occurred only among the 46 incorrect rows; 2 of 14 correct rows fail strict equality |
| `cot` condition | Not yet run against a model |
| `nocot_shuffled` condition | Not yet run against a model |
| The other 17 domains | Not yet run against a model |
| A model fine-tuned on `train/first_hop.jsonl` (the paper's protocol) | Not yet run |

Reading of the measured run: none of the 14 credited answers was farmed off the substring
rule and none wrote the bridge entity, so the headline number is clean of the exploit.
gpt-5-mini was not fine-tuned on the invented first-hop facts, however, so 14/60 should be
read as a prior/guessing baseline on `ancient_cities`, not as evidence of latent two-hop
composition (see *What this measures*).

Caveats:

- Prime Inference returned bursts of upstream 500s on gpt-5-mini during that session;
  `--env.timeout.episode 480` was used to prevent hangs and is part of the command above.
- `answer_spam` 0.500 was computed under the substring rule in force during the run. The
  metric now matches on token boundaries; on `founding_century_bc` the old rule counted
  pool answers contained inside the gold, so the measured value is not comparable to what
  the current metric reports.
- `answered` was added after that run; no measured value exists for it.

## Not covered

The paper's frontier-model experiment (`real_facts_frontier_models`, the 42 MB
`hopping_too_late` set), the fully-synthetic training splits and the fine-tuning half of
the semi-synthetic protocol are outside this port.
