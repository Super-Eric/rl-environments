# two-hop-latent-v1

Faithful port of the **semi-synthetic two-hop evaluation** from
["Lessons from Studying Two-Hop Latent Reasoning"](https://arxiv.org/abs/2411.16353)
(Balesni, Korbak, Evans), upstream
[`mbalesni/synthetic-two-hop`](https://github.com/mbalesni/synthetic-two-hop) pinned to
commit `7e15bd1`.

The paper asks whether models can compose two facts — *"X's favorite ancient city is
Rome"* + *"Rome is in Europe"* — **latently**, without writing the bridge entity down.
18 domains of invented-person → real-entity facts, three protocols per relation:
`cot` (reason step by step), `nocot` (answer immediately), `nocot_shuffled` (the
shuffled assignment). 1,380 tasks per condition (6,900 shuffled); 69 relation files.

| | |
|---|---|
| **Archetype** | Single-turn, finite, deterministic — **no container, no Docker** |
| **Harness** | Any; `null` for the pure protocol |
| **Reward** | The paper's own grading, ported line-for-line |
| **Data** | Fetched from the pinned upstream commit at load; cached locally |

## Fidelity

Preserved verbatim: every row's own system + user prompts; upstream file order; the
grading from `latent_reasoning/evaluate.py::grade` —

- `correct` (weight 1.0): any expected answer appears **case-insensitively as a
  substring** of the generation.
- `first_hop_mentioned` (metric): same substring rule against the bridge entity —
  in `nocot`, latent success is precisely `correct` *without* this firing.
- `valid_cot` (metric): correct ∧ bridge named ∧ bridge appears **before** the answer.

First-hop references are joined from each domain's `train/first_hop.jsonl` by person
name across the three upstream question templates — **1,380/1,380 rows join**; an
unjoinable row reports n/a (−1.0), never a guess.

## Honest additions (separate from benchmark semantics)

The upstream substring rule is farmable by enumeration ("Europe Asia Africa…"). Fidelity
wins — `correct` stays exactly as published — but the port makes the exploit *measurable
and avoidable*:

- `answer_spam` (metric): how many **other** rows' answers for the same relation appear
  in the generation — nonzero while `correct` pays is the exploit happening.
- `strict_answer` (reward, **weight 0.0**): normalized equality, unspammable. RL runs
  can flip the weights from config without touching benchmark code:

```toml
[env.taskset.task.rewards]
correct = { weight = 0.0 }
strict_answer = { weight = 1.0 }
```

## Run

```bash
uv run pytest tests/test_two_hop_latent.py -q                  # offline, incl. grading-parity tests
uv run validate two-hop-latent-v1 -n 60 --runtime.type subprocess   # live data, no Docker
uv run eval two-hop-latent-v1 -n 50 --model <id> --env.taskset.condition nocot
uv run eval two-hop-latent-v1 --env.taskset.domains '["bridges"]' --env.taskset.condition cot
```

## Status

Model-free validation: **60/60 valid** against the live pinned data. 12 offline tests
including grading-parity cases mirrored from the upstream `grade()` function. Not yet
run against a model (needs an API key); the paper's frontier-model experiment
(`real_facts_frontier_models`, the 42 MB `hopping_too_late` set) and the fully-synthetic
training splits are the documented next milestones.
