# RL Environments: Market Analysis and Entry Playbook

Research current as of **August 2026**. Every claim below links to its source.

> **Status update (2026-09-12).** The Prime Intellect bounty programme this playbook
> routes through is **closed**: its task sheet now reads *"This bounty program is closed
> and is not accepting new applications or bounty claims"* for all three tiers (Open
> Access, Application-Only, Call for Partners), and `PrimeIntellect-ai/community-environments`
> — the PR route whose merge hook paid out — has ~200 open PRs and no merges since
> March 2026. What remains open, and is what §5 Phase 1 actually needs: publishing to the
> Environments Hub (`prime env push`) and attaching measured evals to a published
> environment (a v1 `eval` run with `push = true`). The analysis below is left as
> written; only §6 is revised.

---

## 1. Why this market exists

Post-training has shifted from static preference data to **environments**: executable
tasks with programmatic reward functions that an agent can be trained and graded against.
The scarce input is no longer labelled text, it is *verifiable work*.

The money follows. Leaders at Anthropic have discussed spending **more than $1 billion on
RL environments** over a year, and the incumbent data-labelling oligopoly — Scale AI,
Surge AI, Mercor, Handshake, which between them hold **75%+ of labelling revenue** — has
moved to capture the environment layer rather than cede it.

Three supplier groups now compete:

| Group | Examples | How you enter |
|---|---|---|
| Labellers that added environments | Scale, Surge, Mercor, Turing, Centific | Apply as a contractor; vetted, matched, paid hourly or per project |
| Environment-native startups | Mechanize, Fleet AI, HUD, Veris AI, Plato, Bespoke Labs | Direct hire; small teams, very high bar, **Mechanize pays ~$500k** |
| Open ecosystem | **Prime Intellect Environments Hub** | Publish publicly, claim bounties, build attributable track record |

Rates for reference: Mercor's average pay rate **passed $100/hour**, its "Fullstack
Engineer, RL Environments" postings run **$130k–$500k**, and its platform-wide contractor
average is ~$85/hour.

**The structural read:** the first two groups hire on *evidence*, and the third group is
where evidence is manufactured in public. That ordering is the whole strategy.

---

## 2. The on-ramp that is actually open

Prime Intellect runs an **Open-Source Environments Program** with published, claimable
bounties — the only route in this market that does not require passing someone's
recruiting funnel first.

| Tier | Amount | Access |
|---|---|---|
| Open Access | **$100–500** | Claim directly from the [public task sheet](https://docs.google.com/spreadsheets/d/13UDfRDjgIZXsMI2s9-Lmn8KSMMsgk2_zsfju6cx_pNU/edit). Explicitly aimed at first-time builders. |
| Application-Only | **$1,000–5,000+** | Approval basis via [this form](https://form.typeform.com/to/jLfT7v7o). Larger scope, sometimes third-party sponsored. |

Prime Intellect has committed **hundreds of thousands of dollars** in bounties, and has
crowdsourced **400+ environments** through bounties and an RL residency that drew **500+
applicants for 80+ reviewed implementations**.

Note the implied acceptance rate. Volume is not the constraint; **passing review** is.

The stated acceptance bar for a benchmark port is unambiguous: *"full reproduction of
known evaluation scores for a relevant subset of models."* Not "it runs." Reproduces.

---

## 3. The technical edge: v0 is deprecated and almost nobody has noticed

This is the highest-leverage finding in the whole analysis, and it is verifiable in about
two minutes.

`verifiers` shipped a **v1** API — tasksets, harnesses, traces — and marked v0
(`import verifiers as vf`, `SingleTurnEnv`, `MultiTurnEnv`, `ToolEnv`, `Rubric`, `Parser`,
`load_environment()`) **deprecated, to be removed in a future release**.

Now count what the ecosystem actually uses. In Prime Intellect's own production catalogue,
[`research-environments`](https://github.com/PrimeIntellect-ai/research-environments):

```
117   import verifiers.v1 as vf
  0   import verifiers as vf
 88/89 packages tagged "v1"
```

**One hundred percent v1.** Meanwhile essentially every tutorial, blog post, Medium guide
and LLM completion on the subject still teaches v0 — because v0 is what was in the
training data.

So the default output of anyone entering this market right now — including anyone using an
AI assistant to help — is code written against a deprecated API that the buyer's own
catalogue no longer contains. Writing v1 natively is not an optimisation; it is the
difference between a submission that reads as current and one that reads as stale on the
first import line.

**The v0 → v1 mapping**, if you meet legacy code:

| v0 | v1 |
|---|---|
| Dataset row | typed `vf.TaskData` subclass |
| `load_environment(**kwargs)` | exported `vf.Taskset` class + typed config |
| `Rubric` reward function | `@vf.reward` method on the task |
| `Parser` object | ordinary parsing inside task scoring |
| `ToolEnv` tools | `vf.Toolset` on `Task.toolsets` / `Taskset.toolsets` |
| `MultiTurnEnv.env_response` | an interaction loop in `Env.run()` |
| dict state | typed `vf.State` |
| Sandbox subclass | runtime config + task hooks |

---

## 4. What gets accepted

Distilled from Prime Intellect's `BENCHMARK_PORTING.md`, the `create-environments` skill,
and the reward-hacking literature.

**Fidelity is the product.** Prompts, rubrics, model choices, sampling parameters, task
assets, scoring semantics and workspace layout are *benchmark data* — copied verbatim
unless there is a hard technical reason not to, and any deviation documented. "Extra
helpful context" leaking into a prompt is a correctness bug, not a nicety. Files *removed*
before a run matter as much as files present.

**Smallest faithful abstraction wins.** The default question is "which existing verifiers
primitive is closest?", never "how would I implement this?". A port should be mostly
mapping code. Do not add agent machinery to a task that does not need it — it is far
easier to add runtime capability later than to un-leak benchmark logic from an overbuilt
agent environment. `terminal-agentic-v1` in this repo is nine lines for exactly this reason.

**Verification quality decides everything.** The research is blunt: imperfect verifiers
that check only *extensional* correctness admit false positives and actively induce reward
hacking, while *isomorphic* verification prevents it. Deterministic, rule-grounded checks
beat LLM judges; use a judge only when the judgement is irreducibly semantic.

**The review standard.** A reviewer must be able to answer: what came from upstream, what
is framework wiring, what changed deliberately, and why does any custom code exist. If the
code and README do not make those obvious, the port is too complicated.

### The failure modes that get work rejected

1. Written against v0. Stale on the first import line.
2. Reward hackable — partial credit on exact answers, recall-only audit scores, judges
   where a rule would do.
3. Answer key derived from generator intent rather than from the shipped task, so a
   generator bug ships as a confidently wrong key at scale.
4. Grader and solver share a library, so the library's defaults silently become the spec.
5. Reimplements framework plumbing instead of using it — behaviourally close, but it
   bypasses the shared path and duplicates logic that will drift.
6. No evidence it reproduces anything.

---

## 5. Route to revenue

Two goals were selected: **bounty track record first, then a role**, and **an environment
vendor business**. They share the first two phases; only the last diverges.

### Phase 1 — Evidence (weeks 1–4)

Public, attributable, technically current work. This repository is that artifact: four
environments across four archetypes, v1-native, each with adversarial reward tests and a
model-free answer-key check wired into CI.

Then claim **two Open Access bounties ($100–500)** from the public sheet. The money is
incidental. What matters is a completed-and-accepted review under your name, which is the
only credential this market reads.

### Phase 2 — Scale the bar (weeks 4–10)

Apply for **Application-Only bounties ($1,000–5,000+)**. Prefer a benchmark *port* over an
original environment: the acceptance criterion is objective (reproduce known scores), so
review is a measurement rather than a judgement call. Bank the reproduction numbers — a
port that lands published scores within noise is the single most legible artifact you can
own.

Run each finished environment against two or three models and publish the eval curves. An
environment nobody has measured is a claim; one with curves is a result.

### Phase 3a — Role

With accepted bounties and reproduction numbers, apply to Mercor (RL Environments roles,
$130k–$500k), Prime Intellect, Mechanize, Fleet AI, HUD, and the labs' post-training data
teams. Lead with the accepted-work links, not the repo.

### Phase 3b — Vendor

The moat is **domain access, not framework skill** — framework skill is now a weekend, and
this document just proved the whole ecosystem is one library. Sell where verifiable ground
truth requires expertise or licensed data that a generalist contractor cannot synthesise:
regulated compliance work, geospatial and remote sensing, instrumented scientific
pipelines. Charge for the pipeline (generation → verification → reproduction evidence),
not per environment. Approach sponsors through Prime Intellect's
[Call for Partners](https://form.typeform.com/to/DkpaAX0d), which is a warm channel to
organisations already budgeting for exactly this.

---

## 6. Do this next

*(Revised 2026-09-12 — the bounty sheet and application form are closed; see the status
note at the top.)*

1. `prime login`, then run a real eval on every container-free environment here so the
   repo carries measured numbers rather than only structural claims — **done 2026-09-11**
   (see the root README's verification table).
2. Choose a Prime username (it is permanent and public; `prime env push` prompts for it)
   and `prime env push` all six environments, publicly.
3. Attach the measured runs to the published environments: re-enter each finished run
   with `uv run eval @ outputs/<run>/configs/resolved/eval.json --resume` — a complete run
   re-runs nothing and uploads via the config's `push = true`.
4. Lead with the Hub pages and eval curves when approaching the employers in §5 Phase 3a;
   watch the [Prime Intellect blog](https://www.primeintellect.ai/blog) for a reopened
   programme rather than the closed sheet.

## Sources

- [Silicon Valley bets big on 'environments' to train AI agents — TechCrunch](https://techcrunch.com/2025/09/21/silicon-valley-bets-big-on-environments-to-train-ai-agents/)
- [Scaling Our Open-Source Environments Program — Prime Intellect](https://www.primeintellect.ai/blog/scaling-environments-program)
- [Environments Hub: A Community Hub To Scale RL To Open AGI — Prime Intellect](https://www.primeintellect.ai/blog/environments)
- [PrimeIntellect-ai/verifiers](https://github.com/PrimeIntellect-ai/verifiers) — v1 docs, `AGENTS.md`, bundled skills
- [PrimeIntellect-ai/research-environments](https://github.com/PrimeIntellect-ai/research-environments) — production catalogue, `BENCHMARK_PORTING.md`
- [From Data Labeling to RL Environments: The 2026 Market Map — Pebblous](https://blog.pebblous.ai/blog/labeling-to-rl-environments/en/)
- [RL Environment Companies in 2026 — Troveo](https://www.troveo.ai/resources/rl-environment-companies)
- [Mercor Business Breakdown — Contrary Research](https://research.contrary.com/company/mercor)
- [AI Trainer Salary: Hourly Rates — Mercor](https://www.mercor.com/resources/experts/ai-trainer-salary-hourly-rates/)
- [LLMs Gaming Verifiers: RLVR can Lead to Reward Hacking — arXiv:2604.15149](https://arxiv.org/abs/2604.15149)
- [Reward Hacking Mitigation using Verifiable Composite Rewards — arXiv:2509.15557](https://arxiv.org/pdf/2509.15557)
- [When you cannot claim a GST credit — ATO](https://www.ato.gov.au/businesses-and-organisations/gst-excise-and-indirect-taxes/gst/claiming-gst-credits/when-you-cannot-claim-a-gst-credit) (the $82.50 threshold in `au-bas-reconciliation-v1`)
