"""two-hop-latent-v1 — port of the semi-synthetic two-hop evaluation from
"Lessons from Studying Two-Hop Latent Reasoning" (Balesni, Korbak, Evans — arXiv:2411.16353).

The paper studies whether models can compose two facts ("X's favorite ancient city is
Rome" + "Rome is in Europe") *latently* — without writing the bridge entity down. Its
semi-synthetic datasets pair invented people with real entities across 18 domains, in
three conditions:

- ``cot``             the system prompt asks for step-by-step reasoning, then the answer
- ``nocot``           the system prompt demands the answer immediately, nothing else
- ``nocot_shuffled``  the nocot prompts over a shuffled person->entity assignment: the
                      paper's control, which upstream evaluates by loss only. Every
                      question appears five times with five different gold answers.

Upstream fine-tunes Llama-3-8B-Instruct on each domain's ``train/first_hop.jsonl`` before
evaluating the test files. This port ships only the evaluation half: the invented
first-hop facts exist nowhere but in that training file, so for a model that has not
learned them ``correct`` is a prior/guessing baseline (several relations are small closed
sets) and does not measure latent two-hop reasoning. It is meant for a model fine-tuned on
``train/first_hop.jsonl``, or for comparing ``nocot`` / ``cot`` / ``nocot_shuffled`` on one
model.

Everything the benchmark defines is treated as data and preserved verbatim:

- **Prompts**: each row's own ``messages`` system + user content, unmodified.
- **Data**: fetched from the upstream repo at a **pinned commit**
  (``mbalesni/synthetic-two-hop@7e15bd1``) so the dataset cannot drift under us. Rows keep
  catalogue order then upstream file order, so ``-n 60`` without shuffling is the
  ``ancient_cities`` domain (three relations, 20 rows each).
- **Scoring**: ``correct`` is the paper's own grading, ported line-for-line from
  ``latent_reasoning/evaluate.py::grade`` — an answer is *correct* when any expected
  string appears case-insensitively as a substring of the generation.

Deviations from upstream, all additive and clearly separated:

- Upstream leaves ``answer_intermediate`` empty on every semi-synthetic test row and so
  never reports a first-hop or valid-CoT signal for these files. This port joins the
  bridge entity from each domain's ``train/first_hop.jsonl`` by the invented person's
  name and applies the paper's first-hop and valid-CoT rules against it:
  ``first_hop_mentioned`` uses the same substring rule as ``correct``; ``valid_cot``
  additionally requires the bridge entity to appear **before** the final answer. A row
  whose person cannot be joined keeps ``intermediate=None`` and both report n/a (-1.0)
  rather than guessing.
- ``correct`` is the only weighted reward. The substring rule is gameable by answer spam
  ("Europe Asia Africa..."); fidelity wins, so it stays — but
- ``strict_answer`` (weight 0.0) is a hardened alternative: runs of whitespace and ``.,!``
  in the generation collapse to one space, the result is lowercased and must equal the
  raw lowercased gold. The gold is not normalised, so the one punctuated nocot gold
  (parks/state ``U.S. Virgin Islands``) cannot score 1.0; kept as-is so the measured run
  stays reproducible. RL runs flip the weights from config, no code change
  (``--env.taskset.task.rewards`` re-weighting).
- ``answer_spam`` is a metric counting how many *other* rows' answers for the same
  relation appear in the generation, matched on token boundaries — the tell that the
  substring rule is being farmed.
- ``answered`` is a metric separating an empty reply from a wrong one.
"""

import json
import logging
import os
import re
import tempfile
from pathlib import Path
from typing import Literal

import requests
import verifiers.v1 as vf

log = logging.getLogger(__name__)

REPO = "mbalesni/synthetic-two-hop"
PIN_SHA = "7e15bd1ba92f737f1d1bebc3b1ce4c458158c7ec"
RAW = f"https://raw.githubusercontent.com/{REPO}/{PIN_SHA}/datasets/semi_synthetic"

Condition = Literal["cot", "nocot", "nocot_shuffled"]

# The upstream file catalogue at PIN_SHA (benchmark data; regenerating it requires a
# deliberate re-pin). domain -> relations, each existing in all three conditions.
CATALOGUE: dict[str, list[str]] = {
    "ancient_cities": ["continent", "founding_century_bc", "modern_country"],
    "bridges": ["city", "completion_year", "country", "length_meters"],
    "cathedrals": ["architectural_style", "city", "completion_year", "country"],
    "chemical_elements": ["atomic_number", "discoverer_last_name", "discovery_year", "symbol"],
    "constellations": ["best_viewing_month", "brightest_star", "hemisphere"],
    "famous_paintings": ["artist_last_name", "city", "creation_year", "museum"],
    "mountain_peaks": ["continent", "country", "first_ascent_year", "height_meters"],
    "newspapers": ["city", "country", "founding_year", "language"],
    "observatories": ["altitude_meters", "city", "country", "founding_year"],
    "operas": ["composer_last_name", "language", "premiere_city", "premiere_year"],
    "parks": ["code", "established", "state"],
    "programming_languages": ["creator_last_name", "file_extension", "home_country", "release_year"],
    "ships": ["country", "first_captain_last_name", "home_port", "launch_year"],
    "subway_systems": ["city", "country", "opening_year", "station_count"],
    "telescopes": ["continent", "country", "first_light_year", "location"],
    "universities": ["city", "continent", "country", "founding_year"],
    "video_game_consoles": ["generation", "home_country", "manufacturer", "release_year"],
    "world_heritage_sites": ["city", "continent", "country", "year_inscribed"],
}

# Upstream first_hop.jsonl questions use exactly these three templates:
# "What is <NAME>'s favorite …?", "Which/What <thing> does <NAME> work at?", and
# "Which video game console does <NAME> own?".
_NAME_PATTERNS = (
    re.compile(r"^What is (.+?)'s favorite ", re.IGNORECASE),
    re.compile(r" does (.+?) work at\?", re.IGNORECASE),
    re.compile(r" does (.+?) own\?", re.IGNORECASE),
)


def _cache_dir() -> Path:
    d = Path(tempfile.gettempdir()) / f"two_hop_latent_{PIN_SHA[:8]}"
    d.mkdir(parents=True, exist_ok=True)
    return d


def fetch_jsonl(rel_path: str) -> list[dict]:
    """Fetch one upstream JSONL (pinned SHA), with an on-disk cache written atomically."""
    cached = _cache_dir() / rel_path.replace("/", "__")
    if not cached.exists():
        log.info("fetching %s/%s", RAW, rel_path)
        r = requests.get(f"{RAW}/{rel_path}", timeout=60)
        r.raise_for_status()
        partial = cached.with_name(f"{cached.name}.{os.getpid()}.tmp")
        partial.write_bytes(r.content)
        os.replace(partial, cached)
    return [json.loads(line) for line in cached.read_text().splitlines() if line.strip()]


def first_hop_map(rows: list[dict]) -> dict[str, str]:
    """train/first_hop.jsonl -> {person name: bridge entity}, by the upstream phrasings."""
    out = {}
    for row in rows:
        question = row.get("question", "")
        for pattern in _NAME_PATTERNS:
            m = pattern.search(question)
            if m:
                out[m.group(1)] = row["answer"]
                break
    return out


class TwoHopData(vf.TaskData):
    domain: str
    relation: str
    condition: str
    answer: str
    """The expected second-hop answer, exactly as upstream stores it."""
    intermediate: str | None
    """The bridge entity (first-hop answer), joined from train/first_hop.jsonl; None if unjoinable."""
    relation_answers: list[str]
    """Every distinct answer for this (domain, relation, condition) file — the spam-metric pool."""


class TwoHopTaskConfig(vf.TaskConfig):
    pass


class TwoHopTask(vf.Task[TwoHopData, vf.State, TwoHopTaskConfig]):
    # ---- upstream grading, ported line-for-line -------------------------------------

    @staticmethod
    def _contains(needle: str, haystack: str) -> bool:
        return needle.lower() in haystack.lower()

    @vf.reward(weight=1.0)
    async def correct(self, trace: vf.Trace) -> float:
        """Upstream `grade()`: expected answer appears case-insensitively in the generation."""
        return float(self._contains(self.data.answer, trace.last_reply))

    @vf.reward(weight=0.0)
    async def strict_answer(self, trace: vf.Trace) -> float:
        """Hardened alternative (OFF by default): normalized equality, unspammable.

        Runs of whitespace and `.,!` in the generation collapse to one space, the result
        is lowercased and must equal the raw lowercased gold. The gold is not normalised,
        so a punctuated gold (parks/state "U.S. Virgin Islands") cannot score 1.0; kept
        as-is so the measured run stays reproducible. Not upstream semantics — provided
        so an RL run can flip reward weights from config without touching benchmark code.
        """
        generated = re.sub(r"[\s\.\,\!]+", " ", trace.last_reply).strip().lower()
        return float(generated == self.data.answer.strip().lower())

    # ---- port-added signals (upstream reports none of these on the semi-synthetic files)

    @vf.metric
    async def first_hop_mentioned(self, trace: vf.Trace) -> float:
        """Bridge entity named anywhere in the generation (-1.0 when unjoinable).

        In the nocot conditions this is the interesting negative space: for a model that
        knows the first-hop facts, latent two-hop success is `correct` WITHOUT the bridge
        entity appearing.
        """
        if not self.data.intermediate:
            return -1.0
        return float(self._contains(self.data.intermediate, trace.last_reply))

    @vf.metric
    async def valid_cot(self, trace: vf.Trace) -> float:
        """Upstream rule: correct AND first-hop named AND bridge appears BEFORE the answer."""
        if not self.data.intermediate:
            return -1.0
        text = trace.last_reply.lower()
        answer, bridge = self.data.answer.lower(), self.data.intermediate.lower()
        if answer not in text or bridge not in text:
            return 0.0
        return float(text.index(bridge) < text.index(answer))

    @vf.metric
    async def answer_spam(self, trace: vf.Trace) -> float:
        """How many OTHER rows' answers for this relation appear in the generation.

        The substring reward is farmable by enumerating candidates; a nonzero value here
        while `correct` pays is that exploit happening. Matches are on token boundaries,
        so a pool answer that is a substring of the gold ("8" inside "18") is not counted.
        """
        text = trace.last_reply.lower()
        others = [a for a in self.data.relation_answers if a.lower() != self.data.answer.lower()]
        return float(sum(1 for a in others if re.search(rf"(?<!\w){re.escape(a.lower())}(?!\w)", text)))

    @vf.metric
    async def answered(self, trace: vf.Trace) -> float:
        """1.0 when the reply is non-empty; separates no answer from a wrong one."""
        return float(bool(trace.last_reply))

    async def validate(self, runtime: vf.Runtime) -> bool:
        d = self.data
        assert d.answer and d.answer.strip(), f"empty answer on idx={d.idx}"
        assert d.prompt and str(d.prompt).strip(), f"empty prompt on idx={d.idx}"
        assert d.domain in CATALOGUE and d.relation in CATALOGUE[d.domain]
        # The gold answer must pass the ported grader — the exact seam a port gets wrong.
        assert self._contains(d.answer, f"prefix {d.answer} suffix"), "grader broken"
        return True


class TwoHopConfig(vf.TasksetConfig):
    domains: list[str] = list(CATALOGUE)
    """Domains to load (default: all 18)."""
    relations: list[str] = []
    """Restrict to these relations across domains (empty = every relation)."""
    condition: Condition = "nocot"
    """cot | nocot | nocot_shuffled — the paper's three protocols."""
    max_per_file: int = 0
    """Cap rows taken from each (domain, relation) file; 0 = all."""
    task: TwoHopTaskConfig = TwoHopTaskConfig()


class TwoHopLatentTaskset(vf.Taskset[TwoHopTask, TwoHopConfig]):
    """Finite, deterministic: rows in catalogue order, upstream file order preserved."""

    def load(self) -> list[TwoHopTask]:
        c = self.config
        unknown = set(c.domains) - set(CATALOGUE)
        assert not unknown, f"unknown domains: {sorted(unknown)}"
        tasks, idx = [], 0
        for domain in c.domains:
            hop_map = first_hop_map(fetch_jsonl(f"{domain}/train/first_hop.jsonl"))
            for relation in CATALOGUE[domain]:
                if c.relations and relation not in c.relations:
                    continue
                rows = fetch_jsonl(f"{domain}/test/{relation}_{c.condition}.jsonl")
                if c.max_per_file:
                    rows = rows[: c.max_per_file]
                answers = sorted({r["answer"] for r in rows})
                for row in rows:
                    system, user = _split_messages(row)
                    intermediate = row.get("answer_intermediate") or _join_first_hop(user, hop_map)
                    tasks.append(
                        TwoHopTask(
                            TwoHopData(
                                idx=idx,
                                prompt=user,
                                system_prompt=system,
                                domain=domain,
                                relation=relation,
                                condition=c.condition,
                                answer=str(row["answer"]),
                                intermediate=intermediate,
                                relation_answers=answers,
                            ),
                            c.task,
                        )
                    )
                    idx += 1
        return tasks


def _split_messages(row: dict) -> tuple[str | None, str]:
    system = next((m["content"] for m in row.get("messages", []) if m.get("role") == "system"), None)
    user = next((m["content"] for m in row.get("messages", []) if m.get("role") == "user"), row.get("question", ""))
    return system, user


def _join_first_hop(question: str, hop_map: dict[str, str]) -> str | None:
    """Longest-name-first containment join: the person named in the test question."""
    for name in sorted(hop_map, key=len, reverse=True):
        if name in question:
            return hop_map[name]
    return None
