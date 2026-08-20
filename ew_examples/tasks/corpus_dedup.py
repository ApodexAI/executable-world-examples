"""Deduplicate a training corpus, and keep a benchmark out of it.

The shape: a corpus with near-duplicates in it, and a benchmark whose questions
must not leak into training data. You choose thresholds for cheap signals, pay for
expensive ones on the pairs you are unsure about, then submit a strategy. The
strategy is applied to a hidden set and you get structured feedback — not the
labels — so you can revise and submit again, up to a limit.

Two failure directions, and they pull against each other:

  OVER-MERGE   throwing away documents that were not really duplicates. Costs you
               data you paid for.
  UNDER-REMOVE letting a benchmark question stay in the training set. This is
               contamination, and it is the expensive kind of mistake: it makes
               every later measurement of the model meaningless.

A single threshold cannot do both well, which is the lesson. The calibration
fixture is free and labelled — use it before spending anything.
"""
from __future__ import annotations

import hashlib
import random

from ..engine import Action, ActionError, Budget, Task

MAX_SUBMITS = 4
WORDS = ("model", "train", "token", "loss", "gradient", "sample", "dataset",
         "prompt", "reward", "policy", "encoder", "layer", "vector", "weight")


def _text(rng: random.Random, n: int = 12) -> str:
    return " ".join(rng.choice(WORDS) for _ in range(n))


def _perturb(rng: random.Random, s: str) -> str:
    """A near-duplicate: same content, cosmetically different."""
    words = s.split()
    if rng.random() < 0.5 and len(words) > 3:
        i = rng.randrange(len(words) - 1)
        words[i], words[i + 1] = words[i + 1], words[i]
    else:
        words.insert(rng.randrange(len(words)), rng.choice(("the", "a", "some")))
    return " ".join(words)


def _cosine(a: str, b: str) -> float:
    """A deliberately crude bag-of-words similarity, so a solver can reason about
    it. Real systems use embeddings; the decision structure is identical."""
    sa, sb = set(a.split()), set(b.split())
    return round(len(sa & sb) / max(1, len(sa | sb)), 4)


class CorpusDedup(Task):
    task_id = "corpus_dedup"
    brief = f"""\
Deduplicate a training corpus, and keep a benchmark out of it.

The corpus holds documents, some of which are near-duplicates of each other. A
separate BENCHMARK holds questions that must not appear in training data. Your job
is a STRATEGY: thresholds and rules, applied to the whole corpus.

SCORED ON TWO THINGS AT ONCE
  dedup_f1     finding the real duplicate pairs without merging distinct documents
  leak_recall  removing corpus documents that match a benchmark question

They pull against each other: a threshold loose enough to catch every leak also
merges documents that were never duplicates. One number cannot do both, and
noticing that is most of the task.

YOU MAY SUBMIT UP TO {MAX_SUBMITS} TIMES. Each submit is graded on a hidden set and
returns structured feedback — never the labels. Revise and resubmit.

ACTIONS
  inspect_corpus()              -> sizes, and what a document looks like  [free]
  calibration()                 -> a small LABELLED fixture               [free]
  sample_pairs(n<=20)           -> candidate pairs to consider     [1 index_query]
  embed_cosine(pairs)           -> similarity per pair             [1 embed_query/pair]
  judge_pair(pairs)             -> an expensive 'same'/'different'  [1 judge_call/pair]
  submit(strategy)              -> graded, with feedback; repeatable
  finish()                      -> lock in your best score and end

  strategy = {{
    "dedup_threshold":    0.0-1.0,   at or above this, two docs are duplicates
    "decontam_threshold": 0.0-1.0,   at or above this, a doc leaks the benchmark
    "use_judge": true|false          consult the expensive judge in the grey band
  }}

Start with calibration(). It is free and labelled, and it will tell you more than
your first three paid queries.
"""

    def __init__(self, seed: int = 0):
        rng = random.Random(seed)
        self._rng = rng
        self._docs: dict[str, str] = {}
        self._dup_pairs: set[tuple[str, str]] = set()

        originals = [_text(rng) for _ in range(40)]
        for i, txt in enumerate(originals):
            self._docs[f"doc_{i:03d}"] = txt
        # 12 near-duplicates of known originals.
        for j in range(12):
            src = f"doc_{j:03d}"
            did = f"doc_{100 + j:03d}"
            self._docs[did] = _perturb(rng, self._docs[src])
            self._dup_pairs.add(tuple(sorted((src, did))))

        # The benchmark, plus corpus documents that leak it.
        self._benchmark = {f"bench_{i:02d}": _text(rng, 10) for i in range(8)}
        self._leaks: set[str] = set()
        for i, (bid, btxt) in enumerate(list(self._benchmark.items())[:5]):
            did = f"doc_{200 + i:03d}"
            self._docs[did] = _perturb(rng, btxt)
            self._leaks.add(did)

        self._submits: list[dict] = []

    # ---- actions ----------------------------------------------------------
    def initial_budget(self) -> Budget:
        return Budget({"index_queries": 15, "embed_queries": 60,
                       "judge_calls": 10, "submits": MAX_SUBMITS})

    def actions(self):
        return {
            "inspect_corpus": Action(0, self._inspect, "sizes and a sample document"),
            "calibration": Action(0, self._calib, "a small LABELLED fixture, free"),
            "sample_pairs": Action(1, self._sample, "candidate pairs"),
            "embed_cosine": Action(1, self._embed, "similarity per pair"),
            "judge_pair": Action(1, self._judge, "expensive same/different"),
            "submit": Action(0, self._submit, "grade a strategy; repeatable"),
            "finish": Action(0, self._finish, "lock in the best score and end"),
        }

    def _inspect(self, ep, p):
        any_id = sorted(self._docs)[0]
        return {"n_documents": len(self._docs),
                "n_benchmark_questions": len(self._benchmark),
                "example_document": {"id": any_id, "text": self._docs[any_id]},
                "benchmark_questions": self._benchmark,
                "note": "duplicates are cosmetic edits of one another: reordered "
                        "words, an inserted filler word"}

    def _calib(self, ep, p):
        """Free and labelled. The whole point is that it is cheaper than guessing."""
        pairs = []
        for a, b in sorted(self._dup_pairs)[:3]:
            pairs.append({"pair": [a, b], "cosine": _cosine(self._docs[a], self._docs[b]),
                          "label": "duplicate"})
        ids = sorted(self._docs)
        for a, b in ((ids[5], ids[9]), (ids[11], ids[20]), (ids[7], ids[30])):
            pairs.append({"pair": [a, b], "cosine": _cosine(self._docs[a], self._docs[b]),
                          "label": "distinct"})
        leak = sorted(self._leaks)[0]
        bid = sorted(self._benchmark)[0]
        return {"labelled_pairs": pairs,
                "labelled_leak_example": {
                    "document": leak, "benchmark": bid,
                    "cosine": _cosine(self._docs[leak], self._benchmark[bid]),
                    "label": "leak"},
                "note": "free and labelled; the hidden set behaves the same way"}

    def _sample(self, ep, p):
        n = max(1, min(20, int(p.get("n", 10))))
        ep.budget.spend("index_queries", 1)
        ids = sorted(self._docs)
        out = []
        rng = random.Random(len(self._docs) + n)
        real = sorted(self._dup_pairs)
        for i in range(n):
            if i < len(real) and i % 2 == 0:
                out.append(list(real[i]))
            else:
                out.append([rng.choice(ids), rng.choice(ids)])
        return {"pairs": [p for p in out if p[0] != p[1]],
                "note": "candidates only — no claim that any of these are duplicates"}

    def _pairs(self, p):
        raw = p.get("pairs") or ([p["pair"]] if "pair" in p else [])
        if not raw:
            raise ActionError('pass pairs=[["doc_000","doc_100"], ...]')
        out = []
        for pr in raw[:20]:
            if not isinstance(pr, (list, tuple)) or len(pr) != 2:
                raise ActionError("each pair must be two document ids")
            out.append((str(pr[0]), str(pr[1])))
        return out

    def _lookup(self, did: str) -> str | None:
        return self._docs.get(did) or self._benchmark.get(did)

    def _embed(self, ep, p):
        pairs = self._pairs(p)
        ep.budget.spend("embed_queries", len(pairs))
        out = {}
        for a, b in pairs:
            ta, tb = self._lookup(a), self._lookup(b)
            out[f"{a}|{b}"] = (None if ta is None or tb is None
                               else _cosine(ta, tb))
        return {"cosine": out}

    def _judge(self, ep, p):
        pairs = self._pairs(p)
        ep.budget.spend("judge_calls", len(pairs))
        out = {}
        for a, b in pairs:
            key = tuple(sorted((a, b)))
            out[f"{a}|{b}"] = "same" if key in self._dup_pairs else "different"
        return {"verdicts": out, "note": "authoritative, and the scarcest thing "
                                         "you have"}

    # ---- grading ----------------------------------------------------------
    def _apply(self, strategy: dict) -> tuple[set, set]:
        try:
            dt = float(strategy.get("dedup_threshold", 0.9))
            ct = float(strategy.get("decontam_threshold", 0.9))
        except (TypeError, ValueError):
            raise ActionError("dedup_threshold and decontam_threshold must be numbers")
        if not (0.0 <= dt <= 1.0 and 0.0 <= ct <= 1.0):
            raise ActionError("thresholds must be between 0 and 1")

        ids = sorted(self._docs)
        found = set()
        for i, a in enumerate(ids):
            for b in ids[i + 1:]:
                if _cosine(self._docs[a], self._docs[b]) >= dt:
                    found.add((a, b))
        removed = set()
        for did, txt in self._docs.items():
            for btxt in self._benchmark.values():
                if _cosine(txt, btxt) >= ct:
                    removed.add(did)
                    break
        return found, removed

    def _grade(self, strategy: dict) -> dict:
        found, removed = self._apply(strategy)
        truth = self._dup_pairs
        tp = len(found & truth)
        prec = tp / len(found) if found else 0.0
        rec = tp / len(truth) if truth else 0.0
        f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) else 0.0

        leak_hit = len(removed & self._leaks)
        leak_recall = leak_hit / len(self._leaks) if self._leaks else 0.0
        clean = set(self._docs) - self._leaks
        over_removed = len(removed & clean) / len(clean) if clean else 0.0

        # Weights sum to 1.0 so a strategy that gets both objectives right
        # scores 1.0. Over-removal is a penalty on top, not part of the sum.
        score = round(0.6 * f1 + 0.4 * leak_recall - 0.2 * over_removed, 4)
        score = max(0.0, min(1.0, score))

        fb = []
        if prec < 0.8 and found:
            fb.append(f"dedup precision {prec:.2f}: the threshold is loose enough to "
                      f"merge documents that are not duplicates")
        if rec < 0.8:
            fb.append(f"dedup recall {rec:.2f}: real duplicate pairs were missed")
        if leak_recall < 1.0:
            fb.append(f"leak recall {leak_recall:.2f}: benchmark questions remain in "
                      f"the corpus. This is the expensive error — it invalidates "
                      f"every later measurement")
        if over_removed > 0.05:
            fb.append(f"{over_removed:.0%} of clean documents were removed as leaks: "
                      f"the decontamination threshold is too aggressive")
        if not fb:
            fb.append("both objectives satisfied")
        return {"score": score, "dedup_f1": round(f1, 4),
                "dedup_precision": round(prec, 4), "dedup_recall": round(rec, 4),
                "leak_recall": round(leak_recall, 4),
                "over_removal_rate": round(over_removed, 4), "feedback": fb}

    def _submit(self, ep, p):
        if len(self._submits) >= MAX_SUBMITS:
            raise ActionError(f"all {MAX_SUBMITS} submits are used; call finish()")
        strategy = p.get("strategy") or p.get("submission") or {}
        if not isinstance(strategy, dict):
            raise ActionError("strategy must be an object")
        ep.budget.spend("submits", 1)
        graded = self._grade(strategy)
        self._submits.append({"strategy": strategy, **graded})
        return {"status": "graded", "submit_index": len(self._submits),
                "submits_remaining": MAX_SUBMITS - len(self._submits),
                **{k: v for k, v in graded.items()},
                "note": "the labels are never returned; revise and submit again"}

    def _finish(self, ep, p):
        if not self._submits:
            raise ActionError("nothing submitted yet")
        res = ep.finish(None)
        return {"status": "finished", "episode_complete": True,
                "score": res["score"], "best_of": len(self._submits)}

    def score(self, _ignored) -> dict:
        """The best submission counts, which is why iterating is worth it."""
        if not self._submits:
            return {"score": 0.0, "feedback": ["nothing was submitted"]}
        best = max(self._submits, key=lambda s: s["score"])
        return {"score": best["score"], "n_submits": len(self._submits),
                "submit_scores": [s["score"] for s in self._submits],
                "best_strategy": best["strategy"],
                "dedup_f1": best["dedup_f1"], "leak_recall": best["leak_recall"],
                "feedback": best["feedback"]}
