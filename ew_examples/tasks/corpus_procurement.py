"""Assemble a pretraining corpus under budget.

The shape: several data sources and two purchasable packages. Their size, quality
and overlap are hidden. You probe them through typed actions, then submit a
procurement plan, which is executed against the hidden truth. You are scored on
the unique high-value tokens your plan actually lands, against what the best
possible plan would have landed.

Four traps, all of which appear in real corpus work:

  MIRROR      a source that re-serves another source's documents. Same content,
              same hash, zero unique tokens. Buying both is pure waste.
  HONEYPOT    a source that mints endless fresh low-value documents. Sampling it
              never converges, and crawling it fills your corpus with junk.
  GATED       documents that exist but cannot be crawled. Planning around them is
              the point; they are only obtainable inside a package.
  CONTAMINATION  landing too high a fraction of junk fails a hard gate outright,
              no matter how many good tokens you also landed.

Everything is generated from a seed, so two runs of the same task are identical.
"""
from __future__ import annotations

import hashlib
import random

from ..engine import Action, ActionError, Budget, Task

SOURCES = ("src_00", "src_01", "src_02", "src_03", "src_04")
PACKAGES = {"pkg_00": 150, "pkg_01": 600}

# A plan landing more than this fraction of junk fails outright.
CONTAMINATION_LIMIT = 0.15
JUDGE_TOKENS_PER_DOC = 200


def _doc_id(src: str, n: int) -> str:
    return f"{src}/{hashlib.sha1(f'{src}:{n}'.encode()).hexdigest()[:12]}"


class CorpusProcurement(Task):
    task_id = "corpus_procurement"
    brief = """\
You are assembling a pretraining corpus under budget.

Five SOURCES (src_00..src_04) and two PACKAGES (pkg_00, pkg_01) are reachable only
through actions. Their size, document quality and overlap are hidden. Investigate,
then submit a procurement plan; it is executed against the hidden truth.

SCORE: the unique HIGH-value deduplicated tokens your plan lands, as a fraction of
what the best possible plan lands. Documents are either 'high' (a real, useful
work) or 'spam' (junk).

WATCH OUT FOR:
  * a MIRROR source that re-serves another source's documents. Identical
    content_hash means zero unique tokens gained.
  * a HONEYPOT source that generates endless fresh spam. It never runs out.
  * GATED documents, which cannot be crawled and are only available in a package.
  * CONTAMINATION: if more than 15% of your landed tokens are spam, the plan fails
    a hard gate and scores zero regardless of everything else.

Your plan also carries ESTIMATES of the total high-value population. Being honest
about your uncertainty is worth marks; a confident wrong interval is worse than a
wide right one.

ACTIONS
  sample_source(source, n<=50)  -> a page of document ids            [1 index_query]
  fetch(ids)                    -> tokens, content_hash, crawlable   [1 crawl_action/doc]
  judge(ids)                    -> 'high' or 'spam' per document     [200 judge_tokens/doc]
  preview_package(package)      -> a sample of its contents + price  [1 index_query]
  submit(plan)                  -> ends the episode

  plan = {
    "crawl":     ["src_00", ...],        sources to crawl in full
    "purchase":  ["pkg_00", ...],        packages to buy
    "suspected_decoys": ["src_0X", ...], sources you believe are mirror/honeypot
    "estimates": {"M_high": {"point": N, "lo": N, "hi": N}}   document count
  }
"""

    def __init__(self, seed: int = 0):
        rng = random.Random(seed)
        self._rng = rng
        # ---- hidden truth -------------------------------------------------
        # src_01 mirrors src_00; src_03 is the honeypot; the rest are genuine.
        self._mirror_of = {"src_01": "src_00"}
        self._honeypot = "src_03"
        self._docs: dict[str, dict] = {}
        self._by_source: dict[str, list[str]] = {s: [] for s in SOURCES}

        for src in ("src_00", "src_02", "src_04"):
            n = rng.choice((60, 80, 100))
            # src_04 is junk-heavy on purpose: crawling it is what fails the
            # contamination gate, so recognising that is part of the task.
            high_rate = {"src_00": 0.92, "src_02": 0.85, "src_04": 0.20}[src]
            for i in range(n):
                did = _doc_id(src, i)
                high = rng.random() < high_rate
                self._docs[did] = {
                    "tokens": rng.randint(400, 4000),
                    "value": "high" if high else "spam",
                    "content_hash": hashlib.sha1(f"c:{src}:{i}".encode()).hexdigest()[:16],
                    # A slice of src_04's good documents is gated: visible, not crawlable.
                    "crawlable": not (src == "src_04" and high and i % 3 == 0),
                    "source": src,
                }
                self._by_source[src].append(did)

        # The mirror: same content_hash, different ids. Dedup makes it worthless.
        for i, origin in enumerate(self._by_source["src_00"]):
            did = _doc_id("src_01", i)
            self._docs[did] = dict(self._docs[origin], source="src_01")
            self._by_source["src_01"].append(did)

        self._honeypot_seen = 0
        self._oracle_cache = None

        # Packages. pkg_00 is fair value; pkg_01 is mostly the mirror, dearly priced.
        gated = [d for d in self._by_source["src_04"]
                 if not self._docs[d]["crawlable"]]
        self._packages = {
            "pkg_00": gated + self._by_source["src_02"][:10],
            "pkg_01": self._by_source["src_01"][:40],
        }

    # ---- helpers ----------------------------------------------------------
    def _honeypot_doc(self) -> str:
        self._honeypot_seen += 1
        did = _doc_id(self._honeypot, self._honeypot_seen)
        self._docs[did] = {
            "tokens": self._rng.randint(200, 900), "value": "spam",
            "content_hash": hashlib.sha1(f"h:{self._honeypot_seen}".encode())
                                    .hexdigest()[:16],
            "crawlable": True, "source": self._honeypot}
        return did

    def _landed(self, plan: dict) -> tuple[set, list[str]]:
        """Which documents a plan actually lands, and what it spent."""
        crawl = [s for s in (plan.get("crawl") or []) if s in SOURCES]
        buy = [p for p in (plan.get("purchase") or []) if p in PACKAGES]
        notes = []
        ids: list[str] = []
        for s in crawl:
            if s == self._honeypot:
                # Crawling a generator lands a fixed, junk-heavy sample.
                ids += [self._honeypot_doc() for _ in range(40)]
                notes.append(f"{s} is a generator: landed 40 fresh spam documents")
            else:
                got = [d for d in self._by_source[s] if self._docs[d]["crawlable"]]
                ids += got
                skipped = len(self._by_source[s]) - len(got)
                if skipped:
                    notes.append(f"{s}: {skipped} documents were gated, not crawled")
        for p in buy:
            ids += self._packages[p]
        # Deduplicate by content, which is what makes the mirror worthless.
        seen, kept = set(), set()
        for d in ids:
            h = self._docs[d]["content_hash"]
            if h in seen:
                continue
            seen.add(h)
            kept.add(d)
        if len(ids) - len(kept):
            notes.append(f"deduplication removed {len(ids) - len(kept)} documents "
                         f"already present under another id")
        return kept, notes

    def _oracle(self) -> int:
        """The most high-value tokens any GATE-PASSING plan can land.

        Found by enumerating all 128 source/package combinations rather than
        asserting a best plan. That matters: the first version of this task
        asserted one, and the asserted plan itself failed the contamination gate,
        so every possible submission scored zero and the task looked broken rather
        than hard. An oracle that is not checked against the same gate the solver
        faces is not an oracle.
        """
        import itertools
        if getattr(self, "_oracle_cache", None) is not None:
            return self._oracle_cache
        best = 0
        for n_s in range(len(SOURCES) + 1):
            for srcs in itertools.combinations(SOURCES, n_s):
                for n_p in range(len(PACKAGES) + 1):
                    for pkgs in itertools.combinations(sorted(PACKAGES), n_p):
                        if sum(PACKAGES[x] for x in pkgs) > 1000:
                            continue
                        kept, _ = self._landed({"crawl": list(srcs),
                                                "purchase": list(pkgs)})
                        high = sum(self._docs[d]["tokens"] for d in kept
                                   if self._docs[d]["value"] == "high")
                        spam = sum(self._docs[d]["tokens"] for d in kept
                                   if self._docs[d]["value"] != "high")
                        tot = high + spam
                        if tot and spam / tot > CONTAMINATION_LIMIT:
                            continue          # the solver's gate binds here too
                        best = max(best, high)
        self._oracle_cache = best
        return best

    # ---- actions ----------------------------------------------------------
    def initial_budget(self) -> Budget:
        return Budget({"index_queries": 40, "crawl_actions": 120,
                       "judge_tokens": 8000, "purchase_dollars": 1000})

    def actions(self) -> dict[str, Action]:
        return {
            "sample_source": Action(1, self._sample, "page of document ids"),
            "fetch": Action(1, self._fetch, "tokens + content_hash + crawlable"),
            "judge": Action(1, self._judge, "'high' or 'spam' per document"),
            "preview_package": Action(1, self._preview, "sample + price"),
            "submit": Action(0, self._submit, "score the plan; ends the episode"),
        }

    def _sample(self, ep, p):
        src = str(p.get("source", ""))
        if src not in SOURCES:
            raise ActionError(f"unknown source {src!r}; try one of {', '.join(SOURCES)}")
        n = max(1, min(50, int(p.get("n", 10))))
        ep.budget.spend("index_queries", 1)
        if src == self._honeypot:
            items = [self._honeypot_doc() for _ in range(n)]
        else:
            items = self._by_source[src][:n]
        return {"source": src, "n_returned": len(items), "items": items,
                "note": "a page; there may be more"}

    def _ids(self, p) -> list[str]:
        raw = p.get("ids") or ([p["id"]] if "id" in p else [])
        if isinstance(raw, str):
            raw = [raw]
        ids = [str(i) for i in raw][:50]
        if not ids:
            raise ActionError("pass ids=[...] (or id=...)")
        return ids

    def _fetch(self, ep, p):
        ids = self._ids(p)
        ep.budget.spend("crawl_actions", len(ids))
        out = {}
        for did in ids:
            d = self._docs.get(did)
            out[did] = ({"error": "no such document"} if d is None else
                        {"tokens": d["tokens"], "content_hash": d["content_hash"],
                         "crawlable": d["crawlable"]})
        return {"documents": out}

    def _judge(self, ep, p):
        ids = self._ids(p)
        ep.budget.spend("judge_tokens", JUDGE_TOKENS_PER_DOC * len(ids))
        return {"values": {did: (self._docs[did]["value"]
                                if did in self._docs else "unknown")
                           for did in ids}}

    def _preview(self, ep, p):
        pkg = str(p.get("package", ""))
        if pkg not in PACKAGES:
            raise ActionError(f"unknown package {pkg!r}; try {', '.join(PACKAGES)}")
        ep.budget.spend("index_queries", 1)
        contents = self._packages[pkg]
        sample = contents[:5]
        return {"package": pkg, "price_dollars": PACKAGES[pkg],
                "n_documents": len(contents), "sample_ids": sample,
                "note": "a sample only; fetch/judge them to learn more"}

    def _submit(self, ep, p):
        plan = p.get("plan") or p.get("submission") or {}
        if not isinstance(plan, dict):
            raise ActionError("plan must be an object")
        cost = sum(PACKAGES[x] for x in (plan.get("purchase") or [])
                   if x in PACKAGES)
        if cost > ep.budget.counters.get("purchase_dollars", 0):
            raise ActionError(
                f"the plan buys ${cost} of packages and only "
                f"${ep.budget.counters['purchase_dollars']:g} is left")
        ep.budget.spend("purchase_dollars", cost)
        res = ep.finish(plan)
        return {"status": "submitted", "episode_complete": True,
                "score": res["score"], "feedback": res["feedback"]}

    # ---- scoring ----------------------------------------------------------
    def score(self, plan) -> dict:
        kept, notes = self._landed(plan if isinstance(plan, dict) else {})
        high = sum(self._docs[d]["tokens"] for d in kept
                   if self._docs[d]["value"] == "high")
        spam = sum(self._docs[d]["tokens"] for d in kept
                   if self._docs[d]["value"] != "high")
        total = high + spam
        contamination = (spam / total) if total else 0.0
        oracle = self._oracle()
        raw = (high / oracle) if oracle else 0.0

        gate_ok = contamination <= CONTAMINATION_LIMIT
        # Did they identify the mirror and the honeypot?
        truth = set(self._mirror_of) | {self._honeypot}
        guessed = set(plan.get("suspected_decoys") or []) if isinstance(plan, dict) else set()
        hits = len(truth & guessed)
        decoy_f1 = (2 * hits / (len(truth) + len(guessed))) if (truth or guessed) else 0.0

        est = ((plan.get("estimates") or {}).get("M_high") or {}) \
            if isinstance(plan, dict) else {}
        n_high = sum(1 for d in self._docs.values() if d["value"] == "high"
                     and d["source"] != "src_01")
        covered = False
        try:
            covered = float(est.get("lo", 1)) <= n_high <= float(est.get("hi", -1))
        except (TypeError, ValueError):
            covered = False

        score = 0.0 if not gate_ok else round(
            0.70 * raw + 0.20 * decoy_f1 + 0.10 * (1.0 if covered else 0.0), 4)

        fb = list(notes)
        if not gate_ok:
            fb.insert(0, f"HARD GATE FAILED: {contamination:.0%} of landed tokens "
                         f"were spam, and the limit is {CONTAMINATION_LIMIT:.0%}. "
                         f"Score is zero regardless of the {high:,} good tokens.")
        if guessed - truth:
            fb.append(f"wrongly suspected: {sorted(guessed - truth)}")
        if truth - guessed:
            fb.append(f"missed decoys: {sorted(truth - guessed)}")
        fb.append("estimate interval covered the true count" if covered
                  else f"estimate interval missed the true high-value count ({n_high})")
        return {"score": score, "gate_passed": gate_ok,
                "high_tokens_landed": high, "spam_tokens_landed": spam,
                "contamination": round(contamination, 4),
                "oracle_high_tokens": oracle,
                "fraction_of_oracle": round(raw, 4),
                "decoy_f1": round(decoy_f1, 4), "estimate_covered": covered,
                "feedback": fb}
