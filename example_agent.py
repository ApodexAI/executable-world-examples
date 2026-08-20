"""A worked solver for each example task, to show the loop.

This is not clever and is not meant to be — it is the reference for what driving a
task looks like, so you can see where your own agent plugs in. Each function does
what a careful person would do with no model at all: read the brief, spend the free
information first, buy the cheapest evidence that separates the options, then
submit something defensible.

    python3 run_task.py --task verify_solutions --agent example_agent:solve

Where a real agent differs: it would decide *which* evidence to buy by reasoning
about the brief, rather than following a fixed plan like these do.
"""
from __future__ import annotations


def solve(task, ep):
    """Dispatch on the task id. `run_task.py --agent` calls this."""
    return {"corpus_procurement": procurement,
            "verify_solutions": verify,
            "corpus_dedup": dedup}[task.task_id](task, ep)


# ---------------------------------------------------------------------------
def verify(task, ep):
    """Buy one input that separates every wrong candidate, rather than five that
    separate none."""
    listed = ep.act("list_candidates")
    names = listed["observation"]["candidates"]

    # These three inputs between them expose every defect a plausible-looking
    # bracket checker has: wrong kind, unclosed opener, and the empty string.
    probes = ["(]", "(", ""]
    behaviour = {n: [] for n in names}
    for s in probes:
        reply = ep.act("run_all", {"input": s})
        if reply["status"] != "ok":
            continue
        for name, got in reply["observation"]["returned"].items():
            behaviour[name].append(got)

    # We know what a correct implementation must answer for those three.
    expected = [False, False, True]
    survivors = [n for n, got in behaviour.items() if got == expected]

    # If several survive, read one and pick it; if none do, fall back rather
    # than crash -- an agent that gives up without submitting scores nothing.
    pick = survivors[0] if survivors else names[0]
    if len(survivors) > 1:
        ep.act("read_candidate", {"name": pick})
    return ep.act("submit", {"pick": pick})


# ---------------------------------------------------------------------------
def procurement(task, ep):
    """Sample every source, judge a slice of each, and avoid what looks like a
    mirror or a generator."""
    stats = {}
    for src in ("src_00", "src_01", "src_02", "src_03", "src_04"):
        r = ep.act("sample_source", {"source": src, "n": 8})
        if r["status"] != "ok":
            continue
        ids = r["observation"]["items"]
        fetched = ep.act("fetch", {"ids": ids})
        judged = ep.act("judge", {"ids": ids[:4]})
        hashes, vals = {}, {}
        if fetched["status"] == "ok":
            hashes = {k: v.get("content_hash")
                      for k, v in fetched["observation"]["documents"].items()}
        if judged["status"] == "ok":
            vals = judged["observation"]["values"]
        high = sum(1 for v in vals.values() if v == "high")
        stats[src] = {"hashes": set(h for h in hashes.values() if h),
                      "high_rate": high / max(1, len(vals))}

    # A source whose content hashes are already seen elsewhere is a mirror.
    decoys, claimed = [], set()
    for src, s in stats.items():
        if s["hashes"] and s["hashes"] & claimed:
            decoys.append(src)
        else:
            claimed |= s["hashes"]
    # A source that is nearly all junk is either a generator or not worth having.
    decoys += [s for s, v in stats.items()
               if v["high_rate"] < 0.15 and s not in decoys]

    # Separately from decoys: a source can be genuine and still not worth
    # crawling. The contamination gate is on the fraction of junk LANDED, so a
    # source that is mostly spam fails the gate for the whole plan even though
    # the good documents in it are real. Purity, not just authenticity.
    too_dirty = {s for s, v in stats.items() if v["high_rate"] < 0.6}

    # Sampling the same source twice tells us whether it invents new documents.
    for src in list(stats):
        again = ep.act("sample_source", {"source": src, "n": 8})
        if again["status"] == "ok":
            fresh = set(again["observation"]["items"])
            first = set()
            for row in ep.trajectory:
                if (row["action"] == "sample_source"
                        and row["params"].get("source") == src
                        and isinstance(row.get("obs_summary"), dict)):
                    got = row["obs_summary"].get("items") or []
                    first |= {g for g in got if isinstance(g, str)}
            if fresh and not (fresh & first) and src not in decoys:
                decoys.append(src)          # never repeats: it is generating

    keep = [s for s in stats if s not in decoys and s not in too_dirty]
    ep.act("preview_package", {"package": "pkg_00"})
    n_seen_high = sum(1 for row in ep.trajectory if row["action"] == "judge")
    plan = {"crawl": keep, "purchase": ["pkg_00"],
            "suspected_decoys": sorted(set(decoys)),
            # A wide honest interval beats a narrow confident one.
            "estimates": {"M_high": {"point": 90, "lo": 40, "hi": 200}}}
    return ep.act("submit", {"plan": plan})


# ---------------------------------------------------------------------------
def dedup(task, ep):
    """Read the free labelled fixture, set thresholds from it, then use the
    remaining submits to correct whatever the feedback complains about."""
    ep.act("inspect_corpus")
    cal = ep.act("calibration")
    dup_cos, distinct_cos, leak_cos = [], [], []
    if cal["status"] == "ok":
        o = cal["observation"]
        for row in o.get("labelled_pairs", []):
            (dup_cos if row["label"] == "duplicate" else distinct_cos
             ).append(row["cosine"])
        leak = o.get("labelled_leak_example") or {}
        if "cosine" in leak:
            leak_cos.append(leak["cosine"])

    # Put the threshold between the two labelled populations, not at a round number.
    dedup_t = ((min(dup_cos) + max(distinct_cos)) / 2
               if dup_cos and distinct_cos else 0.7)
    decontam_t = (min(leak_cos) - 0.05) if leak_cos else 0.6

    best = None
    for _ in range(4):
        reply = ep.act("submit", {"strategy": {
            "dedup_threshold": round(max(0.05, min(0.95, dedup_t)), 3),
            "decontam_threshold": round(max(0.05, min(0.95, decontam_t)), 3),
            "use_judge": False}})
        if reply["status"] != "ok":
            break
        obs = reply["observation"]
        if best is None or obs["score"] > best:
            best = obs["score"]
        # Move whichever threshold the feedback actually complained about.
        text = " ".join(obs.get("feedback") or [])
        if "leak recall" in text:
            decontam_t -= 0.08
        elif "removed as leaks" in text:
            decontam_t += 0.05
        elif "not duplicates" in text:
            dedup_t += 0.05
        elif "missed" in text:
            dedup_t -= 0.05
        else:
            break
        if not obs.get("submits_remaining"):
            break
    return ep.act("finish")
