"""Tests for the example tasks.

These are aimed at the ways an example task goes quietly wrong: an unreachable
score, a leaked answer, a budget that does not bind, a task that is not actually
deterministic, or an oracle that disagrees with the gate the solver faces. Any of
those makes a task look broken or trivial instead of hard, and none of them is
visible from a passing run.

    python3 -m pytest tests/ -q      (or: python3 tests/test_examples.py)
"""
from __future__ import annotations

import json
import os
import subprocess
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import example_agent                                    # noqa: E402
from ew_examples import Episode, load_task              # noqa: E402
from ew_examples.engine import _assert_no_hidden        # noqa: E402
from ew_examples.tasks import TASKS                     # noqa: E402

ALL = sorted(TASKS)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ---- the pack does what it says --------------------------------------------

@pytest.mark.parametrize("tid", ALL)
def test_every_task_is_solvable_to_a_real_score(tid):
    """The example agent uses no model. If it cannot score above zero, the task is
    not hard, it is broken -- which is exactly how the first version of
    corpus_procurement shipped: its own best plan failed its own gate."""
    task = load_task(tid)
    ep = Episode(task)
    example_agent.solve(task, ep)
    assert ep.result is not None, "the agent finished without being scored"
    assert ep.result["score"] > 0.0, f"{tid} scored {ep.result['score']}"


@pytest.mark.parametrize("tid", ALL)
def test_a_full_score_is_reachable_at_all(tid):
    """A task nobody can max out teaches the wrong thing about the ceiling."""
    task = load_task(tid)
    if tid == "verify_solutions":
        ep = Episode(task)
        ep.act("submit", {"pick": "cand_a"})
        assert ep.result["score"] == 1.0
    elif tid == "corpus_procurement":
        res = task.score({"crawl": ["src_00", "src_02"], "purchase": ["pkg_00"],
                          "suspected_decoys": ["src_01", "src_03"],
                          "estimates": {"M_high": {"point": 90, "lo": 40, "hi": 200}}})
        assert res["score"] == 1.0, res
    elif tid == "corpus_dedup":
        # Measured, not guessed: (0.82, 0.80) is the point where both objectives
        # are satisfied at once. If a future edit makes the ceiling unreachable
        # this fails, which is the whole reason the number is pinned here.
        res = task._grade({"dedup_threshold": 0.82, "decontam_threshold": 0.80})
        assert res["score"] == 1.0, res
        assert res["dedup_f1"] == 1.0 and res["leak_recall"] == 1.0
        assert res["over_removal_rate"] == 0.0
    elif tid == "clinical_signal":
        rr = task._rr([p for p in task._patients if p["dose"] == "dose_high"],
                      "hepatic_enzyme_rise")[0]
        res = task.score({"event": "hepatic_enzyme_rise", "subgroup": "dose_high",
                          "effect": {"risk_ratio": rr}, "excluded": ["site_03"]})
        assert res["score"] == 1.0, res
        assert res["gate_passed"] is True
    elif tid == "treatment_response":
        # The ceiling is above 1.0 by design: 1.0 means matching the
        # every-allowed-column baseline, and beating it means finding which
        # columns carry no signal. So a full score IS reachable and then some.
        res = task.score(["age", "bmi_base", "hba1c", "sex_male",
                          "genetic_risk_score"])
        assert res["gate_passed"] is True
        assert res["score"] >= 1.0, res
    else:
        raise AssertionError(
            f"no ceiling check for {tid!r}. Every task needs one: a task whose top "
            f"score is unreachable looks hard when it is broken, and this branch "
            f"exists so adding a task without checking cannot pass silently.")


@pytest.mark.parametrize("tid", ALL)
def test_tasks_are_deterministic(tid):
    """Same id and seed, same task -- otherwise these cannot be used as fixtures
    and two people comparing notes are comparing different problems."""
    a, b = load_task(tid, seed=5), load_task(tid, seed=5)
    assert a.brief == b.brief
    ea, eb = Episode(a), Episode(b)
    for name in sorted(a.actions()):
        if name in ("submit", "finish"):
            continue
        ra, rb = ea.act(name, {}), eb.act(name, {})
        assert ra["status"] == rb["status"]
        assert ra.get("observation") == rb.get("observation"), name


def test_different_seeds_give_different_tasks():
    t0, t1 = load_task("corpus_procurement", 0), load_task("corpus_procurement", 11)
    assert t0._oracle() != t1._oracle()


# ---- the guard that matters ------------------------------------------------

@pytest.mark.parametrize("tid", ALL)
def test_no_action_ever_serves_a_hidden_key(tid):
    """Underscore-prefixed keys are the answer. Serving one hands over the task and
    nothing looks wrong, so the engine refuses and this proves it stays refused
    across every action each task offers."""
    task = load_task(tid)
    ep = Episode(task)
    for name, spec in task.actions().items():
        if name in ("submit", "finish"):
            continue
        reply = ep.act(name, {"source": "src_00", "n": 3, "name": "cand_a",
                              "input": "()", "package": "pkg_00",
                              "ids": ["nope"], "pairs": [["doc_000", "doc_100"]]})
        if reply["status"] == "ok":
            _assert_no_hidden(reply["observation"])


def test_the_guard_actually_fires():
    """A guard nobody has seen refuse is not known to work."""
    with pytest.raises(RuntimeError) as e:
        _assert_no_hidden({"fine": 1, "nested": {"_truth": "the answer"}})
    assert "_truth" in str(e.value)


@pytest.mark.parametrize("tid", ALL)
def test_the_brief_does_not_contain_the_answer(tid):
    task = load_task(tid)
    low = task.brief.lower()
    for giveaway in ("cand_a is correct", "src_01 is the mirror",
                     "src_03 is the honeypot"):
        assert giveaway not in low


# ---- budgets bind ----------------------------------------------------------

def test_an_exhausted_budget_refuses_but_does_not_crash():
    task = load_task("verify_solutions")
    ep = Episode(task)
    for _ in range(20):
        ep.act("run_all", {"input": "()"})
    assert ep.budget.counters["test_runs"] == 0
    last = ep.act("run_all", {"input": "()"})
    assert last["status"] == "error" and last["error"] == "ActionRefused"
    # ...and the episode is still usable, which is the part that matters.
    assert ep.act("submit", {"pick": "cand_a"})["status"] == "ok"


def test_an_unknown_action_is_charged_and_explains_itself():
    ep = Episode(load_task("verify_solutions"))
    r = ep.act("teleport", {})
    assert r["status"] == "error" and r["error"] == "UnknownAction"
    assert r["cost_charged"] == 1, "probing for action names must not be free"
    assert "run_all" in r["message"]


def test_acting_after_the_episode_ends_is_refused():
    ep = Episode(load_task("verify_solutions"))
    ep.act("submit", {"pick": "cand_a"})
    assert ep.act("run_all", {"input": "()"})["error"] == "EpisodeComplete"


def test_a_malformed_submission_is_recoverable():
    """A solver that submits the wrong shape must be able to try again rather than
    lose the episode."""
    ep = Episode(load_task("verify_solutions"))
    assert ep.act("submit", {"pick": "not_a_candidate"})["status"] == "error"
    assert ep.act("submit", {"pick": "cand_a"})["status"] == "ok"


# ---- task-specific properties worth pinning -------------------------------

def test_the_mirror_source_adds_nothing():
    """The whole lesson of that trap: identical content, zero unique tokens."""
    task = load_task("corpus_procurement")
    alone, _ = task._landed({"crawl": ["src_00"]})
    both, _ = task._landed({"crawl": ["src_00", "src_01"]})
    high_alone = sum(task._docs[d]["tokens"] for d in alone
                     if task._docs[d]["value"] == "high")
    high_both = sum(task._docs[d]["tokens"] for d in both
                    if task._docs[d]["value"] == "high")
    assert high_both == high_alone


def test_the_honeypot_never_runs_out():
    task = load_task("corpus_procurement")
    ep = Episode(task)
    seen = set()
    for _ in range(3):
        r = ep.act("sample_source", {"source": "src_03", "n": 10})
        seen |= set(r["observation"]["items"])
    assert len(seen) == 30, "a generator must never repeat itself"


def test_the_oracle_obeys_the_same_gate_as_the_solver():
    """The bug this exists for: an asserted oracle plan that itself failed the
    contamination gate made every submission score zero."""
    task = load_task("corpus_procurement")
    oracle = task._oracle()
    assert oracle > 0
    best = task.score({"crawl": ["src_00", "src_02"], "purchase": ["pkg_00"],
                       "suspected_decoys": ["src_01", "src_03"],
                       "estimates": {"M_high": {"point": 90, "lo": 40, "hi": 200}}})
    assert best["gate_passed"] is True
    assert best["high_tokens_landed"] == oracle


def test_every_wrong_candidate_is_wrong_and_the_right_one_is_right():
    """Verified against the specification, not against a stored label."""
    from ew_examples.tasks.verify_solutions import CANDIDATES, _reference
    cases = ["", "()", "([{}])", "(]", "(", ")(", "{[}]", "((()))", "[](){}"]
    for name, fn in CANDIDATES.items():
        agrees = all(fn(c) == _reference(c) for c in cases)
        assert agrees == (name == "cand_a"), name


def test_dedup_thresholds_trade_off_against_each_other():
    """If one threshold could max both objectives the task would be trivial."""
    task = load_task("corpus_dedup")
    loose = task._grade({"dedup_threshold": 0.2, "decontam_threshold": 0.2})
    tight = task._grade({"dedup_threshold": 0.9, "decontam_threshold": 0.9})
    assert loose["leak_recall"] >= tight["leak_recall"]
    assert loose["over_removal_rate"] >= tight["over_removal_rate"]


def test_dedup_keeps_the_best_submission_not_the_last():
    task = load_task("corpus_dedup")
    ep = Episode(task)
    ep.act("submit", {"strategy": {"dedup_threshold": 0.55,
                                   "decontam_threshold": 0.5}})
    ep.act("submit", {"strategy": {"dedup_threshold": 0.99,
                                   "decontam_threshold": 0.99}})
    ep.act("finish")
    assert ep.result["score"] == max(ep.result["submit_scores"])


# ---- trajectories ----------------------------------------------------------

def test_the_trajectory_is_written_and_parsable(tmp_path):
    path = str(tmp_path / "traj.jsonl")
    task = load_task("verify_solutions")
    ep = Episode(task, trajectory_path=path)
    example_agent.solve(task, ep)
    rows = [json.loads(l) for l in open(path)]
    assert len(rows) == len(ep.trajectory) > 0
    assert [r["t"] for r in rows] == list(range(1, len(rows) + 1))
    for r in rows:
        assert {"t", "ts", "action", "params", "status", "cost",
                "budget_remaining"} <= set(r)


def test_the_trajectory_never_contains_a_hidden_key(tmp_path):
    """The summary is derived from observations, so the guard must hold here too."""
    for tid in ALL:
        path = str(tmp_path / f"{tid}.jsonl")
        task = load_task(tid)
        ep = Episode(task, trajectory_path=path)
        example_agent.solve(task, ep)
        raw = open(path).read()
        for k in ('"_truth"', '"_oracle"', '"_reference"', '"_true"'):
            assert k not in raw


# ---- it works as shipped ---------------------------------------------------

@pytest.mark.parametrize("tid", ALL)
def test_the_cli_runs_a_task_end_to_end(tid, tmp_path):
    """Everything above imports the package. This proves the documented command
    works from a clean shell, which is what a newcomer actually types."""
    out = str(tmp_path / tid)
    r = subprocess.run(
        [sys.executable, "run_task.py", "--task", tid,
         "--agent", "example_agent:solve", "--out", out],
        cwd=ROOT, capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, r.stderr[-800:]
    assert "score" in r.stdout
    res = json.load(open(os.path.join(out, "result.json")))
    assert res["score"] > 0
    assert os.path.getsize(os.path.join(out, "trajectory.jsonl")) > 0


def test_the_pack_imports_nothing_outside_the_standard_library():
    """The promise on the front of the README. A stray dependency breaks the one
    claim that makes this downloadable."""
    import ast
    allowed = {"ew_examples", "example_agent"}
    stdlib = set(getattr(sys, "stdlib_module_names", ())) | {
        "argparse", "ast", "dataclasses", "hashlib", "importlib", "itertools",
        "json", "os", "random", "shlex", "subprocess", "sys", "time", "typing",
        "inspect", "__future__"}
    offenders = []
    for dirpath, _dirs, files in os.walk(os.path.join(ROOT, "ew_examples")):
        for f in files:
            # Skip OS/editor droppings: an AppleDouble "._foo.py" is not source,
            # and trying to parse one fails with a UnicodeDecodeError that looks
            # nothing like the dependency problem this test is about.
            if not f.endswith(".py") or f.startswith("._"):
                continue
            tree = ast.parse(open(os.path.join(dirpath, f)).read())
            for node in ast.walk(tree):
                mods = []
                if isinstance(node, ast.Import):
                    mods = [a.name.split(".")[0] for a in node.names]
                elif isinstance(node, ast.ImportFrom) and node.level == 0:
                    mods = [(node.module or "").split(".")[0]]
                for m in mods:
                    if m and m not in stdlib and m not in allowed:
                        offenders.append(f"{f}: {m}")
    assert offenders == [], offenders


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))


# ---- the model loop (offline) ----------------------------------------------

def test_the_model_loop_plays_a_task_with_a_stubbed_model(monkeypatch):
    """Proves the loop, not the model: prompt built, JSON parsed out of prose and
    a code fence, action dispatched, observation fed back, submission reached."""
    import llm_agent
    monkeypatch.setenv("EW_MODEL_HOOK", "tests.stub_model:reply")
    task = load_task("verify_solutions")
    ep = Episode(task)
    llm_agent.solve(task, ep, verbose=False)
    assert ep.done and ep.result["score"] == 1.0
    assert len(ep.trajectory) == 5


def test_the_action_parser_takes_the_last_object_not_the_first(monkeypatch):
    """A model that names an option and then rejects it must not have the rejected
    one executed. That failure looks like a bad decision by the model rather than
    a bug in the harness, which is why it is pinned here."""
    import llm_agent
    monkeypatch.setenv("EW_MODEL_HOOK", "tests.stub_model:chatty_reply")
    task = load_task("verify_solutions")
    ep = Episode(task)
    llm_agent.solve(task, ep, verbose=False)
    assert [r["action"] for r in ep.trajectory][0] == "list_candidates"
    assert "read_candidate" not in [r["action"] for r in ep.trajectory]
    assert ep.result["score"] == 1.0


def test_call_model_without_an_implementation_says_what_to_do():
    import llm_agent
    with pytest.raises(NotImplementedError) as e:
        llm_agent.call_model("sys", [])
    assert "call_model" in str(e.value)


def test_the_harness_adapter_exposes_every_action():
    """A framework that owns its own loop gets tool schemas plus a dispatcher."""
    import llm_agent
    task = load_task("corpus_dedup")
    ep = Episode(task)
    schemas, call = llm_agent.tools_for(ep)
    assert {s["function"]["name"] for s in schemas} == set(task.actions())
    assert call("calibration", {})["status"] == "ok"
    assert call("embed_cosine", '{"pairs": [["doc_000", "doc_100"]]}')["status"] == "ok"


# ---- the clinical gate -----------------------------------------------------

def test_a_correct_finding_fails_the_gate_if_nothing_was_checked():
    """The property the whole task exists for. Being right is not sufficient: a
    number nobody can audit is not a result. If this ever passes, the task has
    stopped teaching the only thing it was built to teach."""
    task = load_task("clinical_signal")
    ep = Episode(task)
    ep.act("submit", {"finding": {"event": "hepatic_enzyme_rise",
                                  "subgroup": "dose_high",
                                  "effect": {"risk_ratio": 3.4}}})
    assert ep.result["right_event"] is True          # the answer was correct
    assert ep.result["right_subgroup"] is True
    assert ep.result["gate_passed"] is False         # and it still scores nothing
    assert ep.result["score"] == 0.0


def test_one_free_declaration_rescues_the_same_finding():
    """The identical submission that scored zero above, plus one free action."""
    task = load_task("clinical_signal")
    ep = Episode(task)
    ep.act("declare_limitation",
           {"issue": "alt_value units are not harmonised across sites"})
    ep.act("submit", {"finding": {"event": "hepatic_enzyme_rise",
                                  "subgroup": "dose_high",
                                  "effect": {"risk_ratio": 3.4}}})
    assert ep.result["gate_passed"] is True
    assert ep.result["score"] > 0.8


def test_declaring_a_limitation_is_free_and_unlimited():
    """If it cost anything, the task would be teaching people to stay quiet."""
    task = load_task("clinical_signal")
    ep = Episode(task)
    before = dict(ep.budget.counters)
    for i in range(30):
        r = ep.act("declare_limitation", {"issue": f"concern {i}"})
        assert r["status"] == "ok"
    assert ep.budget.counters == before


def test_excluding_the_bad_site_also_passes_the_gate():
    """Two honest routes: report around the problem, or declare it."""
    task = load_task("clinical_signal")
    ep = Episode(task)
    ep.act("submit", {"finding": {"event": "hepatic_enzyme_rise",
                                  "subgroup": "dose_high",
                                  "effect": {"risk_ratio": 3.4},
                                  "excluded": ["site_03"]}})
    assert ep.result["gate_passed"] is True


def test_the_field_metadata_discloses_the_problem_rather_than_hiding_it():
    """The task must be fair: the ambiguity is discoverable in one query. A trap
    that cannot be found by looking is not a lesson, it is a coin flip."""
    task = load_task("clinical_signal")
    ep = Episode(task)
    obs = ep.act("field_metadata", {"field": "alt_value"})["observation"]
    assert obs["consistent_across_sites"] is False
    assert "NOT HARMONISED" in obs["units"]


def test_the_spurious_signal_scores_far_below_the_real_one():
    task = load_task("clinical_signal")
    real = task.score({"event": "hepatic_enzyme_rise", "subgroup": "dose_high",
                       "effect": {"risk_ratio": task._rr(
                           [p for p in task._patients
                            if p["dose"] == "dose_high"], "hepatic_enzyme_rise")[0]},
                       "excluded": ["site_03"]})
    spurious = task.score({"event": "rash", "subgroup": "age_65_plus",
                           "excluded": ["site_03"]})
    assert real["score"] == 1.0
    assert spurious["score"] < 0.3


# ---- the leakage gate ------------------------------------------------------

def test_a_post_treatment_column_scores_zero():
    """The one mistake this task exists for."""
    task = load_task("treatment_response")
    res = task.score(["bmi_base", "POST_early_weight_change_4w"])
    assert res["gate_passed"] is False
    assert res["score"] == 0.0
    assert res["leaked_columns"] == ["POST_early_weight_change_4w"]


def test_the_forbidden_columns_are_genuinely_more_tempting():
    """If the trap were weak the task would teach nothing. The forbidden column
    must look BETTER than every allowed one, because that is the situation a real
    extract puts an analyst in."""
    task = load_task("treatment_response")
    ep = Episode(task)
    from ew_examples.tasks.treatment_response import ALLOWED, FORBIDDEN
    def corr(col):
        return abs(ep.act("correlation", {"column": col})
                   ["observation"]["correlation_with_outcome"])
    worst_forbidden = min(corr(c) for c in FORBIDDEN)
    best_allowed = max(corr(c) for c in ALLOWED)
    assert worst_forbidden > best_allowed, (
        f"the trap is not tempting: forbidden {worst_forbidden} vs allowed "
        f"{best_allowed}")


def test_the_task_warns_before_it_punishes():
    """A gate that fires without warning is a trick. Both the correlation and the
    fit report say plainly that the column is forbidden and why."""
    task = load_task("treatment_response")
    ep = Episode(task)
    c = ep.act("correlation", {"column": "POST_adherence_proxy"})["observation"]
    assert "FORBIDDEN" in c["warning"]
    f = ep.act("fit_report", {"features": ["POST_adherence_proxy"]})["observation"]
    assert "forbidden" in f["warning"].lower()
    assert "zero" in f["warning"].lower()


def test_the_baselines_form_a_real_ladder():
    """Each rung must add something, or the score means nothing."""
    task = load_task("treatment_response")
    b = task._baselines()
    assert b["mean_only"] < b["clinical_only"] < b["clinical_plus_genetic"]
    assert b["all_allowed"] > b["clinical_only"]


def test_leaky_features_fit_better_and_still_score_zero():
    """The uncomfortable fact the task is built around: the leaky model really is
    a better fit on cohort A. That is why warnings are not enough on their own."""
    task = load_task("treatment_response")
    ep = Episode(task)
    honest = ep.act("fit_report", {"features": ["bmi_base", "genetic_risk_score",
                                                "hba1c"]})["observation"]
    leaky = ep.act("fit_report", {"features": ["POST_early_weight_change_4w"]}
                   )["observation"]
    assert leaky["train_r2"] > honest["train_r2"]
    assert task.score(["POST_early_weight_change_4w"])["score"] == 0.0
