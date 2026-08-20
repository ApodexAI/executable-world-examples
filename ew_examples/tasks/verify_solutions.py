"""Decide which candidate solution is correct.

The shape: one programming problem, five candidate implementations, exactly one of
which is correct. You may run each candidate against inputs you choose, but only a
limited number of times, and you are not told the expected outputs. Then you name
the one you believe is right.

This is a verification task, not a coding task. The interesting part is that the
wrong candidates are not obviously wrong: each passes the cases a hurried reader
would try, and fails on something specific. Finding the input that separates them
is the whole job, and a fixed test budget means guessing costs you.

The reference answer is derived from a real specification rather than stored, so
'the correct one' is genuinely correct and not merely labelled.
"""
from __future__ import annotations

import random

from ..engine import Action, ActionError, Budget, Task

PROBLEM = """\
Write is_balanced(s) -> bool.

s contains only the characters ()[]{}. Return True if every bracket is closed by
the matching kind, in the right order, and every opening bracket is closed.

Examples:  "()[]"    -> True
           "([{}])"  -> True
           "(]"      -> False
           "("       -> False
           ""        -> True
"""


def _reference(s: str) -> bool:
    pairs = {")": "(", "]": "[", "}": "{"}
    stack = []
    for ch in s:
        if ch in "([{":
            stack.append(ch)
        elif ch in pairs:
            if not stack or stack.pop() != pairs[ch]:
                return False
    return not stack


# Each candidate is a real implementation with a real, specific defect. The
# comment on each says what it is; the solver never sees these comments.
CANDIDATES: dict[str, object] = {}


def _cand_a(s):                      # CORRECT
    pairs = {")": "(", "]": "[", "}": "{"}
    stack = []
    for ch in s:
        if ch in "([{":
            stack.append(ch)
        elif ch in pairs:
            if not stack or stack.pop() != pairs[ch]:
                return False
    return not stack


def _cand_b(s):                      # counts only; ignores ORDER and KIND
    return (s.count("(") == s.count(")") and s.count("[") == s.count("]")
            and s.count("{") == s.count("}"))


def _cand_c(s):                      # forgets unclosed openers: "(" -> True
    pairs = {")": "(", "]": "[", "}": "{"}
    stack = []
    for ch in s:
        if ch in "([{":
            stack.append(ch)
        elif ch in pairs:
            if not stack or stack.pop() != pairs[ch]:
                return False
    return True


def _cand_d(s):                      # rejects the empty string
    if not s:
        return False
    return _reference(s)


def _cand_e(s):                      # treats all bracket kinds as interchangeable
    depth = 0
    for ch in s:
        if ch in "([{":
            depth += 1
        else:
            depth -= 1
            if depth < 0:
                return False
    return depth == 0


CANDIDATES.update({"cand_a": _cand_a, "cand_b": _cand_b, "cand_c": _cand_c,
                   "cand_d": _cand_d, "cand_e": _cand_e})


def _source(fn) -> str:
    """The candidate's code, with the giveaway comment stripped."""
    import inspect
    lines = inspect.getsource(fn).splitlines()
    out = []
    for ln in lines:
        if "#" in ln and "def " not in ln:
            ln = ln.split("#")[0].rstrip()
            if not ln.strip():
                continue
        out.append(ln)
    return "\n".join(out).replace("_cand_", "candidate_")


class VerifySolutions(Task):
    task_id = "verify_solutions"
    brief = f"""\
Five candidate implementations are offered. EXACTLY ONE is correct. Name it.

THE PROBLEM THE CANDIDATES CLAIM TO SOLVE
{PROBLEM}
You may read each candidate's source, and you may run any candidate on inputs you
choose — but you are NOT told the expected output, and you have a limited number of
test runs. Work out what the answer should be yourself, then compare.

Every wrong candidate passes the cases a hurried reader would try. Each fails on
something specific. Finding an input that separates them is the task.

ACTIONS
  list_candidates()                   -> the candidate names          [free]
  read_candidate(name)                -> its source code              [1 read]
  run_candidate(name, input)          -> what it returns              [1 test_run]
  run_all(input)                      -> what every candidate returns [1 test_run]
  submit(pick)                        -> ends the episode

`run_all` costs the same as `run_candidate`, so prefer it: one well-chosen input
tested against all five is worth five separate runs.
"""

    def __init__(self, seed: int = 0):
        self._rng = random.Random(seed)
        self._correct = "cand_a"

    def initial_budget(self) -> Budget:
        return Budget({"reads": 5, "test_runs": 12})

    def actions(self) -> dict[str, Action]:
        return {
            "list_candidates": Action(0, self._list, "the candidate names"),
            "read_candidate": Action(1, self._read, "source of one candidate"),
            "run_candidate": Action(1, self._run_one, "run one candidate on an input"),
            "run_all": Action(1, self._run_all, "run every candidate on an input"),
            "submit": Action(0, self._submit, "name the correct one; ends the episode"),
        }

    def _list(self, ep, p):
        return {"candidates": sorted(CANDIDATES), "exactly_one_is_correct": True,
                "problem": PROBLEM}

    def _name(self, p) -> str:
        name = str(p.get("name", ""))
        if name not in CANDIDATES:
            raise ActionError(
                f"unknown candidate {name!r}; try {', '.join(sorted(CANDIDATES))}")
        return name

    def _read(self, ep, p):
        name = self._name(p)
        ep.budget.spend("reads", 1)
        return {"candidate": name, "source": _source(CANDIDATES[name])}

    def _input(self, p) -> str:
        if "input" not in p:
            raise ActionError('pass input="..." (the string to test)')
        s = p["input"]
        if not isinstance(s, str):
            raise ActionError("input must be a string")
        if len(s) > 200:
            raise ActionError("input must be at most 200 characters")
        bad = set(s) - set("()[]{}")
        if bad:
            raise ActionError(
                f"input may only contain ()[]{{}} — found {sorted(bad)}")
        return s

    @staticmethod
    def _call(name: str, s: str):
        try:
            return CANDIDATES[name](s)
        except Exception as e:                 # a candidate may genuinely crash
            return f"raised {type(e).__name__}"

    def _run_one(self, ep, p):
        name = self._name(p)
        s = self._input(p)
        ep.budget.spend("test_runs", 1)
        return {"candidate": name, "input": s, "returned": self._call(name, s),
                "note": "the expected answer is not shown; that is the task"}

    def _run_all(self, ep, p):
        s = self._input(p)
        ep.budget.spend("test_runs", 1)
        return {"input": s,
                "returned": {n: self._call(n, s) for n in sorted(CANDIDATES)},
                "note": "disagreement between candidates on one input is the "
                        "cheapest evidence you can buy"}

    def _submit(self, ep, p):
        pick = p.get("pick") or p.get("submission") or p.get("candidate")
        if not isinstance(pick, str) or pick not in CANDIDATES:
            raise ActionError(
                f"pick must name a candidate: {', '.join(sorted(CANDIDATES))}")
        res = ep.finish(pick)
        return {"status": "submitted", "episode_complete": True,
                "score": res["score"], "feedback": res["feedback"]}

    def score(self, pick) -> dict:
        correct = (pick == self._correct)
        fb = []
        if correct:
            fb.append("correct")
        else:
            defects = {
                "cand_b": 'counts brackets only, so order and kind are ignored: '
                          '"(]" and ")(" both pass',
                "cand_c": 'never checks for unclosed openers, so "(" passes',
                "cand_d": 'rejects the empty string, which is balanced',
                "cand_e": 'treats all bracket kinds as interchangeable: "(]" passes',
            }
            fb.append(f"{pick} is wrong: {defects.get(pick, 'it has a defect')}")
            fb.append(f"the correct candidate was {self._correct}")
            fb.append('an input separating every wrong candidate at once: "(]" and '
                      '"(" and "" between them expose all four')
        return {"score": 1.0 if correct else 0.0, "correct": correct,
                "picked": pick, "feedback": fb}
