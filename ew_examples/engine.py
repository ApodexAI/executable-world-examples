"""The episode loop: budgets, action dispatch, trajectory recording, scoring.

Self-contained on purpose. This package shares no code with the Executable World
implementation it imitates; it exists so anyone can see the *shape* of an EW task
and drive one from a laptop with nothing installed. The action names, the reply
envelope and the trajectory format match the real thing, so a harness built here
needs no changes to run against a real environment later.

Three ideas carry over from the real system and are worth understanding, because
they are what make these tasks different from a chat benchmark:

  * You cannot see the world. Everything is behind typed actions, and every action
    costs something from a finite budget. Deciding what to look at IS the task.
  * The reply is always the same envelope. Only `observation` differs per task, so
    one parser handles every task you will ever be given.
  * A submission is executed against hidden truth. You are scored on what your plan
    actually lands, not on what you claimed it would.
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from typing import Any, Callable

PROTOCOL = 1


class ActionError(Exception):
    """The action was understood but refused. Charged, not fatal."""


@dataclass
class Action:
    """One thing a solver may do. `cost` is charged whether or not it helps."""
    cost: int
    fn: Callable[["Episode", dict], dict]
    doc: str


@dataclass
class Budget:
    """What a solver may spend. Named counters, because a single number teaches
    the wrong lesson: real tasks meter several scarce things at once and the
    interesting decisions are about trading them off."""
    counters: dict[str, float] = field(default_factory=dict)

    def spend(self, name: str, amount: float) -> None:
        if name not in self.counters:
            return
        if self.counters[name] < amount:
            raise ActionError(
                f"{name} exhausted: {self.counters[name]:g} left, {amount:g} needed")
        self.counters[name] -= amount

    def snapshot(self) -> dict:
        return {k: (int(v) if float(v).is_integer() else round(v, 4))
                for k, v in self.counters.items()}


class Task:
    """Base class for an example task.

    Subclasses set `task_id`, write `brief`, implement `actions()` and `score()`,
    and keep everything a solver must not see in attributes prefixed with `_`.
    That underscore is not decoration: `Episode` refuses to put such a key into an
    observation, so a task cannot leak its own answer through a careless reply.
    """

    task_id = "unnamed"
    brief = ""

    def actions(self) -> dict[str, Action]:
        raise NotImplementedError

    def initial_budget(self) -> Budget:
        raise NotImplementedError

    def score(self, submission: Any) -> dict:
        raise NotImplementedError


def _assert_no_hidden(obj: Any, path: str = "observation") -> None:
    """Refuse to serve any key starting with `_`, at any depth.

    The real system does this and it is the single most useful guard in it: a task
    author adding a debug field called `_truth` to a reply would otherwise hand the
    answer to every solver, and nothing would look wrong.
    """
    if isinstance(obj, dict):
        for k, v in obj.items():
            if isinstance(k, str) and k.startswith("_"):
                raise RuntimeError(
                    f"task tried to serve hidden key {path}.{k} — this is a bug in "
                    f"the task, not in your solver")
            _assert_no_hidden(v, f"{path}.{k}")
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            _assert_no_hidden(v, f"{path}[{i}]")


class Episode:
    """One attempt at one task."""

    def __init__(self, task: Task, *, trajectory_path: str | None = None,
                 clock: Callable[[], float] = time.time):
        self.task = task
        self.budget = task.initial_budget()
        self.actions = task.actions()
        self.t = 0
        self.done = False
        self.result: dict | None = None
        self._clock = clock
        self._rows: list[dict] = []
        self.trajectory_path = trajectory_path
        if trajectory_path:
            d = os.path.dirname(os.path.abspath(trajectory_path))
            if d:
                os.makedirs(d, exist_ok=True)
            open(trajectory_path, "w").close()

    # ---- the one entry point ---------------------------------------------
    def act(self, name: str, params: dict | None = None) -> dict:
        """Take one action. Always returns an envelope; never raises for a bad
        action, because a solver that mistypes an action name should be able to
        read the error and carry on, exactly as against the real API."""
        params = dict(params or {})
        self.t += 1
        if self.done:
            return self._record(name, params, "error", 0,
                                error="EpisodeComplete",
                                message="this episode has already finished")
        spec = self.actions.get(name)
        if spec is None:
            # Charged: probing for action names is not free in the real system.
            return self._record(name, params, "error", 1,
                                error="UnknownAction",
                                message=f"no action {name!r}; available: "
                                        f"{', '.join(sorted(self.actions))}")
        try:
            obs = spec.fn(self, params)
        except ActionError as e:
            return self._record(name, params, "error", spec.cost,
                                error="ActionRefused", message=str(e))
        _assert_no_hidden(obs)
        return self._record(name, params, "ok", spec.cost, observation=obs)

    # ---- bookkeeping ------------------------------------------------------
    def _record(self, name: str, params: dict, status: str, cost: int,
                **rest) -> dict:
        env = {"protocol": PROTOCOL, "status": status, "cost_charged": cost,
               "budget_remaining": self.budget.snapshot()}
        env.update(rest)
        row = {"t": self.t, "ts": round(self._clock(), 2), "action": name,
               "params": params, "status": status, "cost": cost,
               "budget_remaining": env["budget_remaining"]}
        obs = rest.get("observation")
        if obs is not None:
            row["obs_summary"] = _summarise(obs)
        if "error" in rest:
            row["error"] = rest["error"]
        self._rows.append(row)
        if self.trajectory_path:
            with open(self.trajectory_path, "a") as fh:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        return env

    def finish(self, submission: Any) -> dict:
        """Score and end. Called by a task's own `submit` action."""
        self.result = self.task.score(submission)
        self.done = True
        return self.result

    @property
    def trajectory(self) -> list[dict]:
        return list(self._rows)


def _summarise(obs: Any, limit: int = 400) -> Any:
    """What goes in the trajectory: enough to see what happened, not a transcript.

    The real system keeps a full verbatim log beside the summary. Here one file is
    enough, but the truncation is the same idea — a trajectory you cannot read is
    a trajectory nobody checks.
    """
    if isinstance(obs, dict):
        return {k: _summarise(v, limit) for k, v in obs.items()}
    if isinstance(obs, list):
        head = [_summarise(v, limit) for v in obs[:5]]
        return head + [f"...{len(obs) - 5} more"] if len(obs) > 5 else head
    if isinstance(obs, str) and len(obs) > limit:
        return obs[:limit] + f"...[{len(obs)} chars]"
    return obs
