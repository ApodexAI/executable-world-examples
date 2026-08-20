#!/usr/bin/env python3
"""Run an example task. Standard library only; nothing to install.

    python3 run_task.py --list
    python3 run_task.py --task verify_solutions --brief
    python3 run_task.py --task verify_solutions --interactive
    python3 run_task.py --task verify_solutions --agent example_agent:solve

Every run writes a trajectory to runs/<task>-<timestamp>/trajectory.jsonl in the
same format the real system records, so whatever you build here produces artifacts
of the right shape.

To drive a task from your own code, skip this file and use the API directly:

    from ew_examples import load_task, Episode
    task = load_task("verify_solutions")
    ep = Episode(task, trajectory_path="traj.jsonl")
    print(task.brief)
    reply = ep.act("run_all", {"input": "(]"})
    ...
    ep.result        # populated once the task's submit action has run
"""
from __future__ import annotations

import argparse
import importlib
import json
import os
import shlex
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ew_examples import Episode, load_task              # noqa: E402
from ew_examples.tasks import TASKS                     # noqa: E402


def _run_dir(task_id: str) -> str:
    d = os.path.join("runs", f"{task_id}-{time.strftime('%Y%m%d-%H%M%S')}")
    os.makedirs(d, exist_ok=True)
    return d


def _parse_params(tokens: list[str]) -> dict:
    """`key=value` pairs; a value that parses as JSON is used as JSON.

    So n=5 is an int, input="(]" is a string, and pairs=[["a","b"]] is a list --
    without needing a different flag for each type.
    """
    out: dict = {}
    for tok in tokens:
        if "=" not in tok:
            raise ValueError(f"expected key=value, got {tok!r}")
        k, v = tok.split("=", 1)
        try:
            out[k] = json.loads(v)
        except ValueError:
            out[k] = v
    return out


def interactive(task, ep) -> None:
    print(task.brief)
    print("Type: <action> key=value ...    ('actions' to list, 'quit' to stop)\n")
    while not ep.done:
        try:
            line = input("> ").strip()
        except EOFError:
            print()
            break
        if not line:
            continue
        if line in ("quit", "exit"):
            break
        if line == "actions":
            for name, spec in sorted(task.actions().items()):
                print(f"  {name:18} cost {spec.cost}  {spec.doc}")
            continue
        parts = shlex.split(line)
        try:
            params = _parse_params(parts[1:])
        except ValueError as e:
            print(f"  {e}")
            continue
        reply = ep.act(parts[0], params)
        print(json.dumps(reply, indent=2, ensure_ascii=False)[:2000])


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="Run an Executable World example task.")
    ap.add_argument("--list", action="store_true", help="list the tasks and exit")
    ap.add_argument("--task", help="task id")
    ap.add_argument("--seed", type=int, default=0,
                    help="same id + seed always gives the same task")
    ap.add_argument("--brief", action="store_true",
                    help="print the task brief and exit")
    ap.add_argument("--interactive", action="store_true",
                    help="drive it by hand")
    ap.add_argument("--agent", metavar="module:function",
                    help="drive it with a callable taking (task, episode)")
    ap.add_argument("--out", help="directory for the trajectory (default: runs/...)")
    a = ap.parse_args(argv)

    if a.list or not a.task:
        print("Example tasks:\n")
        for tid, cls in TASKS.items():
            first = (cls.brief or "").strip().splitlines()[0]
            print(f"  {tid:22} {first}")
        print("\n  python3 run_task.py --task <id> --brief")
        return 0 if a.list else 1

    task = load_task(a.task, seed=a.seed)
    if a.brief:
        print(task.brief)
        return 0

    out = a.out or _run_dir(a.task)
    os.makedirs(out, exist_ok=True)
    ep = Episode(task, trajectory_path=os.path.join(out, "trajectory.jsonl"))

    if a.agent:
        if ":" not in a.agent:
            print("--agent takes module:function, e.g. example_agent:solve")
            return 2
        mod_name, fn_name = a.agent.split(":", 1)
        try:
            fn = getattr(importlib.import_module(mod_name), fn_name)
        except (ImportError, AttributeError) as e:
            print(f"cannot load agent {a.agent}: {e}")
            return 2
        fn(task, ep)
    elif a.interactive:
        interactive(task, ep)
    else:
        print(task.brief)
        print("Nothing to drive it with. Add --interactive or "
              "--agent module:function.")
        return 1

    result = ep.result or {"score": None,
                           "note": "the episode never submitted, so it was not scored"}
    with open(os.path.join(out, "result.json"), "w") as fh:
        json.dump(result, fh, indent=2, ensure_ascii=False)

    print(f"\n{'-' * 62}")
    print(f"actions taken : {len(ep.trajectory)}")
    print(f"score         : {result.get('score')}")
    for line in result.get("feedback") or []:
        print(f"  - {line}")
    print(f"trajectory    : {os.path.join(out, 'trajectory.jsonl')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
