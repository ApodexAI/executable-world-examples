# Executable World — example tasks

Four example tasks in the shape of an Executable World environment, so you can see
what one feels like and run an agent against it on your laptop.

**No install, no account, no network, no Docker.** Python 3.9+ and the standard
library.

```bash
python3 run_task.py --list
python3 run_task.py --task verify_solutions --brief
python3 run_task.py --task verify_solutions --interactive
python3 run_task.py --task verify_solutions --agent example_agent:solve
```

## What makes these different from a chat benchmark

**You cannot see the world.** Everything is behind typed actions, and every action
spends from a finite budget. There is no context window containing the problem —
there is a brief, and a set of things you may ask. Deciding *what to look at* is
most of the task.

**Your submission is executed against hidden truth.** You are scored on what your
plan actually lands, not on what you claimed it would. A confident wrong answer
scores worse than an honest uncertain one.

**Some of what you are shown is a trap.** Not to be unfair — because real data has
mirrors, generators, and things that look valuable and are not. An agent that
believes everything it is served does badly, and that is the point.

## The four tasks

| task | what you do | the lesson |
|---|---|---|
`corpus_procurement` | probe hidden data sources, submit a purchasing plan | a source can be genuine and still not worth having |
`verify_solutions` | decide which of five implementations is correct | one well-chosen test beats five careless ones |
`corpus_dedup` | deduplicate a corpus, keep a benchmark out of it | two objectives that pull against each other |
`clinical_signal` | find the safety signal in a trial and report it | a correct number you cannot justify is not a result |

`clinical_signal` is the one to try if you only try one. It is the only task here
with a **gate**: a finding that rests on data you neither checked nor flagged scores
zero even when the finding is right. Declaring a limitation is free and unlimited,
and it is the only thing that can save the marks. Most agents fail it the first time
by being fluent and confident over a field whose own metadata says it was never
harmonised — which is exactly how capable analysts produce unusable work.

Each is one *shape* of problem. The real environments have many instances of each
shape, and more shapes than these.

## Driving a task from your own code

```python
from ew_examples import load_task, Episode

task = load_task("verify_solutions")          # same id+seed -> same task, always
ep = Episode(task, trajectory_path="traj.jsonl")

print(task.brief)                             # what an agent should be told

reply = ep.act("run_all", {"input": "(]"})
print(reply["status"], reply["budget_remaining"])
print(reply["observation"])

ep.act("submit", {"pick": "cand_a"})
print(ep.result["score"])
```

Every reply has the same outer shape, whatever the task:

```json
{
  "protocol": 1,
  "status": "ok",
  "cost_charged": 1,
  "budget_remaining": {"reads": 5, "test_runs": 11},
  "observation": {"...": "task-specific"}
}
```

`status` is `"ok"` or `"error"`. An error is charged but not fatal — a mistyped
action name, an exhausted budget and a malformed submission all come back as
something you can read and recover from, which is deliberate. An agent that dies on
the first refusal will do badly for reasons that have nothing to do with its
reasoning.

To plug in your own agent, write a function taking `(task, episode)` and pass
`--agent yourmodule:yourfunction`. See `example_agent.py` — it solves all four
with no model at all, so you can see the loop before adding one.

## Trajectories

Every run writes `runs/<task>-<timestamp>/trajectory.jsonl`, one line per action:

```json
{"t": 3, "ts": 1787252231.07, "action": "run_all", "params": {"input": "(]"},
 "status": "ok", "cost": 1, "budget_remaining": {"reads": 5, "test_runs": 10},
 "obs_summary": {"input": "(]", "returned": {"cand_a": false, "cand_b": true}}}
```

This is the same format the real system records, which is the useful part: whatever
you build here already emits artifacts of the right shape.

## What is not in here

The real environments, their instances, their verifiers, the process-scoring
apparatus, and the framework that runs them. These four tasks were written for
public release; they imitate the shape of real environments and share no code or
data with them.

So: a good score here means your harness works. It does not mean anything about how
you would do on a real environment, and it is not a submission to anything.

## Going further

The real environments cover pretraining data work, post-training, RL recipes,
solution verification, SWE tasks, determinism debugging, clinical trial analysis and
protein design. Running against those is a separate, formal arrangement — get in
touch.

## Licence and reuse

Use these as a fixture in your own test suite if useful. They are deterministic, so
they make stable tests: `load_task("corpus_dedup", seed=7)` is the same task on
every machine, forever.

A different `seed` gives a genuinely different instance of `corpus_procurement`,
`corpus_dedup` and `clinical_signal` — new sources, new corpus, new trial — so you
can check whether a strategy generalises or was fitted to one draw.
`verify_solutions` is the exception: its five candidates are fixed, so it has exactly
one instance and the seed does nothing.
