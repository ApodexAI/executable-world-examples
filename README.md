# Executable World — example tasks

Five example tasks built in the shape of an Executable World environment, so you can
see what one feels like and get an agent running against one on your laptop this
afternoon.

**Nothing to install.** Python 3.9+ and the standard library. No account, no API key,
no network, no Docker.

```bash
git clone git@github.com:ApodexAI/executable-world-examples.git
cd executable-world-examples

python3 run_task.py --list
python3 run_task.py --task clinical_signal --brief
python3 run_task.py --task clinical_signal --interactive
python3 run_task.py --task clinical_signal --agent example_agent:solve
```

---

## Read this before anything else

**These are not our environments.** They were written for this repository, they
share no code and no data with the real ones, and they are far simpler. A perfect
score here means your harness works. It says nothing about how you would do on a
real environment, and it is not a submission to anything.

What they *are* is a faithful model of the **interface and the habits of mind** a
real environment rewards. Same reply envelope, same action-and-cost structure, same
trajectory format. Build against these and the real thing will not surprise you
structurally — only in difficulty.

[What the real environments add](#what-the-real-thing-adds) is at the bottom, and
worth reading before you assume this is the whole picture.

---

## What makes these different from a chat benchmark

**You cannot see the world.** Everything is behind typed actions, and every action
spends from a finite budget. There is no context window containing the problem —
there is a brief, and a set of things you may ask. Deciding *what to look at* is
most of the task.

**Your submission is executed against hidden truth.** You are scored on what your
plan actually lands, not on what you claimed it would. A confident wrong answer
scores worse than an honest uncertain one.

**Some of what you are shown is a trap.** Not to be unfair — because real data has
mirrors, generators, columns you must not touch, and fields whose units nobody
harmonised. An agent that believes everything it is served does badly, and that is
the point.

## The five tasks

| task | what you do | the lesson |
|---|---|---|
`corpus_procurement` | probe hidden data sources, submit a purchasing plan | a source can be genuine and still not worth having |
`verify_solutions` | decide which of five implementations is correct | one well-chosen test beats five careless ones |
`corpus_dedup` | deduplicate a corpus, keep a benchmark out of it | two objectives that pull against each other |
`clinical_signal` | find the safety signal in a trial and report it | a correct number you cannot justify is not a result |
`treatment_response` | pick columns to predict treatment response | the strongest predictors in the table are the ones you may not use |

Two of them have a **gate** — a rule that zeroes an otherwise good answer:

- `clinical_signal` fails a finding that rests on data you neither checked nor
  flagged, *even when the finding is correct*. Declaring a limitation is free and
  unlimited, and it is the only thing that can save the marks.
- `treatment_response` fails any model built on a post-treatment column, however
  well it fits.

Both gates warn you first. If you only try one task, try `clinical_signal` — most
agents fail it the first time by being fluent and confident over a field whose own
metadata says it was never verified, which is exactly how capable analysts produce
unusable work.

A different `--seed` gives a genuinely different instance of every task except
`verify_solutions`, whose five candidates are fixed.

---

## How to use it

### 1. Play a task yourself

```
$ python3 run_task.py --task verify_solutions --interactive

> run_all input="(]"
{
  "protocol": 1,
  "status": "ok",
  "cost_charged": 1,
  "budget_remaining": {"reads": 5, "test_runs": 11},
  "observation": {
    "input": "(]",
    "returned": {"cand_a": false, "cand_b": false, "cand_c": false,
                 "cand_d": false, "cand_e": true},
    "note": "disagreement between candidates on one input is the cheapest evidence you can buy"
  }
}

> submit pick=cand_a
{"status": "ok", "observation": {"score": 1.0, "feedback": ["correct"]}}
```

You type the action; the environment cannot tell you apart from a model. Ten minutes
of this is worth more than reading the rest of this file.

### 2. Watch the bundled agent

`example_agent.py` solves all five with **no model at all** — just fixed logic — so
you can see the loop before adding anything.

```bash
python3 run_task.py --task treatment_response --agent example_agent:solve
```

```
actions taken : 12
score         : 1.0067
  - external R² 0.3321 against baselines mean=-0.013, clinical=0.1994,
    clinical+genetic=0.3321, all_allowed=0.3298
  - beat the all-columns baseline — dropping columns that carry no signal is the
    actual skill here
trajectory    : runs/treatment_response-.../trajectory.jsonl
```

### 3. Drive it from your own code

```python
from ew_examples import load_task, Episode

task = load_task("clinical_signal", seed=3)
ep = Episode(task, trajectory_path="traj.jsonl")

print(task.brief)                       # what your agent should be told

reply = ep.act("field_metadata", {"field": "alt_value"})
print(reply["status"], reply["budget_remaining"], reply["observation"])

ep.act("submit", {"finding": {...}})
print(ep.result["score"], ep.result["feedback"])
```

Every reply has the same outer shape, whatever the task:

```json
{
  "protocol": 1,
  "status": "ok",
  "cost_charged": 1,
  "budget_remaining": {"queries": 24},
  "observation": {"...": "task-specific"}
}
```

`status` is `"ok"` or `"error"`. An error is charged but never fatal — a mistyped
action, an exhausted budget and a malformed submission all come back as something
you can read and recover from. That is deliberate: an agent that dies on the first
refusal does badly for reasons unrelated to its reasoning.

### 4. Plug in a model

Implement one function in `llm_agent.py`:

```python
def call_model(system, transcript) -> str:
    from openai import OpenAI
    r = OpenAI().chat.completions.create(
        model="your-model",
        messages=[{"role": "system", "content": system}] + transcript)
    return r.choices[0].message.content
```

Then:

```bash
python3 run_task.py --task corpus_dedup --agent llm_agent:solve
```

`llm_agent.solve` plays **any** of the five without knowing anything about them: it
reads the brief, lists the actions with their costs, and asks for one action at a
time as JSON. That generality is the point — a real environment hands you a brief
you have never seen, so an agent that needs per-task code is not an agent.

The docstring has the same five lines for Anthropic and for a plain HTTP endpoint
with no SDK, so this repo stays dependency-free until you choose otherwise.

### 5. Point your own harness at it

If your framework owns its loop, take the tools and a dispatcher:

```python
from llm_agent import tools_for

schemas, call = tools_for(ep)     # schemas in the shape OpenAI/Anthropic accept
# register `schemas` with your framework, route tool calls to `call`
# your loop ends when ep.done; ep.result holds the score
```

It holds no state — the episode does — so your loop can retry, branch, or use
several models and the accounting stays correct.

---

## Trajectories

Every run writes `runs/<task>-<timestamp>/trajectory.jsonl` plus `result.json`. One
line per action:

```json
{"t": 3, "ts": 1787252231.07, "action": "run_all", "params": {"input": "(]"},
 "status": "ok", "cost": 1, "budget_remaining": {"reads": 5, "test_runs": 10},
 "obs_summary": {"returned": {"cand_a": false, "cand_b": true}}}
```

This is the format the real system records. Whatever you build here already emits
artifacts of the right shape, so nothing has to be rewritten later.

## A note on the scoring

Every task scores itself locally, and the scoring code ships in the same file you
can read. That is deliberate for practice — reading a scorer is a fast way to learn
what a task values. It also means these scores are **not adversarially robust**:
anyone who wants to game them can, trivially. The real environments keep their
verifier where a solver cannot see it, which is the whole difference between a
practice score and a measurement.

---

## What the real thing adds

Everything below exists in the real system and is deliberately absent here. This is
the honest gap, not a teaser.

**Many more environments.** Seventeen in current scope rather than five, spanning
pretraining data acquisition and filtering, post-training data work, RL recipe
design, solution verification, real software-engineering tasks, inference
determinism debugging, clinical trial analysis, and protein and capsid design. They
are not variations on a theme — each was authored by someone who does that work.

**Real environments, with real data.** Several ship substantial datasets and real
artifacts: actual open-source repositories with their test suites, real preprint
corpora, real clinical datasets, real biological assay data. The examples here are
synthetic because synthetic is all a laptop needs.

**Genuine execution.** Most real environments give your agent a shell in an isolated
container — read a repository, apply a patch, run its tests, train something, inspect
what broke. Several provision GPUs through a metered jobs interface. Nothing here has
a container or a GPU; the tasks are typed queries over generated tables.

**Much tighter episode control.** Per-episode credentials with a two-tier privilege
split, request idempotency, rate limiting, wall-clock enforcement with reaping,
metered model access with a frozen price snapshot, per-submitter instance assignment
that never serves the same task twice, and a hidden verifier the solver cannot read.
The engine in this repo is 194 lines. The real equivalent is a couple of thousand,
and the difference is almost entirely this list.

**HDS6 process scoring.** The outcome score answers *did it work*. HDS6 answers
*was the work sound* — a structured, item-by-item judgement of how a trajectory
reached its result, scored by a panel. It is why a run can land the right answer and
still be marked down: for evidence it never gathered, alternatives it never
considered, or a claim its own logs do not support. Nothing in this repo attempts
this, and it is usually the more informative of the two numbers.

**Held-out instances.** Real evaluation runs on instances you will never have seen,
assigned per submitter with no repeats. Practice here is unlimited and repeatable
precisely because none of it counts.

Running against the real environments is a separate, formal arrangement. Get in
touch.

---

## Licence and reuse

Use these as fixtures in your own test suite if useful. They are deterministic, so
they make stable tests: `load_task("corpus_dedup", seed=7)` is the same task on every
machine, forever.

```bash
python3 -m pytest tests/ -q      # 60 tests, no network, no key
```
