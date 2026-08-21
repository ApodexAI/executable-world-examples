"""Drive any of these tasks with a language model, or from your own harness.

Two entry points, for the two ways people arrive here.

1. YOU HAVE A MODEL AND WANT A LOOP.
   Implement `call_model` below — about five lines with any SDK — and run:

       export OPENAI_API_KEY=...          # or whatever your client needs
       python3 run_task.py --task verify_solutions --agent llm_agent:solve

   `solve` works for all three tasks without knowing anything about them: it reads
   the brief, lists the actions, and asks the model for one action at a time. That
   generality is the point. A real environment hands you a brief you have never
   seen, so an agent that needs per-task code is not an agent.

2. YOU HAVE A HARNESS AND WANT TO POINT IT AT THIS.
   Use `tools_for(episode)` to get your loop's tool definitions and a dispatcher.
   Your harness keeps its own loop, prompt and memory; this just adapts the task's
   actions to the shape most frameworks expect. See the bottom of this file.

This module needs no third-party package until YOU add one. The pack itself stays
standard-library only.
"""
from __future__ import annotations

import json
import os
import re

MAX_STEPS = 40


# ---------------------------------------------------------------------------
# 1. The one function you implement
# ---------------------------------------------------------------------------
def call_model(system: str, transcript: list[dict]) -> str:
    """Send the conversation to your model, return its reply as text.

    `transcript` is a list of {"role": "user"|"assistant", "content": str}.

    OpenAI:
        from openai import OpenAI
        client = OpenAI()
        r = client.chat.completions.create(
            model="your-model",
            messages=[{"role": "system", "content": system}] + transcript)
        return r.choices[0].message.content

    Anthropic:
        import anthropic
        client = anthropic.Anthropic()
        r = client.messages.create(
            model="your-model", max_tokens=2048,
            system=system, messages=transcript)
        return r.content[0].text

    A local server, via the stdlib only (no SDK needed):
        import json, urllib.request
        body = json.dumps({"model": "your-model",
                           "messages": [{"role": "system", "content": system}]
                                       + transcript}).encode()
        req = urllib.request.Request(
            os.environ["MODEL_URL"] + "/chat/completions", data=body,
            headers={"Content-Type": "application/json",
                     "Authorization": "Bearer " + os.environ["MODEL_KEY"]})
        with urllib.request.urlopen(req, timeout=120) as r:
            return json.loads(r.read())["choices"][0]["message"]["content"]
    """
    hook = os.environ.get("EW_MODEL_HOOK")
    if hook:
        # Escape hatch for testing: "module:function" with the same signature.
        import importlib
        mod, fn = hook.split(":", 1)
        return getattr(importlib.import_module(mod), fn)(system, transcript)
    raise NotImplementedError(
        "Implement call_model() in llm_agent.py — see the docstring for a "
        "five-line version for OpenAI, Anthropic, or a plain HTTP endpoint.")


# ---------------------------------------------------------------------------
# 2. A task-agnostic loop
# ---------------------------------------------------------------------------
SYSTEM = """\
You are operating inside a sandboxed environment. You cannot see it directly. You
act only through the actions listed below, and every action spends from a finite
budget shown to you after each step.

Reply with ONE action as JSON and nothing else:

    {"action": "<name>", "params": {...}}

Rules that decide whether you do well here:
  * Spend free actions before paid ones. Anything labelled free or cost 0 is
    information you are being given; not taking it is simply worse.
  * Prefer the action that separates several possibilities at once over several
    that each settle one.
  * An error reply is not fatal. Read it, fix the call, continue.
  * You must finish by submitting. An unsubmitted episode scores nothing, however
    good your reasoning was.
"""


def _describe_actions(task) -> str:
    return "\n".join(f"  {name}  (cost {spec.cost})  {spec.doc}"
                     for name, spec in task.actions().items())


def _extract_action(text: str) -> dict | None:
    """Find the action object in a model reply.

    Models wrap JSON in prose and code fences, and they emit several objects when
    they think out loud. Take the LAST well-formed object that has an "action"
    key: an earlier one is usually the model considering an option it then talks
    itself out of, and acting on that is a bug that looks like a bad decision.
    """
    if not text:
        return None
    found = None
    for m in re.finditer(r"\{", text):
        depth, end = 0, None
        for i in range(m.start(), len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    end = i + 1
                    break
        if end is None:
            continue
        try:
            obj = json.loads(text[m.start():end])
        except ValueError:
            continue
        if isinstance(obj, dict) and isinstance(obj.get("action"), str):
            found = obj
    return found


def solve(task, ep, *, max_steps: int = MAX_STEPS, verbose: bool = True):
    """Play any task with a model. Task-agnostic by construction."""
    system = SYSTEM + "\nACTIONS AVAILABLE\n" + _describe_actions(task)
    transcript = [{"role": "user", "content":
                   f"TASK BRIEF\n{task.brief}\n\nTake your first action."}]

    for step in range(max_steps):
        if ep.done:
            break
        try:
            reply = call_model(system, transcript)
        except NotImplementedError:
            raise
        except Exception as e:
            # A model call failing is not the task's fault, and giving up here
            # would score zero. Tell it what happened and let it retry.
            transcript.append({"role": "user",
                               "content": f"Your last call failed ({e}). Retry."})
            continue

        transcript.append({"role": "assistant", "content": reply})
        chosen = _extract_action(reply)
        if chosen is None:
            transcript.append({"role": "user", "content":
                               'I could not find an action in that. Reply with '
                               'exactly {"action": "...", "params": {...}} and '
                               'nothing else.'})
            continue

        env = ep.act(chosen["action"], chosen.get("params") or {})
        if verbose:
            print(f"  [{step + 1}] {chosen['action']:18} {env['status']}"
                  f"  budget={env['budget_remaining']}")
        transcript.append({"role": "user", "content": json.dumps(
            {k: v for k, v in env.items() if k != "protocol"},
            ensure_ascii=False)[:6000]})

    if not ep.done and verbose:
        print("  never submitted — that scores nothing, whatever the reasoning was")
    return ep.result


# ---------------------------------------------------------------------------
# 3. Adapter for your own harness
# ---------------------------------------------------------------------------
def tools_for(episode) -> tuple[list[dict], "callable"]:
    """Tool schemas plus a dispatcher, for a harness that owns its own loop.

    Returns (schemas, call) where `schemas` is a list of JSON-schema tool
    definitions in the shape OpenAI and Anthropic both accept, and `call(name,
    params)` runs one and returns the reply envelope as a dict.

        schemas, call = tools_for(ep)
        # register `schemas` with your framework, route tool calls to `call`
        # your loop ends when ep.done is True; ep.result holds the score

    Nothing here holds state: the episode does, so your loop can be as strange as
    it likes -- retries, branching, several models -- and the accounting stays
    correct.
    """
    task = episode.task
    schemas = []
    for name, spec in task.actions().items():
        schemas.append({
            "type": "function",
            "function": {
                "name": name,
                "description": f"{spec.doc} (costs {spec.cost})",
                "parameters": {
                    "type": "object",
                    # Params are per-action and documented in the brief rather
                    # than typed here, because a task may take a nested plan
                    # object and pinning a schema per task would defeat the
                    # point of one adapter for all of them.
                    "properties": {},
                    "additionalProperties": True,
                },
            },
        })

    def call(name: str, params: dict | str | None = None) -> dict:
        if isinstance(params, str):          # frameworks often hand back a string
            try:
                params = json.loads(params or "{}")
            except ValueError:
                params = {}
        return episode.act(name, params or {})

    return schemas, call
