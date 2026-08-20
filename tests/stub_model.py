"""A scripted stand-in for call_model, so the model loop can be tested offline.

Not a model and not pretending to be one: it returns the actions a competent
solver would choose for verify_solutions, in order. Its job is to prove the LOOP
works -- prompt built, JSON parsed, action dispatched, observation fed back,
submission reached -- without a network call or an API key in CI.
"""
import json

SCRIPT = [
    {"action": "list_candidates", "params": {}},
    {"action": "run_all", "params": {"input": "(]"}},
    {"action": "run_all", "params": {"input": "("}},
    {"action": "run_all", "params": {"input": ""}},
    {"action": "submit", "params": {"pick": "cand_a"}},
]


def reply(system, transcript):
    n = sum(1 for m in transcript if m["role"] == "assistant")
    if n >= len(SCRIPT):
        return json.dumps({"action": "submit", "params": {"pick": "cand_a"}})
    # Wrapped in prose and a code fence, which is how models really answer.
    return (f"Looking at the budget, the cheapest evidence is one input against "
            f"all five.\n\n```json\n{json.dumps(SCRIPT[n])}\n```\n")


def chatty_reply(system, transcript):
    """Emits a rejected option BEFORE its real choice, to prove the parser takes
    the last action object rather than the first."""
    real = json.loads(reply(system, transcript).split("```json\n")[1].split("\n```")[0])
    return (f'I could do {{"action": "read_candidate", "params": {{"name": "cand_b"}}}} '
            f'but reading costs the same as testing and tells me less, so instead:\n'
            f'{json.dumps(real)}')
