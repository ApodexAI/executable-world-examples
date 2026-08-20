"""The example tasks. Each is one shape of problem, not one instance of it."""
from .corpus_dedup import CorpusDedup
from .corpus_procurement import CorpusProcurement
from .verify_solutions import VerifySolutions

TASKS = {
    "corpus_procurement": CorpusProcurement,
    "verify_solutions": VerifySolutions,
    "corpus_dedup": CorpusDedup,
}


def load_task(task_id: str, seed: int = 0):
    """Build a task by id. Same id and seed always give the same task."""
    try:
        cls = TASKS[task_id]
    except KeyError:
        raise KeyError(
            f"no task {task_id!r}; available: {', '.join(sorted(TASKS))}") from None
    return cls(seed=seed)


__all__ = ["TASKS", "load_task", "CorpusProcurement", "VerifySolutions",
           "CorpusDedup"]
