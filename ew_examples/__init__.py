"""Example tasks in the shape of an Executable World environment.

Self-contained: standard library only, no network, no Docker, no account. See
README.md. The real environments this imitates are not included, and neither is
any part of their implementation.
"""
from .engine import Action, ActionError, Budget, Episode, Task
from .tasks import TASKS, load_task

__all__ = ["Action", "ActionError", "Budget", "Episode", "Task", "TASKS",
           "load_task"]
