"""Parteek Own LLM: a small bilingual language model trained from scratch."""

from typing import Any

__all__ = ["ModelConfig", "TrainingConfig", "ProjectConfig", "ParteekLLM"]
__version__ = "0.1.0"


def __getattr__(name: str) -> Any:
    """Lazy public imports keep lightweight utilities usable independently."""
    if name in {"ModelConfig", "TrainingConfig", "ProjectConfig"}:
        from . import config

        return getattr(config, name)
    if name == "ParteekLLM":
        from .model import ParteekLLM

        return ParteekLLM
    raise AttributeError(name)
