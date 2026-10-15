"""
Tool registry.

Each tool entry is defined as::

    {
        "key": "cells_morph",
        "label": "Phase-contrast cell segmentation + morphology clustering",
        "description": "Segment dense phase-contrast cells with watershed, extract morphological features and cluster with KMeans.",
        "params": [ ParamSchema, ... ],
        "run": callable(params_dict) -> str (returns a result summary / output path),
    }
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass
class Param:
    """Schema for a single parameter."""
    key: str
    label: str
    type: str = "str"           # file / files / dir / int / float / str / bool / choice
    default: Any = ""
    help: str = ""
    optional: bool = False      # when False the UI performs required/existence checks
    # file / dir
    file_mode: str = "open"     # open / save
    filetypes: list[tuple[str, str]] = field(default_factory=list)
    # int / float
    min: float | None = None
    max: float | None = None
    # choice
    choices: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "key": self.key,
            "label": self.label,
            "type": self.type,
            "default": self.default,
            "help": self.help,
            "optional": self.optional,
            "file_mode": self.file_mode,
            "filetypes": self.filetypes,
            "min": self.min,
            "max": self.max,
            "choices": self.choices,
        }


@dataclass
class Tool:
    key: str
    label: str
    description: str
    params: list[Param]
    run: Callable[[dict[str, Any]], str]

    def to_dict(self) -> dict:
        return {
            "key": self.key,
            "label": self.label,
            "description": self.description,
            "params": [p.to_dict() for p in self.params],
        }


# Registered by each modules/*.py at import time
REGISTRY: dict[str, Tool] = {}


def register(tool: Tool) -> None:
    REGISTRY[tool.key] = tool


def list_tools() -> list[Tool]:
    # Trigger module registration
    from . import modules  # noqa: F401
    return list(REGISTRY.values())
