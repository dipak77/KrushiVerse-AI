"""Autonomous execution layer for the KrushiVerseAI Mini factory.

The package schedules existing ``mini`` workers; it deliberately does not
duplicate the model, data-lake, or release implementations.
"""

from __future__ import annotations

__all__ = ["__version__"]

__version__ = "0.1.0"
