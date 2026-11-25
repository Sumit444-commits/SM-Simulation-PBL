"""Minimal discrete-event simulation kernel (event calendar built on a binary heap).

SimPy could not be installed in the build environment (no network), so the
kernel is implemented directly.  It provides the same core idea -- a time-ordered
future-event list -- in about 30 lines, and keeps the project dependency-free
apart from NumPy/SciPy/Matplotlib.
"""
from __future__ import annotations

import heapq
import itertools
from typing import Callable, Optional


class Simulator:
    def __init__(self) -> None:
        self.now: float = 0.0
        self._heap: list = []
        self._seq = itertools.count()   # FIFO tie-break for simultaneous events
        self.events_processed = 0

    def schedule(self, delay: float, fn: Callable, *args) -> None:
        """Schedule ``fn(*args)`` to run ``delay`` minutes from now."""
        heapq.heappush(self._heap, (self.now + delay, next(self._seq), fn, args))

    def schedule_at(self, time: float, fn: Callable, *args) -> None:
        heapq.heappush(self._heap, (time, next(self._seq), fn, args))

    def run(self, until: float = float("inf"), stop: Optional[Callable[[], bool]] = None) -> None:
        """Process events in time order until the calendar is empty, ``until`` is
        passed, or ``stop()`` returns True (checked after every event)."""
        while self._heap and self._heap[0][0] <= until:
            t, _, fn, args = heapq.heappop(self._heap)
            self.now = t
            fn(*args)
            self.events_processed += 1
            if stop is not None and stop():
                break
