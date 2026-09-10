"""Sparse accumulation of explicitly estimated bandwidth on one host clock."""
from collections import defaultdict
import math


class WindowSeries:
    """Distribute chunk bytes uniformly in time and also count completion bursts."""

    def __init__(self, width_ns):
        if width_ns <= 0:
            raise ValueError("Window width must be positive")
        self.width = width_ns
        self.partial = defaultdict(float)
        self.delta = defaultdict(float)
        self.completed = defaultdict(int)
        self.expected_bytes = 0

    def add(self, start_ns, end_ns, byte_count):
        """Use half-open transfer intervals; no cross-host clock subtraction occurs."""
        if not 0 <= start_ns < end_ns or byte_count <= 0:
            raise ValueError("Invalid transfer interval or size")
        width = self.width
        first, last = start_ns // width, (end_ns - 1) // width
        rate = byte_count / (end_ns - start_ns)
        if first == last:
            self.partial[first] += byte_count
        else:
            self.partial[first] += ((first + 1) * width - start_ns) * rate
            self.partial[last] += (end_ns - last * width) * rate
            if last > first + 1:
                self.delta[first + 1] += width * rate
                self.delta[last] -= width * rate
        self.completed[last] += byte_count
        self.expected_bytes += byte_count

    def rows(self):
        """Yield sparse nonzero bins and verify conservation after interpolation."""
        points = set(self.partial) | set(self.delta) | set(self.completed)
        if not points:
            return
        running, total, completed_total = 0.0, 0.0, 0
        tolerance = max(1e-6, self.expected_bytes * 1e-12)
        for index in range(min(points), max(points) + 1):
            running += self.delta.get(index, 0.0)
            value = self.partial.get(index, 0.0) + running
            completion = self.completed.get(index, 0)
            if -tolerance < value < 0:
                value = 0.0
            if value < 0:
                raise ValueError("Negative estimated window bytes")
            total += value
            completed_total += completion
            if value or completion:
                yield (index * self.width, (index + 1) * self.width, value,
                       value * 8 / self.width, completion, completion * 8 / self.width)
        if not math.isclose(total, self.expected_bytes, rel_tol=1e-9, abs_tol=1e-4):
            raise ValueError(f"Window byte conservation failed: {total} != {self.expected_bytes}")
        if completed_total != self.expected_bytes:
            raise ValueError("Completion byte conservation failed")
