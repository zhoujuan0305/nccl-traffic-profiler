"""Optional process-wide enqueue tags; no training-framework dependency."""
import ctypes
import os
from pathlib import Path


class ProfilerContext:
    """Load the SAME absolute plugin path used by NCCL before issuing operations."""

    def __init__(self):
        path = Path(os.environ["NCCL_PROFILER_PLUGIN"])
        if not path.is_absolute() or not path.is_file():
            raise ValueError("NCCL_PROFILER_PLUGIN must be an existing absolute path")
        self.lib = ctypes.CDLL(str(path), mode=ctypes.RTLD_GLOBAL)
        self.lib.fine_set_context.argtypes = [ctypes.c_int] * 3
        self.lib.fine_set_context.restype = None
        self.lib.fine_flush.argtypes = []
        self.lib.fine_flush.restype = None

    def set(self, step, microbatch=-1, phase=0):
        """Tag subsequently enqueued operations; phase: 0 unknown, 1 forward, 2 backward."""
        self.lib.fine_set_context(step, microbatch, phase)

    def flush(self):
        """Flush current records; this does not wait for communication or finalize NCCL."""
        self.lib.fine_flush()
