"""Verify delayed proxy children retain the correct application message tags."""
import json
import os
import subprocess
import sys
import tempfile
from collections import defaultdict
from pathlib import Path


def main():
    """Run the CPU-only lifetime regression in a disposable build subdirectory."""
    binary = Path(sys.argv[1]).resolve()
    with tempfile.TemporaryDirectory(prefix="check-", dir=binary.parent) as output:
        subprocess.run([str(binary)], check=True,
                       env={**os.environ, "RANK": "0", "FINE_TRACE_DIR": output})
        events = [json.loads(line) for line in
                  (Path(output) / "rank-0/nccl-events.jsonl").read_text().splitlines()]
        totals = defaultdict(int)
        for event in events:
            if event["type"] == "chunk":
                totals[event["step"]] += event["bytes"]
        assert dict(totals) == {1: 2048, 2: 2048, 3: 2048}, dict(totals)
        print("PASS: delayed proxy children preserve all three message identities")


if __name__ == "__main__":
    main()
