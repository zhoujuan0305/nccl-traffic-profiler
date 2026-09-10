"""Aggregate NCCL send chunks into global-rank matrices and host-local windows."""
import argparse
import csv
import gzip
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from time_windows import WindowSeries
from trace_reader import Traces, require, sha256


def write_csv(path, fields, rows):
    """Write CSV, optionally gzip-compressed, without materializing all rows."""
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "wt", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(fields)
        writer.writerows(rows)


class Aggregation:
    """Count chunks once; proxy totals are consistency checks, not extra bytes."""

    def __init__(self, data, step_min=None, step_max=None, window_ns=None):
        self.data = data
        self.step_min, self.step_max, self.window_ns = step_min, step_max, window_ns
        self.total, self.by_step, self.by_operation = Counter(), Counter(), Counter()
        self.windows = {}
        self.selected_bytes = self.excluded_bytes = 0
        self.counts = Counter()
        self._read_transfers()

    def _read_transfers(self):
        seen = set(self.data.apis)
        proxies, chunk_totals, identities = {}, Counter(), {}
        for event in self.data.events():
            kind = event["type"]
            if kind not in ("chunk", "proxy"):
                continue
            rank, identity = event["rank"], (event["rank"], event["id"])
            require(identity not in seen, "Duplicate event ID")
            seen.add(identity)
            root = (rank, event["api_id"])
            require(root in self.data.apis, "Missing parent API")
            api = self.data.apis[root]
            require(event["comm"] == api["comm"], "Transfer/API communicator mismatch")
            require(all(event[k] == api[k] for k in ("step", "microbatch", "phase", "tag_id")),
                    "Transfer/API context mismatch")
            size = event["bytes"]
            require(isinstance(size, int) and size > 0, "Invalid transfer byte count")
            peer = self.data.peer(event)
            signature = (root, event["comm"], event["peer_local"], event["channel"])
            self.counts[kind] += 1
            if kind == "proxy":
                require(event["parent_id"] == event["api_id"] and event["proxy_id"] == event["id"],
                        "Wrong proxy parent/identity")
                proxies[identity] = (size, signature)
                continue
            require(event["parent_id"] == event["proxy_id"], "Wrong chunk parent")
            require(0 < event["start_ns"] <= event["submit_ns"] < event["end_ns"], "Invalid chunk interval")
            proxy = (rank, event["proxy_id"])
            require(identities.setdefault(proxy, signature) == signature, "Inconsistent chunk identity")
            chunk_totals[proxy] += size
            self._accumulate(event, api, peer)
        require(proxies and set(proxies) == set(chunk_totals), "No send chunks or incomplete proxy/chunk coverage")
        for identity, (size, signature) in proxies.items():
            require(chunk_totals[identity] == size, "Proxy/chunk byte conservation failed")
            require(identities[identity] == signature, "Proxy/chunk root or peer/channel mismatch")
        require(self.selected_bytes > 0, "No bytes selected; check step tags/filter and transport visibility")

    def _accumulate(self, event, api, peer):
        step, size, src = event["step"], event["bytes"], event["rank"]
        if ((self.step_min is not None and step < self.step_min) or
                (self.step_max is not None and step > self.step_max)):
            self.excluded_bytes += size
            return
        self.selected_bytes += size
        self.total[src, peer] += size
        self.by_step[step, src, peer] += size
        self.by_operation[event["comm"], api["kind"], api["function"], src, peer] += size
        if self.window_ns:
            key = (self.data.hosts[src], step, src, peer)
            series = self.windows.setdefault(key, WindowSeries(self.window_ns))
            series.add(event["submit_ns"], event["end_ns"], size)

    def export(self, output):
        """Export selected totals and optional windows in separate host clock domains."""
        ranks = range(self.data.world_size)
        write_csv(output / "matrix_bytes.csv", ["src_rank"] + list(ranks),
                  ([src] + [self.total[src, dst] for dst in ranks] for src in ranks))
        for name, fields, values in (
            ("matrix_by_step.csv", ["step", "src_rank", "dst_rank", "bytes"], self.by_step),
            ("matrix_by_operation.csv", ["comm", "kind", "function", "src_rank", "dst_rank", "bytes"],
             self.by_operation)):
            write_csv(output / name, fields, (list(k) + [v] for k, v in sorted(values.items())))
        if self.window_ns:
            def rows():
                for key, series in sorted(self.windows.items()):
                    for row in series.rows():
                        yield list(key) + list(row)
            write_csv(output / "bandwidth_windows.csv.gz",
                      ["clock_domain", "step", "src_rank", "dst_rank", "start_monotonic_ns", "end_monotonic_ns",
                       "estimated_bytes", "estimated_gbps", "completion_bytes", "completion_gbps"], rows())
        runtime = {"rank_hosts": self.data.hosts, "communicators": self.data.comms,
                   "window_ns": self.window_ns, "clock": "host-local CLOCK_MONOTONIC"}
        (output / "runtime.json").write_text(json.dumps(runtime, indent=2) + "\n")
        return {"selected_send_bytes": self.selected_bytes, "excluded_send_bytes": self.excluded_bytes,
                "transport_event_counts": dict(self.counts),
                "observed_selected_step_tags": sorted({key[0] for key in self.by_step})}


def main():
    """Validate a complete job, then write a fresh output directory and provenance."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", action="append", required=True, metavar="HOST=DIRECTORY")
    parser.add_argument("--world-size", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--step-min", type=int)
    parser.add_argument("--step-max", type=int)
    parser.add_argument("--window-us", type=int, help="Optional host-local window width in integer microseconds")
    args = parser.parse_args()
    if args.world_size <= 0 or (args.window_us is not None and args.window_us <= 0):
        parser.error("world-size and window-us must be positive")
    if args.step_min is not None and args.step_max is not None and args.step_min > args.step_max:
        parser.error("step-min must not exceed step-max")
    args.output.mkdir(parents=True, exist_ok=False)
    record = {"date": datetime.now(timezone.utc).isoformat(), "status": "invalidated",
              "arguments": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
              "code_sha256": {name: sha256(Path(__file__).with_name(name)) for name in
                              ("aggregate.py", "trace_reader.py", "time_windows.py")},
              "scope": "Observed NCCL send proxy chunks; not physical wire counters or GPU kernel time"}
    try:
        data = Traces(args.input, args.world_size)
        record["source_sha256"] = data.hashes
        result = Aggregation(data, args.step_min, args.step_max,
                             args.window_us * 1000 if args.window_us is not None else None)
        record.update(result.export(args.output))
        data.check_unchanged()
        record["status"] = "completed"
    except Exception as error:
        record["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        (args.output / "validation.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps({"status": record["status"], "selected_send_bytes": record["selected_send_bytes"],
                      "output": str(args.output)}))


if __name__ == "__main__":
    main()
