"""CPU-only conservation, subgroup mapping and incomplete-trace regressions."""
import csv
import gzip
import json
import tempfile
import unittest
from pathlib import Path

from aggregate import Aggregation
from trace_reader import Traces


class AggregateTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.events = {}
        # The sender's local peer=1 denotes global rank=2, not global rank=1.
        for rank in range(3):
            members = [0, 2] if rank != 1 else [1]
            comm = "subgroup" if rank != 1 else "singleton"
            self.events[rank] = [dict(type="comm_member", rank=rank, comm=comm,
                                      local_rank=members.index(rank), size=len(members), nodes=2)]
        for rank, peer, size, step in [(0, 1, 200, 1), (2, 0, 300, 2)]:
            shared = dict(rank=rank, comm="subgroup", step=step, microbatch=-1, phase=0, tag_id=0)
            self.events[rank].append(dict(shared, type="api", id=1, kind="collective", function="AllReduce",
                                          start_ns=100, end_ns=150))
            self.events[rank].append(dict(shared, type="chunk", id=3, api_id=1, parent_id=2, proxy_id=2,
                                          peer_local=peer, channel=0, bytes=size, start_ns=190,
                                          submit_ns=200, end_ns=2200))
            self.events[rank].append(dict(shared, type="proxy", id=2, api_id=1, parent_id=1, proxy_id=2,
                                          peer_local=peer, channel=0, bytes=size))
        for rank in range(3):
            self.events[rank].append(dict(type="comm_finalize", rank=rank,
                                          comm="subgroup" if rank != 1 else "singleton", live_events=0,
                                          write_failures=0, allocation_failures=0, anomalies=0))
        self.save()

    def save(self):
        for rank, events in self.events.items():
            path = self.root / ("host-a" if rank < 2 else "host-b") / f"rank-{rank}"
            path.mkdir(parents=True, exist_ok=True)
            (path / "nccl-events.jsonl").write_text("".join(json.dumps(e) + "\n" for e in events))

    def data(self):
        return Traces([f"host-a={self.root / 'host-a'}", f"host-b={self.root / 'host-b'}"], 3)

    def test_subgroup_mapping_and_no_proxy_double_count(self):
        result = Aggregation(self.data())
        self.assertEqual(dict(result.total), {(0, 2): 200, (2, 0): 300})
        self.assertEqual(result.selected_bytes, 500)

    def test_step_filter_and_fractional_windows_conserve(self):
        result = Aggregation(self.data(), step_min=2, window_ns=1000)
        out = self.root / "out"
        out.mkdir()
        result.export(out)
        with gzip.open(out / "bandwidth_windows.csv.gz", "rt") as stream:
            rows = list(csv.DictReader(stream))
        self.assertAlmostEqual(sum(float(r["estimated_bytes"]) for r in rows), 300)
        self.assertEqual(sum(int(r["completion_bytes"]) for r in rows), 300)
        self.assertEqual({r["clock_domain"] for r in rows}, {"host-b"})
        self.assertEqual(result.excluded_bytes, 200)

    def test_missing_finalize_rejected(self):
        self.events[2].pop()
        self.save()
        with self.assertRaisesRegex(ValueError, "finalization"):
            self.data()

    def test_missing_rank_rejected(self):
        with self.assertRaisesRegex(ValueError, "global ranks"):
            Traces([f"a={self.root / 'host-a'}"], 3)

    def test_missing_chunk_rejected(self):
        self.events[0] = [e for e in self.events[0] if e["type"] != "chunk"]
        self.save()
        with self.assertRaisesRegex(ValueError, "coverage"):
            Aggregation(self.data())

    def test_wrong_proxy_total_rejected(self):
        self.events[0][-2]["bytes"] += 1
        self.save()
        with self.assertRaisesRegex(ValueError, "conservation"):
            Aggregation(self.data())

    def test_wrong_peer_rejected(self):
        self.events[0][2]["peer_local"] = 9
        self.save()
        with self.assertRaisesRegex(ValueError, "peer"):
            Aggregation(self.data())

    def test_duplicate_api_rejected(self):
        self.events[0].insert(2, self.events[0][1].copy())
        self.save()
        with self.assertRaisesRegex(ValueError, "Duplicate API"):
            self.data()

    def test_profiler_anomaly_rejected(self):
        self.events[0][-1]["anomalies"] = 1
        self.save()
        with self.assertRaisesRegex(ValueError, "Profiler errors"):
            self.data()

    def test_unmarked_trace_and_empty_filter(self):
        for events in self.events.values():
            for event in events:
                if "step" in event:
                    event["step"] = -1
        self.save()
        self.assertEqual(Aggregation(self.data()).selected_bytes, 500)
        with self.assertRaisesRegex(ValueError, "No bytes selected"):
            Aggregation(self.data(), step_min=1)

    def test_mutating_input_rejected(self):
        data = self.data()
        self.events[0][0]["name"] = "changed"
        self.save()
        with self.assertRaisesRegex(ValueError, "Input changed"):
            data.check_unchanged()


if __name__ == "__main__":
    unittest.main()
