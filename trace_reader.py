"""Read plain plugin traces without Megatron manifests or fixed rank placement."""
import hashlib
import json
from collections import defaultdict
from pathlib import Path


def require(condition, message):
    """Reject incomplete evidence instead of silently dropping events."""
    if not condition:
        raise ValueError(message)


def sha256(path):
    """Hash large traces without loading the entire file into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class Traces:
    """One unique global rank per file, all members of every communicator required."""

    def __init__(self, inputs, world_size):
        require(world_size > 0, "world-size must be positive")
        self.world_size = world_size
        self.files = {}
        self.hosts = {}
        self.hashes = {}
        domains = set()
        for item in inputs:
            host, separator, directory = item.partition("=")
            require(separator and host and directory, "Use --input HOST=DIRECTORY")
            require(host not in domains, "Give each host exactly one input directory")
            domains.add(host)
            root = Path(directory).resolve()
            require(root.is_dir(), f"Missing input directory: {root}")
            paths = sorted(root.rglob("nccl-events.jsonl"))
            require(paths, f"No nccl-events.jsonl under {root}")
            for path in paths:
                require(path.parent.name.startswith("rank-"), f"Expected rank-N directory: {path}")
                rank = int(path.parent.name[5:])
                require(rank not in self.files, f"Duplicate global rank {rank}; mixed runs or duplicate inputs")
                self.files[rank] = path
                self.hosts[rank] = host
                self.hashes[str(path)] = sha256(path)
        require(set(self.files) == set(range(world_size)), "Missing/unexpected global ranks for world-size")
        self.comms = defaultdict(dict)
        self.sizes = {}
        self.apis = {}
        self.members = set()
        self.finals = set()
        self._read_metadata()

    def events(self):
        """Stream records and report source locations for malformed/truncated JSONL."""
        for rank, path in sorted(self.files.items()):
            with path.open() as stream:
                for number, line in enumerate(stream, 1):
                    try:
                        event = json.loads(line)
                        require(event["rank"] == rank, "File/global rank mismatch")
                        require(event["type"] in {"comm_member", "comm_finalize", "api", "chunk", "proxy"},
                                "Unknown event schema")
                    except (ValueError, KeyError, TypeError) as error:
                        raise ValueError(f"{path}:{number}: {error}") from error
                    yield event

    def _read_metadata(self):
        for event in self.events():
            rank, comm, kind = event["rank"], event["comm"], event["type"]
            key = (rank, comm)
            if kind == "comm_member":
                require(key not in self.members, "Duplicate communicator member; output directory reused?")
                self.members.add(key)
                size, local = event["size"], event["local_rank"]
                require(size > 0 and 0 <= local < size, "Invalid communicator local rank/size")
                require(self.sizes.setdefault(comm, size) == size, "Conflicting communicator sizes")
                require(local not in self.comms[comm], "Duplicate communicator local rank")
                self.comms[comm][local] = rank
            elif kind == "comm_finalize":
                require(key not in self.finals, "Duplicate communicator finalization")
                self.finals.add(key)
                require(all(event[k] == 0 for k in
                            ("live_events", "write_failures", "allocation_failures", "anomalies")),
                        f"Profiler errors in rank {rank}, communicator {comm}")
            elif kind == "api":
                identity = (rank, event["id"])
                require(identity not in self.apis, "Duplicate API ID; mixed runs?")
                require(0 < event["start_ns"] <= event["end_ns"], "Invalid API interval")
                self.apis[identity] = event
        require(self.members and self.members == self.finals, "Missing communicator initialization/finalization")
        require({rank for rank, _ in self.members} == set(self.files), "A rank has no communicator records")
        for comm, mapping in self.comms.items():
            require(set(mapping) == set(range(self.sizes[comm])), f"Incomplete communicator map: {comm}")
        for event in self.apis.values():
            require((event["rank"], event["comm"]) in self.members, "API has unknown communicator")

    def peer(self, event):
        """Map a subgroup-local peer through observed communicator membership."""
        mapping = self.comms.get(event["comm"], {})
        require(event["peer_local"] in mapping, "Unknown communicator-local peer")
        return mapping[event["peer_local"]]

    def check_unchanged(self):
        """Catch files still being written while aggregation is running."""
        for path, digest in self.hashes.items():
            require(sha256(Path(path)) == digest, f"Input changed during aggregation: {path}")
