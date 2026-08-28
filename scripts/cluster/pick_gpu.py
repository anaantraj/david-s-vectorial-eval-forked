#!/usr/bin/env python3
"""Find free GPUs across the cthulhu cluster.

    scripts/cluster/pick_gpu.py [--n K] [--host H] [--format plain|shell|json]

A GPU counts as free when its used memory is below the threshold (default 500
MiB, which is above the few tens of MiB an idle card reports) and when no live
job launched by ``launch.sh`` has claimed it. The claim check reads the job
metadata files under ``runs/cluster_logs`` on each node and tests whether the
recorded process is still alive, so two launches issued back to back do not land
on the same card.

Output is one ``<host> <gpu-index>`` line per free GPU, best candidate first,
which the shell scripts consume with ``read``. ``--format shell`` emits
``HOST=... GPU=...`` for a single GPU and ``--format json`` emits the full
survey including busy cards.

Exit status is 0 when at least ``--n`` free GPUs were found and 1 otherwise, so
a launcher can fail fast rather than start on a contended card.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import shlex
import subprocess
import sys
from dataclasses import dataclass, field

DEFAULT_NODES = [f"cthulhu{i}" for i in range(1, 7)]
DOMAIN = "ist.berkeley.edu"
REMOTE_ROOT = "Projects/vectorial"
FREE_MIB = 500
SSH_TIMEOUT = 15

# Probe run on each node. It prints two record types:
#   GPU <index> <used-MiB> <total-MiB> <utilization-percent>
#   CLAIM <index> <job-name>            (one per live job pinned to this node)
# Claims are read from the metadata files launch.sh writes. The home directory
# is NFS-shared, so every node sees every metadata file; only the ones whose
# recorded host is this node are checked for liveness here.
PROBE = r"""
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu \
           --format=csv,noheader,nounits 2>/dev/null \
  | awk -F', *' '{print "GPU", $1, $2, $3, $4}'
me=$(hostname -s)
for meta in "$HOME/REMOTE_ROOT/runs/cluster_logs"/*.meta; do
  [ -e "$meta" ] || continue
  job=""; host=""; gpu=""; pid=""
  while IFS='=' read -r k v; do
    case "$k" in job) job=$v ;; host) host=$v ;; gpu) gpu=$v ;; pid) pid=$v ;; esac
  done < "$meta"
  case "$host" in "$me"|"$me".*) ;; *) continue ;; esac
  [ -n "$pid" ] || continue
  # The environment of the process is checked, not just its existence, so a
  # recycled PID cannot make a finished job look like a live claim.
  [ -r "/proc/$pid/environ" ] || continue
  if tr '\0' '\n' < "/proc/$pid/environ" | grep -qx "VECTORIAL_JOB=$job"; then
    echo "CLAIM $gpu $job"
  fi
done
""".replace("REMOTE_ROOT", REMOTE_ROOT)


@dataclass
class Gpu:
    host: str
    index: int
    used_mib: int
    total_mib: int
    util: int
    claimed_by: str | None = None
    threshold_mib: int = FREE_MIB

    @property
    def free(self) -> bool:
        """Free means idle memory and no live job of ours already on the card."""
        return self.claimed_by is None and self.used_mib < self.threshold_mib

    def as_dict(self) -> dict:
        return {
            "host": self.host,
            "gpu": self.index,
            "used_mib": self.used_mib,
            "total_mib": self.total_mib,
            "utilization": self.util,
            "claimed_by": self.claimed_by,
            "free": self.free,
        }


@dataclass
class NodeReport:
    host: str
    gpus: list[Gpu] = field(default_factory=list)
    error: str | None = None


def _int(text: str, default: int = -1) -> int:
    """Parse an nvidia-smi field. Some cards report '[N/A]' for utilization."""
    try:
        return int(text)
    except ValueError:
        return default


def fqdn(node: str) -> str:
    return node if "." in node else f"{node}.{DOMAIN}"


def probe(node: str, threshold_mib: int = FREE_MIB) -> NodeReport:
    """Survey one node. Any failure is recorded on the report, never raised, so
    one unreachable node does not hide the free GPUs on the other five."""
    host = fqdn(node)
    report = NodeReport(host=host)
    try:
        return _probe(host, threshold_mib, report)
    except Exception as exc:  # noqa: BLE001
        report.error = f"{type(exc).__name__}: {exc}"
        return report


def _probe(host: str, threshold_mib: int, report: NodeReport) -> NodeReport:
    cmd = [
        "ssh", "-o", "BatchMode=yes", "-o", f"ConnectTimeout={SSH_TIMEOUT}",
        host, "bash -s",
    ]
    try:
        proc = subprocess.run(
            cmd, input=PROBE, capture_output=True, text=True, timeout=SSH_TIMEOUT + 20
        )
    except subprocess.TimeoutExpired:
        report.error = "ssh timed out"
        return report
    if proc.returncode != 0:
        report.error = (proc.stderr or "ssh failed").strip().splitlines()[-1][:200]
        return report

    by_index: dict[int, Gpu] = {}
    claims: dict[int, str] = {}
    for line in proc.stdout.splitlines():
        parts = line.split()
        if not parts:
            continue
        if parts[0] == "GPU" and len(parts) >= 5:
            idx = _int(parts[1])
            if idx < 0:
                continue
            by_index[idx] = Gpu(
                host, idx, _int(parts[2], 10**9), _int(parts[3]), _int(parts[4]),
                threshold_mib=threshold_mib,
            )
        elif parts[0] == "CLAIM" and len(parts) >= 3:
            claim_idx = _int(parts[1])
            if claim_idx >= 0:
                claims[claim_idx] = parts[2]
    for idx, job in claims.items():
        if idx in by_index:
            by_index[idx].claimed_by = job
    report.gpus = [by_index[i] for i in sorted(by_index)]
    if not report.gpus and report.error is None:
        report.error = "no GPUs reported by nvidia-smi"
    return report


def survey(nodes: list[str], threshold_mib: int = FREE_MIB) -> list[NodeReport]:
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(nodes)) as pool:
        return list(pool.map(lambda n: probe(n, threshold_mib), nodes))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--n", type=int, default=1, help="number of free GPUs required")
    ap.add_argument("--host", help="restrict the search to one node")
    ap.add_argument(
        "--format", choices=["plain", "shell", "json"], default="plain",
        help="plain: one '<host> <gpu>' line per GPU; shell: HOST=/GPU= for the first; "
             "json: the whole survey",
    )
    ap.add_argument(
        "--threshold", type=int, default=FREE_MIB,
        help=f"used-memory ceiling in MiB for a GPU to count as free (default {FREE_MIB})",
    )
    ap.add_argument(
        "--spread", action="store_true",
        help="prefer one GPU per node before taking a second from any node",
    )
    args = ap.parse_args(argv)

    nodes = [args.host] if args.host else DEFAULT_NODES
    reports = survey(nodes, args.threshold)

    for r in reports:
        if r.error:
            print(f"warning: {r.host}: {r.error}", file=sys.stderr)

    free = [g for r in reports for g in r.gpus if g.free]
    # Emptiest card first; ties broken by host then index so the ordering is
    # stable across calls and two agents reading the same survey agree.
    free.sort(key=lambda g: (g.used_mib, g.host, g.index))

    if args.spread:
        buckets: dict[str, list[Gpu]] = {}
        for g in free:
            buckets.setdefault(g.host, []).append(g)
        interleaved: list[Gpu] = []
        while any(buckets.values()):
            for host in sorted(buckets):
                if buckets[host]:
                    interleaved.append(buckets[host].pop(0))
        free = interleaved

    if args.format == "json":
        payload = {
            "threshold_mib": args.threshold,
            "requested": args.n,
            "free": [g.as_dict() for g in free],
            "nodes": [
                {"host": r.host, "error": r.error, "gpus": [g.as_dict() for g in r.gpus]}
                for r in reports
            ],
        }
        print(json.dumps(payload, indent=2, sort_keys=True))
    elif args.format == "shell":
        if free:
            g = free[0]
            print(f"HOST={shlex.quote(g.host)}")
            print(f"GPU={g.index}")
    else:
        for g in free[: max(args.n, 0)] if args.n > 0 else free:
            print(f"{g.host} {g.index}")

    if len(free) < args.n:
        print(
            f"error: requested {args.n} free GPU(s), found {len(free)}",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
