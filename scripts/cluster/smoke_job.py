#!/usr/bin/env python3
"""A fake training job used to test the cluster harness end to end.

It does no useful learning. It counts steps, checkpoints through
``checkpointing.save_checkpoint``, and resumes through
``checkpointing.resume_or_start``, which is exactly the contract the real
training scripts follow. Killing it and re-issuing the same launch command must
continue from the last checkpointed step.

    scripts/cluster/launch.sh -j smoke -- scripts/cluster/smoke_job.py --steps 200

With torch present it also allocates a small tensor on the visible GPU and prints
the device, which confirms that ``CUDA_VISIBLE_DEVICES`` pinned the job to the
card that ``launch.sh`` chose.
"""

from __future__ import annotations

import argparse
import os
import socket
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from checkpointing import resume_or_start, save_checkpoint  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=60)
    ap.add_argument("--save-every", type=int, default=5)
    ap.add_argument("--sleep", type=float, default=0.5)
    ap.add_argument("--keep-last", type=int, default=3)
    ap.add_argument("--ckpt-dir", default=os.environ.get("VECTORIAL_CKPT_DIR", "runs/checkpoints/smoke"))
    ap.add_argument(
        "--oom", action="store_true",
        help="request an impossible allocation, to check that status.sh reports OOM",
    )
    args = ap.parse_args()

    print(f"host={socket.gethostname()} pid={os.getpid()} "
          f"cuda_visible={os.environ.get('CUDA_VISIBLE_DEVICES')} ckpt_dir={args.ckpt_dir}",
          flush=True)

    device = "cpu"
    try:
        import torch

        if torch.cuda.is_available():
            device = torch.cuda.get_device_name(0)
            _ = torch.zeros(256, 256, device="cuda")
            if args.oom:
                _ = torch.zeros(2**36, device="cuda")
    except ImportError:
        pass
    print(f"device={device}", flush=True)

    state = {"counter": 0.0}

    def restore(ckpt):
        state.update(ckpt.payload["state"])

    start_step, ckpt = resume_or_start(args.ckpt_dir, restore=restore)
    print(f"start_step={start_step} counter={state['counter']}", flush=True)

    for step in range(start_step, args.steps):
        state["counter"] += 1.0
        time.sleep(args.sleep)
        if (step + 1) % args.save_every == 0:
            path = save_checkpoint(
                args.ckpt_dir,
                step + 1,
                payload={"state": dict(state)},
                meta={"counter": state["counter"]},
                keep_last=args.keep_last,
            )
            print(f"step={step + 1} counter={state['counter']} saved={path}", flush=True)

    print("done", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
