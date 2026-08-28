# Cluster job harness

Everything the three adaptation methods (soft prompt, LoRA, activation steering)
need in order to train on the cthulhu GPUs and survive being killed. The GPUs are
shared with their owners, who kill jobs without warning, so the design assumption
is that every run will be interrupted and that relaunching the same command must
continue rather than restart.

`/home/davidchan` is NFS-shared across cthulhu1 through cthulhu6, so one sync
serves every node and logs, metadata, and checkpoints written on one node are
readable from all of them.

## The normal sequence

```bash
scripts/cluster/sync.sh                       # push the repo (any node; NFS)
.venv/bin/python scripts/cluster/pick_gpu.py --n 3 --spread   # see what is free
scripts/cluster/launch.sh -j lora -- scripts/train/lora.py --epochs 3
scripts/cluster/status.sh                     # poll this
scripts/cluster/launch.sh -j lora -- scripts/train/lora.py --epochs 3   # after a kill: identical command
```

The last line is the point of the harness. The relaunch is byte-identical to the
original launch; the training script resumes from its latest checkpoint because
it calls `resume_or_start`.

## Files

| file | purpose |
| --- | --- |
| `sync.sh` | rsync the repo to a node. Idempotent, and does not delete remote logs or checkpoints. |
| `pick_gpu.py` | free `(host, gpu)` pairs across all six nodes, excluding cards already claimed by a live job of ours. |
| `launch.sh` | start a job detached on a chosen GPU with a log, a PID file, and metadata. Refuses to start a job that is already running. |
| `status.sh` | one compact line per job: alive or not, latest checkpoint step, and whether the log ends in OOM, a kill, or a traceback. |
| `checkpointing.py` | atomic checkpoint writes, retention of the last K, and `resume_or_start`. Imported by the training scripts. |
| `smoke_job.py` | a fake training job used to test this harness end to end. |

## What a training script must do

Two things, and nothing else is required:

```python
from checkpointing import resume_or_start, save_checkpoint   # on PYTHONPATH already

ckpt_dir = os.environ.get("VECTORIAL_CKPT_DIR", "runs/checkpoints/<job>")
start_step, _ = resume_or_start(ckpt_dir, restore=restore_fn)

for step in range(start_step, total_steps):
    ...
    if (step + 1) % save_every == 0:
        save_checkpoint(ckpt_dir, step + 1, payload=..., meta={"loss": ...}, keep_last=3)
```

Checkpoint at least every few hundred steps and at every epoch boundary. Save
adapters, soft prompts, or steering vectors only: the shared disk is at 91
percent and a full copy of an 8B model does not belong on it.

`launch.sh` sets `VECTORIAL_CKPT_DIR` (absolute), `VECTORIAL_JOB`,
`CUDA_VISIBLE_DEVICES` (so the visible device is always index 0 inside the
process), `HF_HOME` pointing at the populated cache, and `PYTHONPATH` covering
`src/` and this directory. Hugging Face downloads are disabled unless
`--allow-download` is passed.

## Guarantees and their limits

- A checkpoint is either complete or invisible. Writes go to `step_NNNNNNNN.tmp`
  and are moved into place with a single rename, and the `latest.json` pointer is
  replaced atomically afterwards. A kill mid-write costs at most the interval
  since the last checkpoint.
- A checkpoint that fails to load is skipped and the previous one is used, so
  corruption costs one interval rather than the run.
- Liveness is decided by reading `/proc/<pid>/environ` and matching
  `VECTORIAL_JOB`, not by the existence of the PID, so a recycled PID cannot be
  reported as a running job.
- `pick_gpu.py` excludes cards held by a live job of ours, which keeps two
  launches issued back to back off the same card. It cannot see a card that
  another user is about to claim; that race is inherent to a shared cluster.
- `status.sh` reports `STOPPED` rather than `KILLED` when a job is killed by a
  signal, because the shell message that names the signal goes to the launching
  shell, which is already gone. `STOPPED` with a recent checkpoint and no
  traceback is the normal appearance of an owner kill.

## Testing it

`tests/test_checkpointing.py` covers the resume path, retention, partial writes,
and a real SIGKILL followed by a relaunch. Against the cluster, `smoke_job.py`
exercises the whole harness:

```bash
scripts/cluster/launch.sh -j smoke -- scripts/cluster/smoke_job.py --steps 400 --sleep 1
scripts/cluster/status.sh --job smoke
ssh cthulhu1.ist.berkeley.edu 'pkill -f smoke_job.py'
scripts/cluster/launch.sh -j smoke -- scripts/cluster/smoke_job.py --steps 400 --sleep 1
# the log now shows a second start_step= line at the step where the kill landed
```

`--oom` makes the smoke job request an impossible allocation, which is how the
`OOM` state in `status.sh` was verified.
