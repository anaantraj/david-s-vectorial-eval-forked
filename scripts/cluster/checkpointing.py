"""Crash-safe checkpointing for cluster training jobs.

The GPUs are shared and a job may be killed at any moment, including in the
middle of a write. Two properties follow from that and this module exists to
provide them.

**A partially written checkpoint is never visible.** Every checkpoint is built
inside a ``step_XXXXXXXX.tmp`` directory and moved into place with a single
``os.replace`` once it is complete. A process killed mid-write leaves a ``.tmp``
directory that no reader will ever select, and the pointer file ``latest.json``
is itself replaced atomically after the directory is in place.

**Relaunching the same command resumes.** ``resume_or_start`` is the ordinary
path, not an option: it returns the step to continue from, which is zero on the
first launch and the last completed step afterwards. A checkpoint that fails to
load, for any reason, is skipped and the next most recent one is tried, so a
corrupt directory costs one checkpoint interval rather than the whole run.

Only the last ``keep_last`` checkpoints are retained because the shared disk sits
at 91 percent. Training scripts should save adapters, soft prompts, or steering
vectors, never a full copy of the base model.

Usage in a training script::

    from checkpointing import resume_or_start, save_checkpoint

    def restore(ckpt):
        model.load_state_dict(ckpt.payload["model"])
        optimizer.load_state_dict(ckpt.payload["optimizer"])

    start_step, ckpt = resume_or_start(ckpt_dir, restore=restore)
    for step in range(start_step, total_steps):
        ...
        if step % save_every == 0:
            save_checkpoint(
                ckpt_dir, step + 1,
                payload={"model": model.state_dict(), "optimizer": optimizer.state_dict()},
                meta={"loss": float(loss)},
            )

``payload`` is written with ``torch.save`` when torch is importable and with
``pickle`` otherwise. For PEFT adapters, pass ``save_fn=lambda d:
model.save_pretrained(d)`` instead and reload with the matching ``load_fn``.
"""

from __future__ import annotations

import json
import logging
import os
import pickle
import re
import shutil
import tempfile
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

__all__ = [
    "Checkpoint",
    "checkpoint_dirs",
    "latest_checkpoint_step",
    "load_latest_checkpoint",
    "prune_checkpoints",
    "resume_or_start",
    "save_checkpoint",
]

LOG = logging.getLogger(__name__)

STEP_DIR = "step_{step:08d}"
STEP_RE = re.compile(r"^step_(\d{8})$")
META_NAME = "meta.json"
PAYLOAD_NAME = "payload.pt"
POINTER_NAME = "latest.json"
DEFAULT_KEEP = 3


@dataclass
class Checkpoint:
    """One loaded checkpoint."""

    step: int
    path: Path
    meta: dict[str, Any] = field(default_factory=dict)
    payload: Any | None = None


# --------------------------------------------------------------------------- #
# internals
# --------------------------------------------------------------------------- #


def _torch():
    try:
        import torch  # noqa: PLC0415
    except ImportError:
        return None
    return torch


def _write_payload(path: Path, payload: Any) -> None:
    torch = _torch()
    if torch is not None:
        torch.save(payload, path)
    else:
        with open(path, "wb") as fh:
            pickle.dump(payload, fh, protocol=pickle.HIGHEST_PROTOCOL)
            fh.flush()
            os.fsync(fh.fileno())


def _read_payload(path: Path, map_location: str = "cpu") -> Any:
    torch = _torch()
    if torch is not None:
        try:
            return torch.load(path, map_location=map_location, weights_only=False)
        except Exception:
            # A file written by the pickle fallback still loads here.
            with open(path, "rb") as fh:
                return pickle.load(fh)
    with open(path, "rb") as fh:
        return pickle.load(fh)


def _fsync_dir(path: Path) -> None:
    """Flush a directory entry so a rename survives a machine-level failure."""
    try:
        fd = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    except OSError:
        pass
    finally:
        os.close(fd)


def _write_json_atomic(path: Path, obj: dict[str, Any]) -> None:
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as fh:
            json.dump(obj, fh, indent=2, sort_keys=True)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise
    _fsync_dir(path.parent)


def checkpoint_dirs(ckpt_dir: str | os.PathLike) -> list[Path]:
    """Completed checkpoint directories, oldest first.

    Incomplete ``.tmp`` directories and anything not matching ``step_XXXXXXXX``
    are ignored, which is what makes an interrupted write invisible.
    """
    root = Path(ckpt_dir)
    if not root.is_dir():
        return []
    found = [p for p in root.iterdir() if p.is_dir() and STEP_RE.match(p.name)]
    return sorted(found, key=lambda p: int(STEP_RE.match(p.name).group(1)))


def _step_of(path: Path) -> int:
    m = STEP_RE.match(path.name)
    if m is None:
        raise ValueError(f"not a checkpoint directory: {path}")
    return int(m.group(1))


# --------------------------------------------------------------------------- #
# public API
# --------------------------------------------------------------------------- #


def save_checkpoint(
    ckpt_dir: str | os.PathLike,
    step: int,
    payload: Any | None = None,
    meta: dict[str, Any] | None = None,
    *,
    save_fn: Callable[[Path], None] | None = None,
    keep_last: int = DEFAULT_KEEP,
) -> Path:
    """Write checkpoint ``step`` atomically and return its directory.

    ``payload`` is serialised with torch (or pickle). ``save_fn`` is called with
    the staging directory and is the hook for writers that produce their own
    files, such as ``PeftModel.save_pretrained``. Both may be given.

    The directory appears under its final name only once every byte is on disk,
    so a kill at any point during this call leaves the previous checkpoint as the
    newest visible one.
    """
    if step < 0:
        raise ValueError("step must be non-negative")
    root = Path(ckpt_dir)
    root.mkdir(parents=True, exist_ok=True)

    final = root / STEP_DIR.format(step=step)
    staging = root / (final.name + ".tmp")
    if staging.exists():
        shutil.rmtree(staging, ignore_errors=True)
    staging.mkdir()

    started = time.time()
    if save_fn is not None:
        save_fn(staging)
    if payload is not None:
        _write_payload(staging / PAYLOAD_NAME, payload)

    record = {
        "step": step,
        "saved_at": time.time(),
        "save_seconds": round(time.time() - started, 3),
        "has_payload": payload is not None,
        "user": dict(meta or {}),
    }
    with open(staging / META_NAME, "w") as fh:
        json.dump(record, fh, indent=2, sort_keys=True, default=str)
        fh.flush()
        os.fsync(fh.fileno())

    # Flush the staged files before the rename, otherwise the directory entry can
    # reach disk ahead of its contents.
    for child in staging.iterdir():
        try:
            fd = os.open(child, os.O_RDONLY)
        except OSError:
            continue
        try:
            os.fsync(fd)
        except OSError:
            pass
        finally:
            os.close(fd)
    _fsync_dir(staging)

    if final.exists():
        # Re-saving the same step (a resumed job that repeats a step) replaces it.
        shutil.rmtree(final, ignore_errors=True)
    os.replace(staging, final)
    _fsync_dir(root)

    _write_json_atomic(
        root / POINTER_NAME,
        {"step": step, "dir": final.name, "saved_at": record["saved_at"]},
    )
    prune_checkpoints(root, keep_last=keep_last)
    LOG.info("checkpoint written: %s", final)
    return final


def prune_checkpoints(ckpt_dir: str | os.PathLike, keep_last: int = DEFAULT_KEEP) -> list[Path]:
    """Delete all but the newest ``keep_last`` checkpoints; return what was deleted.

    Stale ``.tmp`` directories left by a killed process are removed as well. The
    newest checkpoint is never deleted, whatever ``keep_last`` is set to.
    """
    root = Path(ckpt_dir)
    removed: list[Path] = []
    if not root.is_dir():
        return removed

    for p in root.iterdir():
        if p.is_dir() and p.name.endswith(".tmp") and STEP_RE.match(p.name[: -len(".tmp")]):
            shutil.rmtree(p, ignore_errors=True)
            removed.append(p)

    keep = max(int(keep_last), 1)
    dirs = checkpoint_dirs(root)
    for p in dirs[:-keep]:
        shutil.rmtree(p, ignore_errors=True)
        removed.append(p)
    return removed


def latest_checkpoint_step(ckpt_dir: str | os.PathLike) -> int | None:
    """Step of the newest complete checkpoint, or ``None`` if there is none."""
    dirs = checkpoint_dirs(ckpt_dir)
    return _step_of(dirs[-1]) if dirs else None


def load_latest_checkpoint(
    ckpt_dir: str | os.PathLike,
    *,
    load_fn: Callable[[Path], None] | None = None,
    map_location: str = "cpu",
    load_payload: bool = True,
) -> Checkpoint | None:
    """Load the newest checkpoint that reads back cleanly, or ``None``.

    Directories are tried newest first. One that raises while loading is reported
    and skipped, so a checkpoint truncated by a filesystem-level failure costs a
    single interval instead of the run.
    """
    for path in reversed(checkpoint_dirs(ckpt_dir)):
        try:
            with open(path / META_NAME) as fh:
                record = json.load(fh)
            payload = None
            payload_path = path / PAYLOAD_NAME
            if load_payload and payload_path.exists():
                payload = _read_payload(payload_path, map_location=map_location)
            if load_fn is not None:
                load_fn(path)
        except Exception as exc:  # noqa: BLE001 - any failure means "try the previous one"
            LOG.warning("skipping unreadable checkpoint %s: %s", path, exc)
            continue
        return Checkpoint(
            step=int(record.get("step", _step_of(path))),
            path=path,
            meta=dict(record.get("user", {})) | {"_record": record},
            payload=payload,
        )
    return None


def resume_or_start(
    ckpt_dir: str | os.PathLike,
    *,
    restore: Callable[[Checkpoint], None] | None = None,
    load_fn: Callable[[Path], None] | None = None,
    map_location: str = "cpu",
) -> tuple[int, Checkpoint | None]:
    """Return ``(start_step, checkpoint)``, resuming when a checkpoint exists.

    This is the default path for every training script. On a fresh run it returns
    ``(0, None)``. After a kill it returns the step after the last completed
    checkpoint, having first called ``restore`` (and ``load_fn``, for writers
    that manage their own files) so the caller's model and optimiser are in the
    matching state.
    """
    ckpt = load_latest_checkpoint(ckpt_dir, load_fn=load_fn, map_location=map_location)
    if ckpt is None:
        LOG.info("no checkpoint in %s; starting from step 0", ckpt_dir)
        return 0, None
    if restore is not None:
        restore(ckpt)
    LOG.info("resuming from %s at step %d", ckpt.path, ckpt.step)
    return ckpt.step, ckpt
