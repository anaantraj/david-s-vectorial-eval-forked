"""Tests for the cluster checkpointing module.

The cluster GPUs are shared and jobs are killed without warning, so the two
properties tested here are the ones the training runs depend on: a checkpoint is
either complete or invisible, and relaunching a killed job resumes from the last
complete step rather than from zero.
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

CLUSTER_DIR = Path(__file__).resolve().parents[1] / "scripts" / "cluster"
sys.path.insert(0, str(CLUSTER_DIR))

from checkpointing import (  # noqa: E402
    checkpoint_dirs,
    latest_checkpoint_step,
    load_latest_checkpoint,
    resume_or_start,
    save_checkpoint,
)


def test_save_and_load_roundtrip(tmp_path):
    save_checkpoint(tmp_path, 7, payload={"w": [1, 2, 3]}, meta={"loss": 0.5})
    ckpt = load_latest_checkpoint(tmp_path)
    assert ckpt is not None
    assert ckpt.step == 7
    assert ckpt.payload == {"w": [1, 2, 3]}
    assert ckpt.meta["loss"] == 0.5
    assert latest_checkpoint_step(tmp_path) == 7


def test_empty_directory_starts_at_zero(tmp_path):
    assert load_latest_checkpoint(tmp_path) is None
    assert latest_checkpoint_step(tmp_path) is None
    assert resume_or_start(tmp_path) == (0, None)


def test_retention_keeps_only_last_k(tmp_path):
    for step in range(1, 8):
        save_checkpoint(tmp_path, step, payload={"step": step}, keep_last=3)
    kept = [p.name for p in checkpoint_dirs(tmp_path)]
    assert kept == ["step_00000005", "step_00000006", "step_00000007"]


def test_partial_write_is_invisible(tmp_path):
    """A staging directory left behind by a kill is never selected or resumed."""
    save_checkpoint(tmp_path, 4, payload={"v": 4})
    staging = tmp_path / "step_00000009.tmp"
    staging.mkdir()
    (staging / "payload.pt").write_bytes(b"truncated garbage")

    assert [p.name for p in checkpoint_dirs(tmp_path)] == ["step_00000004"]
    assert latest_checkpoint_step(tmp_path) == 4
    ckpt = load_latest_checkpoint(tmp_path)
    assert ckpt.step == 4 and ckpt.payload == {"v": 4}


def test_corrupt_checkpoint_falls_back_to_previous(tmp_path):
    save_checkpoint(tmp_path, 1, payload={"v": 1}, keep_last=5)
    save_checkpoint(tmp_path, 2, payload={"v": 2}, keep_last=5)
    (tmp_path / "step_00000002" / "payload.pt").write_bytes(b"not a pickle")

    ckpt = load_latest_checkpoint(tmp_path)
    assert ckpt is not None
    assert ckpt.step == 1 and ckpt.payload == {"v": 1}


def test_stale_staging_dirs_are_cleaned(tmp_path):
    (tmp_path / "step_00000003.tmp").mkdir(parents=True)
    save_checkpoint(tmp_path, 1, payload={"v": 1})
    assert not (tmp_path / "step_00000003.tmp").exists()


def test_resume_restores_state(tmp_path):
    save_checkpoint(tmp_path, 12, payload={"counter": 12.0})
    seen = {}
    start, ckpt = resume_or_start(tmp_path, restore=lambda c: seen.update(c.payload))
    assert start == 12
    assert ckpt.step == 12
    assert seen == {"counter": 12.0}


@pytest.mark.skipif(os.name != "posix", reason="uses SIGKILL")
def test_killed_job_resumes_from_last_checkpoint(tmp_path):
    """The path the cluster actually exercises: SIGKILL, then relaunch."""
    cmd = [
        sys.executable, str(CLUSTER_DIR / "smoke_job.py"),
        "--steps", "40", "--save-every", "2", "--sleep", "0.05",
        "--ckpt-dir", str(tmp_path),
    ]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    deadline = time.time() + 30
    while time.time() < deadline:
        if (latest_checkpoint_step(tmp_path) or 0) >= 4:
            break
        time.sleep(0.05)
    proc.send_signal(signal.SIGKILL)
    proc.wait(timeout=10)

    killed_at = latest_checkpoint_step(tmp_path)
    assert killed_at is not None and killed_at >= 4

    out = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    assert out.returncode == 0, out.stdout + out.stderr
    assert f"start_step={killed_at}" in out.stdout
    # The counter is carried across the kill rather than restarted, which is what
    # distinguishes a resume from a fresh run that happens to finish.
    ckpt = load_latest_checkpoint(tmp_path)
    assert ckpt.step == 40
    assert ckpt.payload["state"]["counter"] == 40.0
