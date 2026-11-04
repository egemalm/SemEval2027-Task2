"""Small, file-based experiment records; no tracking service required."""

from __future__ import annotations

import hashlib
import json
import platform
import shutil
import subprocess
import sys
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from datetime import datetime, timezone
from importlib.metadata import distributions
from pathlib import Path
from uuid import uuid4

import torch


def write_json(path: Path, payload: dict) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8"
    )


def fingerprint(path: Path) -> dict:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return {"path": str(path.resolve()), "sha256": digest.hexdigest()}


def dataset_provenance(path: Path) -> dict:
    record = fingerprint(path)
    # Record the actual bytes and the declared upstream provenance independently.
    for parent in (path.parent, path.parent.parent):
        manifest = parent / "provenance.json"
        if manifest.is_file():
            record["upstream_manifest"] = json.loads(manifest.read_text(encoding="utf-8"))
            break
    return record


def create_run(output_root: Path) -> Path:
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid4().hex[:8]
    run_dir = output_root / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def capture_environment(run_dir: Path, device: torch.device) -> dict:
    root = Path(__file__).resolve().parents[2]

    def git(*args):
        result = subprocess.run(["git", "-C", str(root), *args], capture_output=True, text=True)
        return result.stdout.strip() if result.returncode == 0 else None

    status = git("status", "--porcelain")
    snapshot = run_dir / "source_snapshot"
    snapshot.mkdir()
    for name in (
        "src/dico_nli",
        "scripts",
        "pyproject.toml",
        "Makefile",
        "requirements-lock.txt",
        "AGENTS.md",
    ):
        source = root / name
        target = snapshot / name
        if source.is_dir():
            shutil.copytree(source, target, ignore=shutil.ignore_patterns("__pycache__"))
        elif source.is_file():
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
    versions = {d.metadata["Name"]: d.version for d in distributions() if d.metadata["Name"]}
    write_json(run_dir / "software_versions.json", versions)
    hardware = {
        "device": str(device),
        "platform": platform.platform(),
        "processor": platform.processor(),
        "python": sys.version,
        "cuda_runtime": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "mps_available": torch.backends.mps.is_available(),
    }
    if device.type == "cuda":
        props = torch.cuda.get_device_properties(device)
        hardware.update({"gpu_name": props.name, "gpu_memory_bytes": props.total_memory})
    return {
        "git_commit": git("rev-parse", "HEAD"),
        "git_dirty": bool(status),
        "git_status": status,
        "hardware": hardware,
        "source_snapshot": "source_snapshot",
        "source_files_sha256": {
            str(p.relative_to(snapshot)): fingerprint(p)["sha256"]
            for p in sorted(snapshot.rglob("*"))
            if p.is_file()
        },
        "software_versions": "software_versions.json",
    }


class _Tee:
    """Non-owning console proxy; training_log owns the lifetime of the log file.

    Third-party logging handlers may retain this proxy after redirection ends,
    then flush or close it at interpreter shutdown. Those calls must neither
    access a closed log file nor close the original console stream.
    """

    def __init__(self, original, log):
        self.original, self.log = original, log

    def write(self, value):
        count = self.original.write(value)
        if not self.log.closed:
            self.log.write(value)
            self.log.flush()
        return count

    def flush(self):
        if not getattr(self.original, "closed", False):
            self.original.flush()
        if not self.log.closed:
            self.log.flush()

    def close(self):
        # The console and log file belong to their respective outer contexts.
        self.flush()

    def __getattr__(self, name):
        return getattr(self.original, name)


@contextmanager
def training_log(run_dir: Path):
    with (run_dir / "training.log").open("x", encoding="utf-8") as log:
        with redirect_stdout(_Tee(sys.stdout, log)), redirect_stderr(_Tee(sys.stderr, log)):
            yield
