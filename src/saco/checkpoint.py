"""Local-filesystem checkpointing, replacing the notebook's Google Drive mount.
Same per-image incremental save behavior, so a job killed by walltime resumes
where it left off on resubmission."""
import json
from pathlib import Path


def load_checkpoint(path: Path) -> dict:
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {}


def save_checkpoint(results: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(".tmp")
    with open(tmp_path, "w") as f:
        json.dump(results, f, indent=2)
    tmp_path.replace(path)
