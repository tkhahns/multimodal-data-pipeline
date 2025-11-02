"""Clone and prepare upstream audio model repositories.

All audio feature extractors in this package must rely on the original
implementations referenced in ``requirements.csv``.  To keep the main project
lightweight we clone those repositories on demand into the local workspace and
expose a tiny API so wrappers can resolve their paths.

The helper functions here intentionally avoid third-party Git bindings.  Using
``subprocess`` keeps the runtime dependencies minimal and works both on Windows
(through WSL) and POSIX environments.
"""
from __future__ import annotations

import logging
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ExternalRepo:
    """Descriptor for an external repository."""

    name: str
    url: str
    commit: Optional[str] = None
    subdirectory: Optional[str] = None
    env_var: Optional[str] = None

    def destination(self, base_dir: Path) -> Path:
        return base_dir / self.name

    def resolved_path(self, base_dir: Path) -> Path:
        repo_path = self.destination(base_dir)
        if self.subdirectory:
            return repo_path / self.subdirectory
        return repo_path


# Root where we keep external checkouts (project_root/external/audio).
BASE_DIR = Path(__file__).resolve().parents[3] / "external" / "audio"
BASE_DIR = BASE_DIR.resolve()

# Repositories required by the audio stack.  Commits are pinned for
# reproducibility so future upstream changes do not silently alter behaviour.
AUDIO_REPOS: Dict[str, ExternalRepo] = {
    "emotion_recognition": ExternalRepo(
        name="emotion-recognition-using-speech",
        url="https://github.com/x4nth055/emotion-recognition-using-speech.git",
        commit=None,
        subdirectory=None,
        env_var="EMOTION_RECOGNITION_REPO",
    ),
    "fairseq": ExternalRepo(
        name="fairseq",
        url="https://github.com/facebookresearch/fairseq.git",
        commit=None,
        subdirectory=None,
        env_var="FAIRSEQ_REPO",
    ),
    "whisperx": ExternalRepo(
        name="whisperx",
        url="https://github.com/m-bain/whisperX.git",
        commit=None,
        subdirectory=None,
        env_var="WHISPERX_REPO",
    ),
    "speechbrain": ExternalRepo(
        name="speechbrain",
        url="https://github.com/speechbrain/speechbrain.git",
        commit=None,
        subdirectory=None,
        env_var="SPEECHBRAIN_REPO",
    ),
}


def _run_git(args: list[str], cwd: Optional[Path] = None) -> None:
    command = ["git"] + args
    logger.debug("Running git command: %s", " ".join(command))
    subprocess.run(command, check=True, cwd=str(cwd) if cwd else None)


def ensure_base_dir() -> Path:
    BASE_DIR.mkdir(parents=True, exist_ok=True)
    return BASE_DIR


def ensure_repo(repo_key: str, *, base_dir: Optional[Path] = None, force_update: bool = False) -> Path:
    """Clone the repository identified by ``repo_key`` if necessary.

    Args:
        repo_key: Key in :data:`AUDIO_REPOS` identifying the repository.
        base_dir: Optional override for the clone root directory.
        force_update: When ``True`` the repository is fetched and reset even if
            it already exists locally.

    Returns:
        Path pointing at the repository checkout (or subdirectory when defined).
    """

    repo = AUDIO_REPOS[repo_key]

    # Allow callers to override the location through environment variables.
    if repo.env_var:
        env_path = os.getenv(repo.env_var)
        if env_path:
            candidate = Path(env_path)
            if candidate.exists():
                logger.debug("Using %s from %s", repo.name, repo.env_var)
                return repo.resolved_path(candidate)
            logger.warning("%s points to missing path %s", repo.env_var, candidate)

    root_dir = base_dir or ensure_base_dir()
    destination = repo.destination(root_dir)

    if destination.exists():
        if force_update:
            logger.info("Updating %s", repo.name)
            _run_git(["fetch", "--all", "--tags"], cwd=destination)
            if repo.commit:
                _run_git(["reset", "--hard", repo.commit], cwd=destination)
            else:
                _run_git(["pull", "--ff-only"], cwd=destination)
    else:
        logger.info("Cloning %s into %s", repo.url, destination)
        _run_git(["clone", repo.url, str(destination)])
        if repo.commit:
            _run_git(["checkout", repo.commit], cwd=destination)

    resolved = repo.resolved_path(root_dir)
    if not resolved.exists():
        raise FileNotFoundError(f"Expected {resolved} to exist after cloning {repo.name}")

    return resolved


def remove_repo(repo_key: str, *, base_dir: Optional[Path] = None) -> None:
    """Delete a managed repository checkout."""
    repo = AUDIO_REPOS[repo_key]
    root_dir = base_dir or ensure_base_dir()
    destination = repo.destination(root_dir)
    if destination.exists():
        logger.info("Removing repository checkout %s", destination)
        shutil.rmtree(destination, ignore_errors=True)


__all__ = ["ensure_repo", "remove_repo", "AUDIO_REPOS", "ExternalRepo"]
