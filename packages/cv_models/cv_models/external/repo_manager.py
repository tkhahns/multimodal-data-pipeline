"""Clone and manage external computer-vision repositories.

CV analyzers must rely on the original upstream projects rather than heuristic
re-implementations.  This module mirrors the audio stack's helper by cloning
repositories into ``external/vision`` (or a caller-specified path) and exposing
helpers that analyzers can use to resolve stable import paths.

The implementation intentionally sticks to the standard library to avoid
introducing additional runtime dependencies.
"""
from __future__ import annotations

import logging
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ExternalRepo:
    """Descriptor for a computer-vision repository."""

    name: str
    url: str
    commit: Optional[str] = None
    subdirectory: Optional[str] = None
    env_var: Optional[str] = None
    init_submodules: bool = False

    def destination(self, base_dir: Path) -> Path:
        return base_dir / self.name

    def resolved_path(self, base_dir: Path) -> Path:
        repo_path = self.destination(base_dir)
        return repo_path / self.subdirectory if self.subdirectory else repo_path


# Default root for cloned repositories (project_root/external/vision).
DEFAULT_BASE_DIR = Path(__file__).resolve().parents[3] / "external" / "vision"
DEFAULT_BASE_DIR = DEFAULT_BASE_DIR.resolve()


CV_REPOS: Dict[str, ExternalRepo] = {
    "openpose": ExternalRepo(
        name="openpose",
        url="https://github.com/CMU-Perceptual-Computing-Lab/openpose.git",
        init_submodules=True,
        env_var="OPENPOSE_REPO",
    ),
    "pyfeat": ExternalRepo(
        name="py-feat",
        url="https://github.com/cosanlab/py-feat.git",
        env_var="PYFEAT_REPO",
    ),
    "pare": ExternalRepo(
        name="PARE",
        url="https://github.com/mkocabas/PARE.git",
    ),
    "vitpose": ExternalRepo(
        name="ViTPose",
        url="https://github.com/ViTAE-Transformer/ViTPose.git",
    ),
    "psa": ExternalRepo(
        name="PSA",
        url="https://github.com/DeLightCMU/PSA.git",
    ),
    "rsn": ExternalRepo(
        name="RSN",
        url="https://github.com/caiyuanhao1998/RSN.git",
    ),
    "me_graphau": ExternalRepo(
        name="ME-GraphAU",
        url="https://github.com/CVI-SZU/ME-GraphAU.git",
    ),
    "dan": ExternalRepo(
        name="DAN",
        url="https://github.com/yaoing/DAN.git",
    ),
    "fact": ExternalRepo(
        name="CVPR2024-FACT",
        url="https://github.com/ZijiaLewisLu/CVPR2024-FACT.git",
    ),
    "emotiefflib": ExternalRepo(
        name="EmotiEffLib",
        url="https://github.com/sb-ai-lab/EmotiEffLib.git",
    ),
    "mediapipe": ExternalRepo(
        name="mediapipe",
        url="https://github.com/google/mediapipe.git",
    ),
    "hrnet_pose": ExternalRepo(
        name="deep-high-resolution-net.pytorch",
        url="https://github.com/leoxiaobin/deep-high-resolution-net.pytorch.git",
    ),
    "simple_baselines_pose": ExternalRepo(
        name="human-pose-estimation.pytorch",
        url="https://github.com/Microsoft/human-pose-estimation.pytorch.git",
    ),
    "ganimation": ExternalRepo(
        name="GANimation",
        url="https://github.com/albertpumarola/GANimation.git",
    ),
    "arbex": ExternalRepo(
        name="ARBEx",
        url="https://github.com/takihasan/ARBEx.git",
    ),
    "instadm": ExternalRepo(
        name="Insta-DM",
        url="https://github.com/SeokjuLee/Insta-DM.git",
    ),
    "optical_flow": ExternalRepo(
        name="optical-flow",
        url="https://github.com/chuanenlin/optical-flow.git",
    ),
    "crowdflow": ExternalRepo(
        name="CrowdFlow",
        url="https://github.com/tsenst/CrowdFlow.git",
    ),
    "videofinder": ExternalRepo(
        name="VideoFinder-Llama3.2-vision-Ollama",
        url="https://github.com/win4r/VideoFinder-Llama3.2-vision-Ollama.git",
    ),
    "smoothnet": ExternalRepo(
        name="SmoothNet",
        url="https://github.com/cure-lab/SmoothNet.git",
    ),
    "rife": ExternalRepo(
        name="ECCV2022-RIFE",
        url="https://github.com/hzwer/ECCV2022-RIFE.git",
    ),
    "lanegcn": ExternalRepo(
        name="LaneGCN",
        url="https://github.com/uber-research/LaneGCN.git",
    ),
    "av_hubert": ExternalRepo(
        name="av_hubert",
        url="https://github.com/facebookresearch/av_hubert.git",
        env_var="AVHUBERT_REPO",
    ),
}


def _run_git(args: Iterable[str], *, cwd: Optional[Path] = None) -> None:
    command = ["git", *args]
    logger.debug("Running git command: %s", " ".join(command))
    subprocess.run(command, check=True, cwd=str(cwd) if cwd else None)


def ensure_base_dir(base_dir: Optional[Path] = None) -> Path:
    root = base_dir or DEFAULT_BASE_DIR
    root.mkdir(parents=True, exist_ok=True)
    return root


def ensure_repo(
    repo_key: str,
    *,
    base_dir: Optional[Path] = None,
    force_update: bool = False,
) -> Path:
    """Clone the repository identified by ``repo_key`` if required.

    Args:
        repo_key: Key in :data:`CV_REPOS`.
        base_dir: Optional override for the clone root (defaults to
            ``external/vision``).
        force_update: When ``True`` the repository is fetched and reset even if
            it already exists locally.

    Returns:
        Path pointing to the repository checkout or the declared subdirectory.
    """

    if repo_key not in CV_REPOS:
        raise KeyError(f"Unknown CV repository key: {repo_key}")

    repo = CV_REPOS[repo_key]

    if repo.env_var:
        env_path = os.getenv(repo.env_var)
        if env_path:
            candidate = Path(env_path).expanduser()
            if candidate.exists():
                logger.debug("Using %s from %s", repo.name, repo.env_var)
                return repo.resolved_path(candidate)
            logger.warning("%s points to missing path %s", repo.env_var, candidate)

    root_dir = ensure_base_dir(base_dir)
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

    if repo.init_submodules:
        _run_git(["submodule", "update", "--init", "--recursive"], cwd=destination)

    resolved = repo.resolved_path(root_dir)
    if not resolved.exists():
        raise FileNotFoundError(f"Expected {resolved} to exist after cloning {repo.name}")

    return resolved


def remove_repo(repo_key: str, *, base_dir: Optional[Path] = None) -> None:
    """Delete a managed repository checkout."""

    if repo_key not in CV_REPOS:
        raise KeyError(f"Unknown CV repository key: {repo_key}")

    repo = CV_REPOS[repo_key]
    root_dir = ensure_base_dir(base_dir)
    destination = repo.destination(root_dir)

    if destination.exists():
        logger.info("Removing repository checkout %s", destination)
        shutil.rmtree(destination, ignore_errors=True)


__all__ = ["CV_REPOS", "ExternalRepo", "ensure_repo", "remove_repo", "ensure_base_dir"]
