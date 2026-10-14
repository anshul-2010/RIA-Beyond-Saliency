"""Reproducible runs: JSON configs, fixed seeds, and JSONL result logs.

Every run gets its own directory under results/:

    results/<YYYYmmdd-HHMMSS>_<name>/
        config.json     arguments + git commit, dirty flag, library versions, device
        results.jsonl   one JSON object per logged record

Usage:
    cfg = load_config(parser)              # --config file.json, CLI flags override it
    with Run(cfg["name"], cfg) as run:
        run.log({"example": 0, "G3": 0.01})
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import random
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def load_config(parser: argparse.ArgumentParser, argv: list[str] | None = None) -> dict[str, Any]:
    """Parse args; a --config JSON file supplies defaults that explicit CLI flags override."""
    parser.add_argument("--config", type=Path, help="JSON file with default arguments")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--name", default=None, help="run name (default: script name)")
    parser.add_argument("--results-dir", type=Path, default=REPO_ROOT / "results")
    pre, _ = parser.parse_known_args(argv)
    if pre.config:
        file_cfg = json.loads(pre.config.read_text(encoding="utf-8"))
        unknown = set(file_cfg) - {a.dest for a in parser._actions}
        if unknown:
            raise ValueError(f"unknown keys in {pre.config}: {sorted(unknown)}")
        parser.set_defaults(**file_cfg)
    return vars(parser.parse_args(argv))


def _git(*args: str) -> str | None:
    try:
        return subprocess.run(["git", *args], cwd=REPO_ROOT, capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def environment() -> dict[str, Any]:
    env: dict[str, Any] = {
        # Remote runs (e.g. Kaggle) get code via `git archive`, so the runner passes the commit in.
        "git_commit": _git("rev-parse", "HEAD") or os.environ.get("RIA_GIT_COMMIT"),
        "git_dirty": bool(_git("status", "--porcelain")),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy": np.__version__,
    }
    try:
        import torch
        env.update(torch=torch.__version__, cuda=torch.cuda.is_available(),
                   device_name=torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu")
    except ImportError:
        pass
    try:
        import transformers
        env["transformers"] = transformers.__version__
    except ImportError:
        pass
    return env


def _jsonable(x: Any) -> Any:
    if isinstance(x, Path):
        return str(x)
    if isinstance(x, (np.floating, np.integer)):
        return x.item()
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (set, frozenset)):
        return sorted(x)
    raise TypeError(f"not JSON serialisable: {type(x).__name__}")


class Run:
    """A results directory with config.json and an append-only results.jsonl."""

    def __init__(self, name: str, config: dict[str, Any], results_dir: Path | None = None,
                 run_dir: Path | None = None):
        """run_dir: a fixed directory to resume into (append; `done_keys` lists finished items)."""
        stamp = time.strftime("%Y%m%d-%H%M%S")
        if run_dir:
            self.dir = Path(run_dir)
            self.dir.mkdir(parents=True, exist_ok=True)
        else:
            results_dir = Path(results_dir or config.get("results_dir") or REPO_ROOT / "results")
            self.dir = results_dir / f"{stamp}_{name}"
            suffix = 1
            while self.dir.exists():
                self.dir = results_dir / f"{stamp}_{name}-{suffix}"
                suffix += 1
            self.dir.mkdir(parents=True)
        self.config = config
        if "seed" in config:
            set_seed(int(config["seed"]))
        meta = {"name": name, "started": time.strftime("%Y-%m-%dT%H:%M:%S"), "args": config, "env": environment()}
        cfg_path = self.dir / "config.json"
        if cfg_path.exists():                       # resuming: keep the first config, log this one too
            cfg_path = self.dir / f"config_resume_{stamp}.json"
        cfg_path.write_text(json.dumps(meta, indent=2, default=_jsonable), encoding="utf-8")
        self._fh = open(self.dir / "results.jsonl", "a", encoding="utf-8")
        self.n = 0

    def done_keys(self, field: str = "key") -> set:
        path = self.dir / "results.jsonl"
        if not path.exists():
            return set()
        keys = set()
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                keys.add(json.loads(line).get(field))
            except json.JSONDecodeError:                 # a line cut off by a disconnect
                continue
        return keys

    def log(self, record: dict[str, Any]) -> None:
        self._fh.write(json.dumps(record, default=_jsonable) + "\n")
        self._fh.flush()
        self.n += 1

    def close(self) -> None:
        if not self._fh.closed:
            self._fh.close()

    def __enter__(self) -> "Run":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def read_results(run_dir: Path) -> list[dict[str, Any]]:
    with open(Path(run_dir) / "results.jsonl", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]
