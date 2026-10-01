"""Helpers for invoking Blender as a subprocess for offline rendering.

Reuses `blender_dataset/blender_launcher.py` to locate / install Blender 4.5.
"""

from __future__ import annotations

import json
import os
import shlex
import subprocess
import sys


def _ensure_blender_launcher_on_path():
    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    bd = os.path.join(repo_root, "evaluation", "utils")
    if bd not in sys.path:
        sys.path.insert(0, bd)


def find_blender_binary(install_root: str | None = None) -> str:
    _ensure_blender_launcher_on_path()
    from blender_launcher import ensure_blender_binary, CONFIG_JSON_ENV_VAR  # noqa: F401  # type: ignore
    cfg = {"basic": {}}
    if install_root:
        cfg["basic"]["blender_installation_path"] = install_root
    return ensure_blender_binary(cfg)


def run_blender_worker(worker_script: str, render_config: dict, install_root: str | None = None) -> int:
    """Launch Blender in background mode running `worker_script` with `render_config` passed via env."""
    _ensure_blender_launcher_on_path()
    from blender_launcher import CONFIG_JSON_ENV_VAR  # type: ignore

    blender_path = find_blender_binary(install_root)
    env = os.environ.copy()
    env[CONFIG_JSON_ENV_VAR] = json.dumps(render_config)
    cmd = [blender_path, "-b", "-P", os.path.abspath(worker_script)]
    print(f"[INFO] Launching Blender: {' '.join(shlex.quote(p) for p in cmd)}")
    return subprocess.call(cmd, env=env)
