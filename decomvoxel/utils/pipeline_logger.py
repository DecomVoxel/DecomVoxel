"""Simple step-timing logger for the DecomVoxel full pipeline.

Writes two artefacts under ``<model_path>/log/``:

* ``run_<timestamp>.log``  — human-readable lines (also echoed to stdout).
* ``run_<timestamp>.json`` — structured record with start/end times,
  per-step durations, total duration, and the CLI args.

A ``latest.log`` / ``latest.json`` symlink (or copy on platforms without
symlink support) is also maintained for convenience.
"""

from __future__ import annotations

import json
import os
import shutil
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

from decomvoxel.utils.gpu_monitor import GpuMonitor


def _fmt_duration(seconds: float) -> str:
    seconds = float(seconds)
    h, rem = divmod(seconds, 3600.0)
    m, s = divmod(rem, 60.0)
    if h >= 1:
        return f"{int(h)}h{int(m):02d}m{s:05.2f}s ({seconds:.2f}s)"
    if m >= 1:
        return f"{int(m)}m{s:05.2f}s ({seconds:.2f}s)"
    return f"{seconds:.2f}s"


class PipelineLogger:
    """Lightweight pipeline-step logger.

    Usage::

        logger = PipelineLogger(model_path, cli_args=vars(args))
        logger.start_step("Step 1/5: Train GeoSVR")
        ...
        logger.end_step()
        ...
        logger.finalize()
    """

    def __init__(
        self,
        model_path: str,
        cli_args: Optional[Dict[str, Any]] = None,
        tag: str = "run",
        gpu_sample_interval: float = 5.0,
    ) -> None:
        self.model_path = model_path
        self.log_dir = os.path.join(model_path, "log")
        os.makedirs(self.log_dir, exist_ok=True)

        # Choose a unique timestamped filename so prior log files are never
        # overwritten or removed. If two runs start in the same second (or a
        # file with the same name already exists for any reason), append a
        # numeric suffix until both the .log and .json paths are free.
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        base = f"{tag}_{ts}"
        log_path = os.path.join(self.log_dir, f"{base}.log")
        json_path = os.path.join(self.log_dir, f"{base}.json")
        suffix = 1
        while os.path.exists(log_path) or os.path.exists(json_path):
            suffix += 1
            candidate = f"{base}_{suffix}"
            log_path = os.path.join(self.log_dir, f"{candidate}.log")
            json_path = os.path.join(self.log_dir, f"{candidate}.json")
        self.timestamp = ts
        self.log_path = log_path
        self.json_path = json_path
        # ``latest.{log,json}`` are *pointer* files (symlink, or a copy as
        # fallback). They are refreshed on ``finalize()`` to point at this
        # run, but the underlying timestamped files of past runs are kept
        # untouched.
        self.latest_log = os.path.join(self.log_dir, "latest.log")
        self.latest_json = os.path.join(self.log_dir, "latest.json")

        self._start_wall = time.time()
        self._start_dt = datetime.now()
        self._steps: List[Dict[str, Any]] = []
        self._current: Optional[Dict[str, Any]] = None

        self.cli_args = self._jsonable(cli_args) if cli_args is not None else None

        # Initialise the .log file with a header
        with open(self.log_path, "w") as f:
            f.write(f"# DecomVoxel pipeline log\n")
            f.write(f"# model_path : {model_path}\n")
            f.write(f"# started_at : {self._start_dt.isoformat()}\n")
            if self.cli_args is not None:
                f.write(f"# cli_args   : {json.dumps(self.cli_args, ensure_ascii=False)}\n")
            f.write("\n")

        self._flush_json()
        self._log(f"[Pipeline Logger] Logs → {self.log_path}")
        self._log(f"[Pipeline Logger] Started at {self._start_dt.isoformat()}")
        if self.cli_args is not None:
            self._log(f"[Pipeline Logger] CLI args: {json.dumps(self.cli_args, ensure_ascii=False)}")

        # Background GPU usage sampler. Writes its own timestamped JSON +
        # PNG into the same log dir on ``finalize()``; previous monitor
        # artefacts are never touched.
        self.gpu_monitor = GpuMonitor(
            output_dir=self.log_dir,
            tag=tag,
            interval=gpu_sample_interval,
            timestamp=self.timestamp if suffix == 1 else f"{self.timestamp}_{suffix}",
        )
        self.gpu_monitor.start()

    # ------------------------------------------------------------------ utils

    @staticmethod
    def _jsonable(obj: Any) -> Any:
        try:
            json.dumps(obj)
            return obj
        except (TypeError, ValueError):
            pass
        if isinstance(obj, dict):
            return {str(k): PipelineLogger._jsonable(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [PipelineLogger._jsonable(v) for v in obj]
        return str(obj)

    def _log(self, msg: str) -> None:
        print(msg)
        with open(self.log_path, "a") as f:
            f.write(msg.rstrip("\n") + "\n")

    def _flush_json(self) -> None:
        end_wall = time.time()
        payload = {
            "model_path": self.model_path,
            "start_time": self._start_dt.isoformat(),
            "start_unix": self._start_wall,
            "cli_args": self.cli_args,
            "steps": self._steps,
            "current_step": self._current,
            "end_time": datetime.now().isoformat(),
            "end_unix": end_wall,
            "elapsed_seconds": end_wall - self._start_wall,
            "elapsed_pretty": _fmt_duration(end_wall - self._start_wall),
        }
        gpu = getattr(self, "_gpu_summary", None)
        if gpu is not None:
            payload["gpu_usage"] = gpu
        with open(self.json_path, "w") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)

    # ------------------------------------------------------------------ api

    def start_step(self, name: str) -> None:
        if self._current is not None:
            # Auto-close any open step (defensive)
            self.end_step()
        now = time.time()
        self._current = {
            "name": name,
            "start_time": datetime.fromtimestamp(now).isoformat(),
            "start_unix": now,
        }
        self._log("\n" + "=" * 60)
        self._log(f"[Pipeline] BEGIN  {name}   (at {self._current['start_time']})")
        self._log("=" * 60)
        self._flush_json()

    def end_step(self, extra: Optional[Dict[str, Any]] = None) -> None:
        if self._current is None:
            return
        now = time.time()
        duration = now - self._current["start_unix"]
        self._current.update(
            end_time=datetime.fromtimestamp(now).isoformat(),
            end_unix=now,
            duration_seconds=duration,
            duration_pretty=_fmt_duration(duration),
        )
        if extra:
            self._current["extra"] = self._jsonable(extra)
        self._log(
            f"[Pipeline] END    {self._current['name']}   "
            f"(at {self._current['end_time']}, took {_fmt_duration(duration)})"
        )
        self._steps.append(self._current)
        self._current = None
        self._flush_json()

    def finalize(self) -> None:
        if self._current is not None:
            self.end_step()
        end_wall = time.time()
        total = end_wall - self._start_wall
        # Stop the GPU sampler first so its summary lines appear inside the
        # pipeline summary block.
        gpu_summary = None
        try:
            gpu_summary = self.gpu_monitor.stop()
        except Exception as exc:
            self._log(f"[Pipeline] GPU monitor stop failed: {exc}")
        self._log("\n" + "=" * 60)
        self._log("[Pipeline] Summary")
        self._log("=" * 60)
        for s in self._steps:
            self._log(f"  - {s['name']}: {s.get('duration_pretty', '?')}")
        self._log(f"  TOTAL: {_fmt_duration(total)}")
        self._log(f"  Finished at {datetime.fromtimestamp(end_wall).isoformat()}")
        if gpu_summary and gpu_summary.get("per_gpu_stats"):
            self._log("  GPU usage (peak / mean):")
            for idx, st in gpu_summary["per_gpu_stats"].items():
                self._log(
                    f"    GPU {idx}: util {st['util_peak_pct']:.1f}% / "
                    f"{st['util_mean_pct']:.1f}%  |  mem "
                    f"{st['mem_peak_mib']:.0f} MiB / {st['mem_mean_mib']:.0f} MiB "
                    f"(n={st['samples']})"
                )
            self._log(f"  GPU usage JSON : {self.gpu_monitor.json_path}")
            self._log(f"  GPU usage plot : {self.gpu_monitor.png_path}")
            # Persist a compact reference inside the pipeline JSON too.
            self._gpu_summary = {
                "json_path": self.gpu_monitor.json_path,
                "png_path": self.gpu_monitor.png_path,
                "per_gpu_stats": gpu_summary["per_gpu_stats"],
                "backend": gpu_summary.get("backend"),
                "interval_seconds": gpu_summary.get("interval_seconds"),
                "num_samples": gpu_summary.get("num_samples"),
            }
        self._flush_json()
        self._update_latest()

    def _update_latest(self) -> None:
        # Refresh ``latest.log`` / ``latest.json`` to point at *this* run.
        # Only the pointer files themselves are ever touched here; the
        # timestamped historical logs of previous runs are never modified
        # or deleted.
        for src, dst in ((self.log_path, self.latest_log), (self.json_path, self.latest_json)):
            assert os.path.basename(dst).startswith("latest."), (
                f"_update_latest refused to touch non-latest file: {dst}"
            )
            try:
                if os.path.islink(dst) or os.path.exists(dst):
                    os.remove(dst)
                os.symlink(os.path.basename(src), dst)
            except OSError:
                try:
                    shutil.copy2(src, dst)
                except OSError:
                    pass
