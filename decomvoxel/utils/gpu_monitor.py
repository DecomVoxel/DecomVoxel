"""Background GPU utilization / memory sampler.

Samples GPU utilization (%) and memory usage (MiB) for every CUDA device
visible to this process (i.e. those exposed via ``CUDA_VISIBLE_DEVICES``)
on a fixed interval. At ``stop()`` time it writes a JSON summary
(``gpu_usage_<timestamp>.json``) and a line-chart PNG
(``gpu_usage_<timestamp>.png``) into the same directory.

Uses ``pynvml`` if available; otherwise falls back to shelling out to
``nvidia-smi``. If neither is available, the monitor degrades gracefully
to a no-op.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import threading
import time
from datetime import datetime
from typing import Dict, List, Optional


def _resolve_visible_gpu_indices() -> List[int]:
    """Return the *physical* GPU indices visible to this process."""
    raw = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if raw.strip() == "":
        # No restriction set: monitor every GPU on the host (if any).
        try:
            import pynvml  # type: ignore

            pynvml.nvmlInit()
            count = pynvml.nvmlDeviceGetCount()
            pynvml.nvmlShutdown()
            return list(range(count))
        except Exception:
            return []
    indices: List[int] = []
    for tok in raw.split(","):
        tok = tok.strip()
        if not tok:
            continue
        try:
            indices.append(int(tok))
        except ValueError:
            # UUID form; pynvml path will handle it, fallback can't map.
            indices.append(-1)
    return indices


class GpuMonitor:
    """Sample GPU utilization in a background thread.

    Parameters
    ----------
    output_dir: directory in which the JSON + PNG summary will be written.
    tag: filename prefix (e.g. ``"run"``); files become
        ``gpu_usage_<tag>_<timestamp>.{json,png}``.
    interval: seconds between samples.
    """

    def __init__(
        self,
        output_dir: str,
        tag: str = "run",
        interval: float = 5.0,
        timestamp: Optional[str] = None,
    ) -> None:
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.interval = max(0.5, float(interval))

        ts = timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
        base = f"gpu_usage_{tag}_{ts}"
        json_path = os.path.join(self.output_dir, f"{base}.json")
        png_path = os.path.join(self.output_dir, f"{base}.png")
        suffix = 1
        while os.path.exists(json_path) or os.path.exists(png_path):
            suffix += 1
            candidate = f"{base}_{suffix}"
            json_path = os.path.join(self.output_dir, f"{candidate}.json")
            png_path = os.path.join(self.output_dir, f"{candidate}.png")
        self.json_path = json_path
        self.png_path = png_path

        self._gpu_indices = _resolve_visible_gpu_indices()
        # ``samples`` is a list of dicts: {"t": rel_seconds, "util": {idx: %},
        # "mem_used_mib": {idx: MiB}, "mem_total_mib": {idx: MiB}}.
        self._samples: List[Dict] = []
        self._stop_evt = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._start_unix: Optional[float] = None

        # Choose backend
        self._backend = "noop"
        self._pynvml = None
        self._handles: Dict[int, object] = {}
        if self._gpu_indices:
            try:
                import pynvml  # type: ignore

                pynvml.nvmlInit()
                for idx in self._gpu_indices:
                    if idx < 0:
                        # UUID form – best-effort: cannot resolve via int idx
                        continue
                    try:
                        self._handles[idx] = pynvml.nvmlDeviceGetHandleByIndex(idx)
                    except Exception:
                        pass
                if self._handles:
                    self._pynvml = pynvml
                    self._backend = "pynvml"
                else:
                    pynvml.nvmlShutdown()
            except Exception:
                self._pynvml = None
            if self._backend == "noop" and shutil.which("nvidia-smi") is not None:
                self._backend = "nvidia-smi"

    # ------------------------------------------------------------------ api

    def start(self) -> None:
        if self._thread is not None:
            return
        if self._backend == "noop":
            print(
                "[GpuMonitor] No GPU backend available (neither pynvml nor "
                "nvidia-smi); GPU usage sampling disabled."
            )
            return
        self._start_unix = time.time()
        self._stop_evt.clear()
        self._thread = threading.Thread(
            target=self._run, name="GpuMonitor", daemon=True
        )
        self._thread.start()
        print(
            f"[GpuMonitor] Sampling GPUs {self._monitored_indices()} every "
            f"{self.interval:.1f}s via {self._backend}."
        )

    def stop(self) -> Optional[Dict]:
        if self._thread is None:
            return None
        self._stop_evt.set()
        self._thread.join(timeout=max(5.0, self.interval * 2))
        self._thread = None
        if self._pynvml is not None:
            try:
                self._pynvml.nvmlShutdown()
            except Exception:
                pass
        summary = self._write_summary()
        self._plot()
        return summary

    # ----------------------------------------------------------------- core

    def _monitored_indices(self) -> List[int]:
        if self._backend == "pynvml":
            return sorted(self._handles.keys())
        return [i for i in self._gpu_indices if i >= 0]

    def _run(self) -> None:
        # Take an immediate first sample so very short runs still produce data
        self._sample_once()
        while not self._stop_evt.wait(self.interval):
            self._sample_once()

    def _sample_once(self) -> None:
        now = time.time()
        rel = now - (self._start_unix or now)
        util: Dict[int, float] = {}
        mem_used: Dict[int, float] = {}
        mem_total: Dict[int, float] = {}
        try:
            if self._backend == "pynvml":
                for idx, h in self._handles.items():
                    try:
                        u = self._pynvml.nvmlDeviceGetUtilizationRates(h)
                        m = self._pynvml.nvmlDeviceGetMemoryInfo(h)
                        util[idx] = float(u.gpu)
                        mem_used[idx] = m.used / (1024 ** 2)
                        mem_total[idx] = m.total / (1024 ** 2)
                    except Exception:
                        continue
            elif self._backend == "nvidia-smi":
                idx_list = ",".join(str(i) for i in self._monitored_indices())
                cmd = [
                    "nvidia-smi",
                    f"--query-gpu=index,utilization.gpu,memory.used,memory.total",
                    "--format=csv,noheader,nounits",
                ]
                if idx_list:
                    cmd.insert(1, f"-i={idx_list}")
                out = subprocess.check_output(cmd, timeout=5).decode("utf-8", "ignore")
                for line in out.strip().splitlines():
                    parts = [p.strip() for p in line.split(",")]
                    if len(parts) < 4:
                        continue
                    try:
                        idx = int(parts[0])
                        util[idx] = float(parts[1])
                        mem_used[idx] = float(parts[2])
                        mem_total[idx] = float(parts[3])
                    except ValueError:
                        continue
        except Exception as exc:  # pragma: no cover - defensive
            print(f"[GpuMonitor] sample failed: {exc}")
            return
        if util:
            self._samples.append(
                {
                    "t": rel,
                    "unix": now,
                    "util": util,
                    "mem_used_mib": mem_used,
                    "mem_total_mib": mem_total,
                }
            )

    # -------------------------------------------------------------- summary

    def _per_gpu_stats(self) -> Dict[int, Dict[str, float]]:
        per_gpu: Dict[int, Dict[str, List[float]]] = {}
        for s in self._samples:
            for idx, v in s["util"].items():
                d = per_gpu.setdefault(int(idx), {"util": [], "mem": []})
                d["util"].append(float(v))
                d["mem"].append(float(s["mem_used_mib"].get(idx, 0.0)))
        out: Dict[int, Dict[str, float]] = {}
        for idx, d in per_gpu.items():
            u = d["util"]
            m = d["mem"]
            out[idx] = {
                "samples": len(u),
                "util_peak_pct": max(u) if u else 0.0,
                "util_mean_pct": (sum(u) / len(u)) if u else 0.0,
                "mem_peak_mib": max(m) if m else 0.0,
                "mem_mean_mib": (sum(m) / len(m)) if m else 0.0,
            }
        return out

    def _write_summary(self) -> Dict:
        stats = self._per_gpu_stats()
        payload = {
            "backend": self._backend,
            "interval_seconds": self.interval,
            "monitored_gpu_indices": self._monitored_indices(),
            "start_unix": self._start_unix,
            "end_unix": time.time(),
            "num_samples": len(self._samples),
            "per_gpu_stats": stats,
            "samples": self._samples,
        }
        try:
            with open(self.json_path, "w") as f:
                json.dump(payload, f, indent=2)
        except OSError as exc:
            print(f"[GpuMonitor] failed to write JSON summary: {exc}")
        # Human-readable line
        for idx, s in stats.items():
            print(
                f"[GpuMonitor] GPU {idx}: peak util {s['util_peak_pct']:.1f}% / "
                f"mean util {s['util_mean_pct']:.1f}% | peak mem "
                f"{s['mem_peak_mib']:.0f} MiB / mean mem {s['mem_mean_mib']:.0f} MiB "
                f"(n={s['samples']})"
            )
        return payload

    def _plot(self) -> None:
        if not self._samples:
            return
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except Exception as exc:
            print(f"[GpuMonitor] matplotlib unavailable, skipping plot: {exc}")
            return

        ts = [s["t"] for s in self._samples]
        idx_set = sorted({int(i) for s in self._samples for i in s["util"].keys()})
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
        for idx in idx_set:
            util_series = [float(s["util"].get(idx, float("nan"))) for s in self._samples]
            mem_series = [float(s["mem_used_mib"].get(idx, float("nan"))) for s in self._samples]
            ax1.plot(ts, util_series, label=f"GPU {idx}")
            ax2.plot(ts, mem_series, label=f"GPU {idx}")
        ax1.set_ylabel("GPU Utilization (%)")
        ax1.set_ylim(0, 105)
        ax1.grid(True, alpha=0.3)
        ax1.legend(loc="best")
        ax1.set_title("GPU Usage During Pipeline Run")
        ax2.set_ylabel("GPU Memory Used (MiB)")
        ax2.set_xlabel("Elapsed Time (s)")
        ax2.grid(True, alpha=0.3)
        ax2.legend(loc="best")
        try:
            fig.tight_layout()
            fig.savefig(self.png_path, dpi=120)
            print(f"[GpuMonitor] usage plot → {self.png_path}")
        except Exception as exc:
            print(f"[GpuMonitor] failed to save plot: {exc}")
        finally:
            plt.close(fig)
