"""SatisDive flux: profiling. See README.md for the supported interface."""
from __future__ import annotations

import subprocess
import threading


class GpuSampler:
    """Background nvidia-smi poller. Records per-GPU memory.used (MiB) and
    utilization (%) every `period_s`. NOTE: memory.used is box-wide (A100s are
    shared with other users), so the clean per-process figure is still
    torch.cuda.max_memory_allocated(); this sampler's value-add is GPU UTILIZATION
    (which torch cannot report) and a coarse memory trace. Filter to the GPUs this
    run actually uses via gpu_ids."""

    def __init__(self, gpu_ids=None, period_s=2.0):
        self.gpu_ids = set(gpu_ids) if gpu_ids is not None else None
        self.period = period_s
        self._stop = threading.Event()
        self._samples = []   # list of {gid: (mem_mb, util_pct)}
        self._thr = None

    def _poll(self):
        q = "index,memory.used,utilization.gpu"
        while not self._stop.is_set():
            try:
                out = subprocess.check_output(
                    ["nvidia-smi", f"--query-gpu={q}",
                     "--format=csv,noheader,nounits"], timeout=5).decode()
                row = {}
                for ln in out.strip().splitlines():
                    parts = [p.strip() for p in ln.split(",")]
                    if len(parts) < 3:
                        continue
                    gid = int(parts[0])
                    if self.gpu_ids is not None and gid not in self.gpu_ids:
                        continue
                    row[gid] = (float(parts[1]), float(parts[2]))
                if row:
                    self._samples.append(row)
            except Exception:
                pass
            self._stop.wait(self.period)

    def start(self):
        self._stop.clear()
        self._thr = threading.Thread(target=self._poll, daemon=True)
        self._thr.start()
        return self

    def stop(self):
        self._stop.set()
        if self._thr is not None:
            self._thr.join(timeout=3)
        return self.summary()

    def summary(self):
        if not self._samples:
            return {"n_samples": 0}
        gids = sorted({g for s in self._samples for g in s})
        per = {}
        for g in gids:
            mems = [s[g][0] for s in self._samples if g in s]
            utils = [s[g][1] for s in self._samples if g in s]
            per[g] = {
                "mem_used_mb_peak": round(max(mems), 1),
                "mem_used_mb_mean": round(sum(mems) / len(mems), 1),
                "util_pct_peak": round(max(utils), 1),
                "util_pct_mean": round(sum(utils) / len(utils), 1),
            }
        return {"n_samples": len(self._samples), "period_s": self.period,
                "per_gpu": per}
