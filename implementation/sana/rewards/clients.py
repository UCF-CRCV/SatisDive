"""SatisDive sana: clients. See README.md for the supported interface."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import List, Sequence

from PIL import Image
import torch


HF_HOME = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))

_WORKERS = Path(__file__).resolve().parent / "workers"
# The sana package root, placed on the worker's PYTHONPATH so it can import the
# rewards package (e.g. image_reward_compat). We pass ONLY this plus any explicit
# worker site-packages, never the launcher's own PYTHONPATH, so the launching
# venv's site-packages cannot shadow the worker venv's packages.
_PKG_ROOT = str(Path(__file__).resolve().parents[1])


def _resolve_worker_python(explicit: str | None, env_var: str) -> str:
    """Resolve the worker interpreter (see module docstring). Never a hardcode."""
    cand = explicit or os.environ.get(env_var) or sys.executable
    return cand


class RewardClient:
    """A long-lived reward-model subprocess with a JSON-over-stdio protocol."""

    def __init__(self, proc: subprocess.Popen, tmp_dir: Path):
        self.proc = proc
        self.tmp_dir = Path(tmp_dir)
        self._req = 0

    # ------------------------------------------------------------------ spawn
    @classmethod
    def _spawn(cls, worker_script, worker_python, gpu_id, tmp_prefix,
               shm: bool, warmup_s: float, extra_pythonpath: str | None = None) -> "RewardClient":
        if not Path(worker_python).exists():
            raise FileNotFoundError(
                f"reward worker Python not found: {worker_python}. Set "
                f"SANA_IMAGEREWARD_PYTHON to an interpreter that can import "
                f"ImageReward (see clients.py docstring).")
        if not Path(worker_script).exists():
            raise FileNotFoundError(f"reward worker script not found: {worker_script}")

        tmp_parent = os.environ.get("SANA_REWARD_TMPDIR")
        if tmp_parent:
            Path(tmp_parent).mkdir(parents=True, exist_ok=True)
        elif shm and Path("/dev/shm").exists():
            tmp_parent = "/dev/shm"
        tmp_dir = Path(tempfile.mkdtemp(prefix=tmp_prefix, dir=tmp_parent))
        env = os.environ.copy()


        pp = [_PKG_ROOT]
        if extra_pythonpath:
            pp.append(extra_pythonpath)
        env["PYTHONPATH"] = os.pathsep.join(pp)
        env["HF_HOME"] = HF_HOME
        env["TRANSFORMERS_VERBOSITY"] = "error"
        # gpu_id None -> force CPU; else pin the worker to one card so it never
        # collides with the SANA process (which runs on a different card).
        env["CUDA_VISIBLE_DEVICES"] = "" if gpu_id is None else str(gpu_id)

        proc = subprocess.Popen(
            [worker_python, str(worker_script)],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=sys.stderr,
            env=env, text=True, bufsize=1,
        )
        client = cls(proc, tmp_dir)
        client._await_ready(warmup_s)
        return client

    def _await_ready(self, warmup_s: float) -> None:


        warmup_s = float(os.environ.get("REWARD_WORKER_WARMUP_S", warmup_s))
        deadline = time.time() + warmup_s
        while time.time() < deadline:
            line = self.proc.stdout.readline()
            if not line:
                if self.proc.poll() is not None:
                    raise RuntimeError(
                        f"reward worker exited during startup, code {self.proc.returncode}")
                continue
            try:
                msg = json.loads(line)
            except json.JSONDecodeError:
                continue
            if msg.get("ready"):
                return
            if "error" in msg:
                self.proc.terminate()
                raise RuntimeError(f"reward worker startup error: {msg['error']}")
        self.proc.terminate()
        raise TimeoutError(f"reward worker did not become ready within {warmup_s}s")

    # --------------------------------------------------------------- backends
    @classmethod
    def imagereward(cls, gpu_id: int | None = 0, warmup_s: float = float(os.environ.get("REWARD_WORKER_WARMUP_S", "300")),
                    worker_python: str | None = None) -> "RewardClient":
        """Score-only ImageReward (FK value resampling). Reward = ImageReward."""
        wp = _resolve_worker_python(worker_python, "SANA_IMAGEREWARD_PYTHON")
        return cls._spawn(_WORKERS / "ir_worker.py", wp, gpu_id, "ir_",
                          shm=False, warmup_s=warmup_s,
                          extra_pythonpath=os.environ.get("SANA_IMAGEREWARD_PYTHONPATH"))

    @classmethod
    def imagereward_grad(cls, gpu_id: int | None = 0, warmup_s: float = float(os.environ.get("REWARD_WORKER_WARMUP_S", "300")),
                         worker_python: str | None = None) -> "RewardClient":
        """Differentiable ImageReward (DAS reward-gradient proposal)."""
        wp = _resolve_worker_python(worker_python, "SANA_IMAGEREWARD_PYTHON")
        return cls._spawn(_WORKERS / "ir_grad_worker.py", wp, gpu_id, "ir_grad_",
                          shm=True, warmup_s=warmup_s,
                          extra_pythonpath=os.environ.get("SANA_IMAGEREWARD_PYTHONPATH"))

    # ----------------------------------------------------------- capabilities
    def score(self, prompt: str, images: Sequence[Image.Image]) -> List[float]:
        """Score K PIL images for one prompt (value-only). Returns K floats."""
        self.tmp_dir.mkdir(parents=True, exist_ok=True)
        paths = []
        for img in images:
            p = self.tmp_dir / f"req_{self._req:08d}.png"
            self._req += 1
            img.save(p)
            paths.append(str(p))
        self._send({"prompt": prompt, "image_paths": paths})
        msg = self._recv()
        for p in paths:
            try:
                os.unlink(p)
            except OSError:
                pass
        return msg["scores"]

    def reward_and_grad(self, images: "torch.Tensor", prompts: Sequence[str]):
        """Differentiable score of K images (K,3,H,W in [0,1]).

        Returns (rewards: list[float], grad: tensor (K,3,H,W) = d sum_k r / d image,
        on CPU). The autograd graph on the SANA side is preserved by the caller,
        which uses ``grad`` as grad_outputs to backprop through DC-AE-decode + the
        SANA transformer.
        """
        if images.dim() == 3:
            images = images.unsqueeze(0)
        k = images.shape[0]
        if len(prompts) != k:
            raise ValueError(f"prompts {len(prompts)} != images {k}")
        rid = self._req
        self._req += 1
        self.tmp_dir.mkdir(parents=True, exist_ok=True)
        image_pt = str(self.tmp_dir / f"img_{rid:08d}.pt")
        out_pt = str(self.tmp_dir / f"grad_{rid:08d}.pt")
        torch.save(images.detach().cpu(), image_pt)
        self._send({"image_pt": image_pt, "prompts": list(prompts), "out_pt": out_pt})
        msg = self._recv()
        grad = torch.load(msg["grad_pt"], map_location="cpu")
        for p in (image_pt, out_pt):
            try:
                os.unlink(p)
            except OSError:
                pass
        return msg["rewards"], grad

    # -------------------------------------------------------------------- io
    def _send(self, req: dict) -> None:
        self.proc.stdin.write(json.dumps(req) + "\n")
        self.proc.stdin.flush()

    def _recv(self) -> dict:
        line = self.proc.stdout.readline()
        if not line:
            raise RuntimeError(f"reward worker closed stdout (exit {self.proc.poll()})")
        msg = json.loads(line)
        if "error" in msg:
            raise RuntimeError(f"reward worker error: {msg['error']}\n{msg.get('trace', '')}")
        return msg

    def close(self) -> None:
        if self.proc.poll() is None:
            try:
                self._send({"shutdown": True})
                try:
                    self.proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self.proc.terminate()
            except (BrokenPipeError, OSError):
                self.proc.terminate()
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


class RewardScorer:
    """Adapter exposing ``score(images, prompt) -> tensor`` for the FK callback."""

    def __init__(self, client: RewardClient):
        self.client = client

    def score(self, images, prompt: str):
        return torch.tensor(self.client.score(prompt, images))
