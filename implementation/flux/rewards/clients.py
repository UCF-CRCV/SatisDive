"""SatisDive flux: clients. See README.md for the supported interface."""

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


def _envvar(name, default):
    return os.environ.get(name, default)

HPSV3_PYTHON = _envvar("FLUX_HPSV3_PYTHON", sys.executable)
HPSV3_SITE = _envvar("FLUX_HPSV3_SITE",
                     "")
IMAGEREWARD_PYTHON = _envvar("FLUX_IMAGEREWARD_PYTHON", sys.executable)
IMAGEREWARD_SITE = _envvar("FLUX_IMAGEREWARD_SITE",
                           "")
HF_HOME = _envvar("HF_HOME", os.path.expanduser("~/.cache/huggingface"))

_WORKERS = Path(__file__).resolve().parent / "workers"
# The package root, placed on the worker's PYTHONPATH so it can import the
# rewards package (e.g. image_reward_compat). We pass ONLY this, never the
# launcher's own PYTHONPATH, so the flux venv's site-packages cannot shadow the
# worker venv's packages.
_PKG_ROOT = str(Path(__file__).resolve().parents[1])


class RewardClient:
    """A long-lived reward-model subprocess with a JSON-over-stdio protocol."""

    def __init__(self, proc: subprocess.Popen, tmp_dir: Path):
        self.proc = proc
        self.tmp_dir = Path(tmp_dir)
        self._req = 0

    # ------------------------------------------------------------------ spawn
    @classmethod
    def _spawn(cls, worker_script, worker_python, worker_site, gpu_id, tmp_prefix,
               shm: bool, warmup_s: float) -> "RewardClient":
        if not Path(worker_python).exists():
            raise FileNotFoundError(f"reward worker Python not found: {worker_python}")
        if not Path(worker_script).exists():
            raise FileNotFoundError(f"reward worker script not found: {worker_script}")
        if worker_site and not Path(worker_site).exists():
            raise FileNotFoundError(f"reward worker site-packages not found: {worker_site}")

        tmp_dir = Path(tempfile.mkdtemp(prefix=tmp_prefix,
                                        dir="/dev/shm" if shm else None))
        env = os.environ.copy()
        # site-packages of the worker's venv FIRST (so its torch/transformers/
        # ImageReward win), then the rewards package root (for image_reward_compat).
        # We never pass the launcher's own PYTHONPATH, so the flux venv's
        # site-packages cannot shadow the worker venv's packages.
        env["PYTHONPATH"] = os.pathsep.join(p for p in (worker_site, _PKG_ROOT) if p)
        env["HF_HOME"] = HF_HOME
        env["TRANSFORMERS_VERBOSITY"] = "error"
        # gpu_id is a logical index into the GPUs already exposed to this job.
        # Slurm may expose physical devices such as ``2,3``; preserve that
        # allocation instead of accidentally addressing physical GPU 1.
        if gpu_id is None:
            env["CUDA_VISIBLE_DEVICES"] = ""
        else:
            visible = os.environ.get("CUDA_VISIBLE_DEVICES")
            if visible:
                devices = [token.strip() for token in visible.split(",") if token.strip()]
                if gpu_id < 0 or gpu_id >= len(devices):
                    raise ValueError(
                        f"reward worker GPU index {gpu_id} is outside "
                        f"CUDA_VISIBLE_DEVICES={visible!r}"
                    )
                env["CUDA_VISIBLE_DEVICES"] = devices[gpu_id]
            else:
                env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

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
    def hpsv3(cls, gpu_id: int = 0, warmup_s: float = 300.0) -> "RewardClient":
        """Score-only HPSv3 (FK / VASR value resampling)."""
        return cls._spawn(_WORKERS / "hpsv3_worker.py", HPSV3_PYTHON, HPSV3_SITE, gpu_id,
                          "hpsv3_", shm=False, warmup_s=warmup_s)

    @classmethod
    def hpsv3_grad(cls, gpu_id: int = 0, warmup_s: float = 300.0) -> "RewardClient":
        """Differentiable HPSv3 (ours floor term / DAS reward-gradient proposal)."""
        return cls._spawn(_WORKERS / "hpsv3_grad_worker.py", HPSV3_PYTHON, HPSV3_SITE, gpu_id,
                          "hpsv3_grad_", shm=True, warmup_s=warmup_s)

    @classmethod
    def imagereward_grad(cls, gpu_id: int = 0, warmup_s: float = 300.0) -> "RewardClient":
        """Differentiable ImageReward (ours floor term / DAS reward-gradient proposal)."""
        return cls._spawn(_WORKERS / "ir_grad_worker.py", IMAGEREWARD_PYTHON, IMAGEREWARD_SITE, gpu_id,
                          "ir_grad_", shm=True, warmup_s=warmup_s)

    @classmethod
    def imagereward(cls, gpu_id: int = 0, warmup_s: float = 300.0) -> "RewardClient":
        """Score-only ImageReward (FK / VASR value resampling)."""
        return cls._spawn(_WORKERS / "ir_worker.py", IMAGEREWARD_PYTHON, IMAGEREWARD_SITE, gpu_id,
                          "ir_", shm=False, warmup_s=warmup_s)


    # ----------------------------------------------------------- capabilities
    def score(self, prompt: str, images: Sequence[Image.Image]) -> List[float]:
        """Score K PIL images for one prompt (value-only). Returns K floats."""
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
        on CPU). The autograd graph on the flux side is preserved by the caller,
        which uses ``grad`` as grad_outputs to backprop through VAE-decode + FLUX.
        """
        if images.dim() == 3:
            images = images.unsqueeze(0)
        k = images.shape[0]
        if len(prompts) != k:
            raise ValueError(f"prompts {len(prompts)} != images {k}")
        rid = self._req
        self._req += 1
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
    """Adapter exposing ``score(images, prompt) -> tensor`` for FK/VASR callbacks."""

    def __init__(self, client: RewardClient):
        self.client = client

    def score(self, images, prompt: str):
        return torch.tensor(self.client.score(prompt, images))
