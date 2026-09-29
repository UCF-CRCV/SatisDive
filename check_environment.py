"""Check dependencies and the reward API without loading model weights."""
from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import importlib.util
import inspect
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
PREFIX = "SATISDIVE_ENV="


def inspect_worker(model):
    import torch
    sys.path.insert(0, str(ROOT / "implementation" / model / "rewards"))
    if model == "flux":
        from hpsv3 import HPSv3RewardInferencer
        if "differentiable" not in inspect.signature(HPSv3RewardInferencer).parameters:
            raise ValueError("HPSv3 is missing the differentiable API; install the bundled source")
        if not callable(getattr(HPSv3RewardInferencer, "prepare_batch", None)):
            raise ValueError("HPSv3 is missing prepare_batch; install the bundled source")
    else:
        from image_reward_compat import _install_stubs
        _install_stubs()
        # Same inference-only import strategy as the compatibility shim, but do
        # not invoke load(), the tokenizer, or anything that fetches weights.
        import types
        spec = importlib.util.find_spec("ImageReward")
        if spec is None or not spec.submodule_search_locations:
            raise ValueError("ImageReward is not installed in the reward environment")
        package = types.ModuleType("ImageReward")
        package.__path__ = list(spec.submodule_search_locations)
        sys.modules["ImageReward"] = package
        importlib.import_module("ImageReward.utils")
    packages = {}
    for name in ("torch", "torchvision", "transformers", "tokenizers", "hpsv3" if model == "flux" else "image-reward"):
        packages[name] = importlib.metadata.version(name)
    if torch.version.cuda is None:
        raise ValueError("The reward environment has CPU-only PyTorch; install a CUDA build")
    return {"python": sys.version.split()[0], "packages": packages}


def check_worker(interpreter, model):
    env = os.environ.copy()
    # Avoid inheriting generation-environment packages into the reward process.
    env.pop("PYTHONPATH", None)
    env["HF_HUB_OFFLINE"] = "1"
    env["TRANSFORMERS_OFFLINE"] = "1"
    try:
        result = subprocess.run([interpreter, str(Path(__file__).absolute()), "--worker", model], env=env, capture_output=True, text=True, timeout=90)
    except subprocess.TimeoutExpired as exc:
        raise ValueError("Reward import check exceeded 90 seconds; no generation started") from exc
    if result.returncode:
        raise ValueError("Reward environment check failed before model loading:\n" + (result.stderr or result.stdout)[-4000:])
    lines = [line[len(PREFIX):] for line in result.stdout.splitlines() if line.startswith(PREFIX)]
    if len(lines) != 1:
        raise ValueError("Reward environment check returned no unambiguous result")
    return json.loads(lines[0])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", nargs="?", choices=("flux", "sana", "sd15"), default="sd15")
    parser.add_argument("--worker", choices=("flux", "sana", "sd15"), help=argparse.SUPPRESS)
    args = parser.parse_args()
    try:
        if args.worker:
            print(PREFIX + json.dumps(inspect_worker(args.worker)))
        else:
            import torch
            import diffusers
            import transformers
            if not torch.cuda.is_available():
                raise ValueError("Generation requires CUDA-enabled PyTorch and an accessible GPU")
            interpreter = os.environ.get("SATISDIVE_REWARD_PYTHON")
            if not interpreter or not Path(interpreter).expanduser().is_file():
                raise ValueError("Set SATISDIVE_REWARD_PYTHON to the reward venv's Python executable")
            result = check_worker(str(Path(interpreter).expanduser().absolute()), args.model)
            print(json.dumps({"generation": {"torch": torch.__version__, "diffusers": diffusers.__version__, "transformers": transformers.__version__}, "reward": result}, indent=2))
        return 0
    except Exception as exc:
        print(f"Environment check failed: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
