"""Small public interface for the three fixed SatisDive generation presets."""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import re
import sys

from check_environment import check_worker

ROOT = Path(__file__).resolve().parent


def positive(value):
    value = float(value)
    if not math.isfinite(value) or value < 0:
        raise argparse.ArgumentTypeError("Delta must be finite and nonnegative")
    return value


def parser():
    p = argparse.ArgumentParser(description="Generate four images per prompt with a fixed SatisDive model preset.", allow_abbrev=False)
    p.add_argument("--model", choices=("sd15", "sana", "flux"), default="sd15")
    p.add_argument("--prompts", type=Path, default=ROOT / "example.jsonl", help="JSONL with prompt_id and prompt")
    p.add_argument("--output", type=Path, required=True, help="New output directory (existing directories are refused)")
    p.add_argument("--delta", type=positive, help="Override the preset's reward tolerance Delta")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--dry-run", action="store_true", help="Validate inputs and print the complete configuration without loading models")
    return p


def resolve(args):
    config = json.loads((ROOT / "presets" / f"{args.model}.json").read_text(encoding="utf-8"))
    prompts = args.prompts.expanduser().resolve(strict=True)
    output = args.output.expanduser().resolve()
    if output.exists():
        raise ValueError("Output directory already exists; use a new directory to avoid mixing runs")
    seen = set()
    for line in prompts.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict):
            raise ValueError("Each prompt record must be a JSON object")
        pid = row.get("prompt_id")
        if not isinstance(pid, str) or not re.fullmatch(r"[0-9]{1,9}", pid):
            raise ValueError("Each prompt_id must be a numeric string of 1 to 9 digits")
        if pid in seen or not isinstance(row.get("prompt"), str) or not row["prompt"].strip():
            raise ValueError("Prompt IDs must be unique and prompt text must be nonempty")
        seen.add(pid)
    if not seen:
        raise ValueError("Prompt file is empty")
    if args.seed < 0 or args.seed + max(map(int, seen)) >= 2**63:
        raise ValueError("Seed plus prompt ID must be in [0, 2**63)")
    config["geneval_metadata" if args.model == "flux" else "metadata"] = str(prompts)
    config["output_dir" if args.model == "sana" else "out_root"] = str(output)
    config["seed"] = args.seed
    if args.delta is not None:
        config["tau_relmax_offset"] = args.delta
    worker = os.environ.get("SATISDIVE_WORKER_GPU", str(config["worker_gpu"]))
    if not worker.isdigit():
        raise ValueError("SATISDIVE_WORKER_GPU must be a nonnegative logical GPU index")
    config["worker_gpu"] = int(worker)
    if args.model == "flux" and config["worker_gpu"] == 0:
        raise ValueError("The FLUX/HPSv3 preset requires a separate reward GPU (logical index 1 by default)")
    return config, prompts, output


def worker_environment(model, config):
    interpreter = os.environ.get("SATISDIVE_REWARD_PYTHON")
    if not interpreter or not Path(interpreter).expanduser().is_file():
        raise ValueError("Set SATISDIVE_REWARD_PYTHON to the reward environment's Python executable")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible:
        devices = [v.strip() for v in visible.split(",") if v.strip()]
        index = config["worker_gpu"]
        if index >= len(devices):
            raise ValueError("Reward GPU index is outside CUDA_VISIBLE_DEVICES")
        # FLUX already maps logical indices in its reward client; the other clients
        # expect physical IDs. Normalize here without changing sampling code.
        if model != "flux":
            config["worker_gpu"] = devices[index]
    prefix = {"flux": "FLUX_HPSV3", "sana": "SANA_IMAGEREWARD", "sd15": "SD15_IMAGEREWARD"}[model]
    # Do not resolve the executable symlink: a venv's bin/python often links to
    # system Python, and dereferencing it would discard the virtual environment.
    os.environ[prefix + "_PYTHON"] = str(Path(interpreter).expanduser().absolute())
    os.environ[prefix + ("_SITE" if model == "flux" else "_PYTHONPATH")] = ""
    if "HF_HOME" not in os.environ:
        os.environ["HF_HOME"] = str(Path.home() / ".cache/huggingface")


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        config, prompts, output = resolve(args)
        if args.dry_run:
            print(json.dumps(config, indent=2))
            return 0
        worker_environment(args.model, config)
        worker_info = check_worker(os.environ[{
            "flux": "FLUX_HPSV3_PYTHON", "sana": "SANA_IMAGEREWARD_PYTHON",
            "sd15": "SD15_IMAGEREWARD_PYTHON",
        }[args.model]], args.model)
        # Fail before loading the model if dependencies/CUDA are unavailable.
        import torch
        if not torch.cuda.is_available():
            raise ValueError("A CUDA-enabled generation environment and GPU are required")
        if not os.environ.get("CUDA_VISIBLE_DEVICES") and int(config["worker_gpu"]) >= torch.cuda.device_count():
            raise ValueError("Reward GPU index is not available")
        sys.path.insert(0, str(ROOT / "implementation" / args.model))
        sampler = importlib.import_module("ours")
        output.mkdir(parents=True, exist_ok=False)
        manifest = {"model_preset": args.model, "arguments": config, "prompts_sha256": hashlib.sha256(prompts.read_bytes()).hexdigest(), "torch": torch.__version__, "reward_environment": worker_info, "status": "running"}
        manifest_path = output / "run.json"
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        try:
            sampler.main(argparse.Namespace(**config))
        except BaseException:
            manifest["status"] = "failed"
            raise
        else:
            manifest["status"] = "complete"
        finally:
            manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        return 0
    except (ValueError, FileNotFoundError, ImportError, json.JSONDecodeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
