"""SatisDive sd15: ir worker. See README.md for the supported interface."""

from __future__ import annotations

import json
import sys
import time
import traceback
from pathlib import Path

# rewards/ on sys.path so the ImageReward compat shim imports.
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

from image_reward_compat import load_image_reward


def _send(obj: dict) -> None:
    sys.stdout.write(json.dumps(obj) + "\n")
    sys.stdout.flush()


def _log(msg: str) -> None:
    sys.stderr.write(f"[ir_worker] {msg}\n")
    sys.stderr.flush()


def main():
    _log("loading ImageReward (score-only)...")
    t0 = time.time()
    try:
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = load_image_reward().to(device)
        model.eval()
    except Exception as e:
        _send({"error": f"failed to load ImageReward: {e}",
               "trace": traceback.format_exc()})
        sys.exit(1)
    _log(f"loaded in {time.time()-t0:.1f}s on {device}")
    _send({"ready": True})

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except Exception as e:
            _send({"error": f"bad JSON: {e}"})
            continue

        if req.get("shutdown"):
            _log("shutdown requested")
            _send({"shutdown_ack": True})
            break

        try:
            from PIL import Image
            prompt = req["prompt"]
            started = time.perf_counter()
            if "uint8_npy" in req:
                import numpy as np
                from batched_image_reward import score_images
                pixels = np.load(req["uint8_npy"], allow_pickle=False)
                if pixels.dtype != np.uint8 or pixels.ndim != 4 or pixels.shape[-1] != 3:
                    raise ValueError("Expected NHWC uint8 RGB images")
                images = [Image.fromarray(row) for row in pixels]
                scores = score_images(model, prompt, images, int(req.get("batch_size", 8)))
            else:
                scores = []
                for p in req["image_paths"]:
                    with Image.open(p) as image:
                        scores.append(float(model.score(prompt, image.convert("RGB"))))
            _send({"scores": scores, "worker_seconds": time.perf_counter() - started})
        except Exception as e:
            _send({"error": str(e), "trace": traceback.format_exc()})

    _log("exiting")


if __name__ == "__main__":
    main()
