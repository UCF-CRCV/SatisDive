"""SatisDive flux: ir worker. See README.md for the supported interface."""

from __future__ import annotations

import json
import sys
import time
import traceback
from pathlib import Path

# rewards/ on sys.path so the ImageReward compat shim (image_reward_compat)
# imports. Mirrors ir_grad_worker.py exactly.
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent  # rewards/
sys.path.insert(0, str(ROOT))

from PIL import Image

from image_reward_compat import load_image_reward


def _send(obj: dict) -> None:
    sys.stdout.write(json.dumps(obj) + "\n")
    sys.stdout.flush()


def _log(msg: str) -> None:
    """Log to stderr so it doesn't get mixed with the JSON protocol on stdout."""
    sys.stderr.write(f"[ir_worker] {msg}\n")
    sys.stderr.flush()


def main():
    _log("loading ImageReward...")
    t0 = time.time()
    try:
        model = load_image_reward()
    except Exception as e:
        _send({"error": f"failed to load ImageReward: {e}", "trace": traceback.format_exc()})
        sys.exit(1)
    _log(f"loaded in {time.time()-t0:.1f}s")
    _send({"ready": True})

    # Main request loop
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
            prompt = req["prompt"]
            paths = req["image_paths"]
            scores = []
            for p in paths:
                img = Image.open(p).convert("RGB")
                s = model.score(prompt, img)
                scores.append(float(s))
            _send({"scores": scores})
        except Exception as e:
            _send({"error": str(e), "trace": traceback.format_exc()})

    _log("exiting")


if __name__ == "__main__":
    main()
