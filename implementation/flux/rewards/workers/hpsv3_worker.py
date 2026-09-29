"""SatisDive flux: hpsv3 worker. See README.md for the supported interface."""

from __future__ import annotations

import json
import sys
import time
import traceback


def _send(obj: dict) -> None:
    sys.stdout.write(json.dumps(obj) + "\n")
    sys.stdout.flush()


def _log(msg: str) -> None:
    sys.stderr.write(f"[hpsv3_worker] {msg}\n")
    sys.stderr.flush()


def main():
    _log("loading HPSv3...")
    t0 = time.time()
    try:
        from hpsv3 import HPSv3RewardInferencer
        inferencer = HPSv3RewardInferencer(device="cuda")
    except Exception as e:
        _send({"error": f"failed to load HPSv3: {e}", "trace": traceback.format_exc()})
        sys.exit(1)
    _log(f"loaded in {time.time()-t0:.1f}s")
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
            prompt = req["prompt"]
            paths = req["image_paths"]
            # Score images one at a time. HPSv3 (Qwen2-VL-7B) processes high-
            # resolution image tokens through 28 transformer layers; batching
            # K=4 1024x1024 images at once OOMs on an 80GB GPU. Per-image
            # scoring stays under ~25GB and finishes in ~1.7s/image.
            scores = []
            for p in paths:


                rewards = inferencer.reward([prompt], [p])
                scores.append(float(rewards[0][0].item()))
            _send({"scores": scores})
        except Exception as e:
            _send({"error": str(e), "trace": traceback.format_exc()})

    _log("exiting")


if __name__ == "__main__":
    main()
