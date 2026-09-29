"""SatisDive flux: hpsv3 grad worker. See README.md for the supported interface."""

from __future__ import annotations

import json
import sys
import time
import traceback

import torch


def _send(obj: dict) -> None:
    sys.stdout.write(json.dumps(obj) + "\n")
    sys.stdout.flush()


def _log(msg: str) -> None:
    sys.stderr.write(f"[hpsv3_grad_worker] {msg}\n")
    sys.stderr.flush()


def _differentiable_reward(inferencer, image: torch.Tensor, prompt: str) -> torch.Tensor:
    """Evaluate one tensor image without the released inference-only wrapper."""
    batch = inferencer.prepare_batch([image], [prompt])
    return inferencer.model(return_dict=True, **batch)["logits"]


def main():
    _log("loading differentiable HPSv3...")
    t0 = time.time()
    try:
        from hpsv3 import HPSv3RewardInferencer
        inf = HPSv3RewardInferencer(device="cuda", differentiable=True)
    except Exception as e:
        _send({"error": f"failed to load differentiable HPSv3: {e}",
               "trace": traceback.format_exc()})
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
            image_pt = req["image_pt"]
            prompts = req["prompts"]
            out_pt = req["out_pt"]

            # Load image tensor (K,3,H,W) float [0,1] onto this process's GPU.
            imgs = torch.load(image_pt, map_location="cuda").float()
            if imgs.dim() == 3:
                imgs = imgs.unsqueeze(0)
            K = imgs.shape[0]
            assert len(prompts) == K, f"prompts {len(prompts)} != images {K}"

            rewards = []
            grads = torch.zeros_like(imgs)
            # Score one image at a time: HPSv3 (Qwen2-VL-7B) at high res OOMs if
            # K 1024^2 images are batched (same reason hpsv3_worker scores singly).
            #
            # CRITICAL (verified scripts/_das_reward_path_check.py): HPSv3's
            # differentiable reward() expects pixel values in [0,255], NOT [0,1].
            # Its processor applies rescale_factor=1/255 internally; feeding [0,1]
            # makes it see a ~0.004-range image and returns a constant floor
            # (~-10.2). We accept [0,1] from the flux side (natural VAE-decode
            # range) and multiply by 255 INSIDE the autograd graph, so the
            # returned gradient is correctly w.r.t. the [0,1] input (chain rule:
            # d r / d x_[0,1] = 255 * d r / d x_[0,255]).
            for i in range(K):
                xi = imgs[i].detach().clone().requires_grad_(True)  # (3,H,W) [0,1]
                xi255 = xi * 255.0
                # The released inference helper takes ``(prompts, images)`` and
                # is wrapped in ``torch.inference_mode()``, while the earlier
                # differentiable test helper took ``(images, prompts)``.  Use
                # the shared preparation/model path directly so gradients are
                # retained and this worker does not depend on either wrapper.
                out = _differentiable_reward(inf, xi255, prompts[i])
                # reward() returns (B,2): [miu, sigma]; miu (index 0) is the score.
                ri = out[0][0]
                gi = torch.autograd.grad(ri, xi, retain_graph=False)[0]
                rewards.append(float(ri.item()))
                grads[i] = gi.detach()

            torch.save(grads.cpu(), out_pt)
            _send({"rewards": rewards, "grad_pt": out_pt})
        except Exception as e:
            _send({"error": str(e), "trace": traceback.format_exc()})

    _log("exiting")


if __name__ == "__main__":
    main()
