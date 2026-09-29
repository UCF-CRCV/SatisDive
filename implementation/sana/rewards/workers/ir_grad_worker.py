"""SatisDive sana: ir grad worker. See README.md for the supported interface."""

from __future__ import annotations

import json
import sys
import time
import traceback
from pathlib import Path

# rewards/ on sys.path so the ImageReward compat shim (image_reward_compat) imports.
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

import torch
import torch.nn.functional as F

from image_reward_compat import load_image_reward

# CLIP preprocessing constants (must match ImageReward._transform / BLIP).
_CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
_CLIP_STD = (0.26862954, 0.26130258, 0.27577711)
_RES = 224


def _send(obj: dict) -> None:
    sys.stdout.write(json.dumps(obj) + "\n")
    sys.stdout.flush()


def _log(msg: str) -> None:
    sys.stderr.write(f"[ir_grad_worker] {msg}\n")
    sys.stderr.flush()


def main():
    _log("loading differentiable ImageReward...")
    t0 = time.time()
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = load_image_reward().to(device)
        model.eval()
        mean = torch.tensor(_CLIP_MEAN, device=device).view(1, 3, 1, 1)
        std = torch.tensor(_CLIP_STD, device=device).view(1, 3, 1, 1)
    except Exception as e:
        _send({"error": f"failed to load differentiable ImageReward: {e}",
               "trace": traceback.format_exc()})
        sys.exit(1)
    _log(f"loaded in {time.time()-t0:.1f}s on {device}")
    _send({"ready": True})

    def _preprocess(img):
        # img: (1,3,H,W) float in [0,1]. Differentiable resize+crop+normalize,
        # mirroring torchvision Resize(224, BICUBIC)+CenterCrop(224)+Normalize.
        _, _, H, W = img.shape
        if H <= W:
            newh, neww = _RES, int(round(_RES * W / H))
        else:
            newh, neww = int(round(_RES * H / W)), _RES
        img = F.interpolate(img, size=(newh, neww), mode="bicubic",
                            align_corners=False, antialias=True)
        top = (newh - _RES) // 2
        left = (neww - _RES) // 2
        img = img[:, :, top:top + _RES, left:left + _RES]
        return (img - mean) / std

    def _reward(img01, prompt):
        # img01: (1,3,H,W) [0,1] with requires_grad. Returns scalar reward tensor.
        ti = model.blip.tokenizer(prompt, padding="max_length", truncation=True,
                                  max_length=35, return_tensors="pt").to(img01.device)
        x = _preprocess(img01)
        r = model.score_gard(ti.input_ids, ti.attention_mask, x)  # (1,1) normalized
        return r.reshape(())

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

            dev = next(model.parameters()).device
            imgs = torch.load(image_pt, map_location=dev).float()
            if imgs.dim() == 3:
                imgs = imgs.unsqueeze(0)
            K = imgs.shape[0]
            assert len(prompts) == K, f"prompts {len(prompts)} != images {K}"

            rewards = []
            grads = torch.zeros_like(imgs)
            # One image at a time so each gradient is exact and memory is bounded.
            for i in range(K):
                xi = imgs[i:i + 1].detach().clone().requires_grad_(True)  # (1,3,H,W) [0,1]
                ri = _reward(xi, prompts[i])
                gi = torch.autograd.grad(ri, xi, retain_graph=False)[0]
                rewards.append(float(ri.item()))
                grads[i] = gi.detach()[0]

            torch.save(grads.cpu(), out_pt)
            _send({"rewards": rewards, "grad_pt": out_pt})
        except Exception as e:
            _send({"error": str(e), "trace": traceback.format_exc()})

    _log("exiting")


if __name__ == "__main__":
    main()
