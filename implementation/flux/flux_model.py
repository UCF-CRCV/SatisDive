"""SatisDive flux: flux model. See README.md for the supported interface."""
from __future__ import annotations

import json

import torch

def load_prompts(metadata_path, prompt_ids):
    prompts = []
    with open(metadata_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            pid = str(r.get("prompt_id", r.get("id", f"{len(prompts):05d}")))
            if prompt_ids is not None and pid not in prompt_ids:
                continue
            prompts.append({"id": pid, "prompt": r.get("prompt", "")})
    return prompts


def build_prompt_state(pipe, prompt, k, height, width, device):
    """Conditioning for manual FLUX transformer calls (mirrors _build_prompt_state)."""
    prompt_embeds, pooled, text_ids = pipe.encode_prompt(
        prompt=prompt, prompt_2=None, device=device,
        num_images_per_prompt=k, max_sequence_length=512,
    )
    h_pack = 2 * (int(height) // (pipe.vae_scale_factor * 2))
    w_pack = 2 * (int(width) // (pipe.vae_scale_factor * 2))
    latent_image_ids = pipe._prepare_latent_image_ids(
        k, h_pack // 2, w_pack // 2, device, prompt_embeds.dtype)
    if pipe.transformer.config.guidance_embeds:
        guidance = torch.full([1], 3.5, device=device, dtype=torch.float32).expand(k)
    else:
        guidance = None
    return {
        "prompt_embeds": prompt_embeds, "pooled_prompt_embeds": pooled,
        "text_ids": text_ids, "latent_image_ids": latent_image_ids,
        "guidance": guidance, "joint_attention_kwargs": None,
    }


def predict_velocity(pipe, latents, t, ps):
    """v_theta(x_t, t) for FLUX (guidance-distilled, no CFG split)."""
    timestep = t.expand(latents.shape[0]).to(latents.dtype)
    return pipe.transformer(
        hidden_states=latents,
        timestep=timestep / 1000,
        guidance=ps["guidance"],
        pooled_projections=ps["pooled_prompt_embeds"],
        encoder_hidden_states=ps["prompt_embeds"],
        txt_ids=ps["text_ids"],
        img_ids=ps["latent_image_ids"],
        joint_attention_kwargs=ps["joint_attention_kwargs"],
        return_dict=False,
    )[0]


def decode_to_image(pipe, x0_packed, height, width):
    """Unpack + VAE-decode a packed x0 latent to image in [0,1], (K,3,H,W).

    Differentiable in x0_packed (no torch.no_grad), for the reward-gradient path.
    """
    z = x0_packed
    eh = (height // pipe.vae_scale_factor) * pipe.vae_scale_factor
    ew = (width // pipe.vae_scale_factor) * pipe.vae_scale_factor
    if hasattr(pipe, "_unpack_latents") and z.ndim == 3:
        z = pipe._unpack_latents(z, eh, ew, pipe.vae_scale_factor)
    if getattr(pipe.vae.config, "shift_factor", None) is not None:
        z = (z / pipe.vae.config.scaling_factor) + pipe.vae.config.shift_factor
    else:
        z = z / pipe.vae.config.scaling_factor
    im = pipe.vae.decode(z.to(pipe.vae.dtype), return_dict=False)[0]  # (K,3,H,W) ~[-1,1]
    im = (im / 2 + 0.5).clamp(0, 1)
    return im


def _slice_ps(ps, kk):
    """Slice prompt-state to a single particle kk (embeddings are batch-replicated)."""
    out = dict(ps)
    out["prompt_embeds"] = ps["prompt_embeds"][kk:kk + 1]
    out["pooled_prompt_embeds"] = ps["pooled_prompt_embeds"][kk:kk + 1]
    # text_ids / latent_image_ids are shared (FLUX uses per-token id grids, not batched)
    if ps["guidance"] is not None:
        out["guidance"] = ps["guidance"][kk:kk + 1]
    return out

