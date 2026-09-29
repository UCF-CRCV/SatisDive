"""SatisDive sd15: sd15 model. See README.md for the supported interface."""
from __future__ import annotations

import json
from typing import Callable, List, Optional

import torch


DEFAULT_MODEL_ID = "stable-diffusion-v1-5/stable-diffusion-v1-5"


def load_prompts(metadata_path, prompt_ids=None):
    """Read a metadata jsonl ({"prompt_id", "prompt"} per line).

    Mirrors ``flux_model.load_prompts``: returns ``[{"id", "prompt"}, ...]`` and
    applies NO limit (the caller caps with ``--limit`` if it wants to). Accepts
    ``id`` as a fallback key and synthesises a zero-padded index if neither is
    present.
    """
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


# Type of the per-step hook. Returns either ``None`` (keep the loop's latents)
# or a replacement latents tensor (e.g. FK resampling / floor / repulsion).
StepCallback = Callable[..., Optional[torch.Tensor]]


class SD15Model:
    """Thin wrapper around ``StableDiffusionPipeline`` with an explicit,
    hook-able denoise loop.

    The wrapper owns the UNet / VAE / CLIP / DDIM scheduler and exposes three
    things the later method ports need:
      - ``x0_from_eps``  the one-step Tweedie x̂₀ (the floor term differentiates
                         the reward of this estimate);
      - ``decode``       latents → PIL images (final or x̂₀ proxy);
      - ``denoise``      the K-candidate denoising loop with a ``step_callback``
                         seam fired at every step (the scored-step gate lives in
                         the callback, mirroring FLUX's composed callbacks).
    """

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL_ID,
        revision: str | None = None,
        device: str = "cuda",
        dtype: torch.dtype = torch.float16,
        local_files_only: bool = True,
    ):
        # Imported lazily so this module parses without diffusers installed
        # (local static checks on the Mac; the model only loads on GPU).
        from diffusers import StableDiffusionPipeline, DDIMScheduler

        self.device = torch.device(device)
        self.dtype = dtype

        # Cache-only load (no network); drop the NSFW checker — we score with our
        # own reward models and the checker only wastes VRAM + can blank images.
        pipe = StableDiffusionPipeline.from_pretrained(
            model_id,
            revision=revision,
            torch_dtype=dtype,
            local_files_only=local_files_only,
            safety_checker=None,
            requires_safety_checker=False,
        )
        # Swap to DDIM (ε-prediction, deterministic at η=0) to match the FK
        # reference + the verified one-step x̂₀ identity.
        pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
        pipe = pipe.to(self.device)

        self.pipe = pipe
        self.unet = pipe.unet
        self.vae = pipe.vae
        self.text_encoder = pipe.text_encoder
        self.tokenizer = pipe.tokenizer
        self.scheduler = pipe.scheduler
        self.image_processor = pipe.image_processor
        self.vae_scale_factor = pipe.vae_scale_factor

    # ------------------------------------------------------------------ encode
    def encode_prompt(self, prompt: str, k: int, do_cfg: bool):
        """CLIP-encode one prompt, replicated to K particles.

        Returns ``prompt_embeds`` of shape ``(K, L, D)`` when CFG is off, or
        ``(2K, L, D)`` with the negative ("") embeddings stacked first when CFG
        is on (the diffusers convention consumed by ``UNet`` after a
        ``torch.cat([latents]*2)``).
        """
        neg = "" if do_cfg else None
        prompt_embeds, negative_embeds = self.pipe.encode_prompt(
            prompt=prompt,
            device=self.device,
            num_images_per_prompt=k,
            do_classifier_free_guidance=do_cfg,
            negative_prompt=neg,
        )
        if do_cfg:
            return torch.cat([negative_embeds, prompt_embeds], dim=0)
        return prompt_embeds

    # ------------------------------------------------------------------ x̂₀
    def x0_from_eps(self, x_t: torch.Tensor, eps: torch.Tensor, t) -> torch.Tensor:
        """One-step Tweedie x̂₀ for ε-prediction (SD1.5 / DDIM).

            x̂₀ = (x_t − √(1 − ᾱ_t) · ε) / √(ᾱ_t)

        ``t`` is the current (integer) diffusion timestep. ᾱ_t is read from the
        scheduler's ``alphas_cumprod`` table. Exposed for the later floor term
        (the reward is computed on the decode of this estimate); the base
        sampler does not call it.
        """
        ac = self.scheduler.alphas_cumprod.to(device=x_t.device, dtype=x_t.dtype)
        a_t = ac[t].view(-1, *([1] * (x_t.ndim - 1)))
        return (x_t - (1.0 - a_t).sqrt() * eps) / a_t.sqrt()

    # ------------------------------------------------------------------ decode
    @torch.no_grad()
    def decode(self, latents: torch.Tensor) -> List["object"]:
        """Latents → list of PIL images (SD1.5 4-ch VAE, scaling-factor only)."""
        z = (latents / self.vae.config.scaling_factor).to(self.vae.dtype)
        image = self.vae.decode(z, return_dict=False)[0]
        return self.image_processor.postprocess(image, output_type="pil")

    # ----------------------------------------------------------------- denoise
    @torch.no_grad()
    def denoise(
        self,
        prompt: str,
        k: int = 4,
        num_steps: int = 50,
        seed: int = 0,
        guidance_scale: float = 7.5,
        height: int = 512,
        width: int = 512,
        eta: float = 0.0,
        step_callback: Optional[StepCallback] = None,
    ) -> List["object"]:
        """Run the K-candidate DDIM denoising loop for one prompt.

        K candidates are carried through a single run together (one prompt
        replicated to ``num_images_per_prompt=k``), with K independent initial
        noises drawn from a per-prompt seeded generator — this is what makes the
        reward-agnostic base batch diverse (no SDE needed, unlike FLUX base).

        The ``step_callback`` seam fires AFTER each ``scheduler.step`` with the
        post-step latents and the one-step x̂₀, exactly where the FK reference
        resamples (``fkd_class.FKD.resample`` consumes ``x0_preds``). It is the
        single insertion point for the later floor / repulsion / FK-resample
        hooks. Base passes ``step_callback=None`` and so computes no x̂₀.

        Returns a list of ``k`` PIL images.
        """
        do_cfg = guidance_scale > 1.0
        generator = torch.Generator(device=self.device).manual_seed(seed)

        prompt_embeds = self.encode_prompt(prompt, k, do_cfg)

        # K independent initial latents (the source of base K-diversity).
        num_channels = self.unet.config.in_channels
        shape = (
            k,
            num_channels,
            height // self.vae_scale_factor,
            width // self.vae_scale_factor,
        )
        latents = torch.randn(
            shape, generator=generator, device=self.device, dtype=prompt_embeds.dtype
        )

        self.scheduler.set_timesteps(num_steps, device=self.device)
        latents = latents * self.scheduler.init_noise_sigma

        # Extra DDIM step kwargs (eta only consumed when η > 0).
        extra_step_kwargs = {}
        if eta and "eta" in self.scheduler.step.__code__.co_varnames:
            extra_step_kwargs["eta"] = eta
            extra_step_kwargs["generator"] = generator

        for i, t in enumerate(self.scheduler.timesteps):
            x_t = latents  # pre-step state, used for the x̂₀ identity below
            model_input = torch.cat([x_t] * 2) if do_cfg else x_t
            model_input = self.scheduler.scale_model_input(model_input, t)

            eps = self.unet(
                model_input,
                t,
                encoder_hidden_states=prompt_embeds,
                return_dict=False,
            )[0]

            if do_cfg:
                eps_uncond, eps_text = eps.chunk(2)
                eps = eps_uncond + guidance_scale * (eps_text - eps_uncond)

            step_out = self.scheduler.step(
                eps, t, x_t, **extra_step_kwargs, return_dict=True
            )
            latents = step_out.prev_sample

            if step_callback is not None:
                # x̂₀ from the pre-step latents + CFG ε (our own identity, not
                # the scheduler's pred_original_sample, so the seam matches what
                # the floor term will differentiate).
                x0 = self.x0_from_eps(x_t, eps, t)
                new_latents = step_callback(
                    step=i,
                    t=t,
                    latents=latents,
                    x0=x0,
                    eps=eps,
                    model=self,
                )
                if new_latents is not None:
                    latents = new_latents

        return self.decode(latents)
