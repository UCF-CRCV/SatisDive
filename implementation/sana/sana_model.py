"""SatisDive sana: sana model. See README.md for the supported interface."""
from __future__ import annotations

import json
from typing import Callable, Dict, List, Optional

import torch


DEFAULT_MODEL_ID = "Efficient-Large-Model/Sana_1600M_1024px_diffusers"


def load_prompts(metadata_path, prompt_ids=None):
    """Read a metadata jsonl ({"prompt_id", "prompt"} per line).

    Mirrors ``flux_model.load_prompts`` / ``sd15_model.load_prompts``: returns
    ``[{"id", "prompt"}, ...]`` and applies NO limit (the caller caps with
    ``--limit`` if it wants to). Accepts ``id`` as a fallback key and synthesises
    a zero-padded index if neither is present.
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


# Type of the per-step hook. Matches the diffusers pipeline callback contract
# ``(pipe, step, timestep, callback_kwargs) -> callback_kwargs`` (the pipeline
# does ``latents = callback_outputs.pop("latents", latents)`` after the call).
# ``callback_kwargs`` carries at least ``"latents"`` and (after our patch)
# ``"noise_pred"`` (the CFG-combined velocity v_θ for the just-taken step).
StepCallback = Callable[..., Dict]


class SanaModel:
    """Thin wrapper around ``SanaPipeline`` with a hook-able denoise loop.

    The wrapper owns the SanaTransformer / DC-AE VAE / Gemma text encoder /
    flow scheduler and exposes the pieces the later method ports need:
      - ``x0_from_velocity``  the one-step Tweedie x̂₀ (the floor term
                              differentiates the reward of this estimate);
      - ``decode``            latents → PIL images (final or x̂₀ proxy);
      - ``encode_prompt``     Gemma-encode a prompt to (embeds, mask), replicated
                              to K particles (for the later manual/grad paths);
      - ``predict_velocity``  a manual transformer call (for DAS/ours later);
      - ``denoise``           the K-candidate loop driven through the real
                              pipeline with a ``step_callback`` seam fired after
                              each scheduler step (single insertion point for the
                              later floor / repulsion / FK-resample / SDE hooks).
    """

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL_ID,
        device: str = "cuda",
        dtype: torch.dtype = torch.float16,
        local_files_only: bool = True,
        use_flow_euler: bool = False,
    ):
        # Imported lazily so this module parses without diffusers installed
        # (local static checks on the Mac; the model only loads on GPU).
        from diffusers import SanaPipeline

        self.device = torch.device(device)
        self.dtype = dtype

        # Cache-only load (no network). ``torch_dtype`` sets the transformer
        # (compute) dtype; the DC-AE VAE and Gemma text encoder must NOT be
        # fp16 (HF model card: "text encoder and VAE weights must stay in
        # torch.bfloat16 or torch.float32"), so we cast them to bf16 below.
        pipe = SanaPipeline.from_pretrained(
            model_id,
            torch_dtype=dtype,
            local_files_only=local_files_only,
        )


        if dtype == torch.float16:
            pipe.text_encoder.to(torch.bfloat16)
            pipe.vae.to(torch.bfloat16)

        pipe = pipe.to(self.device)

        # Optional scheduler swap: the shipped DPMSolverMultistepScheduler is a
        # stateful 2nd-order solver, which is faithful for BASE generation but
        # not a clean substrate for the FK-Flow marginal-preserving SDE (that
        # derivation assumes a 1st-order Euler ODE step). Swapping to
        # FlowMatchEulerDiscreteScheduler gives the clean Euler step the SDE
        # callback expects. Off by default (keeps the model-card sampler).
        if use_flow_euler:
            from diffusers import FlowMatchEulerDiscreteScheduler
            pipe.scheduler = FlowMatchEulerDiscreteScheduler.from_config(
                pipe.scheduler.config
            )
        self.use_flow_euler = use_flow_euler

        self.pipe = pipe
        self.transformer = pipe.transformer
        self.vae = pipe.vae
        self.text_encoder = pipe.text_encoder
        self.tokenizer = pipe.tokenizer
        self.scheduler = pipe.scheduler
        self.image_processor = pipe.image_processor
        # DC-AE: 32× spatial compression (vae_scale_factor == 32), 32 latent ch.
        self.vae_scale_factor = pipe.vae_scale_factor
        # SanaTransformer scales the timestep it receives; default 1.0 when the
        # config omits it (the 1600M config does).
        self.timestep_scale = float(
            getattr(self.transformer.config, "timestep_scale", 1.0)
        )

        # Expose the CFG-combined velocity to step callbacks (SDE / future FK)
        # by adding it to the pipeline's allow-list of callback tensor inputs.
        # The pipeline's denoise loop keeps the post-CFG prediction in a local
        # named ``noise_pred``; diffusers copies ``locals()[k]`` for each k in
        # ``callback_on_step_end_tensor_inputs`` — so once it is allow-listed a
        # callback can read ``kw["noise_pred"]`` = v_θ for the step just taken.
        if "noise_pred" not in self.pipe._callback_tensor_inputs:
            self.pipe._callback_tensor_inputs = list(
                self.pipe._callback_tensor_inputs
            ) + ["noise_pred"]

    # ------------------------------------------------------------------ encode
    def encode_prompt(self, prompt: str, k: int, do_cfg: bool):
        """Gemma-encode one prompt, replicated to K particles.

        Thin pass-through to ``SanaPipeline.encode_prompt`` (which prepends the
        "complex human instruction" in-context prefix, tokenizes with the Gemma
        tokenizer, runs the Gemma2 encoder, and selects the ``max_sequence_length``
        window). Returns the diffusers 4-tuple
        ``(prompt_embeds, prompt_attention_mask, neg_embeds, neg_attention_mask)``;
        with CFG on, callers stack ``[neg, pos]`` on dim 0 (the pipeline's own
        convention). Exposed for the later manual/grad (DAS/ours) paths; base
        drives ``self.pipe(prompt=...)`` directly and never calls this.
        """
        return self.pipe.encode_prompt(
            prompt,
            do_classifier_free_guidance=do_cfg,
            negative_prompt="",
            num_images_per_prompt=k,
            device=self.device,
        )

    # ---------------------------------------------------------------- velocity
    def predict_velocity(self, latents, t, prompt_embeds, prompt_attention_mask,
                         guidance_scale=4.5, do_cfg=True):
        """Manual v_θ(x_t, t) for SANA (flow / velocity prediction).

        Replicates the pipeline's transformer call + CFG combine + learned-sigma
        guard, for the later grad-enabled paths (DAS/ours) that cannot ride the
        ``@torch.no_grad`` pipeline. ``prompt_embeds`` / ``prompt_attention_mask``
        must already be the CFG-stacked ``[neg, pos]`` tensors when ``do_cfg``.

        SANA specifics (verified against pipeline_sana.py):
          - the transformer receives ``timestep = t * timestep_scale`` broadcast
            over the (CFG-doubled) batch;
          - CFG: ε = ε_uncond + s·(ε_text − ε_uncond);
          - learned-sigma guard: if ``out_channels // 2 == in_channels`` the
            model predicts (v, logvar) stacked on the channel dim and we keep the
            first half. For the 1600M model out=in=32, so the guard is inert, but
            we keep it for portability to other SANA sizes.
        """
        latent_model_input = torch.cat([latents] * 2) if do_cfg else latents
        timestep = t.expand(latent_model_input.shape[0]) * self.timestep_scale
        tdtype = self.transformer.dtype
        noise_pred = self.transformer(
            latent_model_input.to(dtype=tdtype),
            encoder_hidden_states=prompt_embeds.to(dtype=tdtype),
            encoder_attention_mask=prompt_attention_mask,
            timestep=timestep,
            return_dict=False,
        )[0]
        noise_pred = noise_pred.float()
        if do_cfg:
            uncond, text = noise_pred.chunk(2)
            noise_pred = uncond + guidance_scale * (text - uncond)
        latent_channels = self.transformer.config.in_channels
        if self.transformer.config.out_channels // 2 == latent_channels:
            noise_pred = noise_pred.chunk(2, dim=1)[0]
        return noise_pred

    # ------------------------------------------------------------------ x̂₀
    def sigma_at(self, step_index: int, x_ref: torch.Tensor) -> torch.Tensor:
        """Flow sigma σ at a scheduler ``step_index``, shaped to broadcast on x.

        Reads ``scheduler.sigmas`` (flow sigmas in [~1, 0], with ``flow_shift``
        already applied for the DPM/flow scheduler). Used by ``x0_from_velocity``.
        """
        sig = self.scheduler.sigmas[step_index].to(device=x_ref.device,
                                                    dtype=x_ref.dtype)
        while sig.ndim < x_ref.ndim:
            sig = sig.unsqueeze(-1)
        return sig

    def x0_from_velocity(self, x_t: torch.Tensor, v: torch.Tensor,
                         sigma: torch.Tensor) -> torch.Tensor:
        """One-step Tweedie x̂₀ for flow / velocity prediction (SANA, like FLUX).

            x̂₀ = x_t − σ_t · v_θ(x_t, t)

        This is the SANA analogue of SD1.5's ε-form x̂₀ = (x_t−√(1−ᾱ)ε)/√ᾱ; for
        rectified flow with x_s = (1−s)·x₀ + s·ε the marginal Tweedie mean is
        exactly x_s − s·v. ``sigma`` is the flow sigma at the frame of ``x_t``
        (use ``sigma_at``). Exposed for the later floor term (reward computed on
        the decode of this estimate); the base sampler does not call it.

        NOTE on the post-step frame (mirrors ``flux/baselines.py::_decode_latents``):
        the pipeline callback fires AFTER ``scheduler.step``, so ``kw["latents"]``
        is x_{t+dt} and ``kw["noise_pred"]`` is v_θ(x_t). The Tweedie estimate from
        the POST-step latent is x̂₀ = x_{t+dt} − σ_{t+dt}·v (use the sigma at the
        incremented ``step_index``), which is exact (σ→0 at the final step).
        """
        return x_t - sigma * v

    # ------------------------------------------------------------------ decode
    @torch.no_grad()
    def decode(self, latents: torch.Tensor) -> List["object"]:
        """Latents → list of PIL images (DC-AE, ``scaling_factor`` only).

        DC-AE has NO ``shift_factor`` (unlike the FLUX VAE): the pipeline decodes
        ``latents / vae.config.scaling_factor`` directly.
        """
        z = (latents / self.vae.config.scaling_factor).to(self.vae.dtype)
        image = self.vae.decode(z, return_dict=False)[0]
        return self.image_processor.postprocess(image, output_type="pil")

    # ----------------------------------------------------------------- denoise
    @torch.no_grad()
    def denoise(
        self,
        prompt: str,
        k: int = 4,
        num_steps: int = 20,
        seed: int = 0,
        guidance_scale: float = 4.5,
        height: int = 1024,
        width: int = 1024,
        step_callback: Optional[StepCallback] = None,
        complex_human_instruction: Optional[list] = None,
    ) -> List["object"]:
        """Run the K-candidate SANA denoising loop for one prompt.

        K candidates are carried through a single pipeline run together (one
        prompt with ``num_images_per_prompt=k``), with K independent initial
        latents drawn from a per-prompt seeded generator — the source of base
        K-diversity (same idea as sd15 base; SANA needs no SDE to be diverse
        across the K noises, though the SDE option breaks apart FK duplicates
        later).

        The ``step_callback`` seam is the pipeline's native
        ``callback_on_step_end``: it fires AFTER each ``scheduler.step`` with a
        dict carrying the post-step ``latents`` and (via our patch) the
        CFG-combined ``noise_pred`` (= v_θ). It must return the (possibly
        modified) dict — exactly the contract ``flux/baselines.py`` composes SDE
        + steering callbacks onto. Base passes ``step_callback=None``.

        Returns a list of ``k`` PIL images.
        """
        generator = torch.Generator(device=self.device).manual_seed(seed)

        # SANA's __call__ default for complex_human_instruction is a long
        # in-context prefix; ``None`` here means "use the pipeline default".
        # Pass ``[]`` (empty) to disable the instruction prefix entirely.
        chi = (complex_human_instruction
               if complex_human_instruction is not None else _CHI_DEFAULT)

        output = self.pipe(
            prompt=prompt,
            negative_prompt="",
            num_images_per_prompt=k,
            num_inference_steps=num_steps,
            guidance_scale=guidance_scale,
            height=height,
            width=width,
            generator=generator,
            output_type="pil",
            complex_human_instruction=chi,
            callback_on_step_end=step_callback,
            callback_on_step_end_tensor_inputs=(
                ["latents", "noise_pred"] if step_callback is not None else ["latents"]
            ),
        )
        return list(output.images)


# The SanaPipeline ``complex_human_instruction`` default (verified against the
# diffusers pipeline_sana.py signature). Kept here so ``denoise`` can pass it
# explicitly (and so callers can inspect / override / disable it).
_CHI_DEFAULT = [
    "Given a user prompt, generate an 'Enhanced prompt' that provides detailed "
    "visual descriptions suitable for image generation. Evaluate the level of "
    "detail in the user prompt:",
    "- If the prompt is simple, focus on adding specifics about colors, shapes, "
    "sizes, textures, and spatial relationships to create vivid and concrete scenes.",
    "- If the prompt is already detailed, refine and enhance the existing details "
    "slightly without overcomplicating.",
    "Here are examples of how to transform or refine prompts:",
    "- User Prompt: A cat sleeping -> Enhanced: A small, fluffy white cat curled up "
    "in a round shape, sleeping peacefully on a warm sunny windowsill, surrounded by "
    "pots of blooming red flowers.",
    "- User Prompt: A busy city street -> Enhanced: A bustling city street scene at "
    "dusk, featuring glowing street lamps, a diverse crowd of people in colorful "
    "clothing, and a double-decker bus passing by towering glass skyscrapers.",
    "Please generate only the enhanced description for the prompt below and avoid "
    "including any additional commentary or evaluations:",
    "User Prompt: ",
]
