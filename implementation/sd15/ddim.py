"""Shared DDIM mechanics, extracted from the existing DAS port. See THIRD_PARTY.md."""
import torch

def _ddim_mean_and_variance(scheduler, eps, t, x_t, eta):
    """Return (prev_sample_mean, base_variance, prev_timestep) for one DDIM step.

    Standard η-DDIM (diffusers DDIMScheduler / DAS ddim_with_logprob):
        x̂₀  = (x_t − √(1−ᾱ_t)·ε) / √ᾱ_t
        var  = (1−ᾱ_{t-1})/(1−ᾱ_t) · (1 − ᾱ_t/ᾱ_{t-1})        [get_variance]
        σ_t  = η·√var
        dir  = √(1 − ᾱ_{t-1} − σ_t²) · ε
        mean = √ᾱ_{t-1}·x̂₀ + dir
    The proposal then forms prev_sample = mean + σ_t·noise and shifts by
    variance(=σ_t²)·approx_guidance. We return ``base_variance`` (= var) so the
    caller forms σ_t² = η²·var exactly as the reference does.
    """
    num_train = scheduler.config.num_train_timesteps
    n_steps = scheduler.num_inference_steps
    prev_t = t - num_train // n_steps

    acp = scheduler.alphas_cumprod.to(device=x_t.device, dtype=torch.float32)
    final_acp = torch.tensor(getattr(scheduler, "final_alpha_cumprod", acp[0]),
                             device=x_t.device, dtype=torch.float32)
    a_t = acp[int(t)]
    a_prev = acp[int(prev_t)] if int(prev_t) >= 0 else final_acp

    eps32 = eps.to(torch.float32)
    x32 = x_t.to(torch.float32)
    x0 = (x32 - (1.0 - a_t).sqrt() * eps32) / a_t.sqrt()

    base_variance = (1.0 - a_prev) / (1.0 - a_t) * (1.0 - a_t / a_prev)
    sigma = eta * base_variance.sqrt()
    direction = (1.0 - a_prev - sigma ** 2).clamp_min(0.0).sqrt() * eps32
    mean = a_prev.sqrt() * x0 + direction
    return mean, base_variance, int(prev_t)


def _predict_eps_cfg(model, x, t, embeds_2, guidance_scale):
    """CFG ε prediction for one particle. embeds_2 = cat([neg, pos]) (2,L,D)."""
    do_cfg = guidance_scale > 1.0
    model_input = torch.cat([x] * 2) if do_cfg else x
    model_input = model.scheduler.scale_model_input(model_input, t)
    eps = model.unet(model_input, t, encoder_hidden_states=embeds_2,
                     return_dict=False)[0]
    if do_cfg:
        eps_uncond, eps_text = eps.chunk(2)
        eps = eps_uncond + guidance_scale * (eps_text - eps_uncond)
    return eps


def _decode_to_image(model, latents):
    """Differentiable VAE decode → (K,3,H,W) image in [0,1]. NOT under no_grad."""
    z = (latents / model.vae.config.scaling_factor).to(model.vae.dtype)
    image = model.vae.decode(z, return_dict=False)[0]  # (K,3,H,W) in [-1,1]
    return (image / 2 + 0.5).clamp(0, 1)
