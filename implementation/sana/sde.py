"""Marginal-preserving flow SDE step; Euler discretization."""
import math

def fk_flow_kernel_mean(latents, v, s_curr, ds, sde_a):
    """Return the corrected Euler mean and scalar noise standard deviation."""
    # Deterministic Euler part: x + ds*v (ds < 0). x_pre = current latents.
    euler_mean = latents + ds * v
    sigma_sde = sde_a * math.sqrt(max(s_curr * (1.0 - s_curr), 0.0))
    if sigma_sde <= 0.0 or s_curr <= 1e-3 or s_curr >= 1.0 - 1e-3:
        # boundary: no SDE (matches FKFlowSDECallback's boundary skip)
        return euler_mean, 0.0
    # score = -(x_pre + (1-s)*v)/s ; drift = -ds*(sigma_sde^2/2)*score
    score = -(latents + (1.0 - s_curr) * v) / s_curr
    drift = -ds * (sigma_sde ** 2 / 2.0) * score
    mean = euler_mean + drift
    noise_std = sigma_sde * math.sqrt(abs(ds))
    return mean, noise_std
