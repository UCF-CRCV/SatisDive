"""SatisDive sd15: ours. See README.md for the supported interface."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from sd15_model import SD15Model, load_prompts, DEFAULT_MODEL_ID
# Reuse the verified SD1.5/DDIM mechanics from the DAS port (single source of
# truth for the eps-pred CFG forward, the eta-DDIM mean+variance, and the
# differentiable VAE decode). Importing das.py only defines functions (its main()
# is __main__-guarded), so this is side-effect-free.
from ddim import _ddim_mean_and_variance, _predict_eps_cfg, _decode_to_image
from rewards.clients import RewardClient

ROOT = Path(__file__).resolve().parent


def log(m):
    print(f"[ours_sd15] {m}", flush=True)


# ----------------------------------------------------------------------------
# Diversity feature capture: a forward hook on the UNet mid-block (bottleneck).
# Keeps the autograd graph (the hook receives the live output tensor), so the
# diversity channel can backprop phi(x) -> x. Fully additive: registered from
# here, sd15_model.py is untouched.
# ----------------------------------------------------------------------------
class _MidBlockCapture:
    """Stores the UNet mid-block output of the CURRENT forward (graph kept).

    The mid-block (UNetMidBlock2DCrossAttn) output is (B, C, h, w) -- the deepest
    semantic feature map of the SD1.5 UNet. This is the SD1.5 analogue of FLUX's
    joint-block-12 features (the verified diversity geometry: diversity that moves
    the perceptual metric lives in network FEATURE space, not latent/pixel space).
    """

    def __init__(self):
        self.last = None

    def __call__(self, module, inputs, output):
        # ``output`` is the live tensor returned by mid_block.forward -- part of
        # the autograd graph under enable_grad, detached under no_grad. Mirrors
        # flux _Block12Capture.last_features.
        self.last = output

    def pooled(self, do_cfg: bool) -> torch.Tensor:
        """Mean-pooled feature of the CFG-conditional half. Returns (n, C)."""
        f = self.last.float()                       # (B, C, h, w)
        if do_cfg:
            f = f[f.shape[0] // 2:]                  # conditional (text) half
        return f.mean(dim=(2, 3))                    # (n, C) spatial mean-pool


def _cos_dist(feats: torch.Tensor) -> torch.Tensor:
    """(K,K) cosine DISTANCE (1 - cosine sim) between (K, dim) feature vectors.

    Cosine (not Euclidean) is scale-invariant across prompts and matches the
    verified Repel geometry (identical to flux/ours.py::_cos_dist)."""
    fn = F.normalize(feats, dim=-1)
    sim = fn @ fn.t()
    return (1.0 - sim).clamp_min(0.0)


def _flat(x: torch.Tensor) -> torch.Tensor:
    return x.reshape(x.shape[0], -1).float()


def run_ours_one_prompt(model, client, capture, prompt, k, num_steps, guidance_scale,
                        height, width, seed, score_steps, *, eta, beta_r, beta_d,
                        delta, lam_r, lam_d, step_size, max_step,
                        tau_relmax_offset, delta_scale, reward_ramp,
                        clone_dud, clone_start_p, verbose=False):
    """Generate one batch with reward-floor and gated-diversity updates.

    Returns images, reward traces, and per-step diagnostics. Reward and diversity
    gradients are normalized separately before the latent update. Late-stage
    replacement uses a randomly selected candidate above the current cutoff.
    Null calibration arguments use first-scoring-step batch statistics.
    """
    import numpy as np
    import time as _time

    channel_sched = "separate"
    device = model.device
    dtype = model.dtype
    sched = model.scheduler
    do_cfg = guidance_scale > 1.0

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    # [neg, pos] CLIP embeddings for ONE particle (shared across particles).
    embeds_2 = model.encode_prompt(prompt, k=1, do_cfg=do_cfg)

    gen = torch.Generator(device=device).manual_seed(seed)
    num_ch = model.unet.config.in_channels
    shape = (k, num_ch, height // model.vae_scale_factor, width // model.vae_scale_factor)
    latents = torch.randn(shape, generator=gen, device=device, dtype=dtype)

    sched.set_timesteps(num_steps, device=device)
    latents = latents * sched.init_noise_sigma
    timesteps = sched.timesteps

    rewards_trace = []
    diag = []
    cal = {"beta_r": beta_r, "beta_d": beta_d,
           "delta": delta, "max_step": max_step, "done": False}
    n_fwd = 0
    n_decode = 0
    prev_score_x0 = None
    prev_score_ranks = None
    prev_clone_r = None
    t_run0 = _time.time()

    for i, t in enumerate(timesteps):
        latents = latents.detach()

        eps_all = torch.zeros_like(latents)
        guidance = torch.zeros_like(latents)
        rewards = torch.zeros(k, device=device)

        is_score = i in score_steps
        if is_score:
            t_step0 = _time.time()
            # ---- PASS A (no grad): per particle, get reward, d r/d image, and the
            #      DETACHED mid-block features (the diversity geometry).
            dimg_rew, feat_rows = [], []
            x0_cur_rows = []
            img_min = img_max = None
            with torch.no_grad():
                for kk in range(k):
                    eps_k = _predict_eps_cfg(model, latents[kk:kk + 1], t, embeds_2,
                                             guidance_scale).float()
                    n_fwd += 1
                    feat_k = capture.pooled(do_cfg).detach().float()   # (1, C)
                    x0k = model.x0_from_eps(latents[kk:kk + 1].float(), eps_k, t)
                    x0_cur_rows.append(x0k.detach())
                    imgk = _decode_to_image(model, x0k.to(dtype)).float()  # (1,3,H,W)[0,1]
                    n_decode += 1
                    bmin, bmax = float(imgk.min()), float(imgk.max())
                    img_min = bmin if img_min is None else min(img_min, bmin)
                    img_max = bmax if img_max is None else max(img_max, bmax)
                    rew_k, dimg_k = client.reward_and_grad(imgk, [prompt])
                    rewards[kk] = float(rew_k[0])
                    eps_all[kk] = eps_k.detach().to(dtype)[0]
                    dimg_k = torch.nan_to_num(dimg_k.to(imgk.device, imgk.dtype))
                    dimg_rew.append(dimg_k)
                    feat_rows.append(feat_k)

            feat_det = torch.cat(feat_rows, 0)                 # (K, C) detached features
            r = rewards.float()
            dmat = _cos_dist(feat_det)                         # (K,K) feature cosine dist
            iu = torch.triu_indices(k, k, offset=1, device=device)

            # cross-step proxy drift (diagnostic): how much each x̂₀ moved.
            x0_cur = torch.cat(x0_cur_rows, 0)
            if prev_score_x0 is not None and prev_score_x0.shape == x0_cur.shape:
                x0_drift = (x0_cur - prev_score_x0).flatten(1).norm(dim=1)
                x0_drift_rel = (x0_drift / x0_cur.flatten(1).norm(dim=1).clamp_min(1e-8)).tolist()
            else:
                x0_drift_rel = None
            prev_score_x0 = x0_cur.detach()
            order = torch.argsort(r, descending=True)
            ranks = torch.empty(k, dtype=torch.long); ranks[order] = torch.arange(k)
            rank_churn = (int((ranks != prev_score_ranks).sum())
                          if prev_score_ranks is not None else -1)
            prev_score_ranks = ranks.clone()

            # ---- per-prompt calibration at the first scoring step (no magic const) ----
            if not cal["done"]:
                if cal["beta_r"] is None:
                    cal["beta_r"] = max(float(r.std()), 1e-3)
                if cal["delta"] is None:
                    cal["delta"] = float(dmat[iu[0], iu[1]].median()) * delta_scale
                if cal["beta_d"] is None:
                    cal["beta_d"] = max(0.5 * cal["delta"], 1e-3)
                if cal["max_step"] is None:
                    cal["max_step"] = 0.1 * float(_flat(latents).norm(dim=1).mean())
                cal["done"] = True
                if verbose:
                    log(f"  CAL step={i}: beta_r={cal['beta_r']:.4f} "
                        f"delta={cal['delta']:.4f} beta_d={cal['beta_d']:.4f} "
                        f"max_step={cal['max_step']:.3f}")
            beta_r_v = cal["beta_r"]
            beta_d_v, delta_v = cal["beta_d"], cal["delta"]
            # ORACLE-FREE steering target: tau_t = max(reward) - Delta (ADDITIVE
            # margin, shift-invariant). Keeps the best particle strictly above the
            # bar so the donor pool never empties. Delta in ImageReward units.
            tau_v = float(r.max()) - tau_relmax_offset
            if verbose:
                log(f"  MARGINtau step={i}: rmin={float(r.min()):.3f} rmax={float(r.max()):.3f} "
                    f"rmean={float(r.mean()):.3f} -> tau_v={tau_v:.3f} (Delta={tau_relmax_offset})")

            # ---- set-coupled scalars (reward gate q, feature-space D_q) ----
            qk = torch.sigmoid((r - tau_v) / beta_r_v)         # (K,) soft acceptance
            qk_div = qk
            qq = qk_div[iu[0]] * qk_div[iu[1]]
            dij = dmat[iu[0], iu[1]]
            denom = qq.sum() + 1e-6
            D_q = (qq * dij).sum() / denom
            sig_div = torch.sigmoid((delta_v - D_q) / beta_d_v)

            p = i / max(num_steps - 1, 1)
            # REWARD-channel cotangent scalar dL_reward/dr_k: softplus FLOOR.
            dL_dr = lam_r * (-1.0 / beta_r_v) * torch.sigmoid((tau_v - r) / beta_r_v)

            g_rew_norm = [0.0] * k
            g_div_norm = [0.0] * k
            g_rew_unit = torch.zeros_like(latents)
            g_div_unit = torch.zeros_like(latents)
            w_floor = [0.0] * k

            # ---- PASS B: one forward + two backwards per particle. The capture
            #      holds feat_k WITH graph (diversity channel); the reward channel
            #      rides the image cotangent (cross-process d r/d image). Each
            #      channel unit-normalized (different scales), others' features fixed.
            for kk in range(k):
                xk = latents[kk:kk + 1].detach().float().requires_grad_(True)
                with torch.enable_grad():
                    eps_k = _predict_eps_cfg(model, xk.to(dtype), t, embeds_2,
                                             guidance_scale).float()
                    n_fwd += 1
                    feat_k = capture.pooled(do_cfg).float()        # (1, C) WITH graph
                    x0k = model.x0_from_eps(xk, eps_k, t)
                    imgk = _decode_to_image(model, x0k.to(dtype)).float()  # (1,3,H,W)
                    n_decode += 1
                    # diversity loss = softplus floor on D_q^(k); D_q^(k) uses THIS
                    # particle's live cosine distances to the (fixed) other features.
                    others_n = F.normalize(feat_det, dim=-1)       # (K, C) fixed
                    fk_n = F.normalize(feat_k, dim=-1)             # (1, C) live
                    dk = (1.0 - (fk_n * others_n).sum(1)).clamp_min(0.0)  # (K,)
                    qq_full = (qk_div[iu[0]] * qk_div[iu[1]]).detach()
                    dij_fixed = dmat[iu[0], iu[1]].detach()
                    involves_k = (iu[0] == kk) | (iu[1] == kk)
                    pa, pb = iu[0], iu[1]
                    other_idx = torch.where(pa == kk, pb, pa)
                    dij_live = torch.where(involves_k, dk[other_idx], dij_fixed)
                    D_q_k = (qq_full * dij_live).sum() / denom.detach()
                    L_div_k = F.softplus((delta_v - D_q_k) / beta_d_v)
                cot_img = dL_dr[kk] * dimg_rew[kk]
                g_rew = torch.autograd.grad(imgk, xk, grad_outputs=cot_img,
                                            retain_graph=True)[0]
                g_div = torch.autograd.grad(L_div_k, xk, retain_graph=False)[0]
                g_rew = torch.nan_to_num(g_rew); g_div = torch.nan_to_num(g_div)
                rn = float(g_rew.flatten().norm()); dn = float(g_div.flatten().norm())
                g_rew_norm[kk] = rn; g_div_norm[kk] = dn
                # CROSS-CHANNEL balance: take only the DIRECTION of each channel
                # (unit-normalize) so the much larger reward grad does not drown
                # diversity; re-apply the floor's per-particle weight explicitly below.
                g_rew_u = g_rew / max(rn, 1e-12)
                g_div_u = g_div / max(dn, 1e-12)
                g_rew_unit[kk] = g_rew_u.detach().to(dtype)
                g_div_unit[kk] = g_div_u.detach().to(dtype)
                w_floor[kk] = float(dL_dr[kk].abs())
            max_step_v = cal["max_step"]
        else:
            with torch.no_grad():
                for kk in range(k):
                    eps_all[kk] = _predict_eps_cfg(
                        model, latents[kk:kk + 1], t, embeds_2, guidance_scale)[0]
                    n_fwd += 1

        rewards_trace.append(rewards.detach().cpu().tolist())

        # ---------- joint guidance step: x <- x - step * clip(grad) ----------
        if is_score:
            dist_before = float(_cos_dist(feat_det)[iu[0], iu[1]].mean())

            def _clip_step(g):                                 # per-particle norm clip
                gn = g.flatten(1).norm(dim=1, keepdim=True).clamp_min(1e-12)
                return g * (max_step_v / gn).clamp(max=1.0).view(k, *([1] * (latents.ndim - 1)))

            # FLOOR-WEIGHT the reward channel per particle (worst pushed hardest);
            # clip each channel separately so diversity can't starve the floor.
            wf = torch.tensor(w_floor, device=device, dtype=torch.float32)
            wf_rel = (wf / wf.max().clamp_min(1e-12)).view(k, *([1] * (latents.ndim - 1)))
            g_rew_w = lam_r * wf_rel * g_rew_unit.float()
            g_div_w = lam_d * g_div_unit.float()
            g_rew_c = _clip_step(g_rew_w)
            g_div_c = _clip_step(g_div_w)
            w_div = 1.0
            w_rew = (0.5 + 1.5 * p) if reward_ramp else 1.0
            gclip = (w_rew * g_rew_c + w_div * g_div_c).to(latents.dtype)

            rn_pre = g_rew_w.flatten(1).norm(dim=1)
            dn_pre = g_div_w.flatten(1).norm(dim=1)
            rew_post = (w_rew * g_rew_c).flatten(1).norm(dim=1)
            div_post = (w_div * g_div_c).flatten(1).norm(dim=1)
            n_rew_clipped = int((rn_pre > max_step_v).sum())
            n_div_clipped = int((dn_pre > max_step_v).sum())
            rc = (w_rew * g_rew_c).flatten(1)
            dc = (w_div * g_div_c).flatten(1)
            chan_cos = F.cosine_similarity(rc, dc, dim=1).tolist()
            rew_post = [round(x, 4) for x in rew_post.tolist()]
            div_post = [round(x, 4) for x in div_post.tolist()]
            chan_cos = [round(x, 4) for x in chan_cos]
            gnorm = guidance.flatten(1).norm(dim=1)   # (kept 0 here; per-channel below)

            applied = (step_size * gclip).flatten(1).norm(dim=1)
            applied_norm = [round(x, 4) for x in applied.float().tolist()]
            latents = (latents - step_size * gclip).detach()
            n_nan = int(torch.isnan(latents).sum())

            # ---- late copy (SURGICAL replacement of a gradient-unreachable dud) ----
            clone_log = None
            if clone_dud and k > 1 and p >= clone_start_p:
                with torch.no_grad():
                    r_cd = torch.zeros(k, device=device)
                    for kk in range(k):
                        eps_kk = _predict_eps_cfg(model, latents[kk:kk + 1], t,
                                                  embeds_2, guidance_scale).float()
                        n_fwd += 1
                        x0kk = model.x0_from_eps(latents[kk:kk + 1].float(), eps_kk, t)
                        imgkk = _decode_to_image(model, x0kk.to(dtype)).float()
                        n_decode += 1
                        rk, _ = client.reward_and_grad(imgkk, [prompt])
                        r_cd[kk] = float(rk[0])
                    q_cd = torch.sigmoid((r_cd - tau_v) / beta_r_v)
                    acc = q_cd > 0.5                              # donor pool (r > tau)
                    is_dud = r_cd < tau_v
                    if prev_clone_r is not None:
                        delta_r = r_cd - prev_clone_r
                    else:
                        delta_r = torch.full((k,), float("inf"), device=device)
                    n_remaining = sum(1 for c in score_steps if c > i)
                    # optimal-stopping commit: clone a sub-floor particle only when its
                    # projected reward r + max(0,delta_r)*remaining still can't reach tau.
                    projected = r_cd + delta_r.clamp_min(0.0) * n_remaining
                    rescuable = is_dud & (projected < tau_v)
                    clone_log = {"r_pre": [round(float(x), 4) for x in r_cd.tolist()],
                                 "q": [round(float(x), 3) for x in q_cd.tolist()],
                                 "n_remaining": int(n_remaining), "cloned": []}
                    if acc.any() and rescuable.any():
                        acc_idx = torch.where(acc)[0]
                        clone_gen = torch.Generator().manual_seed(int(seed) + 104729 * (i + 1))
                        for w in torch.where(rescuable)[0].tolist():
                            donor_idx = acc_idx[acc_idx != w]
                            if len(donor_idx) == 0:
                                continue
                            donor = int(donor_idx[int(torch.randint(len(donor_idx), (1,), generator=clone_gen))])
                            latents[w] = latents[donor].clone()       # FULL clone (gamma=1)
                            clone_log["cloned"].append(
                                {"w": int(w), "donor": int(donor),
                                 "r": round(float(r_cd[w]), 4),
                                 "r_donor": round(float(r_cd[donor]), 4)})
                        latents = latents.detach()
                    prev_clone_r = r_cd.detach()
                    if verbose:
                        ncl = len(clone_log["cloned"])
                        log(f"    CLONE step {i}: {ncl}/{int(is_dud.sum())} sub-floor "
                            f"cloned; r_pre={clone_log['r_pre']} "
                            + (f"cloned={clone_log['cloned']}" if ncl else ""))

            rec = {
                "step": int(i), "t": int(t), "p": round(p, 3),
                "reward_min": round(float(r.min()), 4),
                "reward_mean": round(float(r.mean()), 4),
                "reward_max": round(float(r.max()), 4),
                "reward_spread": round(float(r.max() - r.min()), 4),
                "q_gate": [round(x, 3) for x in qk.tolist()],
                "D_q": round(float(D_q), 4), "delta": round(delta_v, 4),
                "div_binding": float(sig_div) > 0.5,
                "feat_cos_mean": round(dist_before, 4),
                "channel_sched": channel_sched, "reward_ramp": bool(reward_ramp),
                "clone": clone_log,
                "w_floor": [round(x, 5) for x in w_floor],
                "w_rew": round(w_rew, 4), "w_div": round(w_div, 4),
                "applied_step_norm": applied_norm,
                "rew_post_norm": rew_post, "div_post_norm": div_post,
                "n_rew_clipped": n_rew_clipped, "n_div_clipped": n_div_clipped,
                "chan_cos": chan_cos,
                "g_reward_norm": [round(x, 4) for x in g_rew_norm],
                "g_div_norm": [round(x, 4) for x in g_div_norm],
                "x0_drift_rel": ([round(x, 4) for x in x0_drift_rel]
                                 if x0_drift_rel is not None else None),
                "rank_churn": rank_churn,
                "img_min": round(img_min, 4), "img_max": round(img_max, 4),
                "guidance_nan": n_nan,
                "step_time_s": round(_time.time() - t_step0, 2),
            }
            diag.append(rec)
            if verbose:
                gs = f" g_rew={np.mean(g_rew_norm):.3f} g_div={np.mean(g_div_norm):.3f}"
                log(f"  step {i} t={int(t)}: r[min/mean/max]="
                    f"{float(r.min()):.3f}/{float(r.mean()):.3f}/{float(r.max()):.3f} "
                    f"D_q={float(D_q):.3f}/delta={delta_v:.3f} bind={float(sig_div) > 0.5}{gs} "
                    f"feat_cos={dist_before:.3f} churn={rank_churn} "
                    f"img[{img_min:.2f},{img_max:.2f}] nan={n_nan} ({rec['step_time_s']}s)")
                log(f"    [sep] w_rew={w_rew:.3f} w_div={w_div:.3f} "
                    f"rew_post={np.mean(rew_post):.3f}(clip {n_rew_clipped}/{k}) "
                    f"div_post={np.mean(div_post):.3f}(clip {n_div_clipped}/{k}) "
                    f"chan_cos={np.mean(chan_cos):+.3f} "
                    f"applied={np.mean(applied_norm):.3f} max_step={max_step_v:.3f}")

        # ---------- ordinary DDIM ancestral (eta) step (all K alive) ----------
        mean, base_var, _ = _ddim_mean_and_variance(sched, eps_all, t, latents, eta)
        variance = (eta ** 2) * base_var
        if float(variance) <= 0.0:
            latents = mean.to(dtype)
            continue
        std = variance.sqrt()
        eps_gen = torch.Generator(device=device).manual_seed(seed + 7919 * (i + 1))
        noise = torch.randn(latents.shape, device=device, dtype=torch.float32,
                            generator=eps_gen)
        latents = (mean + std * noise).detach().to(dtype)

    # ---- FINAL decode + score ----
    if device.type == "cuda":
        torch.cuda.empty_cache()
    with torch.no_grad():
        pil = model.decode(latents)
        # final feature spread + per-image reward on the actual returned images
        final_rewards = None
        try:
            import numpy as _np
            arr = _np.stack([_np.asarray(im).astype("float32") / 255.0 for im in pil], 0)
            imgs = torch.from_numpy(arr).permute(0, 3, 1, 2).contiguous()  # (K,3,H,W)[0,1]
            fr = []
            for j in range(imgs.shape[0]):
                rj, _ = client.reward_and_grad(imgs[j:j + 1], [prompt])
                fr.append(float(rj[0]))
            final_rewards = fr
        except Exception as e:
            log(f"  !! final-image scoring failed ({type(e).__name__}: {e}); images still saved")
        feat_cos_final = None
        try:
            feats_final = []
            for kk in range(k):
                _predict_eps_cfg(model, latents[kk:kk + 1], timesteps[-1], embeds_2, guidance_scale)
                feats_final.append(capture.pooled(do_cfg).detach().float())
                n_fwd += 1
            ff = torch.cat(feats_final, 0)
            iu2 = torch.triu_indices(k, k, offset=1, device=device)
            feat_cos_final = float(_cos_dist(ff)[iu2[0], iu2[1]].mean())
        except Exception:
            pass

    final_summary = None
    if final_rewards is not None:
        fr_t = torch.tensor(final_rewards)
        final_summary = {
            "min_k": round(float(fr_t.min()), 4),
            "mean_k": round(float(fr_t.mean()), 4),
            "max_k": round(float(fr_t.max()), 4),
            "spread_k": round(float(fr_t.max() - fr_t.min()), 4),
            "per_particle": [round(x, 4) for x in final_rewards],
            "feat_cos_final": (round(feat_cos_final, 4) if feat_cos_final is not None else None),
        }
        log(f"  FINAL score: min-K={final_summary['min_k']} mean-K={final_summary['mean_k']} "
            f"max-K={final_summary['max_k']} | per-particle={final_summary['per_particle']} "
            f"| final feat_cos={final_summary['feat_cos_final']}")

    peak_mem_gb = (torch.cuda.max_memory_allocated() / 1e9 if device.type == "cuda" else 0.0)
    cost = {"wall_s": round(_time.time() - t_run0, 1),
            "n_unet_forwards": n_fwd, "n_vae_decodes": n_decode,
            "peak_mem_gb": round(peak_mem_gb, 2),
            "calibration": {kk2: (round(vv, 4) if isinstance(vv, float) else vv)
                            for kk2, vv in cal.items() if kk2 != "done"}}
    return pil, rewards_trace, {"steps": diag, "cost": cost, "final": final_summary}


def _derive_score_steps(spec: str, num_steps: int):
    """Parse --score_steps. 'auto' -> ~8 steps evenly spread in [0.15N, 0.85N]."""
    if spec is None or spec.strip().lower() == "auto":
        import numpy as np
        lo, hi = int(0.15 * num_steps), int(0.85 * num_steps)
        n = min(8, max(2, hi - lo))
        return sorted(set(int(x) for x in np.linspace(lo, hi, n).round().astype(int)))
    return sorted(set(int(x) for x in spec.split(",")))


def main(args):

    score_steps = set(_derive_score_steps(args.score_steps, args.num_steps))
    print(f"[ours config] reward={args.reward} tau_relmax_offset={args.tau_relmax_offset} "
          f"score_steps={sorted(score_steps)} step_size={args.step_size} "
          f"lam_r={args.lam_r} lam_d={args.lam_d} eta={args.eta} "
          f"clone_dud={args.clone_dud} reward_ramp={args.reward_ramp}", flush=True)
    if (not args.clone_dud) or (not args.reward_ramp):
        print("[ours config] WARNING: DEGRADED config vs deployed — the headline method needs "
              "clone_dud=ON + reward_ramp=ON. Disable ONLY for an explicit ablation.", flush=True)

    prompt_ids = set(args.prompt_ids.split(",")) if args.prompt_ids else None
    prompts = load_prompts(ROOT / args.metadata, prompt_ids)
    if args.limit is not None:
        prompts = prompts[:args.limit]
    if not prompts:
        log("ERROR: no prompts loaded"); sys.exit(1)
    log(f"loaded {len(prompts)} prompts: {[p['id'] for p in prompts]}")

    log("loading SD1.5 (cache-only)...")
    model = SD15Model(model_id=args.model_id, device=args.device)
    try:
        model.unet.enable_gradient_checkpointing()
        log("enabled gradient checkpointing on the UNet")
    except Exception as e:
        log(f"WARN: could not enable gradient checkpointing: {e}")

    # Install the mid-block feature capture (additive forward hook; sd15_model.py
    # untouched). Fires on every UNet forward; capture.pooled() = mean-pooled
    # mid-block features (graph kept under enable_grad).
    capture = _MidBlockCapture()
    handle = model.unet.mid_block.register_forward_hook(capture)
    log("installed mid-block feature capture (UNet bottleneck) for the diversity term")

    log(f"starting {args.reward} grad worker on worker_gpu={args.worker_gpu}...")
    client = RewardClient.imagereward_grad(gpu_id=args.worker_gpu)

    out_root = ROOT / args.out_root
    method_dir = out_root / "ours"
    method_dir.mkdir(parents=True, exist_ok=True)
    summary_path = method_dir / "ours_summary.json"
    summary = {}

    try:
        for pi, p in enumerate(prompts):
            pid, prompt = p["id"], p["prompt"]
            t0 = time.time()
            log(f"=== prompt {pid}: {prompt!r} ===")
            try:
                pid_int = int(pid)
            except (TypeError, ValueError):
                pid_int = pi
            per_prompt_seed = args.seed + pid_int
            pil, rewards_trace, diag = run_ours_one_prompt(
                model, client, capture, prompt, args.k, args.num_steps,
                args.guidance_scale, args.height, args.width, per_prompt_seed,
                score_steps, eta=args.eta, beta_r=args.beta_r, beta_d=args.beta_d,
                delta=args.delta, lam_r=args.lam_r, lam_d=args.lam_d,
                step_size=args.step_size, max_step=args.max_step,
                tau_relmax_offset=args.tau_relmax_offset, delta_scale=args.delta_scale,
                reward_ramp=args.reward_ramp, clone_dud=args.clone_dud,
                clone_start_p=args.clone_start_p, verbose=args.verbose)
            pdir = method_dir / pid
            pdir.mkdir(parents=True, exist_ok=True)
            for j, im in enumerate(pil):
                im.save(pdir / f"candidate_{j}.png")
            with open(pdir / "diag.json", "w") as f:
                json.dump(diag, f, indent=2)
            cost = diag["cost"]
            summary[pid] = {"prompt": prompt, "rewards_trace": rewards_trace,
                            "final": diag.get("final"),
                            "elapsed_s": round(time.time() - t0, 1), "cost": cost}
            fin = diag.get("final")
            fin_str = (f" | FINAL min-K={fin['min_k']} mean-K={fin['mean_k']} "
                       f"feat_cos={fin['feat_cos_final']}" if fin else "")
            log(f"  saved {len(pil)} imgs to {pdir} | {cost['wall_s']}s, "
                f"{cost['n_unet_forwards']} fwd, {cost['n_vae_decodes']} decode, "
                f"peak {cost['peak_mem_gb']}GB | diag.json written{fin_str}")
            with open(summary_path, "w") as f:
                json.dump(summary, f, indent=2)
    finally:
        handle.remove()
        client.close()

    log(f"DONE. images under {method_dir}; summary at {summary_path}")


