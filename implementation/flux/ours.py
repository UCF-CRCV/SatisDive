"""SatisDive flux: ours. See README.md for the supported interface."""

from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

# Shared FLUX helpers (single source of truth) + the FK-Flow geom SDE kernel.
from flux_model import (  # noqa: E402
    load_prompts, build_prompt_state, predict_velocity,
    decode_to_image, _slice_ps,
)
from sde import fk_flow_kernel_mean  # noqa: E402
from profiling import GpuSampler  # noqa: E402
from rewards.clients import RewardClient  # noqa: E402

ROOT = Path(__file__).resolve().parent


def log(m):
    print(f"[ours_flux] {m}", flush=True)


class _Block12Capture:
    """Capture pooled FLUX image-token attention features with gradients retained."""

    BLOCK_IDX = 12

    def __init__(self, num_generated_tokens=4096, store=None, block_idx=12,
                 grid=1, h_tok=64, w_tok=64):
        self.num_generated_tokens = num_generated_tokens
        self.store = store if store is not None else {}
        self.block_idx = block_idx
        self.grid = int(grid)        # GxG spatial regions (1 = global token pool)
        self.h_tok = int(h_tok)
        self.w_tok = int(w_tok)

    @property
    def last_features(self):
        # LPIPS-on-FLUX diversity feature. Each store entry is the per-block,
        # per-region L2-NORMALIZED feature flattened to (B, R*dim). Concatenating
        # the unit sub-vectors across the band and over regions, then applying ONE
        # cosine distance (downstream _cos_dist), equals EXACTLY mean over
        # (block, region) of (1 - cos): the LPIPS-style per-location, per-block
        # distance-THEN-average (a valid metric; no incommensurable base-mixing).
        # Single block + grid=1 reproduces the deployed pooled block-12 metric.
        if not self.store:
            return None
        return torch.cat([self.store[b] for b in sorted(self.store)], dim=-1)

    def __call__(self, attn, hidden_states, encoder_hidden_states=None,
                 attention_mask=None, image_rotary_emb=None, *args, **kwargs):
        from diffusers.models.embeddings import apply_rotary_emb
        bsz = hidden_states.shape[0]
        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)
        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads
        query = query.view(bsz, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(bsz, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(bsz, -1, attn.heads, head_dim).transpose(1, 2)
        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)
        n_text = 0
        if encoder_hidden_states is not None:
            enc_q = attn.add_q_proj(encoder_hidden_states)
            enc_k = attn.add_k_proj(encoder_hidden_states)
            enc_v = attn.add_v_proj(encoder_hidden_states)
            enc_q = enc_q.view(bsz, -1, attn.heads, head_dim).transpose(1, 2)
            enc_k = enc_k.view(bsz, -1, attn.heads, head_dim).transpose(1, 2)
            enc_v = enc_v.view(bsz, -1, attn.heads, head_dim).transpose(1, 2)
            if attn.norm_added_q is not None:
                enc_q = attn.norm_added_q(enc_q)
            if attn.norm_added_k is not None:
                enc_k = attn.norm_added_k(enc_k)
            n_text = encoder_hidden_states.shape[1]
            query = torch.cat([enc_q, query], dim=2)
            key = torch.cat([enc_k, key], dim=2)
            value = torch.cat([enc_v, value], dim=2)
        if image_rotary_emb is not None:
            query = apply_rotary_emb(query, image_rotary_emb)
            key = apply_rotary_emb(key, image_rotary_emb)
        out = F.scaled_dot_product_attention(query, key, value, dropout_p=0.0, is_causal=False)
        out = out.transpose(1, 2).reshape(bsz, -1, attn.heads * head_dim).to(query.dtype)
        if encoder_hidden_states is not None:
            sample_states = out[:, n_text:]
            n_gen = min(self.num_generated_tokens, sample_states.shape[1])
            # KEEP graph: mean-pooled generated-token features (the diversity phi).
            gen = sample_states[:, :n_gen, :].float()                  # (B, n_gen, dim)
            Bsz, _, dim = gen.shape
            g = self.grid
            if g <= 1:
                regions = gen.mean(dim=1, keepdim=True)                     # (B,1,dim) global pool
            else:
                sp = gen[:, :self.h_tok * self.w_tok, :].transpose(1, 2)
                sp = sp.reshape(Bsz, dim, self.h_tok, self.w_tok)
                sp = F.adaptive_avg_pool2d(sp, (g, g))                      # (B,dim,g,g)
                regions = sp.reshape(Bsz, dim, g * g).transpose(1, 2)      # (B,R,dim)
            regions = F.normalize(regions, dim=-1)                          # unit per region
            self.store[self.block_idx] = regions.reshape(Bsz, -1)           # (B,R*dim) graph kept
            sample_out = attn.to_out[1](attn.to_out[0](sample_states))
            text_out = attn.to_add_out(out[:, :n_text])
            return sample_out, text_out
        return out


def _cos_dist(feats):
    """(K,K) cosine DISTANCE (1 - cosine sim) between (K,dim) feature vectors.
    Cosine (not Euclidean) is scale-invariant across prompts and matches the
    verified Repel geometry; the toy's cdist is Euclidean only because its 2D
    output points have no meaningful scale to factor out."""
    fn = F.normalize(feats, dim=-1)
    sim = fn @ fn.t()
    return (1.0 - sim).clamp_min(0.0)


def _flat(x):
    """Flatten a packed latent (K, ...) to (K, D) float for the max_step norm calc."""
    return x.reshape(x.shape[0], -1).float()


def run_ours_one_prompt(pipe, client, capture, prompt, k, height, width, num_steps,
                        device, dtype, seed, score_steps, beta_r, beta_d,
                        delta, lam_r, lam_d, step_size, max_step, sde_a,
                        reward_ramp=False, clone_dud=False, clone_start_p=0.5,
                        tau_relmax_offset=1.5, delta_scale=1.0, verbose=False):
    """Generate one batch with reward-floor and gated-diversity updates.

    Returns images, reward traces, and per-step diagnostics. Reward and diversity
    gradients are normalized separately before the latent update. Late-stage
    replacement uses a randomly selected candidate above the current cutoff.
    Null calibration arguments use first-scoring-step batch statistics.
    """
    import numpy as np
    import time as _time

    # Deployed operating point (toggles removed): 2nd-order Heun Tweedie x0 is
    # always on, and the reward/diversity channels always use the separate-budget
    # schedule (the combined/feynman schedules were removed).
    channel_sched = "separate"

    if device == "cuda" or (hasattr(device, "type") and device.type == "cuda"):
        torch.cuda.reset_peak_memory_stats()

    sched = pipe.scheduler
    ps = build_prompt_state(pipe, prompt, k, height, width, device)

    gen = torch.Generator(device=device).manual_seed(seed)
    num_ch = pipe.transformer.config.in_channels // 4
    latents, latent_image_ids = pipe.prepare_latents(
        k, num_ch, height, width, dtype, device, gen, None)
    ps["latent_image_ids"] = latent_image_ids

    def _calc_shift(seq_len, base_seq=256, max_seq=4096, base_shift=0.5, max_shift=1.15):
        m = (max_shift - base_shift) / (max_seq - base_seq)
        b = base_shift - m * base_seq
        return seq_len * m + b
    sigmas_np = np.linspace(1.0, 1.0 / num_steps, num_steps)
    if getattr(sched.config, "use_flow_sigmas", False):
        sigmas_np = None
    mu = _calc_shift(
        latents.shape[1],
        sched.config.get("base_image_seq_len", 256),
        sched.config.get("max_image_seq_len", 4096),
        sched.config.get("base_shift", 0.5),
        sched.config.get("max_shift", 1.15),
    )
    sched.set_timesteps(num_steps, device=device, sigmas=sigmas_np, mu=mu)
    timesteps = sched.timesteps
    sigmas = sched.sigmas

    rewards_trace = []
    diag = []                       # per-step structured debug records
    cal = {"beta_r": beta_r, "beta_d": beta_d,
           "delta": delta, "max_step": max_step, "done": False}
    v_prev = None                   # cached per-particle velocity for 2nd-order x0
    n_fwd = 0                       # FLUX transformer forward counter (cost)
    n_decode = 0                    # VAE decode counter (cost)
    prev_score_x0 = None            # last scoring step's x0 (proxy-drift fidelity)
    prev_score_ranks = None         # last scoring step's reward ranks (rank churn)
    prev_clone_r = None             # per-particle reward at the prev scored step (clone trigger)
    n_rollout = 0                   # retained in the cost dict (always 0; sample-force removed)
    t_run0 = _time.time()

    for i, t in enumerate(timesteps):
        s_curr = float(sigmas[i])
        s_next = float(sigmas[i + 1])
        ds = s_next - s_curr  # < 0
        latents = latents.detach()

        v_all = torch.zeros_like(latents)
        guidance = torch.zeros_like(latents)   # grad_x L assembled here (all K)
        rewards = torch.zeros(k, device=device)

        is_score = i in score_steps
        if is_score:
            t_step0 = _time.time()


            dimg_rew, feat_rows = [], []
            x0_first_norm = []      # ||x0_1st|| per particle (fidelity ref)
            x0_corr_rel = []        # ||x0_2nd - x0_1st|| / ||x0_1st|| (2nd-order correction size)
            used_2nd = bool(v_prev is not None)   # 2nd-order actually applied?
            img_min = img_max = None
            x0_cur_rows = []        # 2nd-order x0 per particle (for cross-step drift)
            with torch.no_grad():
                for kk in range(k):
                    vk = predict_velocity(pipe, latents[kk:kk + 1], t, _slice_ps(ps, kk)).to(torch.float32)
                    n_fwd += 1
                    feat_k = capture.last_features.detach().float()      # (1,dim) block-12 phi
                    x0_1st = latents[kk:kk + 1].to(torch.float32) - s_curr * vk
                    if v_prev is not None:
                        v_use = 0.5 * (vk + v_prev[kk:kk + 1].to(torch.float32))
                    else:
                        v_use = vk
                    x0k = latents[kk:kk + 1].to(torch.float32) - s_curr * v_use
                    # x0-fidelity diagnostics
                    n1 = float(x0_1st.flatten().norm())
                    x0_first_norm.append(n1)
                    x0_corr_rel.append(float((x0k - x0_1st).flatten().norm()) / max(n1, 1e-8))
                    x0_cur_rows.append(x0k.detach())
                    imgk = decode_to_image(pipe, x0k, height, width)     # (1,3,H,W)[0,1]
                    n_decode += 1
                    bmin, bmax = float(imgk.min()), float(imgk.max())
                    img_min = bmin if img_min is None else min(img_min, bmin)
                    img_max = bmax if img_max is None else max(img_max, bmax)
                    rew_k, dimg_k = client.reward_and_grad(imgk.float(), [prompt])
                    rewards[kk] = float(rew_k[0])
                    v_all[kk] = vk.detach().to(dtype)
                    # NaN/inf guard on the worker gradient (cross-process; can return junk)
                    dimg_k = torch.nan_to_num(dimg_k.to(imgk.device, imgk.dtype))
                    dimg_rew.append(dimg_k)                               # d r_k / d image_k
                    feat_rows.append(feat_k)                             # (1,dim)

            feat_det = torch.cat(feat_rows, 0)                     # (K,dim) detached features
            r = rewards.float()
            dmat = _cos_dist(feat_det)                             # (K,K) FEATURE cosine dist
            iu = torch.triu_indices(k, k, offset=1, device=device)
            # cross-step proxy drift: how much each particle's predicted clean image
            # x0 moved since the last scoring step (high => proxy still unreliable).
            x0_cur = torch.cat(x0_cur_rows, 0)
            if prev_score_x0 is not None and prev_score_x0.shape == x0_cur.shape:
                x0_drift = (x0_cur - prev_score_x0).flatten(1).norm(dim=1)
                x0_drift_rel = (x0_drift / x0_cur.flatten(1).norm(dim=1).clamp_min(1e-8)).tolist()
            else:
                x0_drift_rel = None
            prev_score_x0 = x0_cur.detach()
            # reward rank churn vs last scoring step (timing-paradox readout)
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
            # ORACLE-FREE per-step steering target: tau_t = max(reward) - Delta
            # (ADDITIVE margin). SHIFT-INVARIANT (only the gap max-r_k enters), the
            # correct form for HPSv3's unbounded/arbitrary-origin (RankNet/BT) reward,
            # where a multiplicative cap frac*max is ill-defined and goes inert on a
            # tight, high batch. Stays active on a tight high batch (HPDv2) and keeps
            # the best particle strictly above the bar (max - tau_v = Delta > 0) so the
            # donor pool / clone_frac gate never empties. Delta is in reward (HPSv3
            # logit) units (~1-2; a 1-logit gap ~= 73% preference).
            tau_v = float(r.max()) - tau_relmax_offset
            if verbose:
                log(f"  MARGINtau step={i}: rmin={float(r.min()):.3f} rmax={float(r.max()):.3f} "
                    f"rmean={float(r.mean()):.3f} -> tau_v={tau_v:.3f} (Delta={tau_relmax_offset})")

            # ---- set-coupled scalars (reward gate q, feature-space D_q) ----
            qk = torch.sigmoid((r - tau_v) / beta_r_v)             # (K,) soft acceptance
            # diversity pair-gate: reward-gated (spread only the accepted particles).
            qk_div = qk
            qq = qk_div[iu[0]] * qk_div[iu[1]]
            dij = dmat[iu[0], iu[1]]
            denom = qq.sum() + 1e-6
            D_q = (qq * dij).sum() / denom                         # (reward-gated) set diversity
            sig_div = torch.sigmoid((delta_v - D_q) / beta_d_v)    # diversity-floor activation (diag)

            p = i / max(num_steps - 1, 1)
            climb_on = False   # reward climb removed (deployed lam_climb=0)
            # REWARD-channel cotangent scalar dL_reward/dr_k: softplus FLOOR. Pushes
            # hard only below tau; cannot transport the worst particle across a
            # low-density valley to an off-mode optimum (the clone path handles that).
            dL_dr = lam_r * (-1.0 / beta_r_v) * torch.sigmoid((tau_v - r) / beta_r_v)  # floor

            # diagnostic accumulators (RAW per-channel grad norms, before normalize)
            g_rew_norm = [0.0] * k
            g_div_norm = [0.0] * k
            g_rew_unit = torch.zeros_like(latents)   # stored unit grads for counterfactuals
            g_div_unit = torch.zeros_like(latents)
            w_floor = [0.0] * k   # per-particle floor weight |dL_dr| (worst-pushed-hardest)

            # ---- PASS B (1x memory): one forward + two backwards per particle. The
            #      block-12 capture holds feat_k WITH graph (diversity channel); the
            #      reward channel rides the image cotangent (cross-process d r/d image).
            #      We NORMALIZE each channel to unit norm then weight (see above), so
            #      the diversity term is not drowned by the ~1000x larger reward grad.
            #      Others' features held fixed (the toy does the same).
            for kk in range(k):
                xk = latents[kk:kk + 1].detach().to(torch.float32).requires_grad_(True)
                with torch.enable_grad():
                    vk = predict_velocity(pipe, xk.to(dtype), t, _slice_ps(ps, kk)).to(torch.float32)
                    n_fwd += 1
                    feat_k = capture.last_features.float()            # (1,dim) WITH graph
                    if v_prev is not None:
                        v_use = 0.5 * (vk + v_prev[kk:kk + 1].to(torch.float32).detach())
                    else:
                        v_use = vk
                    x0k = xk - s_curr * v_use
                    imgk = decode_to_image(pipe, x0k, height, width)  # (1,3,H,W)[0,1]
                    n_decode += 1
                    # diversity loss = softplus floor on D_q^(k), D_q^(k) uses THIS
                    #   particle's live cosine distances to the (fixed) other features.
                    others_n = F.normalize(feat_det, dim=-1)          # (K,dim) fixed
                    fk_n = F.normalize(feat_k, dim=-1)                # (1,dim) live
                    dk = (1.0 - (fk_n * others_n).sum(1)).clamp_min(0.0)  # (K,) cosine dist
                    qq_full = (qk_div[iu[0]] * qk_div[iu[1]]).detach()
                    dij_fixed = dmat[iu[0], iu[1]].detach()
                    involves_k = (iu[0] == kk) | (iu[1] == kk)
                    pa, pb = iu[0], iu[1]
                    other_idx = torch.where(pa == kk, pb, pa)
                    dij_live = torch.where(involves_k, dk[other_idx], dij_fixed)
                    # q-weighted MEAN pair distance -> the set-diversity scalar D_q_k.
                    D_q_k = (qq_full * dij_live).sum() / denom.detach()
                    # NOTE: no lam_d here -- the channel weight is applied after norm.
                    L_div_k = F.softplus((delta_v - D_q_k) / beta_d_v)
                cot_img = dL_dr[kk] * dimg_rew[kk]
                # reward channel (through image) and diversity channel (through feat)
                g_rew = torch.autograd.grad(imgk, xk, grad_outputs=cot_img,
                                            retain_graph=True)[0]
                g_div = torch.autograd.grad(L_div_k, xk, retain_graph=False)[0]
                g_rew = torch.nan_to_num(g_rew); g_div = torch.nan_to_num(g_div)
                rn = float(g_rew.flatten().norm()); dn = float(g_div.flatten().norm())
                g_rew_norm[kk] = rn; g_div_norm[kk] = dn
                # CROSS-CHANNEL balance: take only the DIRECTION of each channel
                # (unit-normalize) so the ~1000x larger reward grad does not drown
                # diversity, and so FLUX's ~50x particle-to-particle variation in the
                # raw image-gradient magnitude |dr/dimage| (which is NOISE w.r.t. the
                # floor, and is even ANTI-correlated with floor distance -- verified
                # in diag: the worst particle had the SMALLEST raw reward grad) is
                # stripped. THEN re-apply the floor's own per-particle weight as an
                # EXPLICIT scalar (below), so cross-PARTICLE magnitude = floor
                # distance, not image-gradient noise. This is the FLUX analog of the
                # toy's clip-to-ceiling (toy |dr/dx| is ~uniform across particles, so
                # the toy gets "worst pushed hardest" for free; FLUX does not).
                g_rew_u = g_rew / max(rn, 1e-12)
                g_div_u = g_div / max(dn, 1e-12)
                g_rew_unit[kk] = g_rew_u.detach().to(dtype)
                g_div_unit[kk] = g_div_u.detach().to(dtype)
                # per-particle FLOOR WEIGHT (the magnitude the unit-norm destroyed):
                #   w_floor_k = |dL_dr_k| = lam_r/beta * sigmoid((tau-r_k)/beta)  [+climb]
                #   = LARGE when r_k << tau (worst particle pushed hardest),
                #     ~0 once r_k >> tau (good-enough particle left alone -> also
                #     kills the dead-channel unit-noise that fired when raw grad ~0).
                w_floor[kk] = float(dL_dr[kk].abs())
                gk = lam_r * g_rew_u + lam_d * g_div_u
                guidance[kk] = gk.detach().to(dtype)
            max_step_v = cal["max_step"]
        else:
            with torch.no_grad():
                for kk in range(k):
                    v_all[kk] = predict_velocity(pipe, latents[kk:kk + 1], t, _slice_ps(ps, kk))[0]
                    n_fwd += 1

        v_prev = v_all.detach().clone()   # cache for next step's 2nd-order Tweedie x0
        rewards_trace.append(rewards.detach().cpu().tolist())

        # ---------- joint guidance step: x <- x - step * clip(grad) ----------
        if is_score:
            # particle spread BEFORE the guidance step (feature cosine, the metric we steer)
            dist_before = float(_cos_dist(feat_det)[iu[0], iu[1]].mean())
            # raw combined-guidance magnitude (kept as a diagnostic for ALL schemes)
            gnorm = guidance.flatten(1).norm(dim=1, keepdim=True).clamp_min(1e-12)
            n_clipped = int((gnorm.squeeze(1) > max_step_v).sum())
            latents_pre = latents.detach().clone()                 # before the guidance step

            def _clip_step(g):                                     # per-particle norm clip to max_step
                gn = g.flatten(1).norm(dim=1, keepdim=True).clamp_min(1e-12)
                return g * (max_step_v / gn).clamp(max=1.0).view(k, *([1] * (latents.ndim - 1)))

            # 'separate' schedule: clip EACH channel to its OWN max_step so the
            # diversity channel can no longer starve the reward floor (the toy min-K
            # fix that lifted the toy floor 0.16 -> 0.755). g_rew_unit/g_div_unit
            # already hold the unit-normalized per-channel grads; weight by
            # lam_r/lam_d, clip each separately, then sum.
            #
            # FLOOR-WEIGHT the reward channel per particle (the min-K fix): scale
            # each particle's unit reward direction by its RELATIVE floor weight
            # w_floor_k/max_k(w_floor) in [0,1]. The worst particle (largest
            # |dL_dr|) gets the full step; an above-floor particle (|dL_dr|~0)
            # gets ~0 (and its dead-channel unit-noise is multiplied away). This
            # restores "worst pushed hardest" that the per-particle unit-norm had
            # destroyed, WITHOUT reintroducing the 1000x cross-channel imbalance
            # (direction is still unit; only the per-particle SCALE is floor-set).
            wf = torch.tensor(w_floor, device=latents.device, dtype=torch.float32)
            wf_rel = (wf / wf.max().clamp_min(1e-12)).view(k, *([1] * (latents.ndim - 1)))
            g_rew_w = lam_r * wf_rel * g_rew_unit.float()
            g_div_w = lam_d * g_div_unit.float()
            g_rew_c = _clip_step(g_rew_w)
            g_div_c = _clip_step(g_div_w)
            # time-invariant unless reward_ramp ramps the reward weight late.
            w_div = 1.0
            w_rew = (0.5 + 1.5 * p) if reward_ramp else 1.0
            gclip = (w_rew * g_rew_c + w_div * g_div_c).to(latents.dtype)
            # ---- separate-path diagnostics (the quantities that actually drive
            #      the step; the combined `gnorm`/`n_clipped` above do NOT reflect
            #      what is applied in this path) ----
            rn_pre = g_rew_w.flatten(1).norm(dim=1)            # pre-clip weighted norms
            dn_pre = g_div_w.flatten(1).norm(dim=1)
            rew_post = (w_rew * g_rew_c).flatten(1).norm(dim=1)  # post-clip, post-weight
            div_post = (w_div * g_div_c).flatten(1).norm(dim=1)
            n_rew_clipped = int((rn_pre > max_step_v).sum())
            n_div_clipped = int((dn_pre > max_step_v).sum())
            # are the two channels fighting? cosine of their (weighted, clipped)
            # directions per particle; <0 means reward & diversity pull opposite.
            rc = (w_rew * g_rew_c).flatten(1)
            dc = (w_div * g_div_c).flatten(1)
            chan_cos = F.cosine_similarity(rc, dc, dim=1).tolist()
            rew_post = [round(x, 4) for x in rew_post.tolist()]
            div_post = [round(x, 4) for x in div_post.tolist()]
            chan_cos = [round(x, 4) for x in chan_cos]

            applied = (step_size * gclip).flatten(1).norm(dim=1)   # TRUE per-particle step magnitude
            applied_norm = [round(x, 4) for x in applied.float().tolist()]
            latents = (latents - step_size * gclip).detach()
            n_nan = int(torch.isnan(latents).sum())

            # (counterfactual diagnostic removed — was --diag_counterfactual)
            cf = None


            reloc_log = None

            # Late-stage replacement uses a randomly selected above-cutoff candidate.


            clone_log = None
            if clone_dud and k > 1 and p >= clone_start_p:
                with torch.no_grad():
                    r_cd = torch.zeros(k, device=device)
                    for kk in range(k):
                        vkk = predict_velocity(pipe, latents[kk:kk + 1], t, _slice_ps(ps, kk)).to(torch.float32)
                        n_fwd += 1
                        if v_prev is not None:
                            v_use = 0.5 * (vkk + v_prev[kk:kk + 1].to(torch.float32))
                        else:
                            v_use = vkk
                        x0kk = latents[kk:kk + 1].to(torch.float32) - s_curr * v_use
                        imgkk = decode_to_image(pipe, x0kk, height, width)
                        n_decode += 1
                        rk, _ = client.reward_and_grad(imgkk.float(), [prompt])
                        r_cd[kk] = float(rk[0])
                    q_cd = torch.sigmoid((r_cd - tau_v) / beta_r_v)
                    acc = q_cd > 0.5                  # steering-accepted (r > tau) = donor pool
                    # DUD bar: a particle is a clone TARGET only if its reward is below
                    # the steering tau_v. Donors are the steering-accepted (>tau_v) peers.
                    dud_bar = tau_v
                    is_dud = r_cd < dud_bar
                    sub = is_dud
                    # per-step reward improvement vs the previous scored step
                    if prev_clone_r is not None:
                        delta_r = r_cd - prev_clone_r
                    else:
                        delta_r = torch.full((k,), float("inf"), device=device)
                    # remaining scored steps AFTER this one (runway to keep climbing)
                    n_remaining = sum(1 for c in score_steps if c > i)
                    # TRIGGER (optimal-stopping commit): clone a sub-floor particle only
                    # when its projected reward r + Δ*remaining_scored_steps still cannot
                    # close the gap to tau ("remaining value can't reach the bar").
                    projected = r_cd + delta_r.clamp_min(0.0) * n_remaining
                    trig = projected < tau_v
                    rescuable = sub & trig
                    clone_log = {"r_pre": [round(float(x), 4) for x in r_cd.tolist()],
                                 "q": [round(float(x), 3) for x in q_cd.tolist()],
                                 "trigger": "project", "n_remaining": int(n_remaining),
                                 "cloned": []}
                    if acc.any() and rescuable.any():
                        acc_idx = torch.where(acc)[0]
                        # SEEDED clone-donor RNG: deterministic per (prompt-seed, step)
                        # so a fired clone picks a reproducible random accepted peer.
                        clone_gen = torch.Generator().manual_seed(int(seed) + 104729 * (i + 1))
                        for w in torch.where(rescuable)[0].tolist():
                            # never donate a target onto itself (no-op clone)
                            donor_idx = acc_idx[acc_idx != w]
                            if len(donor_idx) == 0:
                                continue
                            # random accepted donor peer
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
                        log(f"    CLONE step {i}: {ncl}/{int(sub.sum())} sub-floor "
                            f"cloned; r_pre={clone_log['r_pre']} "
                            + (f"cloned={clone_log['cloned']}" if ncl else ""))

            rec = {
                "step": int(i), "sigma": round(s_curr, 4), "p": round(p, 3),
                "reward_min": round(float(r.min()), 4),
                "reward_mean": round(float(r.mean()), 4),
                "reward_max": round(float(r.max()), 4),
                "reward_spread": round(float(r.max() - r.min()), 4),
                "q_gate": [round(x, 3) for x in qk.tolist()],
                "D_q": round(float(D_q), 4), "delta": round(delta_v, 4),
                "div_binding": float(sig_div) > 0.5,    # is the diversity floor active?
                "feat_cos_mean": round(dist_before, 4),
                "climb_on": bool(climb_on),
                "grad_norm_mean": round(float(gnorm.mean()), 4),
                "grad_norm_per": [round(float(x), 4) for x in gnorm.squeeze(1).tolist()],
                "n_clipped": n_clipped, "guidance_nan": n_nan,
                "channel_sched": channel_sched, "reward_ramp": bool(reward_ramp),
                "relocate": reloc_log,
                "clone": clone_log,
                "w_floor": [round(x, 5) for x in w_floor],   # per-particle floor weight |dL_dr|
                "w_rew": (round(w_rew, 4) if w_rew is not None else None),
                "w_div": (round(w_div, 4) if w_div is not None else None),
                "applied_step_norm": applied_norm,        # TRUE per-particle step magnitude
                "rew_post_norm": rew_post,                # post-clip,post-weight reward step (separate only)
                "div_post_norm": div_post,                # post-clip,post-weight diversity step (separate only)
                "n_rew_clipped": n_rew_clipped, "n_div_clipped": n_div_clipped,
                "chan_cos": chan_cos,                     # reward<->diversity direction cosine (separate only)
                "g_reward_norm": [round(x, 4) for x in g_rew_norm],
                "g_div_norm": [round(x, 4) for x in g_div_norm],
                "x0_2nd_order": used_2nd,
                "x0_corr_rel_mean": round(float(np.mean(x0_corr_rel)), 4),
                "x0_corr_rel_max": round(float(np.max(x0_corr_rel)), 4),
                "x0_drift_rel": ([round(x, 4) for x in x0_drift_rel]
                                 if x0_drift_rel is not None else None),
                "rank_churn": rank_churn,
                "img_min": round(img_min, 4), "img_max": round(img_max, 4),
                "counterfactual": cf,
                "step_time_s": round(_time.time() - t_step0, 2),
            }
            diag.append(rec)
            if verbose:
                gs = f" g_rew={np.mean(g_rew_norm):.3f} g_div={np.mean(g_div_norm):.3f}"
                log(f"  step {i} s={s_curr:.3f}: r[min/mean/max]="
                    f"{float(r.min()):.3f}/{float(r.mean()):.3f}/{float(r.max()):.3f} "
                    f"D_q={float(D_q):.3f}/delta={delta_v:.3f} bind={float(sig_div)>0.5} "
                    f"|grad|={float(gnorm.mean()):.3e}{gs} clip={n_clipped}/{k} "
                    f"x0corr={np.mean(x0_corr_rel):.3f} churn={rank_churn} "
                    f"img[{img_min:.2f},{img_max:.2f}] nan={n_nan} ({rec['step_time_s']}s)")
                if channel_sched != "combined":
                    # the numbers that actually drive the separate-channel step
                    log(f"    [sep] w_rew={w_rew:.3f} w_div={w_div:.3f} "
                        f"rew_post={np.mean(rew_post):.3f}(clip {n_rew_clipped}/{k}) "
                        f"div_post={np.mean(div_post):.3f}(clip {n_div_clipped}/{k}) "
                        f"chan_cos={np.mean(chan_cos):+.3f} "
                        f"applied={np.mean(applied_norm):.3f} max_step={max_step_v:.3f}")
                    if np.mean(div_post) < 1e-4:
                        log(f"  !! step {i}: diversity channel ~0 after clip "
                            f"(div_post={np.mean(div_post):.2e}) -- floor met or grad vanished")
                    if np.mean(rew_post) < 1e-4:
                        log(f"  !! step {i}: reward channel ~0 after clip "
                            f"(rew_post={np.mean(rew_post):.2e}) -- floor satisfied or grad vanished")
                    if np.mean(chan_cos) < -0.5:
                        log(f"  !! step {i}: reward & diversity channels strongly "
                            f"OPPOSED (chan_cos={np.mean(chan_cos):+.3f})")
                if n_nan:
                    log(f"  !! step {i}: {n_nan} NaN latent entries after guidance")
                if img_max > 1.01 or img_min < -0.01:
                    log(f"  !! step {i}: decoded image out of [0,1]: [{img_min:.3f},{img_max:.3f}]")

        # ---------- ordinary FK-Flow SDE step (all K alive; no clone) ----------
        mean, noise_std = fk_flow_kernel_mean(latents, v_all, s_curr, ds, sde_a)
        if noise_std <= 0.0:
            latents = mean
            continue
        eps_gen = torch.Generator(device=device).manual_seed(seed + 7919 * (i + 1))
        eps = torch.randn(latents.shape, device=device, dtype=latents.dtype, generator=eps_gen)
        latents = (mean + noise_std * eps).detach()

    # FINAL decode: MICROBATCH it. The per-step guidance decodes one particle at a
    # time (PASS A/B), so it fits at any k; but decoding all k latents in ONE VAE
    # call peaks at ~0.5GB/particle of decode activations on top of the ~64GB still
    # held by the run -> OOM at k=16 (verified: the crash was here, line ~730, the
    # single all-k decode, NOT the interacting guidance loop which completed all
    # score steps). Decode in chunks of <=4 with the cache freed first.
    if device == "cuda" or getattr(device, "type", "") == "cuda":
        torch.cuda.empty_cache()
    with torch.no_grad():
        img_chunks = []
        dec_bs = min(4, k)
        for c0 in range(0, k, dec_bs):
            img_chunks.append(
                decode_to_image(pipe, latents[c0:c0 + dec_bs], height, width).float().clamp(0, 1).cpu())
            n_decode += 1
        imgs = torch.cat(img_chunks, 0)
    arr = (imgs.float().clamp(0, 1).numpy() * 255).round().astype("uint8")
    from PIL import Image
    pil = [Image.fromarray(arr[j].transpose(1, 2, 0)) for j in range(arr.shape[0])]

    peak_mem_gb = (torch.cuda.max_memory_allocated() / 1e9
                   if (device == "cuda" or getattr(device, "type", "") == "cuda") else 0.0)

    # ---- FINAL-IMAGE score summary (the headline diagnostic: did the floor lift
    #      and did diversity hold on the ACTUAL returned images?). Score each final
    #      image with the same HPSv3 client; report min-K/mean-K reward + final
    #      block-12 feature spread. Saves a separate scoring pass when triaging runs.
    final_rewards = None
    try:
        with torch.no_grad():
            fr = []
            for j in range(imgs.shape[0]):
                rj, _ = client.reward_and_grad(imgs[j:j + 1].float(), [prompt])
                fr.append(float(rj[0]))
            final_rewards = fr
    except Exception as e:                                  # never let scoring kill a good run
        log(f"  !! final-image scoring failed ({type(e).__name__}: {e}); images still saved")
    final_summary = None
    if final_rewards is not None:
        fr_t = torch.tensor(final_rewards)
        # final feature spread: re-capture block-12 features for the returned latents
        try:
            with torch.no_grad():
                feats_final = []
                for kk in range(k):
                    predict_velocity(pipe, latents[kk:kk + 1], timesteps[-1], _slice_ps(ps, kk))
                    feats_final.append(capture.last_features.detach().float())
                ff = torch.cat(feats_final, 0)
                feat_cos_final = float(_cos_dist(ff)[iu[0], iu[1]].mean())
                n_fwd += k
        except Exception:
            feat_cos_final = None
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

    cost = {
        "wall_s": round(_time.time() - t_run0, 1),
        "n_flux_forwards": n_fwd, "n_vae_decodes": n_decode,
        "n_rollout_forwards": n_rollout,
        "peak_mem_gb": round(peak_mem_gb, 2),
        "second_order": True,
        "calibration": {kk2: (round(vv, 4) if isinstance(vv, float) else vv)
                        for kk2, vv in cal.items() if kk2 != "done"},
    }
    return pil, rewards_trace, {"steps": diag, "cost": cost, "final": final_summary}


def main(args):

    # --- deployed-config guardrail -------------------------------------------------
    # Print the active steering config so a degraded run is LOUD in the log, not silent
    # (a bare invocation must reproduce the deployed method: clone+ramp ON, 6 score steps,
    # step_size 1.0). This is the check that the B2 Pick-a-Pic IR run skipped.
    print(f"[ours config] reward={args.reward} tau_relmax_offset={args.tau_relmax_offset} "
          f"score_steps={args.score_steps} step_size={args.step_size} "
          f"lam_r={args.lam_r} lam_d={args.lam_d} "
          f"clone_dud={args.clone_dud} reward_ramp={args.reward_ramp}", flush=True)
    if (not args.clone_dud) or (not args.reward_ramp):
        print("[ours config] WARNING: DEGRADED config vs deployed — the headline method needs "
              "clone_dud=ON + reward_ramp=ON (the late copy lifts the worst-candidate floor). "
              "Disable ONLY for an explicit ablation.", flush=True)
    # -------------------------------------------------------------------------------

    score_steps = set(int(x) for x in args.score_steps.split(","))
    device = "cuda"
    dtype = torch.bfloat16

    prompts = load_prompts(ROOT / args.geneval_metadata,
                           set(args.prompt_ids.split(",")) if args.prompt_ids else None)
    if args.prompt_slice:
        a, b = args.prompt_slice.split(":")
        prompts = prompts[int(a):int(b)]
    if not prompts:
        log("ERROR: no prompts loaded"); sys.exit(1)
    log(f"loaded {len(prompts)} prompts: {[p['id'] for p in prompts]}")

    from diffusers import FluxPipeline

    log("loading FLUX.1-dev (flux venv)...")
    pipe = FluxPipeline.from_pretrained(
        "black-forest-labs/FLUX.1-dev", torch_dtype=dtype).to(device)
    try:
        pipe.transformer.enable_gradient_checkpointing()
        log("enabled gradient checkpointing on FLUX transformer")
    except Exception as e:
        log(f"WARN: could not enable gradient checkpointing: {e}")
    log("FLUX loaded.")


    n_gen = (args.height // 16) * (args.width // 16)
    div_blocks = [int(b) for b in str(args.div_blocks).split(",") if b.strip() != ""]
    joint_blocks = list(pipe.transformer.transformer_blocks)
    _store = {}
    _caps = []
    for _bi in div_blocks:
        if _bi >= len(joint_blocks):
            log(f"ERROR: block {_bi} out of range ({len(joint_blocks)} joint blocks)"); sys.exit(1)
        _c = _Block12Capture(num_generated_tokens=n_gen, store=_store, block_idx=_bi,
                             grid=args.div_grid, h_tok=args.height // 16, w_tok=args.width // 16)
        joint_blocks[_bi].attn.processor = _c
        _caps.append(_c)
    capture = _caps[0]   # all share _store; capture.last_features = band mean
    log(f"installed feature capture on joint blocks {div_blocks} (n_gen={n_gen})")

    _grad_factory = {"hpsv3": RewardClient.hpsv3_grad,
                     "imagereward": RewardClient.imagereward_grad}[args.reward]
    log(f"starting {args.reward} grad worker on worker_gpu={args.worker_gpu}...")
    client = _grad_factory(gpu_id=args.worker_gpu)

    out_root = ROOT / args.out_root
    method_dir = out_root / "ours"
    method_dir.mkdir(parents=True, exist_ok=True)
    summ_tag = (args.prompt_slice.replace(":", "_") if args.prompt_slice else "all")
    summary_path = method_dir / f"ours_summary_{summ_tag}.json"
    summary = {}
    # ---- GPU-usage sampler (cost tracking). Box is shared (8x A100); the clean
    # per-process figure is peak_mem_gb (torch); this adds GPU UTILIZATION which
    # torch can't report. Scope to the FLUX GPU(s) + the reward worker GPU. ----
    import os as _os
    _vis = _os.environ.get("CUDA_VISIBLE_DEVICES", "")
    _gpu_ids = [int(x) for x in _vis.split(",") if x.strip().isdigit()]
    if args.worker_gpu is not None and int(args.worker_gpu) not in _gpu_ids:
        _gpu_ids.append(int(args.worker_gpu))
    gpu_sampler = GpuSampler(gpu_ids=(_gpu_ids or None), period_s=2.0).start()
    log("RUN CONFIG: " + json.dumps({
        "k": args.k, "num_steps": args.num_steps, "score_steps": sorted(score_steps),
        "reward_ramp": bool(args.reward_ramp),
        "clone_dud": bool(args.clone_dud), "clone_start_p": args.clone_start_p,
        "clone_donor": "random", "clone_trigger": "project",
        "lam_r": args.lam_r, "lam_d": args.lam_d, "step_size": args.step_size,
        "tau_relmax_offset": args.tau_relmax_offset, "beta_r": args.beta_r,
        "beta_d": args.beta_d, "delta": args.delta, "delta_scale": args.delta_scale,
        "max_step": args.max_step, "sde_a": args.sde_a, "seed": args.seed,
    }))

    try:
        for pi, p in enumerate(prompts):
            pid, prompt = p["id"], p["prompt"]
            t0 = time.time()
            log(f"=== prompt {pid}: {prompt!r} ===")
            # PER-PROMPT SEED to match the baseline runner (per_prompt_seed =
            # args.seed + pid_int) so the K initial latents are IDENTICAL to
            # Base/FK/VASR/DAS on each prompt -> a PAIRED comparison. Parse pid as
            # int; fall back to enumeration index for non-integer IDs.
            try:
                pid_int = int(pid)
            except (TypeError, ValueError):
                pid_int = pi
            per_prompt_seed = args.seed + pid_int
            # beta_r/beta_d/delta/max_step auto-calibrate per-prompt at the first
            # scoring step (None -> data-driven, no magic constant); the steering
            # target tau_t = max(reward) - tau_relmax_offset is computed per step.
            pil, rewards_trace, diag = run_ours_one_prompt(
                pipe, client, capture, prompt, args.k, args.height, args.width, args.num_steps,
                device, dtype, per_prompt_seed, score_steps, args.beta_r,
                args.beta_d, args.delta, args.lam_r, args.lam_d, args.step_size,
                args.max_step, args.sde_a, reward_ramp=args.reward_ramp,
                clone_dud=args.clone_dud, clone_start_p=args.clone_start_p,
                tau_relmax_offset=args.tau_relmax_offset,
                delta_scale=args.delta_scale, verbose=args.verbose)
            pdir = method_dir / pid
            pdir.mkdir(parents=True, exist_ok=True)
            for j, im in enumerate(pil):
                im.save(pdir / f"sample_{j}.png")
            # dump the per-prompt structured debug trace next to the images
            with open(pdir / "diag.json", "w") as f:
                json.dump(diag, f, indent=2)
            cost = diag["cost"]
            summary[pid] = {"prompt": prompt, "rewards_trace": rewards_trace,
                            "final": diag.get("final"),
                            "elapsed_s": round(time.time() - t0, 1), "cost": cost}
            fin = diag.get("final")
            fin_str = (f" | FINAL min-K={fin['min_k']} mean-K={fin['mean_k']} "
                       f"feat_cos={fin['feat_cos_final']}" if fin else "")
            log(f"  saved {len(pil)} imgs to {pdir} | "
                f"{cost['wall_s']}s, {cost['n_flux_forwards']} fwd, "
                f"{cost.get('n_rollout_forwards', 0)} rollout-fwd, "
                f"{cost['n_vae_decodes']} decode, peak {cost['peak_mem_gb']}GB | "
                f"diag.json written{fin_str}")
            with open(summary_path, "w") as f:
                json.dump(summary, f, indent=2)
    finally:
        gpu_usage = gpu_sampler.stop()
        summary["_gpu_usage"] = gpu_usage
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        log("GPU usage: " + json.dumps(gpu_usage))
        client.close()

    log(f"DONE. images under {method_dir}; summary at {summary_path}")


