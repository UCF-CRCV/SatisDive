"""SatisDive sana: ours. See README.md for the supported interface."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from sana_model import SanaModel, load_prompts, DEFAULT_MODEL_ID
from sde import fk_flow_kernel_mean
# NOTE: negtome.py is intentionally NOT imported here. The diversity term is a
# differentiable feature-distance gradient (see _BlockFeatureCapture), not a
# NegToMe/attention push. The NegToMe-push variant is the ablation (baselines.py).
from rewards.clients import RewardClient

ROOT = Path(__file__).resolve().parent


def log(m):
    print(f"[ours_sana] {m}", flush=True)


def _flat(x):
    """Flatten a latent (K, ...) to (K, D) float for per-particle norm calcs."""
    return x.reshape(x.shape[0], -1).float()


def _cos_dist(feats):
    """(K,K) cosine DISTANCE (1 - cosine sim) between (K,dim) feature vectors.

    Copied verbatim from ``flux/ours.py``. Cosine (not Euclidean) is scale-
    invariant across prompts and matches the verified Repel geometry; the toy's
    cdist is Euclidean only because its 2D output points have no meaningful scale
    to factor out."""
    fn = F.normalize(feats, dim=-1)
    sim = fn @ fn.t()
    return (1.0 - sim).clamp_min(0.0)


def _decode_to_image(model, latents):
    """Grad-enabled DC-AE decode of latents -> image tensor in [0,1], (K,3,H,W).

    ``SanaModel.decode`` is ``@torch.no_grad`` and returns PIL, so the guidance
    step (which backprops the reward through the decode) cannot use it. This
    replicates the numeric decode: divide by ``scaling_factor`` (DC-AE has NO
    ``shift_factor``, unlike the FLUX VAE), decode, then denormalise ``x/2 + 0.5``
    and clamp — the [0,1] range the ImageReward grad worker expects.


    diffusers VAE convention decode -> ~[-1,1], postprocess does /2 + 0.5); confirm
    on the first smoke that the decoded image is numerically sane in [0,1].
    """
    z = (latents / model.vae.config.scaling_factor).to(model.vae.dtype)
    image = model.vae.decode(z, return_dict=False)[0]  # (K,3,H,W) ~[-1,1]
    image = (image / 2 + 0.5).clamp(0, 1)
    return image.float()


class _BlockFeatureCapture:
    """Capture pooled SANA attn1 output features, preserving gradients and the conditioned CFG half."""

    def __init__(self, store=None):
        self.store = store if store is not None else {}

    def __call__(self, module, inputs, output):
        # The block returns the image-token hidden states; accept a bare tensor or
        # a tuple/list whose first element is that tensor.
        hs = output[0] if isinstance(output, (tuple, list)) else output
        # (B, num_tokens, dim) -> mean-pool over tokens -> (B, dim) -> L2-normalize.
        # KEEP the graph (do NOT detach): PASS B differentiates L_div through this.
        pooled = hs.float().mean(dim=1)                 # (B, dim)
        self.store["feat"] = F.normalize(pooled, dim=-1)  # (B, dim) unit, graph kept
        return None  # read-only: do not modify the block's output

    @property
    def last_features(self):
        """The full (B, dim) capture of the most recent forward (both CFG halves)."""
        return self.store.get("feat", None)

    def cond_features(self, do_cfg):
        """The conditioned-half features of the most recent forward, (B_cond, dim).

        Under CFG the batch is ``[uncond | cond]`` -> return the second half; with
        CFG off the whole batch is already the conditioned forward.
        """
        f = self.store.get("feat", None)
        if f is None:
            return None
        if do_cfg:
            return f[f.shape[0] // 2:]
        return f


class _PromptEmbeds:
    """CFG-stacked Gemma embeddings for one prompt, sliceable per particle.

    ``SanaModel.encode_prompt`` returns the diffusers 4-tuple
    ``(prompt_embeds, prompt_attention_mask, neg_embeds, neg_attention_mask)``,
    each replicated to K rows. ``SanaModel.predict_velocity`` expects the CFG-
    stacked ``[uncond|cond]`` embeds for whatever batch of latents it is handed
    (it does ``torch.cat([latents]*2)`` internally). This helper builds those
    stacks for either the full K batch (``all_k``) or a single particle
    (``one``), so the batched velocity forward and the per-particle grad passes
    share one source of truth.
    """

    def __init__(self, model, prompt, k, do_cfg=True):
        self.do_cfg = do_cfg
        pe, pam, ne, nam = model.encode_prompt(prompt, k, do_cfg=do_cfg)
        # pe/ne: (K, seq, dim); pam/nam: (K, seq). Keep on device/dtype as returned.
        self.pe, self.pam, self.ne, self.nam = pe, pam, ne, nam

    def all_k(self):
        """CFG-stacked embeds/mask for the whole K batch: [uncond(K)|cond(K)]."""
        if not self.do_cfg:
            return self.pe, self.pam
        embeds = torch.cat([self.ne, self.pe], dim=0)
        mask = None
        if self.pam is not None and self.nam is not None:
            mask = torch.cat([self.nam, self.pam], dim=0)
        return embeds, mask

    def one(self, kk):
        """CFG-stacked embeds/mask for a single particle kk: [uncond(1)|cond(1)]."""
        if not self.do_cfg:
            return self.pe[kk:kk + 1], (self.pam[kk:kk + 1] if self.pam is not None else None)
        embeds = torch.cat([self.ne[kk:kk + 1], self.pe[kk:kk + 1]], dim=0)
        mask = None
        if self.pam is not None and self.nam is not None:
            mask = torch.cat([self.nam[kk:kk + 1], self.pam[kk:kk + 1]], dim=0)
        return embeds, mask


def run_ours_one_prompt(
    model, client, capture, prompt, k, num_steps, seed,
    guidance_scale, height, width, score_steps,
    tau_relmax_offset=1.0, beta_r=None, beta_d=None, delta=None, delta_scale=1.0,
    lam_r=1.0, lam_d=1.0, step_size=1.0, max_step=None, sde_a=0.3,
    reward_ramp=True, clone_dud=True, clone_start_p=0.5,
    verbose=False,
):
    """Generate one batch with reward-floor and gated-diversity updates.

    Returns images, reward traces, and per-step diagnostics. Reward and diversity
    gradients are normalized separately before the latent update. Late-stage
    replacement uses a randomly selected candidate above the current cutoff.
    Null calibration arguments use first-scoring-step batch statistics.
    """
    import numpy as np
    import time as _time

    device = model.device
    dtype = model.dtype
    Delta = float(tau_relmax_offset)

    if str(device).startswith("cuda") or getattr(device, "type", "") == "cuda":
        torch.cuda.reset_peak_memory_stats()

    # ---- prompt + K initial latents (paired seeding, like base/negtome) --------
    embeds = _PromptEmbeds(model, prompt, k, do_cfg=(guidance_scale > 1.0))
    gen = torch.Generator(device=device).manual_seed(seed)
    num_ch = model.transformer.config.in_channels


    latents = model.pipe.prepare_latents(
        k, num_ch, height, width, dtype, device, gen)

    # ---- scheduler timesteps (clean Euler substrate for the FK-Flow SDE) -------
    if not getattr(model, "use_flow_euler", False):
        log("WARNING: OURS expects use_flow_euler=True (FlowMatchEuler) so the "
            "manual Euler step + FK-Flow SDE are exact; the shipped DPM solver is "
            "not a clean SDE substrate (see sde.py).")
    sched = model.scheduler


    sched.set_timesteps(num_steps, device=device)
    timesteps = sched.timesteps
    sigmas = sched.sigmas

    rewards_trace = []
    diag = []
    cal = {"beta_r": beta_r, "beta_d": beta_d,
           "delta": delta, "max_step": max_step, "done": False}
    v_prev = None            # cached CLEAN per-particle velocity for 2nd-order x̂₀
    n_fwd = 0                # SANA transformer forward counter (cost)
    n_decode = 0             # DC-AE decode counter (cost)
    prev_clone_r = None      # per-particle reward at the prev scored step (clone trigger)
    t_run0 = _time.time()

    for i, t in enumerate(timesteps):
        s_curr = float(sigmas[i])
        s_next = float(sigmas[i + 1])
        ds = s_next - s_curr  # < 0
        latents = latents.detach()
        is_score = i in score_steps

        v_clean = torch.zeros_like(latents)   # clean per-particle velocity (SDE + v_prev)
        rewards = torch.zeros(k, device=device)

        if is_score:
            t_step0 = _time.time()
            p = i / max(num_steps - 1, 1)
            # ---- PASS A (NO grad): per particle -> reward, worker d r/d image,
            #      clean velocity, the DETACHED block feature φ_k, and the 2nd-order
            #      (Heun) Tweedie x̂₀ used for the reward/feature proxy. -----------
            dimg_rew, feat_rows = [], []
            x0_corr_rel = []
            used_2nd = bool(v_prev is not None)
            img_min = img_max = None
            with torch.no_grad():
                for kk in range(k):
                    e_kk, m_kk = embeds.one(kk)
                    vk = model.predict_velocity(
                        latents[kk:kk + 1], t, e_kk, m_kk,
                        guidance_scale=guidance_scale,
                        do_cfg=embeds.do_cfg).to(torch.float32)
                    n_fwd += 1
                    feat_k = capture.cond_features(embeds.do_cfg).detach().float()  # (1,dim)
                    x0_1st = latents[kk:kk + 1].to(torch.float32) - s_curr * vk
                    if v_prev is not None:
                        v_use = 0.5 * (vk + v_prev[kk:kk + 1].to(torch.float32))
                    else:
                        v_use = vk
                    x0k = latents[kk:kk + 1].to(torch.float32) - s_curr * v_use
                    n1 = float(x0_1st.flatten().norm())
                    x0_corr_rel.append(float((x0k - x0_1st).flatten().norm()) / max(n1, 1e-8))
                    imgk = _decode_to_image(model, x0k.to(dtype))  # (1,3,H,W)[0,1]
                    n_decode += 1
                    bmin, bmax = float(imgk.min()), float(imgk.max())
                    img_min = bmin if img_min is None else min(img_min, bmin)
                    img_max = bmax if img_max is None else max(img_max, bmax)
                    rew_k, dimg_k = client.reward_and_grad(imgk.float(), [prompt])
                    rewards[kk] = float(rew_k[0])
                    v_clean[kk] = vk.detach().to(dtype)
                    dimg_k = torch.nan_to_num(dimg_k.to(imgk.device, imgk.dtype))
                    dimg_rew.append(dimg_k)          # d r_k / d image_k
                    feat_rows.append(feat_k)         # (1,dim) detached block feature

            feat_det = torch.cat(feat_rows, 0)       # (K,dim) detached features
            r = rewards.float()
            dmat = _cos_dist(feat_det)               # (K,K) FEATURE cosine distance
            iu = torch.triu_indices(k, k, offset=1, device=device)

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
            max_step_v = cal["max_step"]

            # ORACLE-FREE additive-margin target: τ_t = max(r) − Δ (shift-invariant;
            # the correct form for ImageReward's arbitrary-origin scale, matching
            # flux/ours.py). Keeps the best particle strictly above the bar
            # (max − τ = Δ > 0) so the donor pool never empties.
            tau_v = float(r.max()) - Delta

            # ---- set-coupled scalars (reward gate q, feature-space D_q) ----
            qk = torch.sigmoid((r - tau_v) / beta_r_v)     # (K,) soft acceptance
            qk_div = qk                                    # reward-gated pair weights
            qq = qk_div[iu[0]] * qk_div[iu[1]]
            dij = dmat[iu[0], iu[1]]
            denom = qq.sum() + 1e-6
            D_q = (qq * dij).sum() / denom                 # (reward-gated) set diversity
            sig_div = torch.sigmoid((delta_v - D_q) / beta_d_v)   # diversity-floor activation (diag)

            # REWARD-channel cotangent scalar dL_reward/dr_k: softplus FLOOR. Pushes
            # hard only below τ; SATURATES at the floor. Same as flux/ours.py.
            dL_dr = lam_r * (-1.0 / beta_r_v) * torch.sigmoid((tau_v - r) / beta_r_v)

            if verbose:
                log(f"  step {i} s={s_curr:.3f}: r[min/mean/max]="
                    f"{float(r.min()):.3f}/{float(r.mean()):.3f}/{float(r.max()):.3f} "
                    f"tau={tau_v:.3f}(Δ={Delta}) D_q={float(D_q):.3f}/delta={delta_v:.3f} "
                    f"bind={float(sig_div) > 0.5} x0corr={np.mean(x0_corr_rel):.3f} "
                    f"img[{img_min:.2f},{img_max:.2f}]")

            # ---- PASS B (grad): one forward + TWO backwards per particle (ALL K).
            #      The forward-hook holds φ_k WITH graph (diversity channel); the
            #      reward channel rides the worker's d r/d image cotangent. Each
            #      channel is unit-normalized (the ~1000x reward/feature scale gap),
            #      then combined into the single objective's latent update below.
            #      Others' features held fixed (the toy/FLUX do the same). --------
            g_rew_unit = torch.zeros_like(latents)
            g_div_unit = torch.zeros_like(latents)
            g_rew_norm = [0.0] * k
            g_div_norm = [0.0] * k
            w_floor = [0.0] * k          # per-particle floor weight |dL_dr|
            others_n = F.normalize(feat_det, dim=-1)       # (K,dim) fixed detached
            dij_fixed_all = dmat[iu[0], iu[1]].detach()
            qq_full = (qk_div[iu[0]] * qk_div[iu[1]]).detach()
            for kk in range(k):
                e_kk, m_kk = embeds.one(kk)
                xk = latents[kk:kk + 1].detach().to(torch.float32).requires_grad_(True)
                with torch.enable_grad():
                    vk = model.predict_velocity(
                        xk.to(dtype), t, e_kk, m_kk,
                        guidance_scale=guidance_scale,
                        do_cfg=embeds.do_cfg).to(torch.float32)
                    n_fwd += 1
                    feat_k = capture.cond_features(embeds.do_cfg).float()  # (1,dim) WITH graph
                    if v_prev is not None:
                        v_use = 0.5 * (vk + v_prev[kk:kk + 1].to(torch.float32).detach())
                    else:
                        v_use = vk
                    x0k = xk - s_curr * v_use
                    imgk = _decode_to_image(model, x0k.to(dtype))  # (1,3,H,W)[0,1]
                    n_decode += 1
                    # diversity loss = softplus floor on D_q^(k); D_q^(k) uses THIS
                    #   particle's live cosine distances to the (fixed) other feats.
                    fk_n = F.normalize(feat_k, dim=-1)             # (1,dim) live
                    dk = (1.0 - (fk_n * others_n).sum(1)).clamp_min(0.0)  # (K,) cosine dist
                    involves_k = (iu[0] == kk) | (iu[1] == kk)
                    other_idx = torch.where(iu[0] == kk, iu[1], iu[0])
                    dij_live = torch.where(involves_k, dk[other_idx], dij_fixed_all)
                    # q-weighted MEAN pair distance -> the set-diversity scalar D_q_k.
                    D_q_k = (qq_full * dij_live).sum() / denom.detach()
                    # NOTE: no lam_d here -- the channel weight is applied after norm.
                    L_div_k = F.softplus((delta_v - D_q_k) / beta_d_v)
                cot_img = dL_dr[kk] * dimg_rew[kk]
                # reward channel (through image) and diversity channel (through feat)
                g_rew = torch.autograd.grad(imgk, xk, grad_outputs=cot_img,
                                            retain_graph=True)[0]
                g_div = torch.autograd.grad(L_div_k, xk, retain_graph=False)[0]
                g_rew = torch.nan_to_num(g_rew)
                g_div = torch.nan_to_num(g_div)
                rn = float(g_rew.flatten().norm())
                dn = float(g_div.flatten().norm())
                g_rew_norm[kk] = rn
                g_div_norm[kk] = dn
                # unit-normalize each channel (strip the ~1000x cross-channel scale
                # gap AND SANA's particle-to-particle image-grad magnitude noise),
                # then re-apply the floor's own per-particle weight below as an
                # explicit scalar (worst particle pushed hardest). Matches flux/ours.
                g_rew_unit[kk] = (g_rew / max(rn, 1e-12)).detach().to(dtype)
                g_div_unit[kk] = (g_div / max(dn, 1e-12)).detach().to(dtype)
                w_floor[kk] = float(dL_dr[kk].abs())

            # ---- joint guidance step: x <- x - step * clip(grad) (separate sched) ----
            # 'separate' schedule (mirrors flux/ours.py): clip EACH channel to its
            # OWN max_step so the diversity channel cannot starve the reward floor.
            # FLOOR-WEIGHT the reward channel per particle by w_floor_k/max_k(w_floor)
            # in [0,1] (restores "worst pushed hardest" that the per-particle unit-norm
            # destroyed, without reintroducing the cross-channel imbalance). reward_ramp
            # ramps the reward-channel weight late (w_rew = 0.5 + 1.5*p) so the floor
            # commits late. Both verbatim from flux/ours.py.
            dist_before = float(dmat[iu[0], iu[1]].mean())     # feature spread pre-step
            wf = torch.tensor(w_floor, device=latents.device, dtype=torch.float32)
            wf_rel = (wf / wf.max().clamp_min(1e-12)).view(k, *([1] * (latents.ndim - 1)))
            g_rew_w = lam_r * wf_rel * g_rew_unit.float()
            g_div_w = lam_d * g_div_unit.float()

            def _clip_step(g):                                 # per-particle norm clip to max_step
                gn = g.flatten(1).norm(dim=1, keepdim=True).clamp_min(1e-12)
                return g * (max_step_v / gn).clamp(max=1.0).view(k, *([1] * (latents.ndim - 1)))

            g_rew_c = _clip_step(g_rew_w)
            g_div_c = _clip_step(g_div_w)
            w_div = 1.0
            w_rew = (0.5 + 1.5 * p) if reward_ramp else 1.0
            gclip = (w_rew * g_rew_c + w_div * g_div_c).to(dtype)

            # separate-path diagnostics (the quantities that actually drive the step)
            rn_pre = g_rew_w.flatten(1).norm(dim=1)
            dn_pre = g_div_w.flatten(1).norm(dim=1)
            rew_post = [round(x, 4) for x in (w_rew * g_rew_c).flatten(1).norm(dim=1).tolist()]
            div_post = [round(x, 4) for x in (w_div * g_div_c).flatten(1).norm(dim=1).tolist()]
            n_rew_clipped = int((rn_pre > max_step_v).sum())
            n_div_clipped = int((dn_pre > max_step_v).sum())
            rc = (w_rew * g_rew_c).flatten(1)
            dc = (w_div * g_div_c).flatten(1)
            chan_cos = [round(x, 4) for x in F.cosine_similarity(rc, dc, dim=1).tolist()]
            applied_norm = [round(x, 4) for x in (step_size * gclip).flatten(1).norm(dim=1).float().tolist()]

            latents = (latents - step_size * gclip).detach()
            n_nan = int(torch.isnan(latents).sum())

        # ---- velocity for the SDE step: batched forward over all K (clean; no
        #      push). At score steps this is taken AFTER the guidance update, so it
        #      is the velocity at the guided latent that drives the trajectory. At
        #      non-score steps it is the ordinary per-step velocity and also serves
        #      as the v_prev proxy. (The former NegToMe repulsion that used to ride
        #      this forward has been removed — diversity is a latent gradient now.) -
        e_all, m_all = embeds.all_k()
        with torch.no_grad():
            v_step = model.predict_velocity(
                latents, t, e_all, m_all,
                guidance_scale=guidance_scale, do_cfg=embeds.do_cfg).to(dtype)
            n_fwd += 1
        if not is_score:
            v_clean = v_step.clone()

        v_prev = v_clean.detach().clone()   # CLEAN velocity cached for next Heun x̂₀
        rewards_trace.append(rewards.detach().cpu().tolist())

        # ---- ONE LATE COPY (clone_dud): surgical replacement of a gradient-
        #      unreachable dud at LATE scored steps. Identical trigger to
        #      flux/ours.py: a below-floor particle whose PROJECTED final reward
        #      (r + Δ·remaining_scored_steps) still cannot reach τ is replaced by a
        #      FULL copy (γ=1) of a random accepted peer's latent; the remaining
        #      SDE noise re-separates the cloned pair. -----------------------------
        clone_log = None
        if clone_dud and is_score and k > 1 and p >= clone_start_p:
            with torch.no_grad():
                r_cd = torch.zeros(k, device=device)
                for kk in range(k):
                    e_kk, m_kk = embeds.one(kk)
                    vkk = model.predict_velocity(
                        latents[kk:kk + 1], t, e_kk, m_kk,
                        guidance_scale=guidance_scale,
                        do_cfg=embeds.do_cfg).to(torch.float32)
                    n_fwd += 1
                    if v_prev is not None:
                        v_use = 0.5 * (vkk + v_prev[kk:kk + 1].to(torch.float32))
                    else:
                        v_use = vkk
                    x0kk = latents[kk:kk + 1].to(torch.float32) - s_curr * v_use
                    imgkk = _decode_to_image(model, x0kk.to(dtype))
                    n_decode += 1
                    rk, _ = client.reward_and_grad(imgkk.float(), [prompt])
                    r_cd[kk] = float(rk[0])
                acc = r_cd > tau_v                       # accepted donor pool (> τ)
                is_dud = r_cd <= tau_v                   # below-floor clone targets
                if prev_clone_r is not None:
                    delta_r = r_cd - prev_clone_r
                else:
                    delta_r = torch.full((k,), float("inf"), device=device)
                n_remaining = sum(1 for c in score_steps if c > i)
                # optimal-stopping commit: projected reward still < τ.
                projected = r_cd + delta_r.clamp_min(0.0) * n_remaining
                rescuable = is_dud & (projected < tau_v)
                clone_log = {"r_pre": [round(float(x), 4) for x in r_cd.tolist()],
                             "n_remaining": int(n_remaining), "cloned": []}
                if acc.any() and rescuable.any():
                    acc_idx = torch.where(acc)[0]
                    clone_gen = torch.Generator().manual_seed(int(seed) + 104729 * (i + 1))
                    for w in torch.where(rescuable)[0].tolist():
                        donor_pool = acc_idx[acc_idx != w]
                        if len(donor_pool) == 0:
                            continue
                        donor = int(donor_pool[int(torch.randint(
                            len(donor_pool), (1,), generator=clone_gen))])
                        latents[w] = latents[donor].clone()   # FULL clone (γ=1)
                        # keep the SDE step consistent with the clone's new position:
                        # the donor's velocity IS correct for the shared latent.
                        v_step[w] = v_step[donor].clone()
                        clone_log["cloned"].append(
                            {"w": int(w), "donor": int(donor),
                             "r": round(float(r_cd[w]), 4),
                             "r_donor": round(float(r_cd[donor]), 4)})
                    latents = latents.detach()
                prev_clone_r = r_cd.detach()
                if verbose and clone_log["cloned"]:
                    log(f"    CLONE step {i}: {len(clone_log['cloned'])} sub-floor "
                        f"cloned; r_pre={clone_log['r_pre']}")

        # ---------- ordinary FK-Flow SDE step (all K alive; no resampling) -------
        mean, noise_std = fk_flow_kernel_mean(latents, v_step.to(torch.float32),
                                              s_curr, ds, sde_a)
        if noise_std <= 0.0:
            latents = mean.to(dtype).detach()
        else:
            eps_gen = torch.Generator(device=device).manual_seed(seed + 7919 * (i + 1))
            eps = torch.randn(latents.shape, device=device, dtype=torch.float32,
                              generator=eps_gen)
            latents = (mean + noise_std * eps).to(dtype).detach()

        if is_score:
            diag.append({
                "step": int(i), "sigma": round(s_curr, 4), "p": round(p, 3),
                "reward_min": round(float(r.min()), 4),
                "reward_mean": round(float(r.mean()), 4),
                "reward_max": round(float(r.max()), 4),
                "reward_spread": round(float(r.max() - r.min()), 4),
                "tau": round(tau_v, 4), "Delta": Delta,
                "q_gate": [round(x, 3) for x in qk.tolist()],
                "D_q": round(float(D_q), 4), "delta": round(delta_v, 4),
                "div_binding": float(sig_div) > 0.5,   # is the diversity floor active?
                "feat_cos_mean": round(dist_before, 4),
                "w_rew": round(float(w_rew), 4), "w_div": round(float(w_div), 4),
                "w_floor": [round(float(x), 5) for x in w_floor],
                "g_reward_norm": [round(x, 4) for x in g_rew_norm],
                "g_div_norm": [round(x, 4) for x in g_div_norm],
                "rew_post_norm": rew_post, "div_post_norm": div_post,
                "n_rew_clipped": n_rew_clipped, "n_div_clipped": n_div_clipped,
                "chan_cos": chan_cos, "applied_step_norm": applied_norm,
                "x0_2nd_order": used_2nd,
                "x0_corr_rel_mean": round(float(np.mean(x0_corr_rel)), 4),
                "guidance_nan": n_nan,
                "clone": clone_log,
                "img_min": round(img_min, 4), "img_max": round(img_max, 4),
                "step_time_s": round(_time.time() - t_step0, 2),
            })

    # ---- FINAL decode (microbatched, cache freed first) ------------------------
    if str(device).startswith("cuda") or getattr(device, "type", "") == "cuda":
        torch.cuda.empty_cache()
    with torch.no_grad():
        pil = []
        dec_bs = min(4, k)
        for c0 in range(0, k, dec_bs):
            imgs = _decode_to_image(model, latents[c0:c0 + dec_bs]).clamp(0, 1).cpu()
            n_decode += 1
            arr = (imgs.numpy() * 255).round().astype("uint8")
            from PIL import Image
            for j in range(arr.shape[0]):
                pil.append(Image.fromarray(arr[j].transpose(1, 2, 0)))

    peak_mem_gb = (torch.cuda.max_memory_allocated() / 1e9
                   if (str(device).startswith("cuda")
                       or getattr(device, "type", "") == "cuda") else 0.0)

    # ---- FINAL-IMAGE score summary (did the floor lift? did diversity hold?) ----
    final_summary = None
    try:
        with torch.no_grad():
            fr = []
            for j in range(len(pil)):
                arr = torch.from_numpy(
                    np.asarray(pil[j]).transpose(2, 0, 1).copy()
                ).float().unsqueeze(0) / 255.0
                rj, _ = client.reward_and_grad(arr, [prompt])
                fr.append(float(rj[0]))
        fr_t = torch.tensor(fr)
        final_summary = {
            "min_k": round(float(fr_t.min()), 4),
            "mean_k": round(float(fr_t.mean()), 4),
            "max_k": round(float(fr_t.max()), 4),
            "spread_k": round(float(fr_t.max() - fr_t.min()), 4),
            "per_particle": [round(x, 4) for x in fr],
        }
        log(f"  FINAL score: min-K={final_summary['min_k']} "
            f"mean-K={final_summary['mean_k']} max-K={final_summary['max_k']} "
            f"| per-particle={final_summary['per_particle']}")
    except Exception as e:                       # never let scoring kill a good run
        log(f"  !! final-image scoring failed ({type(e).__name__}: {e}); images saved")

    cost = {
        "wall_s": round(_time.time() - t_run0, 1),
        "n_sana_forwards": n_fwd, "n_vae_decodes": n_decode,
        "peak_mem_gb": round(peak_mem_gb, 2),
        "second_order": True,
        "calibration": {kk2: (round(vv, 4) if isinstance(vv, float) else vv)
                        for kk2, vv in cal.items() if kk2 != "done"},
    }
    return pil, rewards_trace, {"steps": diag, "cost": cost, "final": final_summary}


DEFAULT_SCORE_STEPS = "3,6,9,12,15,18"

# Diversity feature block/point. CHOSEN EMPIRICALLY by a one-off block-sweep probe in
# the development repo (sana_div_probe.py, not shipped here; 2026-07-13,
# 6 prompts, K=4): the diversity feature is the block's ``attn1`` SUB-OUTPUT (NOT the
# full block/residual-stream output — that is dominated by a large shared component,
# cosine≈0.999 → degenerate D_q≈0.001, the FLUX analog of capturing pre-to_out attn).
# Over all 20 blocks × {full,attn1} × {raw,centered}, ``attn1`` at BLOCK 7 gave the
# best DreamSim alignment (median pairwise cosine 0.34, Spearman ρ=0.91 with DreamSim);
# blocks 4/5/13 also strong (ρ 0.81–0.87). This is the SANA analog of FLUX's
# empirically-established block-12 (there, ρ was validated the same way).
DEFAULT_OURS_BLOCK_IDX = 7


def main(args):

    # deployed-config guardrail: make a degraded run LOUD (mirrors flux/ours.py).
    print(f"[ours_sana config] reward={args.reward} "
          f"tau_relmax_offset={args.tau_relmax_offset} score_steps={args.score_steps} "
          f"step_size={args.step_size} lam_r={args.lam_r} lam_d={args.lam_d} "
          f"ours_block_idx={args.ours_block_idx} "
          f"clone_dud={args.clone_dud} reward_ramp={args.reward_ramp} "
          f"sde_a={args.sde_a}", flush=True)
    if (not args.clone_dud) or (not args.reward_ramp):
        print("[ours_sana config] WARNING: DEGRADED config vs deployed — the headline "
              "method needs clone_dud=ON + reward_ramp=ON. Disable ONLY for an "
              "explicit ablation.", flush=True)

    score_steps = set(int(x) for x in args.score_steps.split(","))

    prompt_ids = set(args.prompt_ids.split(",")) if args.prompt_ids else None
    prompts = load_prompts(ROOT / args.metadata, prompt_ids)
    if args.limit is not None:
        prompts = prompts[:args.limit]
    if not prompts:
        log("ERROR: no prompts loaded (check --metadata / --prompt_ids)"); sys.exit(1)
    log(f"loaded {len(prompts)} prompts: {[p['id'] for p in prompts][:8]}"
        f"{' ...' if len(prompts) > 8 else ''}")

    log("loading SANA (cache-only)...")
    model = SanaModel(model_id=args.model_id, device=args.device,
                      use_flow_euler=(not args.no_flow_euler))
    # NOTE: gradient checkpointing is DELIBERATELY LEFT OFF. The diversity feature is
    # captured from the ``attn1`` SUB-module output (a checkpointed region's
    # intermediate); under gradient checkpointing that forward runs in no_grad and the
    # captured feature comes back DETACHED → the diversity channel would be dead. PASS B
    # does K=1 per-particle forwards, so activation memory fits an A100-80GB without
    # checkpointing (smoke: peak ~31 GB WITH checkpointing; verify the no-checkpoint peak
    # at re-smoke). If memory is tight, checkpoint every block EXCEPT the capture block.

    # ---- install the diversity feature capture: a read-only FORWARD HOOK on the
    #      chosen transformer block. It side-channels φ = normalize(mean_tokens(
    #      block_output)) WITH the autograd graph so PASS B can differentiate the
    #      feature-distance objective. Fires on every SANA forward; leaves the
    #      block output unchanged. This REPLACES the removed NegToMe repulsion. ----
    blocks = list(getattr(model.transformer, "transformer_blocks", []))
    if not blocks:
        log("ERROR: transformer has no `transformer_blocks` attribute"); sys.exit(1)
    if args.ours_block_idx < 0 or args.ours_block_idx >= len(blocks):
        log(f"ERROR: --ours_block_idx {args.ours_block_idx} out of range "
            f"({len(blocks)} transformer_blocks)"); sys.exit(1)
    capture = _BlockFeatureCapture()
    # Hook the block's ``attn1`` SUB-module (the diversity feature = attn1 output),
    # NOT the full block output (residual stream → degenerate; see DEFAULT_OURS_BLOCK_IDX).
    _cap_attn1 = getattr(blocks[args.ours_block_idx], "attn1", None)
    if _cap_attn1 is None:
        log(f"ERROR: transformer_blocks[{args.ours_block_idx}] has no .attn1"); sys.exit(1)
    hook_handle = _cap_attn1.register_forward_hook(capture)
    log(f"installed diversity feature-capture forward hook on "
        f"transformer_blocks[{args.ours_block_idx}].attn1 (of {len(blocks)} blocks)")

    log(f"starting {args.reward} grad worker on worker_gpu={args.worker_gpu}...")
    client = RewardClient.imagereward_grad(gpu_id=args.worker_gpu)

    out_dir = ROOT / args.output_dir
    method_dir = out_dir / "ours" / "images"
    method_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / "ours" / "ours_summary.json"
    if summary_path.exists():
        with open(summary_path) as f:
            summary = json.load(f)
        if not isinstance(summary, dict):
            raise ValueError(f"existing summary is not a mapping: {summary_path}")
        log(f"resuming with {len(summary)} completed prompts from {summary_path}")
    else:
        summary = {}

    try:
        for pi, rec in enumerate(prompts):
            pid, prompt = rec["id"], rec["prompt"]
            prompt_dir = method_dir / pid
            existing = sorted(prompt_dir.glob("candidate_*.png")) if prompt_dir.exists() else []
            if len(existing) == args.k and pid in summary and (prompt_dir / "diag.json").is_file():
                log(f"=== prompt {pid}: already complete; skipping ===")
                continue
            t0 = time.time()
            log(f"=== prompt {pid}: {prompt!r} ===")
            try:
                pid_int = int(pid)
            except (TypeError, ValueError):
                pid_int = pi
            per_prompt_seed = args.seed + pid_int

            pil, rewards_trace, diag = run_ours_one_prompt(
                model, client, capture, prompt, args.k, args.num_steps,
                per_prompt_seed, args.guidance_scale, args.height, args.width,
                score_steps, tau_relmax_offset=args.tau_relmax_offset,
                beta_r=args.beta_r, beta_d=args.beta_d, delta=args.delta,
                delta_scale=args.delta_scale, lam_r=args.lam_r, lam_d=args.lam_d,
                step_size=args.step_size, max_step=args.max_step, sde_a=args.sde_a,
                reward_ramp=args.reward_ramp, clone_dud=args.clone_dud,
                clone_start_p=args.clone_start_p, verbose=args.verbose)

            prompt_dir.mkdir(parents=True, exist_ok=True)
            for j, im in enumerate(pil):
                im.save(prompt_dir / f"candidate_{j}.png")
            with open(prompt_dir / "diag.json", "w") as f:
                json.dump(diag, f, indent=2)
            summary[pid] = {"prompt": prompt, "rewards_trace": rewards_trace,
                            "final": diag.get("final"),
                            "elapsed_s": round(time.time() - t0, 1),
                            "cost": diag["cost"]}
            with open(summary_path, "w") as f:
                json.dump(summary, f, indent=2)
            fin = diag.get("final")
            fin_str = (f" | FINAL min-K={fin['min_k']} mean-K={fin['mean_k']}"
                       if fin else "")
            log(f"  [{pi+1}/{len(prompts)}] saved {len(pil)} imgs to {prompt_dir} | "
                f"{diag['cost']['wall_s']}s{fin_str}")
    finally:
        hook_handle.remove()
        client.close()

    log(f"DONE. images under {method_dir}; summary at {summary_path}")


