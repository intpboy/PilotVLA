"""PilotVLA steering: initial-noise optimization inside pi0.5's flow sampler.

`install_guided_sampler` swaps the pi0.5 model's `sample_actions` for a
steering-aware copy. All steering state lives in the shared `GCTX` dict, which
the rollout tools fill in at every replanning step (goal direction, on/off,
release counters).

Conditions (GCTX["mode"]):
  noiseopt  PilotVLA: optimize ONLY the sampler's initial noise through a
            differentiable Euler unrolling of the flow ODE, minimizing a cosine
            loss between the chunk's displacement and the goal direction
  guide     flow guidance: per-Euler-step normalized gradient update on x_t
            using the closed-form terminal-chunk estimate (a0 = x_t - t*v_t)
  bon       best-of-N: sample N chunks, execute the one with the best cosine
            score (no gradients; measures what plain sampling already contains)

The goal direction enters ONLY as a constant of the steering loss - the model
weights and inputs are never modified. The loss uses the first GCTX["ndim"]
action dimensions (xy by default, xyz with ndim=3).
"""
import types

import numpy as np
import torch

CHUNK = 50
GCTX = {"on": False, "mode": "guide", "g_xy": None, "eta": 0.0, "K": 30, "ndim": 2,
        "scale3": None, "offset3": None, "diag": None,
        "opt_steps": 10, "opt_lr": 0.5, "bon_n": 16,
        # warm-start + fixed-horizon release:
        "warm": False, "warm_x1": None,     # if warm, reuse the previous optimized x1
        "replan_i": 0, "release_after": 0}  # release_after>0: steering OFF after that
                                            # many replans (caller resets per episode)


def eef_xyz(raw):
    sim = raw.sim
    sid = sim.model.site_name2id("gripper0_grip_site")
    return np.array(sim.data.site_xpos[sid][:3], dtype=float, copy=True)


def extract_action_affine(adapter, device):
    """Affine (unnorm = scale*x + offset) of the first 3 action dims, from the
    postprocessor's Unnormalizer stats. pi05 uses QUANTILES normalization, so
    q01/q99 take precedence over mean/std."""
    post = adapter.postprocessor
    steps = getattr(post, "steps", None) or getattr(post, "processors", None) or []
    for s in steps:
        st = getattr(s, "stats", None)
        if not st or "action" not in st:
            continue
        a = st["action"]
        def arr(k):
            v = a[k]
            return torch.as_tensor(np.asarray(v), dtype=torch.float32, device=device)
        keys = set(a.keys())
        if {"q01", "q99"} <= keys:
            q01, q99 = arr("q01"), arr("q99")
            scale, offset = (q99 - q01) / 2, (q99 + q01) / 2
        elif {"mean", "std"} <= keys:
            scale, offset = arr("std"), arr("mean")
        elif {"min", "max"} <= keys:
            lo, hi = arr("min"), arr("max")
            scale, offset = (hi - lo) / 2, (hi + lo) / 2
        else:
            continue
        print(f"[affine] unnormalizer keys={sorted(keys)} scale3={scale[:3].tolist()} "
              f"offset3={offset[:3].tolist()}", flush=True)
        return scale[:3], offset[:3]
    raise RuntimeError(f"action unnormalizer not found in postprocessor ({type(post)}; "
                       f"steps={[type(s).__name__ for s in steps]})")


def install_guided_sampler(model):
    """Replace pi05's sample_actions loop with a steering-aware copy.

    The sampling loop is replicated from lerobot's PI05 modeling code
    (Apache License 2.0) with the steering terms added. Steering is active
    only while GCTX['on'] is set.
    """
    from lerobot.policies.pi05.modeling_pi05 import make_att_2d_masks

    def guided_sample_actions(self, images, img_masks, tokens, masks,
                              noise=None, num_steps=None, **kwargs):
        if num_steps is None:
            num_steps = self.config.num_inference_steps
        bsize = tokens.shape[0]
        device = tokens.device
        with torch.no_grad():
            if noise is None:
                noise = self.sample_noise(
                    (bsize, self.config.chunk_size, self.config.max_action_dim), device)
            prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(
                images, img_masks, tokens, masks)
            prefix_att_2d = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
            prefix_pos = torch.cumsum(prefix_pad_masks, dim=1) - 1
            att4d = self._prepare_attention_masks_4d(prefix_att_2d)
            self.paligemma_with_expert.paligemma.model.language_model.config._attn_implementation = "eager"  # noqa: SLF001
            _, past_kv = self.paligemma_with_expert.forward(
                attention_mask=att4d, position_ids=prefix_pos, past_key_values=None,
                inputs_embeds=[prefix_embs, None], use_cache=True)

        dt = -1.0 / num_steps

        # Fixed-horizon release: count replans; past release_after, run plain.
        GCTX["replan_i"] = GCTX.get("replan_i", 0) + 1
        _released = (GCTX.get("release_after", 0) > 0
                     and GCTX["replan_i"] > GCTX["release_after"])
        _on = GCTX["on"] and not _released

        def unroll(x1, differentiable):
            """Full Euler unrolling from x1 to the terminal chunk x0."""
            ctx = torch.enable_grad() if differentiable else torch.no_grad()
            with ctx:
                x = x1
                for step in range(num_steps):
                    tt = torch.tensor(1.0 + step * dt, dtype=torch.float32,
                                      device=device).expand(bsize)
                    v = self.denoise_step(prefix_pad_masks=prefix_pad_masks,
                                          past_key_values=past_kv,
                                          x_t=x, timestep=tt)
                    x = x + dt * v
            return x

        if _on and GCTX["mode"] == "bon" and GCTX["g_xy"] is not None:
            # best-of-N: no gradients. Sample N chunks, execute the best-scoring one.
            g = torch.as_tensor(GCTX["g_xy"], dtype=torch.float32, device=device)
            g = g / (g.norm() + 1e-9)
            sc3, of3 = GCTX["scale3"], GCTX["offset3"]
            K = GCTX["K"]
            best_x, best_s, scores = None, None, []
            with torch.no_grad():
                for cand in range(GCTX["bon_n"]):
                    n_i = noise if cand == 0 else self.sample_noise(noise.shape, device)
                    x0 = unroll(n_i, differentiable=False)
                    disp3 = (x0[:, :K, :3].float() * sc3 + of3).sum(1)
                    dxy = disp3[:, :GCTX["ndim"]]
                    cosv = float((dxy * g).sum(-1) / (dxy.norm(dim=-1) + 1e-9))
                    scores.append(cosv)
                    if best_s is None or cosv > best_s:
                        best_s, best_x = cosv, x0
            if GCTX["diag"] is not None:
                ss = sorted(scores)
                GCTX["diag"].append({"cos0": ss[len(ss) // 2], "cosF": best_s,
                                     "gnorm": ss[0], "xnorm": float(best_x.norm())})
            return best_x.detach()

        if _on and GCTX["mode"] == "noiseopt" and GCTX["g_xy"] is not None:
            # PilotVLA: optimize ONLY the initial noise - the executed chunk is the
            # unmodified model's own output for the optimized noise, so the
            # search stays on the policy's output manifold.
            g = torch.as_tensor(GCTX["g_xy"], dtype=torch.float32, device=device)
            g = g / (g.norm() + 1e-9)
            sc3, of3 = GCTX["scale3"], GCTX["offset3"]
            K = GCTX["K"]
            init_noise = noise
            # warm-start: reuse the previous replan's optimized x1
            # (the caller resets warm_x1=None at every episode start)
            if GCTX.get("warm") and GCTX.get("warm_x1") is not None:
                init_noise = GCTX["warm_x1"].to(device)
            x1p = torch.nn.Parameter(init_noise.detach().clone().float())
            opt = torch.optim.Adam([x1p], lr=GCTX["opt_lr"])
            cos0 = cosF = None
            for s in range(GCTX["opt_steps"]):
                with torch.enable_grad():   # pierce the caller's no_grad (select_action)
                    x0 = unroll(x1p, differentiable=True)
                    disp3 = (x0[:, :K, :3] * sc3 + of3).sum(1)
                    dxy = disp3[:, :GCTX["ndim"]]
                    cosv = (dxy * g).sum(-1) / (dxy.norm(dim=-1) + 1e-9)
                    loss = (1 - cosv).sum()
                    if s == 0:
                        cos0 = float(cosv.sum())
                    opt.zero_grad()
                    loss.backward()
                opt.step()
                cosF = float(cosv.sum())
            if GCTX["diag"] is not None:
                GCTX["diag"].append({"cos0": cos0, "cosF": cosF,
                                     "gnorm": float(x1p.grad.norm()),
                                     "xnorm": float(x1p.detach().norm())})
            if GCTX.get("warm"):
                GCTX["warm_x1"] = x1p.detach().clone()   # next replan's starting point
            return unroll(x1p.detach().to(noise.dtype), differentiable=False).detach()

        x_t = noise
        guide = _on and GCTX["mode"] == "guide" \
            and GCTX["g_xy"] is not None and GCTX["eta"] > 0
        if guide:
            g = torch.as_tensor(GCTX["g_xy"], dtype=torch.float32, device=device)
            g = g / (g.norm() + 1e-9)
            sc3, of3 = GCTX["scale3"], GCTX["offset3"]
            K = GCTX["K"]
        for step in range(num_steps):
            t_now = 1.0 + step * dt
            t_tensor = torch.tensor(t_now, dtype=torch.float32, device=device).expand(bsize)
            if guide:
                with torch.enable_grad():
                    x_in = x_t.detach().clone().requires_grad_(True)
                    v_t = self.denoise_step(prefix_pad_masks=prefix_pad_masks,
                                            past_key_values=past_kv,
                                            x_t=x_in, timestep=t_tensor)
                    a0 = x_in - t_now * v_t                     # closed-form terminal estimate
                    disp3 = (a0[:, :K, :3] * sc3 + of3).sum(1)  # unnormalized cumulative dxyz
                    dxy = disp3[:, :GCTX["ndim"]]
                    cosv = (dxy * g).sum(-1) / (dxy.norm(dim=-1) + 1e-9)
                    loss = (1 - cosv).sum()
                    grad = torch.autograd.grad(loss, x_in)[0]
                if GCTX["diag"] is not None:
                    if step == 0:
                        GCTX["diag"].append({"cos0": float(cosv.mean()),
                                             "gnorm": float(grad.norm()),
                                             "xnorm": float(x_t.norm())})
                    if step == num_steps - 1:
                        GCTX["diag"][-1]["cosF"] = float(cosv.mean())
                # Normalized gradient: eta is an absolute step size in x-space
                # (the raw gradient norm is ~1e-3 vs ||x|| ~ 40, i.e. inert).
                gdir = grad / (grad.norm() + 1e-9)
                x_t = (x_t + dt * v_t.detach() - GCTX["eta"] * gdir).detach()
            else:
                with torch.no_grad():
                    v_t = self.denoise_step(prefix_pad_masks=prefix_pad_masks,
                                            past_key_values=past_kv,
                                            x_t=x_t, timestep=t_tensor)
                    x_t = x_t + dt * v_t
        return x_t.detach()

    model.sample_actions = types.MethodType(guided_sample_actions, model)
