#!/usr/bin/env python3
"""Swap-benchmark rollouts for a frozen pi0.5 with test-time steering (PilotVLA).

Benchmark: 10 libero_object scenes x all in-scene distractors = 50 swap cells.
  - instruction swap (default): the prompt asks for a distractor instead of the
    scene's canonical target. Success = asked-in-basket (the env's own success
    signal is zero by construction).
  - position swap (--posswap): the prompt is unchanged, but the target and the
    cell's distractor exchange initial xy. Success = env success signal.
  - --noswap: unmodified task (no-regression control).

Conditions (--cond):
  none      frozen policy, no intervention (baseline)
  noiseopt  PilotVLA: optimize ONLY the sampler's initial noise through a
            differentiable Euler unrolling of the flow ODE, minimizing a cosine
            loss between the chunk's planar displacement and the goal direction
  guide     flow guidance: per-Euler-step normalized gradient update on x_t
            using the closed-form terminal-chunk estimate (a0 = x_t - t*v_t)
  bon       best-of-N: sample N chunks, execute the one with the best cosine
            score (no gradients; measures what plain sampling already contains)

The goal direction g = (p_goal - p_eef)_xy is read from the simulator at every
replanning step and enters ONLY as a constant of the steering loss - the model
weights and inputs are never modified. Steering updates only the horizontal
(xy) action dimensions and is released before manipulation (lift latch by
default, or a fixed replanning horizon via --release_replans).

Noise seeds are a deterministic function of the episode index, so steered and
unsteered conditions share identical initial states and sampled noise
(paired comparisons). Episodes with an existing meta.json are skipped, making
every run resumable.

Usage:
  python -m pilotvla.rollout_swap --out DIR --cond noiseopt [--gpu G --ngpu N]
"""
import os
import sys
import json
import argparse
import time
import traceback
import gc
import types

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch

from pilotvla.pi05_adapter import get_adapter, save_video
from pilotvla.libero_utils import build_single_env, all_object_tids, get_env_handles
from pilotvla.metrics import (find_inst, body_xyz, episode_metrics, joint_qpos_slice,
                          LIFT_DZ, BASKET_R)

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


def scene_distractors(raw, target_inst, basket_inst):
    ds = [k for k in raw.obj_body_id
          if k not in (target_inst, basket_inst) and not k.startswith("basket")]
    return sorted(ds)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--ngpu", type=int, default=1)
    ap.add_argument("--checkpoint", default="lerobot/pi05_libero_finetuned")
    ap.add_argument("--cond", default="none",
                    choices=["none", "guide", "noiseopt", "bon"])
    ap.add_argument("--bon_n", type=int, default=16, help="best-of-N: candidate count")
    ap.add_argument("--eta", type=float, default=3.0,
                    help="guide: normalized-gradient step size in x-space")
    ap.add_argument("--opt_steps", type=int, default=10, help="noiseopt: Adam steps")
    ap.add_argument("--opt_lr", type=float, default=0.5, help="noiseopt: Adam lr (x-space)")
    ap.add_argument("--K", type=int, default=30,
                    help="steering loss horizon: first K chunk steps")
    ap.add_argument("--eps", type=int, default=20)
    ap.add_argument("--replan", type=int, default=5)
    ap.add_argument("--max-steps", type=int, default=300)
    ap.add_argument("--scenes", default="", help="explicit comma list of scene names")
    ap.add_argument("--dlimit", type=int, default=0,
                    help="limit distractors per scene (smoke tests)")
    ap.add_argument("--gt_once", action="store_true",
                    help="read the goal object position only once at episode start "
                         "(afterwards the direction updates via the EEF only)")
    ap.add_argument("--gt_noise", type=float, default=0.0,
                    help="Gaussian noise sigma (m) on the goal position - "
                         "robustness ablation for imperfect goal estimates")
    ap.add_argument("--use_z", action="store_true",
                    help="include z in the steering loss (dxyz vs g_xyz) - default xy only")
    ap.add_argument("--posswap", action="store_true",
                    help="position-swap mode: original instruction, target<->distractor "
                         "initial xy exchanged. asked = original target (new position); "
                         "env_success becomes the valid success signal")
    ap.add_argument("--noswap", action="store_true",
                    help="no-swap (original task) mode: original instruction and layout. "
                         "no-regression control on standard LIBERO-object")
    ap.add_argument("--reverse", action="store_true",
                    help="consume cells/episodes in reverse order - lets two jobs share "
                         "one work queue from both ends (meta.json skip avoids overlap)")
    ap.add_argument("--warm", action="store_true",
                    help="warm-start: reuse the previous replan's optimized noise")
    ap.add_argument("--release_replans", type=int, default=0,
                    help=">0: steer only the first N replans (fixed-horizon release; "
                         "10 replans = 50 action steps) instead of the lift latch")
    ap.add_argument("--save_tracks", action="store_true",
                    help="save per-episode EEF/goal xyz tracks to tracks.npz")
    args = ap.parse_args()

    if "CUDA_VISIBLE_DEVICES" not in os.environ:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.out, exist_ok=True)
    vid_dir = os.path.join(args.out, "videos")
    os.makedirs(vid_dir, exist_ok=True)

    tids = all_object_tids()
    all_scenes = sorted(tids)
    scenes = (args.scenes.split(",") if args.scenes else all_scenes)
    if args.reverse:
        scenes = scenes[::-1]
    assert all(s in tids for s in scenes), scenes

    adapter = get_adapter("pi05")
    print(f"[gpu{args.gpu}] loading {args.checkpoint}...", flush=True)
    adapter.load_model(args.checkpoint, device)
    adapter.policy.config.n_action_steps = args.replan
    install_guided_sampler(adapter.policy.model)
    GCTX["K"] = args.K
    GCTX["ndim"] = 3 if args.use_z else 2
    GCTX["eta"] = args.eta
    GCTX["mode"] = args.cond if args.cond != "none" else "guide"
    GCTX["opt_steps"] = args.opt_steps
    GCTX["opt_lr"] = args.opt_lr
    GCTX["bon_n"] = args.bon_n
    GCTX["on"] = args.cond != "none"
    GCTX["warm"] = args.warm
    GCTX["release_after"] = args.release_replans

    from lerobot.envs.utils import preprocess_observation, add_envs_task

    t0, n_done = time.time(), 0
    for O in scenes:
        env = None
        try:
            env, desc = build_single_env(adapter, "libero_object", tids[O], args.max_steps)
            lenv, raw, sim, orig_inits, off = get_env_handles(env)
            if GCTX["scale3"] is None:
                sc3, of3 = extract_action_affine(adapter, device)
                GCTX["scale3"], GCTX["offset3"] = sc3, of3
            o_inst = find_inst(raw, O)
            b_inst = find_inst(raw, "basket")
            ds_all = scene_distractors(raw, o_inst, b_inst)
            d_names = [d.rsplit("_", 1)[0] for d in ds_all]
            print(f"[gpu{args.gpu}] [{O}] tid{tids[O]} '{desc}' | distractors={d_names}",
                  flush=True)

            # cell = (scene, distractor); ngpu sharding by global cell index
            ds_iter = list(enumerate(ds_all[: args.dlimit or None]))
            if args.reverse:
                ds_iter = ds_iter[::-1]
            for di, d_inst in ds_iter:
                cell_idx = all_scenes.index(O) * len(ds_all) + di
                if cell_idx % args.ngpu != args.gpu:
                    continue
                D = d_inst.rsplit("_", 1)[0]
                if args.posswap or args.noswap:
                    prompt = desc                    # original instruction
                    asked_inst, asked_name = o_inst, O
                else:
                    prompt = desc.replace(O.replace("_", " "), D.replace("_", " "))
                    assert prompt != desc, f"prompt swap failed {O}->{D}"
                    asked_inst, asked_name = d_inst, D
                if args.cond == "guide":
                    ctag = f"guide{args.eta:g}"
                elif args.cond == "noiseopt":
                    ctag = f"noiseopt{'z' if args.use_z else ''}{args.opt_steps}x{args.opt_lr:g}"
                elif args.cond == "bon":
                    ctag = f"bon{args.bon_n}"
                else:
                    ctag = "none"
                if args.posswap:
                    ctag = "pos" + ctag
                if args.warm:
                    ctag += "-warm"
                if args.release_replans > 0:
                    ctag += f"-r{args.release_replans}"
                if args.gt_once:
                    ctag += "-once"
                if args.gt_noise > 0:
                    ctag += f"-gtn{args.gt_noise:g}"
                cell_dir = os.path.join(args.out, O, f"{ctag}_{D}")
                ep_iter = range(args.eps - 1, -1, -1) if args.reverse else range(args.eps)
                for k in ep_iter:
                    ep_dir = os.path.join(cell_dir, f"ep{k}")
                    if os.path.exists(os.path.join(ep_dir, "meta.json")):
                        continue
                    inits = orig_inits.copy()
                    if args.posswap:   # exchange target<->distractor xy in the init state
                        # robosuite recreates MjSim on reset -> always use fresh raw.sim
                        st = joint_qpos_slice(raw.sim, o_inst, off)
                        sd = joint_qpos_slice(raw.sim, d_inst, off)
                        row = inits[k % len(inits)]
                        qa, qb = row[st].copy(), row[sd].copy()
                        row[st][:2], row[sd][:2] = qb[:2], qa[:2]
                    lenv._init_states = inits
                    lenv.init_state_id = k % len(inits)
                    adapter.policy.reset()
                    GCTX["replan_i"] = 0           # reset warm/release counters per episode
                    GCTX["warm_x1"] = None
                    torch.manual_seed(1000 + k)   # match denoising noise across conditions
                    obs, _ = env.reset(seed=42 + k)
                    GCTX["diag"] = [] if GCTX["on"] else None

                    insts = {"target": o_inst, "distractor": d_inst,
                             "basket": b_inst, "asked": asked_inst}
                    track = {key: [] for key in insts}
                    eef_track, frames = [], []
                    success = False
                    asked_z0, asked_lifted = None, False
                    d_z0, wrong_lift_step = None, None   # posswap early-exit bookkeeping
                    gt_cache = None                       # for --gt_once (cached per episode)
                    gt_rng = np.random.RandomState(9000 + k)  # for --gt_noise (fixed per episode)
                    for step in range(args.max_steps):
                        if k == 0 and "pixels" in obs and "image" in obs["pixels"]:
                            f = obs["pixels"]["image"]
                            if hasattr(f, "cpu"):
                                f = f.cpu().numpy()
                            if f.ndim == 4:
                                f = f[0]
                            if f.dtype != np.uint8:
                                f = (f * 255).clip(0, 255).astype(np.uint8)
                            frames.append(f[::-1, ::-1].copy())
                        for name, inst in insts.items():
                            track[name].append(body_xyz(raw, inst))
                        ee = eef_xyz(raw)
                        eef_track.append(ee)
                        a_xyz = body_xyz(raw, asked_inst)
                        if asked_z0 is None:
                            asked_z0 = a_xyz[2]
                        if not asked_lifted and a_xyz[2] - asked_z0 > LIFT_DZ:
                            asked_lifted = True   # grasped - hand over to the model's place routine
                        if args.posswap:
                            d_xyz = body_xyz(raw, d_inst)
                            if d_z0 is None:
                                d_z0 = d_xyz[2]
                            if wrong_lift_step is None and d_xyz[2] - d_z0 > LIFT_DZ:
                                wrong_lift_step = step
                            # Decided failures end early (saves 300-step rollouts):
                            # (1) the bait lands in the basket, (2) only the bait was
                            # lifted and nothing changed for 120 steps.
                            if not asked_lifted and wrong_lift_step is not None:
                                bx_ = body_xyz(raw, b_inst)
                                if (np.linalg.norm(d_xyz[:2] - bx_[:2]) < BASKET_R
                                        and d_xyz[2] - d_z0 < LIFT_DZ):
                                    break
                                if step - wrong_lift_step > 120:
                                    break
                        a_gt = a_xyz
                        if args.gt_once:
                            if gt_cache is None:
                                gt_cache = a_xyz.copy()   # episode-start position only
                            a_gt = gt_cache
                        if args.gt_noise > 0:
                            a_gt = a_gt + gt_rng.normal(0, args.gt_noise, 3)
                        if args.release_replans > 0:
                            GCTX["on"] = (args.cond != "none")   # release via replan counter
                        else:
                            GCTX["on"] = (args.cond != "none") and not asked_lifted
                        GCTX["g_xy"] = (a_gt - ee)[:3 if args.use_z else 2]

                        obs_p = preprocess_observation(obs)
                        obs_p = add_envs_task(env, obs_p)
                        obs_p["task"] = [prompt]
                        obs_p = adapter.env_preprocessor(obs_p)
                        obs_p = adapter.preprocessor(obs_p)
                        with torch.no_grad():
                            action = adapter.policy.select_action(obs_p)
                        action = adapter.postprocessor(action)
                        action_np = adapter.env_postprocessor(
                            {"action": action})["action"].cpu().numpy()
                        obs, _, term, trunc, _ = env.step(action_np)
                        if term[0]:
                            success = True
                            break
                        # Swap success (asked in basket + set down) ends the episode
                        # early - the env only detects the ORIGINAL task's success.
                        if asked_lifted:
                            ax = body_xyz(raw, asked_inst)
                            bx = body_xyz(raw, b_inst)
                            if (np.linalg.norm(ax[:2] - bx[:2]) < BASKET_R
                                    and ax[2] - asked_z0 < LIFT_DZ):
                                break

                    met = episode_metrics({m: np.array(v) for m, v in track.items()},
                                          success)
                    ee_arr = np.array(eef_track)
                    ask_arr = np.array(track["asked"])
                    L = min(len(ee_arr), len(ask_arr))
                    dmin = float(np.linalg.norm(
                        (ee_arr[:L] - ask_arr[:L])[:, :3], axis=1).min())
                    met.update({"scene": O, "tid": tids[O], "cond": args.cond,
                                "bon_n": args.bon_n if args.cond == "bon" else 0,
                                "opt_steps": args.opt_steps if args.cond == "noiseopt" else 0,
                                "opt_lr": args.opt_lr if args.cond == "noiseopt" else 0.0,
                                "eta": args.eta if args.cond == "guide" else 0.0,
                                "K": args.K, "distractor": D, "asked": asked_name,
                                "mode": "posswap" if args.posswap else "instrswap",
                                "gt_once": bool(args.gt_once), "gt_noise": args.gt_noise,
                                "prompt": prompt, "ep": k, "seed": 42 + k,
                                "steps": step + 1, "replan": args.replan,
                                "min_eef_asked_dist": round(dmin, 4),
                                "reach": bool(dmin < 0.07),
                                "warm": bool(args.warm),
                                "release_replans": args.release_replans,
                                "guidance_released": bool(asked_lifted)})
                    if GCTX["diag"]:
                        dg = GCTX["diag"]
                        met["diag"] = {
                            "n_inf": len(dg),
                            "cos0_mean": round(float(np.mean([x["cos0"] for x in dg])), 3),
                            "cosF_mean": round(float(np.mean([x.get("cosF", np.nan) for x in dg])), 3),
                            "gnorm_mean": round(float(np.mean([x["gnorm"] for x in dg])), 4),
                            "xnorm0": round(dg[0]["xnorm"], 2)}
                    os.makedirs(ep_dir, exist_ok=True)
                    if args.save_tracks:
                        np.savez_compressed(os.path.join(ep_dir, "tracks.npz"),
                                            eef=ee_arr, asked=ask_arr)
                    json.dump(met, open(os.path.join(ep_dir, "meta.json"), "w"))
                    if frames:
                        tag = "S" if met["asked_in_basket"] else "F"
                        save_video(frames, os.path.join(
                            vid_dir, f"{O}_{ctag}_{D}_ep{k}_{tag}.mp4"), fps=20)
                    n_done += 1
                el = time.time() - t0
                print(f"[gpu{args.gpu}] [{O}/{ctag}_{D}] {args.eps} eps done "
                      f"({n_done} total, {el/60:.1f}min)", flush=True)
        except Exception:
            print(f"[gpu{args.gpu}] !! {O} FAILED:\n{traceback.format_exc()}", flush=True)
        finally:
            try:
                if env is not None:
                    env.close()
            except Exception:
                pass
            del env
            gc.collect(); torch.cuda.empty_cache()

    print(f"[gpu{args.gpu}] DONE: {n_done} eps in {(time.time()-t0)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
