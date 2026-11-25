#!/usr/bin/env python3
"""Task / position perturbation rollouts on LIBERO-Object (PilotVLA).

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
  python tools/rollout_libero.py --out DIR --cond noiseopt [--gpu G --ngpu N]
  (or the pipeline scripts: scripts/task_perturbation.sh, scripts/position_perturbation.sh)
"""
import os
import sys
import json
import argparse
import time
import traceback
import gc

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch

from src.pi05_adapter import get_adapter, save_video
from src.libero_utils import build_single_env, all_object_tids, get_env_handles
from src.metrics import (find_inst, body_xyz, episode_metrics, joint_qpos_slice,
                         LIFT_DZ, BASKET_R)
from src.steering import GCTX, eef_xyz, extract_action_affine, install_guided_sampler


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
                            frames.append(f[::-1].copy())   # obs rows are bottom-up; no mirror
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
