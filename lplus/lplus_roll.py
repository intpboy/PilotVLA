#!/usr/bin/env python
"""LIBERO-Plus steering rollouts (libero_spatial): none / oracle noiseopt.

Run lplus/setup_liberoplus.py once first - it builds the pkgshadow package,
the merged assets view, and the LIBERO config this script relies on.

Isolation: LIBERO_CONFIG_PATH (dedicated config) + stripping libero meta-path
hooks + pkgshadow (a concrete `libero` package) make the LIBERO-plus benchmark
a drop-in without touching the base environment.
Target: the first argument of the BDDL goal (On <bowl_k> <place>) is the
correct bowl instance (among look-alike bowls). Success: the env's own signal.
Seeds are matched across conditions; episodes with meta.json are skipped.

Usage:
  python lplus/lplus_roll.py --axis language --cond noiseopt --eps 10 \
      --gpu G --ngpu N --out DIR
"""
import os
import sys
import json
import argparse
import re

LPD = os.path.dirname(os.path.abspath(__file__))
os.environ["LIBERO_CONFIG_PATH"] = os.path.join(LPD, "libero_config")
sys.meta_path = [f for f in sys.meta_path if "libero" not in repr(f).lower()]
sys.path.insert(0, os.path.join(LPD, "pkgshadow"))
import libero  # noqa: E402  (must resolve to pkgshadow)
assert "pkgshadow" in libero.__file__, libero.__file__

sys.path.insert(0, os.path.dirname(LPD))  # repo root
from pilotvla.rollout_swap import (  # noqa: E402
    install_guided_sampler, GCTX, eef_xyz, extract_action_affine)
from pilotvla.pi05_adapter import get_adapter, save_video  # noqa: E402
from pilotvla.libero_utils import build_single_env, get_env_handles  # noqa: E402
from pilotvla.metrics import find_inst, body_xyz  # noqa: E402
# lerobot's helper assembles init-file paths directly; route through the
# suite's own method instead (LIBERO-plus layout differs).
import lerobot.envs.libero as _lel  # noqa: E402
_orig_gtis = _lel.get_task_init_states
def _patched_gtis(task_suite, i):
    if hasattr(task_suite, "get_task_init_states"):
        import torch as _t
        _ld = _t.load
        _t.load = lambda *a, **k: _ld(*a, **{**k, "weights_only": False})
        try:
            return np.asarray(task_suite.get_task_init_states(i))
        finally:
            _t.load = _ld
    return _orig_gtis(task_suite, i)
_lel.get_task_init_states = _patched_gtis
# LIBERO-plus envs resolve ../assets relative to realpath(__file__), which
# escapes our symlinked view - force DIR_PATH to the pkgshadow package.
import libero.libero.envs.bddl_base_domain as _bbd  # noqa: E402
_bbd.DIR_PATH = os.path.join(LPD, "pkgshadow", "libero", "libero", "envs")
import numpy as np  # noqa: E402
import torch  # noqa: E402

AXIS_MAP = {"layout": "Objects Layout", "language": "Language Instructions",
            "camera": "Camera Viewpoints", "robotinit": "Robot Initial States"}


def goal_target(bddl_path):
    """(:goal (And (On akita_black_bowl_2 plate_1))) -> akita_black_bowl_2"""
    txt = open(bddl_path).read()
    g = txt[txt.index("(:goal"):]
    m = re.search(r"\(On\s+(\S+)\s+\S+\)", g, re.IGNORECASE)
    assert m, bddl_path
    return m.group(1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--axis", required=True,
                    choices=["layout", "language", "camera", "robotinit"])
    ap.add_argument("--cond", default="none", choices=["none", "noiseopt"])
    ap.add_argument("--use_z", action="store_true",
                    help="include z in the steering loss (dxyz vs g_xyz)")
    ap.add_argument("--eps", type=int, default=10)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--ngpu", type=int, default=1)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-steps", type=int, default=300)
    ap.add_argument("--replan", type=int, default=5)
    ap.add_argument("--opt_steps", type=int, default=10)
    ap.add_argument("--opt_lr", type=float, default=0.5)
    ap.add_argument("--K", type=int, default=30)
    ap.add_argument("--limit", type=int, default=0, help="smoke: cap instance count")
    ap.add_argument("--difficulty", type=int, default=0,
                    help="run only this difficulty (difficulty_got; 0 = all)")
    ap.add_argument("--gt_noise_cm", type=float, default=0.0,
                    help="robustness ablation: fixed per-episode xy offset (cm) "
                         "injected into the steering target")
    ap.add_argument("--reverse", action="store_true",
                    help="consume instances/episodes in reverse order - lets two "
                         "jobs share one queue (meta.json skip avoids overlap)")
    ap.add_argument("--checkpoint", default="lerobot/pi05_libero_finetuned")
    ap.add_argument("--warm", action="store_true",
                    help="warm-start: reuse the previous optimized x1")
    ap.add_argument("--release_replans", type=int, default=0,
                    help=">0: steer only the first N replans instead of the "
                         "lift/proximity release")
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.out, exist_ok=True)
    vid_dir = os.path.join(args.out, "videos")
    os.makedirs(vid_dir, exist_ok=True)

    sel = json.load(open(os.path.join(LPD, "selection.json")))
    p2 = os.path.join(LPD, "selection2.json")   # Tier2 (camera/robotinit)
    if os.path.exists(p2):
        sel += json.load(open(p2))
    sel = [s for s in sel if s["axis"] == AXIS_MAP[args.axis]]
    seen, insts = set(), []
    for s in sel:                      # drop duplicate selections
        if s["id"] not in seen:
            seen.add(s["id"])
            insts.append(s)
    if args.difficulty:
        insts = [s for s in insts if s["difficulty_got"] == args.difficulty]
    if args.limit:
        insts = insts[: args.limit]
    if args.reverse:
        insts = insts[::-1]

    from libero.libero import benchmark as lb
    suite = lb.get_benchmark_dict()["libero_spatial"]()
    name2idx = {suite.get_task(i).name: i for i in range(suite.get_num_tasks())}
    from libero.libero import get_libero_path
    bddl_root = get_libero_path("bddl_files")

    adapter = get_adapter("pi05")
    print(f"[gpu{args.gpu}] loading {args.checkpoint}...", flush=True)
    adapter.load_model(args.checkpoint, device)
    adapter.policy.config.n_action_steps = args.replan
    install_guided_sampler(adapter.policy.model)
    GCTX["K"] = args.K
    GCTX["mode"] = "noiseopt"
    GCTX["ndim"] = 3 if args.use_z else 2
    GCTX["opt_steps"] = args.opt_steps
    GCTX["opt_lr"] = args.opt_lr
    GCTX["warm"] = args.warm
    GCTX["release_after"] = args.release_replans

    from lerobot.envs.utils import preprocess_observation, add_envs_task

    ctag = "none" if args.cond == "none" else (
        f"noiseopt{'z' if args.use_z else ''}{args.opt_steps}x{args.opt_lr:g}"
        + ("-warm" if args.warm else "")
        + (f"-r{args.release_replans}" if args.release_replans > 0 else "")
        + (f"-gt{args.gt_noise_cm:g}cm" if args.gt_noise_cm else ""))
    LIFT_DZ = 0.04

    for wi, s in enumerate(insts):
        if wi % args.ngpu != args.gpu:
            continue
        tname = s["name"]
        assert tname in name2idx, tname
        tid = name2idx[tname]
        env = None
        try:
            env, desc = build_single_env(adapter, "libero_spatial", tid, args.max_steps)
            lenv, raw, sim, orig_inits, off = get_env_handles(env)
            if GCTX["scale3"] is None:
                sc3, of3 = extract_action_affine(adapter, device)
                GCTX["scale3"], GCTX["offset3"] = sc3, of3
            tk = suite.get_task(tid)
            bf = tk.bddl_file
            if "_view_" in bf:               # lplus convention: strip _view_ suffix
                bf = bf.split("_view_")[0] + ".bddl"
            tgt_bddl = goal_target(os.path.join(bddl_root, tk.problem_folder, bf))
            t_inst = find_inst(raw, tgt_bddl)
            wrongs = [i2 for i2 in raw.obj_body_id
                      if "akita_black_bowl" in i2 and i2 != t_inst]
            print(f"[gpu{args.gpu}] [{wi}] {tname[:60]} lvl{s['difficulty_got']} "
                  f"| target={t_inst} | '{desc[:50]}'", flush=True)

            cell_dir = os.path.join(args.out, args.axis, tname, ctag)
            ep_iter = range(args.eps - 1, -1, -1) if args.reverse else range(args.eps)
            for k in ep_iter:
                ep_dir = os.path.join(cell_dir, f"ep{k}")
                if os.path.exists(os.path.join(ep_dir, "meta.json")):
                    continue
                lenv._init_states = np.array(lenv._init_states, dtype=np.float64,
                                             copy=True)
                lenv.init_state_id = k % max(1, len(lenv._init_states))
                adapter.policy.reset()
                GCTX["replan_i"] = 0           # reset warm/release counters per episode
                GCTX["warm_x1"] = None
                torch.manual_seed(1000 + k)
                obs, _ = env.reset(seed=42 + k)
                # robustness ablation: fixed per-episode xy offset
                # (direction seed = f(task, ep), paired across conditions)
                gt_off = np.zeros(3)
                if args.gt_noise_cm:
                    th = np.random.default_rng(777000 + tid * 1000 + k).uniform(
                        0, 2 * np.pi)
                    gt_off = np.array([np.cos(th), np.sin(th), 0.0]) \
                        * args.gt_noise_cm / 100.0
                GCTX["diag"] = [] if args.cond != "none" else None

                eef_track, t_track, frames = [], [], []
                success = False
                t_z0, lifted = None, False
                w_z0, wrong_lift_step = {}, None
                for step in range(args.max_steps):
                    if k == 0:
                        f = obs["pixels"]["image"]
                        if hasattr(f, "cpu"):
                            f = f.cpu().numpy()
                        if f.ndim == 4:
                            f = f[0]
                        if f.dtype != np.uint8:
                            f = (f * 255).clip(0, 255).astype(np.uint8)
                        frames.append(f[::-1, ::-1].copy())
                    t_xyz = body_xyz(raw, t_inst)
                    ee = eef_xyz(raw)
                    t_track.append(t_xyz)
                    eef_track.append(ee)
                    if t_z0 is None:
                        t_z0 = t_xyz[2]
                    if not lifted and t_xyz[2] - t_z0 > LIFT_DZ:
                        lifted = True
                    if not lifted:
                        for wj in wrongs:
                            wz = body_xyz(raw, wj)[2]
                            if wj not in w_z0:
                                w_z0[wj] = wz
                            elif wrong_lift_step is None and wz - w_z0[wj] > LIFT_DZ:
                                wrong_lift_step = step
                        if (wrong_lift_step is not None
                                and step - wrong_lift_step > 120):
                            break   # only a wrong bowl lifted, stalled - decided failure
                    a_gt = t_xyz + gt_off   # only the steering target is perturbed
                    # Proximity release: bowls are rim-grasped, so pulling the EEF
                    # toward the center blocks the grasp (dmin stalls at 2-4 cm in
                    # failures) - steering off within 8 cm of the target.
                    near = float(np.linalg.norm((a_gt - ee)[:2])) < 0.08
                    if args.release_replans > 0:
                        GCTX["on"] = (args.cond != "none")  # release via replan counter
                    else:
                        GCTX["on"] = (args.cond != "none") and not lifted and not near
                    GCTX["g_xy"] = (a_gt - ee)[:3 if args.use_z else 2]

                    obs_p = preprocess_observation(obs)
                    obs_p = add_envs_task(env, obs_p)
                    obs_p["task"] = [desc]
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

                ee_arr = np.array(eef_track)
                tt_arr = np.array(t_track)
                L = min(len(ee_arr), len(tt_arr))
                dmin = float(np.linalg.norm((ee_arr[:L] - tt_arr[:L])[:, :3],
                                            axis=1).min())
                met = {"suite": "libero_spatial", "axis": args.axis,
                       "task": tname, "tid": int(tid),
                       "difficulty": s["difficulty_got"],
                       "difficulty_want": s["difficulty_want"],
                       "cond": args.cond,
                       "target_inst": t_inst, "prompt": desc,
                       "ep": k, "seed": 42 + k, "steps": step + 1,
                       "success": bool(success),
                       "min_eef_target_dist": round(dmin, 4),
                       "reach": bool(dmin < 0.07),
                       "lifted": bool(lifted),
                       "wrong_lift": wrong_lift_step is not None}
                if args.gt_noise_cm:
                    met["gt_noise_cm"] = args.gt_noise_cm
                    met["gt_off_xy"] = [round(float(v), 4) for v in gt_off[:2]]
                if GCTX["diag"]:
                    dg = GCTX["diag"]
                    met["diag"] = {"n_inf": len(dg),
                                   "cosF_mean": round(float(np.mean(
                                       [x.get("cosF", np.nan) for x in dg])), 3)}
                os.makedirs(ep_dir, exist_ok=True)
                json.dump(met, open(os.path.join(ep_dir, "meta.json"), "w"))
                if k == 0 and frames:
                    tag = "S" if success else "F"
                    save_video(frames, os.path.join(
                        vid_dir, f"{args.axis}_{tname[:60]}_{ctag}_ep0_{tag}.mp4"),
                        fps=20)
                print(f"[gpu{args.gpu}] {tname[:40]} {ctag} ep{k} "
                      f"{'S' if success else 'F'} steps={step+1} dmin={dmin:.3f}",
                      flush=True)
        except Exception:
            import traceback
            traceback.print_exc()
        finally:
            if env is not None:
                env.close()
    print(f"[gpu{args.gpu}] DONE", flush=True)


if __name__ == "__main__":
    main()
