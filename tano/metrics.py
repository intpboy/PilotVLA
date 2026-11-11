"""Scene-state helpers and episode success metrics for LIBERO swap rollouts."""
import numpy as np

LIFT_DZ = 0.04   # object counts as "lifted" once its z rises by this much (m)
BASKET_R = 0.10  # horizontal distance to basket below which the object is "in basket" (m)


def joint_qpos_slice(sim, inst, flat_offset):
    """Slice of the flattened init-state vector holding `inst`'s free-joint qpos."""
    try:
        names = list(sim.model.joint_names)
    except Exception:
        names = [sim.model.joint_id2name(i) for i in range(sim.model.njnt)]
    jn = next(n for n in names if n and n.startswith(inst) and "joint" in n)
    lo, hi = sim.model.get_joint_qpos_addr(jn)
    return slice(flat_offset + lo, flat_offset + hi)


def body_xyz(raw, inst):
    # robosuite recreates MjSim on env reset -> always fetch raw.sim fresh.
    # body_xpos[id] is a VIEW into the sim buffer -> must copy or every tracked
    # step aliases to the final state.
    return np.array(raw.sim.data.body_xpos[raw.obj_body_id[inst]][:3], dtype=float, copy=True)


def find_inst(raw, obj_type):
    keys = list(raw.obj_body_id.keys())
    return next(k for k in keys if k == obj_type or k.startswith(obj_type + "_"))


def episode_metrics(track, success):
    """Per-episode outcome from object xyz tracks.

    `track` maps names (incl. "basket" and "asked") to (T, 3) arrays. Returns
    env success, per-object lift steps, which object was lifted first, and
    whether the asked object ended up in the basket (the swap success signal).
    """
    tz = {k: v[:, 2] for k, v in track.items() if k not in ("basket", "asked")}
    lift_step = {}
    for k, z in tz.items():
        idx = np.where(z - z[0] > LIFT_DZ)[0]
        lift_step[k] = int(idx[0]) if len(idx) else None
    lifted = [k for k, s in lift_step.items() if s is not None]
    first = min(lifted, key=lambda k: lift_step[k]) if lifted else None
    basket = track["basket"][-1]
    d_ask = float(np.linalg.norm(track["asked"][-1][:2] - basket[:2]))
    return {"env_success": bool(success),
            "lift_step": lift_step,
            "first_lifted": first,
            "asked_final_dist_to_basket": round(d_ask, 4),
            "asked_in_basket": bool(d_ask < BASKET_R)}
