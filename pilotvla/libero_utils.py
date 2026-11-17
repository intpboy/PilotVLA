"""LIBERO environment helpers (single-task env build, task map, env handles)."""
import re

import numpy as np


def build_single_env(adapter, suite, tid, max_steps):
    """Build ONLY the one task's env (task_ids=[tid]) instead of the whole suite.

    lerobot's make_env(task=suite) eagerly builds one vec-env PER TASK in the
    suite and caches them all, which wastes memory; passing task_ids=[tid]
    builds exactly one. The caller must close() it afterwards.
    """
    from lerobot.envs.factory import make_env
    from lerobot.envs.configs import LiberoEnv as LiberoEnvConfig
    adapter._ensure_env_processors(suite, max_steps)
    env_cfg = LiberoEnvConfig(task=suite, task_ids=[tid],
                              observation_height=256, observation_width=256)
    envs_dict = make_env(cfg=env_cfg, n_envs=1, use_async_envs=False)
    env = envs_dict[suite][tid]
    desc = ""
    if hasattr(env, "envs"):
        desc = getattr(env.envs[0], "task_description", f"task_{tid}")
    return env, desc


def all_object_tids():
    """Every libero_object target name -> task id (10 tasks)."""
    from libero.libero import benchmark
    suite = benchmark.get_benchmark_dict()["libero_object"]()
    m = {}
    for i in range(suite.n_tasks):
        lang = suite.get_task(i).language
        obj = re.match(r"pick up the (.+) and place it in the basket", lang).group(1)
        m[obj.replace(" ", "_")] = i
    return m


def get_env_handles(env):
    """Unwrap a lerobot LIBERO vec-env down to the raw robosuite env + init states."""
    lenv = env.envs[0]
    raw = lenv._env
    raw = getattr(raw, "env", raw)
    sim = raw.sim
    inits = np.array(lenv._init_states, dtype=np.float64, copy=True)
    off = len(inits[0]) - sim.model.nq - sim.model.nv
    assert off in (0, 1), f"unexpected init-state layout (off={off})"
    return lenv, raw, sim, inits, off
