#!/usr/bin/env python3
"""Paired statistics for swap rollouts: baseline vs steered.

Reads per-episode meta.json files written by pilotvla/rollout_swap.py and reports
Wilson CIs, the paired success-rate difference, a continuity-corrected McNemar
test, a scene-clustered bootstrap, and a per-scene breakdown. Pairs are formed
over episodes present in BOTH condition cells (identical seeds by design).

Usage:
  python -m pilotvla.paired_stats --root OUT_DIR --key asked_in_basket \
      --none-tag none --steer-tag noiseopt10x0.5-r10
  python -m pilotvla.paired_stats --root OUT_DIR --key env_success \
      --none-tag posnone --steer-tag posnoiseopt10x0.5-r10
"""
import argparse
import glob
import json
import math
from collections import defaultdict


class RNG:
    """Deterministic LCG so bootstrap results are reproducible."""

    def __init__(s, seed):
        s.x = seed & 0xFFFFFFFF

    def rint(s, n):
        s.x = (1103515245 * s.x + 12345) & 0x7FFFFFFF
        return s.x % n


def wilson(k, n, z=1.96):
    if n == 0:
        return (0, 0, 0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (100 * p, 100 * (c - h), 100 * (c + h))


def load(root, key, none_tag, steer_tag):
    """cells[(scene, distractor)][ep] = (baseline_success, steered_success)."""
    nd, od = defaultdict(dict), defaultdict(dict)
    for f in glob.glob(f"{root}/*/*/ep*/meta.json"):
        parts = f.split("/")
        scene, cd, ep = parts[-4], parts[-3], parts[-2]
        for tag, store in ((none_tag, nd), (steer_tag, od)):
            if cd.startswith(tag + "_"):
                distr = cd[len(tag) + 1:]
                try:
                    store[(scene, distr)][ep] = bool(json.load(open(f)).get(key))
                except Exception:
                    pass
    cells = {}
    for cell in set(nd) & set(od):
        eps = set(nd[cell]) & set(od[cell])
        if eps:
            cells[cell] = {e: (nd[cell][e], od[cell][e]) for e in eps}
    return cells


def analyze(name, cells, n_boot=2000):
    rows = []  # (scene, distractor, baseline, steered) per paired episode
    for (scene, distr), eps in cells.items():
        for ep, (n, o) in sorted(eps.items(), key=lambda x: int(x[0][2:])):
            rows.append((scene, distr, n, o))
    N = len(rows)
    if N == 0:
        print(f"=== {name}: no paired episodes found ===")
        return
    nk = sum(r[2] for r in rows)
    ok = sum(r[3] for r in rows)
    # McNemar on discordant pairs (continuity-corrected chi2, df=1)
    b = sum(1 for r in rows if r[2] and not r[3])   # baseline wins
    c = sum(1 for r in rows if not r[2] and r[3])   # steered wins
    mc = (abs(b - c) - 1) ** 2 / (b + c) if (b + c) > 0 else 0.0
    diff = 100 * (ok - nk) / N
    se_diff = 100 * math.sqrt((b + c) - (b - c) ** 2 / N) / N if N > 0 else 0
    # scene-cluster bootstrap (resample scenes with replacement)
    scenes = sorted(set(r[0] for r in rows))
    by_scene = defaultdict(list)
    for r in rows:
        by_scene[r[0]].append(r)
    rng = RNG(12345)
    diffs, orates, nrates = [], [], []
    for _ in range(n_boot):
        bn = bo = bN = 0
        for _ in range(len(scenes)):
            s = scenes[rng.rint(len(scenes))]
            for (_, _, n, o) in by_scene[s]:
                bn += n
                bo += o
                bN += 1
        if bN:
            diffs.append(100 * (bo - bn) / bN)
            orates.append(100 * bo / bN)
            nrates.append(100 * bn / bN)
    diffs.sort(); orates.sort(); nrates.sort()

    def pct(a, p):
        return a[int(p * len(a))] if a else 0

    print(f"\n=== {name} | N={N} pairs, scenes={len(scenes)} ===")
    nn, oo = wilson(nk, N), wilson(ok, N)
    print(f"  baseline: {nn[0]:.1f}%  (Wilson95 {nn[1]:.1f}~{nn[2]:.1f})  [{nk}/{N}]")
    print(f"  steered : {oo[0]:.1f}%  (Wilson95 {oo[1]:.1f}~{oo[2]:.1f})  [{ok}/{N}]")
    print(f"  diff    : {diff:+.1f}%p  (paired-normal95 {diff-1.96*se_diff:.1f}~{diff+1.96*se_diff:.1f})")
    print(f"  McNemar : b(baseline wins)={b} c(steered wins)={c}  chi2={mc:.1f}  (>3.84 -> p<0.05)")
    print(f"  scene-cluster bootstrap ({n_boot} draws, {len(scenes)} clusters):")
    print(f"     baseline {pct(nrates,.025):.1f}~{pct(nrates,.975):.1f}  "
          f"steered {pct(orates,.025):.1f}~{pct(orates,.975):.1f}  "
          f"diff {pct(diffs,.025):.1f}~{pct(diffs,.975):.1f}")
    print("  per scene (baseline -> steered, n episodes):")
    sc = defaultdict(lambda: [0, 0, 0])
    for (scene, distr, n, o) in rows:
        sc[scene][0] += n
        sc[scene][1] += o
        sc[scene][2] += 1
    for s in sorted(sc, key=lambda x: sc[x][1] / max(sc[x][2], 1) - sc[x][0] / max(sc[x][2], 1)):
        a = sc[s]
        print(f"     {s:18s} {100*a[0]//a[2]:3d} -> {100*a[1]//a[2]:3d}%  (n={a[2]})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="rollout output dir (scene/cell/ep*/meta.json)")
    ap.add_argument("--key", default="asked_in_basket",
                    help="success key in meta.json (asked_in_basket | env_success)")
    ap.add_argument("--none-tag", default="none", help="baseline cell tag")
    ap.add_argument("--steer-tag", required=True,
                    help="steered cell tag, e.g. noiseopt10x0.5-r10")
    ap.add_argument("--name", default="", help="label for the printed report")
    args = ap.parse_args()
    cells = load(args.root, args.key, args.none_tag, args.steer_tag)
    analyze(args.name or f"{args.none_tag} vs {args.steer_tag}", cells)


if __name__ == "__main__":
    main()
