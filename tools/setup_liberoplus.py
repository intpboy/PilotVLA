#!/usr/bin/env python3
"""One-time setup for LIBERO-Plus evaluation (tools/rollout_liberoplus.py).

Given a LIBERO-plus checkout (https://github.com/sylvestf/LIBERO-plus, with its
released assets unpacked inside), this script builds - WITHOUT copying any
data - the drop-in package that rollout_liberoplus.py imports, under
work/liberoplus/ (git-ignored, machine-local):

  pkgshadow/libero/            a concrete `libero` package whose contents are
                               symlinks into the LIBERO-plus checkout
  assets_union/                merged view of the base LIBERO assets and the
                               extra LIBERO-plus assets (per-entry symlinks)
  libero_config/config.yaml    LIBERO config pointing at the checkout

Usage:
  python tools/setup_liberoplus.py --root /path/to/LIBERO-plus
"""
import argparse
import glob
import os


def overlay(union_dir, sources):
    """Merge `sources` into union_dir using symlinks; later sources add files
    into entries that exist in several sources (converted to real dirs)."""
    os.makedirs(union_dir, exist_ok=True)
    for src in sources:
        if not os.path.isdir(src):
            continue
        for name in sorted(os.listdir(src)):
            sp = os.path.join(src, name)
            up = os.path.join(union_dir, name)
            if not os.path.lexists(up):
                os.symlink(sp, up)
            elif os.path.isdir(sp) and os.path.isdir(up):
                if os.path.islink(up):     # dir collision: materialize, re-link contents
                    old = os.path.realpath(up)
                    os.unlink(up)
                    os.makedirs(up)
                    for n2 in sorted(os.listdir(old)):
                        os.symlink(os.path.join(old, n2), os.path.join(up, n2))
                for n2 in sorted(os.listdir(sp)):
                    p2 = os.path.join(up, n2)
                    if not os.path.lexists(p2):
                        os.symlink(os.path.join(sp, n2), p2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=os.environ.get("LIBERO_PLUS_ROOT", ""),
                    help="LIBERO-plus checkout root (or set $LIBERO_PLUS_ROOT)")
    args = ap.parse_args()
    assert args.root, "pass --root or set LIBERO_PLUS_ROOT"
    root = os.path.abspath(args.root)
    pkg = os.path.join(root, "libero", "libero")
    assert os.path.isdir(pkg), f"not a LIBERO-plus checkout: {pkg} missing"

    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    lpd = os.path.join(repo, "work", "liberoplus")
    os.makedirs(lpd, exist_ok=True)

    # 1) assets union: base package assets + the extra released asset pack(s)
    extra = sorted(glob.glob(os.path.join(pkg, "**", "LIBERO-plus*", "assets"),
                             recursive=True))
    union = os.path.join(lpd, "assets_union")
    overlay(union, [os.path.join(pkg, "assets")] + extra)
    print(f"[setup] assets_union <- {[os.path.join(pkg, 'assets')] + extra}")

    # 2) pkgshadow: concrete `libero` package made of symlinks into the checkout
    shadow = os.path.join(lpd, "pkgshadow", "libero")
    inner = os.path.join(shadow, "libero")
    os.makedirs(inner, exist_ok=True)
    top_init = os.path.join(shadow, "__init__.py")
    if not os.path.exists(top_init):
        open(top_init, "w").write("\n")
    for name in ("benchmark", "bddl_files", "utils", "envs", "init_files",
                 "__init__.py"):
        sp, up = os.path.join(pkg, name), os.path.join(inner, name)
        if os.path.exists(sp) and not os.path.lexists(up):
            os.symlink(sp, up)
    ap_ = os.path.join(inner, "assets")
    if not os.path.lexists(ap_):
        os.symlink(union, ap_)
    print(f"[setup] pkgshadow -> {shadow}")

    # 3) LIBERO config consumed via LIBERO_CONFIG_PATH
    cfg_dir = os.path.join(lpd, "libero_config")
    os.makedirs(cfg_dir, exist_ok=True)
    with open(os.path.join(cfg_dir, "config.yaml"), "w") as f:
        f.write(f"assets: {union}\n"
                f"bddl_files: {os.path.join(pkg, 'bddl_files')}\n"
                f"benchmark_root: {pkg}\n"
                f"datasets: {os.path.join(pkg, '..', 'datasets')}\n"
                f"init_states: {os.path.join(pkg, 'init_files')}\n")
    print(f"[setup] config -> {os.path.join(cfg_dir, 'config.yaml')}")
    print("[setup] done - tools/rollout_liberoplus.py is ready to run")


if __name__ == "__main__":
    main()
