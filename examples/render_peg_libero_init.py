"""Render agentview + frontview of the PegInsertionVertical LIBERO env at reset.

Saves PNG images for visual inspection and camera comparison with ManiSkill.

Usage (from repo root):
    cd /home/gpu4/yingxi/dsrl_pi0
    python3 examples/render_peg_libero_init.py [--output_dir /tmp/peg_render] [--resolution 256]

Then on local Mac:
    scp H100-SQZ:/tmp/peg_render/*.png /tmp/ && open /tmp/agentview.png /tmp/frontview.png
"""
import argparse
import pathlib
import sys

import numpy as np

# ── Register new objects and problem class (no existing files modified) ───────
from libero.libero.envs.objects.peg_insertion_objects import Peg, BoxWithHole          # noqa: F401
from libero.libero.envs.problems.peg_insertion_tabletop import (                       # noqa: F401
    PegInsertion_Tabletop_Manipulation,
)

from libero.libero.envs import OffScreenRenderEnv


_BDDL_PATH = (
    pathlib.Path(__file__).resolve().parents[1]
    / "LIBERO"
    / "libero"
    / "libero"
    / "bddl_files"
    / "peg_insertion"
    / "TABLETOP_peg_insertion_vertical.bddl"
)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output_dir",  default="/tmp/peg_render",
                        help="Directory to save PNG images")
    parser.add_argument("--resolution",  type=int, default=256,
                        help="Camera resolution (height = width = resolution)")
    parser.add_argument("--n_resets",    type=int, default=1,
                        help="Number of resets; >1 shows object placement variation")
    args = parser.parse_args()

    out = pathlib.Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print(f"BDDL: {_BDDL_PATH}")
    assert _BDDL_PATH.exists(), f"BDDL file not found: {_BDDL_PATH}"

    env = OffScreenRenderEnv(
        bddl_file_name=str(_BDDL_PATH),
        camera_names=["agentview", "frontview"],
        camera_heights=args.resolution,
        camera_widths=args.resolution,
    )

    try:
        from PIL import Image
    except ImportError:
        print("PIL not found — saving raw npy instead")
        Image = None

    for trial in range(args.n_resets):
        obs = env.reset()

        # Print success check result at reset (should always be False)
        success = env.env._check_success() if hasattr(env, "env") else env._check_success()
        print(f"\n[trial {trial}] _check_success() = {success}")

        # Print peg and hole positions
        try:
            peg_id  = env.obj_body_id.get("peg_1")
            hole_id = env.obj_body_id.get("box_with_hole_1")
            if peg_id is not None:
                print(f"  peg_1  pos: {env.sim.data.body_xpos[peg_id]}")
            if hole_id is not None:
                print(f"  hole_1 pos: {env.sim.data.body_xpos[hole_id]}")
        except Exception as e:
            print(f"  (could not read body positions: {e})")

        # Save images
        for cam in ["agentview", "frontview"]:
            key = f"{cam}_image"
            if key not in obs:
                print(f"  WARNING: {key} not in obs; skipping")
                continue
            img = obs[key][::-1]     # robosuite image is upside-down
            fname = f"{cam}_{trial:02d}.png" if args.n_resets > 1 else f"{cam}.png"
            path = out / fname
            if Image is not None:
                Image.fromarray(img.astype(np.uint8)).save(path)
            else:
                np.save(str(path).replace(".png", ".npy"), img)
            print(f"  Saved {path}")
    env.close()
    print(f"\nDone — images in {out}")


if __name__ == "__main__":
    main()
