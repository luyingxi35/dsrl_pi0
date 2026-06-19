"""Plot env_steps vs. success_rate curve from a multi-seed sweep.

Each seed writes an eval_curve.csv to its own output directory:
    $EXP/<run_prefix>_seed<N>_*/eval_curve.csv

This script:
  1. Finds all eval_curve.csv files under --log_dir
  2. Interpolates each seed onto a common env-step grid
  3. Computes mean ± std across seeds
  4. Saves a publication-ready figure as --output

Usage (from repo root, after running run_sim_dino_sweep.sh):
    python3 examples/plot_sim_dino_curve.py \\
        --log_dir  ./logs/DSRL_pi0_SimDino \\
        --output   ./logs/sim_dino_curve.png
"""
import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # headless rendering
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


def load_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "env_steps" not in df.columns or "success_rate" not in df.columns:
        raise ValueError(
            f"{path}: expected columns [env_steps, success_rate], got {list(df.columns)}"
        )
    df = df.sort_values("env_steps").drop_duplicates("env_steps")
    return df


def build_common_grid(dfs: list[pd.DataFrame], n_points: int = 200) -> np.ndarray:
    """Linearly-spaced grid from 0 to the minimum max env_step across seeds."""
    max_common = min(df["env_steps"].max() for df in dfs)
    return np.linspace(0, max_common, n_points)


def interpolate(df: pd.DataFrame, grid: np.ndarray) -> np.ndarray:
    return np.interp(grid, df["env_steps"].values, df["success_rate"].values)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--log_dir",    required=True,
                        help="Base directory containing per-seed run subdirectories.")
    parser.add_argument("--output",     default=None,
                        help="Path for the output PNG. "
                             "Defaults to <log_dir>/sim_dino_curve.png.")
    parser.add_argument("--n_points",   default=200, type=int,
                        help="Number of points in the interpolation grid.")
    parser.add_argument("--stop_line",  default=0.95, type=float,
                        help="Draw a horizontal dashed line at this success rate "
                             "(set to 0 to suppress).")
    parser.add_argument("--title",      default="PegInsertionVertical — DSRL (sim)",
                        help="Plot title.")
    parser.add_argument("--smoothing",  default=1, type=int,
                        help="Rolling-window smoothing over the grid (1 = no smoothing).")
    args = parser.parse_args()

    # ── Discover CSV files ─────────────────────────────────────────────────────
    pattern = os.path.join(args.log_dir, "**", "eval_curve.csv")
    csv_files = sorted(glob.glob(pattern, recursive=True))
    if not csv_files:
        print(f"No eval_curve.csv files found under {args.log_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(csv_files)} seed(s):")
    for f in csv_files:
        print(f"  {f}")

    # ── Load & interpolate ─────────────────────────────────────────────────────
    dfs  = [load_csv(f) for f in csv_files]
    grid = build_common_grid(dfs, n_points=args.n_points)

    curves = np.stack([interpolate(df, grid) for df in dfs], axis=0)  # (n_seeds, n_points)

    mean = np.mean(curves, axis=0)
    std  = np.std(curves,  axis=0)

    # Optional smoothing
    if args.smoothing > 1:
        k    = args.smoothing
        mean = np.convolve(mean, np.ones(k) / k, mode="same")
        std  = np.convolve(std,  np.ones(k) / k, mode="same")

    # ── Plot ──────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 5))

    n_seeds = len(dfs)
    color   = "#1f77b4"

    ax.plot(grid, mean, color=color, linewidth=2.0, label=f"Mean ({n_seeds} seeds)")
    ax.fill_between(grid, mean - std, mean + std,
                    color=color, alpha=0.20, label="±1 std")

    # Individual seed traces (faint)
    for k, c in enumerate(curves):
        seed_label = os.path.basename(os.path.dirname(csv_files[k]))
        ax.plot(grid, c, color=color, linewidth=0.6, alpha=0.35,
                label=seed_label if n_seeds <= 5 else None)

    # Convergence threshold line
    if args.stop_line > 0:
        ax.axhline(args.stop_line, color="crimson", linestyle="--",
                   linewidth=1.0, label=f"Target ({args.stop_line:.0%})")

    ax.set_xlabel("Environment steps", fontsize=12)
    ax.set_ylabel("Success rate",      fontsize=12)
    ax.set_title(args.title,           fontsize=13)
    ax.set_ylim(-0.02, 1.05)
    ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1))
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x/1e3:.0f}k"))
    ax.grid(True, linestyle=":", alpha=0.5)

    # Deduplicate legend entries
    handles, labels = ax.get_legend_handles_labels()
    seen = {}
    uniq_h, uniq_l = [], []
    for h, l in zip(handles, labels):
        if l not in seen:
            seen[l] = True
            uniq_h.append(h)
            uniq_l.append(l)
    ax.legend(uniq_h, uniq_l, loc="lower right", fontsize=10)

    plt.tight_layout()

    out_path = args.output or os.path.join(args.log_dir, "sim_dino_curve.png")
    os.makedirs(os.path.dirname(out_path) if os.path.dirname(out_path) else ".",
                exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nFigure saved → {out_path}")

    # Also print final stats
    print("\n── Per-seed final success rates ──")
    for f, c in zip(csv_files, curves):
        print(f"  {os.path.relpath(f, args.log_dir):50s}  final={c[-1]:.3f}")
    print(f"\nMean final: {mean[-1]:.3f} ± {std[-1]:.3f}")


if __name__ == "__main__":
    main()
