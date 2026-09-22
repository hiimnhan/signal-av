"""Plot training curves from metrics.csv files produced by train.py.

Single seed:   python scripts/plot_curves.py checkpoints/bimodal_seed1/metrics.csv
Mean (N seeds): python scripts/plot_curves.py --seeds-dir checkpoints/
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PANELS = [
    ("reward", "Return"),
    ("crash",  "Crash rate"),
    ("rho",    "ρ (rho)"),
    ("sc",     "SC"),
]


def _make_fig(title: str):
    n = len(PANELS)
    ncols = 2
    nrows = (n + 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(12, nrows * 3))
    axes = axes.flatten()
    fig.suptitle(title, fontsize=12)
    return fig, axes


def plot(csv_path: Path, out_path: Path | None, window: int) -> None:
    """Single-seed plot with rolling average."""
    df = pd.read_csv(csv_path)
    fig, axes = _make_fig(csv_path.parent.name)

    for i, (col, label) in enumerate(PANELS):
        ax = axes[i]
        if col not in df.columns:
            ax.set_visible(False)
            continue
        ax.plot(df["iter"], df[col], alpha=0.3, color="steelblue", linewidth=0.8)
        if len(df) >= window:
            ax.plot(df["iter"], df[col].rolling(window).mean(),
                    color="steelblue", linewidth=1.8)
        ax.set_xlabel("Iteration")
        ax.set_title(label)
        ax.grid(True, alpha=0.3)

    for j in range(len(PANELS), len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout()
    _save_or_show(fig, out_path)


def plot_mean(seeds_dir: Path, out_path: Path | None, window: int) -> None:
    """Mean ± 1 std across all seeds found under seeds_dir."""
    csvs = sorted(seeds_dir.glob("*/metrics.csv"))
    if not csvs:
        raise FileNotFoundError(f"No */metrics.csv found under {seeds_dir}")

    dfs = [pd.read_csv(p).set_index("iter") for p in csvs]
    # inner join on iter so all seeds share same index
    common_iter = dfs[0].index
    for df in dfs[1:]:
        common_iter = common_iter.intersection(df.index)
    dfs = [df.loc[common_iter] for df in dfs]
    stacked = np.stack([df.values for df in dfs], axis=0)   # (N, T, C)
    cols = dfs[0].columns.tolist()
    iters = common_iter.to_numpy()
    n_seeds = len(dfs)

    mean_df = pd.DataFrame(stacked.mean(axis=0), columns=cols, index=iters)
    std_df  = pd.DataFrame(stacked.std(axis=0),  columns=cols, index=iters)

    title = f"{seeds_dir.name}  (n={n_seeds} seeds, mean ± 1 std)"
    fig, axes = _make_fig(title)

    for i, (col, label) in enumerate(PANELS):
        ax = axes[i]
        if col not in mean_df.columns:
            ax.set_visible(False)
            continue

        mu  = mean_df[col]
        sig = std_df[col]

        # raw mean (faint)
        ax.plot(iters, mu, alpha=0.25, color="steelblue", linewidth=0.8)

        # smoothed mean + shaded ±1 std
        if len(mu) >= window:
            mu_smooth  = mu.rolling(window).mean()
            sig_smooth = sig.rolling(window).mean()
            ax.plot(iters, mu_smooth, color="steelblue", linewidth=1.8)
            ax.fill_between(iters,
                            mu_smooth - sig_smooth,
                            mu_smooth + sig_smooth,
                            alpha=0.20, color="steelblue")

        ax.set_xlabel("Iteration")
        ax.set_title(label)
        ax.grid(True, alpha=0.3)

    for j in range(len(PANELS), len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout()
    _save_or_show(fig, out_path)


def _save_or_show(fig, out_path: Path | None) -> None:
    if out_path:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        print(f"Saved: {out_path}")
    else:
        plt.show()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("csv", nargs="?", help="Path to a single metrics.csv")
    p.add_argument("--seeds-dir", default="", help="Dir containing seed subdirs with metrics.csv")
    p.add_argument("--out", default="", help="Save figure to this path instead of displaying")
    p.add_argument("--window", type=int, default=20, help="Rolling average window")
    args = p.parse_args()

    out = Path(args.out) if args.out else None

    if args.seeds_dir:
        plot_mean(Path(args.seeds_dir), out, args.window)
    elif args.csv:
        plot(Path(args.csv), out, args.window)
    else:
        p.error("Provide either a csv path or --seeds-dir")


if __name__ == "__main__":
    main()
