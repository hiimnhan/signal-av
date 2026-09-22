import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

TRAINING = "--training" in sys.argv


RESULTS = Path("results")
OUT = Path("dissertation/figures")
OUT.mkdir(exist_ok=True)

CONDITION = "bimodal"
INTERVENTIONS = ["hide", "randomise", "permute", "desync", "hide_kin", "hide_all"]


def load_data():
    files = sorted(RESULTS.glob(f"{CONDITION}_seed*.json"))
    return {f.stem: json.loads(f.read_text()) for f in files}


def load_condition_rhos(prefix: str) -> list:
    rhos = []
    for f in sorted(RESULTS.glob(f"{prefix}_seed*.json")):
        records = json.loads(f.read_text())
        rho = get(records, "baseline", "rho")
        if rho is not None:
            rhos.append(rho)
    return rhos


def get(records, condition, key):
    for r in records:
        if r["condition"] == condition:
            return r.get(key)
    return None


def ci95(values):
    a = np.array(values, dtype=float)
    m = a.mean()
    se = a.std(ddof=1) / np.sqrt(len(a))
    h = stats.t.ppf(0.975, df=len(a) - 1) * se
    return m, m - h, m + h


data = load_data()
seeds = sorted(data.keys())
labels = [s.replace(f"{CONDITION}_", "") for s in seeds]

rhos = [get(data[s], "baseline", "rho") or 0.0 for s in seeds]
returns = [get(data[s], "baseline", "mean_return") or 0.0 for s in seeds]
crashes = [get(data[s], "baseline", "collision_rate") or 0.0 for s in seeds]

iv_deltas = {
    iv: [
        v - crashes[i]
        for i, s in enumerate(seeds)
        if (v := get(data[s], iv, "collision_rate")) is not None
    ]
    for iv in INTERVENTIONS
}


m, lo, hi = ci95(rhos)
colours = ["#2c7fb8" if r >= 0.1 else "#aec7e8" for r in rhos]

fig, ax = plt.subplots(figsize=(9, 4))
ax.bar(labels, rhos, color=colours, edgecolor="black", linewidth=0.6)
ax.axhspan(lo, hi, alpha=0.15, color="#2c7fb8", label=f"95% CI [{lo:.3f}, {hi:.3f}]")
ax.axhline(m, color="#2c7fb8", linewidth=1.2, label=f"mean = {m:.3f}")
ax.axhline(
    0.1, color="red", linestyle="--", linewidth=0.8, label="pooling threshold (0.1)"
)
ax.set_ylabel(r"$\rho$")
ax.legend(fontsize=8)
plt.xticks(rotation=30, ha="right")
fig.tight_layout()
fig.savefig(OUT / "rho_per_seed.png", dpi=300, bbox_inches="tight")
plt.close()
print("saved rho_per_seed.png")


# rho by condition

COMPARE_CONDITIONS = [
    ("bimodal", "bimodal", "#2c7fb8"),
    ("same_reward", "same_reward", "#fd8d3c"),
    ("random", "random", "#74c476"),
    ("no_signal", "no_signal", "#c7c6c1"),
]

cond_data = []
for label, prefix, color in COMPARE_CONDITIONS[:3]:
    rhos_c = load_condition_rhos(prefix)
    if rhos_c:
        cond_data.append((label, rhos_c, color))

if len(cond_data) >= 2:
    fig, ax = plt.subplots(figsize=(6, 4))
    for i, (label, rho_vals, color) in enumerate(cond_data):
        m, lo, hi = ci95(rho_vals)
        ax.bar(
            i,
            m,
            yerr=[[m - lo], [hi - m]],
            capsize=6,
            color=color,
            edgecolor="black",
            linewidth=0.6,
            alpha=0.85,
            label=f"{label} (n={len(rho_vals)}, mean={m:.3f})",
        )
        ax.scatter(
            [i] * len(rho_vals), rho_vals, color="black", s=18, zorder=5, alpha=0.55
        )
    ax.axhline(
        0.1, color="red", linestyle="--", linewidth=0.8, label="pooling threshold (0.1)"
    )
    ax.set_xticks(range(len(cond_data)))
    ax.set_xticklabels([d[0] for d in cond_data])
    ax.set_ylabel(r"$\rho$")
    ax.legend(fontsize=8)
    ax.set_ylim(bottom=0)
    ax.grid(axis="y", alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / "rho_by_condition.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("saved rho_by_condition.png")
else:
    print("rho_by_condition skipped - fewer than 2 conditions found in results/")


def load_condition_returns(prefix: str) -> list:
    vals = []
    for f in sorted(RESULTS.glob(f"{prefix}_seed*.json")):
        records = json.loads(f.read_text())
        r = get(records, "baseline", "mean_return")
        if r is not None:
            vals.append(r)
    return vals


ret_data = []
for label, prefix, color in COMPARE_CONDITIONS:
    ret_vals = load_condition_returns(prefix)
    if ret_vals:
        ret_data.append((label, ret_vals, color))

if len(ret_data) >= 2:
    fig, ax = plt.subplots(figsize=(6, 4))
    for i, (label, vals, color) in enumerate(ret_data):
        m, lo, hi = ci95(vals)
        ax.bar(
            i,
            m,
            yerr=[[m - lo], [hi - m]],
            capsize=6,
            color=color,
            edgecolor="black",
            linewidth=0.6,
            alpha=0.85,
            label=f"{label} (n={len(vals)}, mean={m:.1f})",
        )
        ax.scatter([i] * len(vals), vals, color="black", s=18, zorder=5, alpha=0.55)
    ax.set_xticks(range(len(ret_data)))
    ax.set_xticklabels([d[0] for d in ret_data])
    ax.set_ylabel("mean episode return")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / "return_by_condition.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("saved return_by_condition.png")
else:
    print("return_by_condition skipped - fewer than 2 conditions found in results/")


# crash by condition
def load_condition_crashes(prefix: str) -> list:
    vals = []
    for f in sorted(RESULTS.glob(f"{prefix}_seed*.json")):
        records = json.loads(f.read_text())
        c = get(records, "baseline", "collision_rate")
        if c is not None:
            vals.append(c)
    return vals


crash_data = []
for label, prefix, color in COMPARE_CONDITIONS:
    crash_vals = load_condition_crashes(prefix)
    if crash_vals:
        crash_data.append((label, crash_vals, color))

if len(crash_data) >= 2:
    fig, ax = plt.subplots(figsize=(6, 4))
    for i, (label, vals, color) in enumerate(crash_data):
        m, lo, hi = ci95(vals)
        ax.bar(
            i,
            m * 100,
            yerr=[[(m - lo) * 100], [(hi - m) * 100]],
            capsize=6,
            color=color,
            edgecolor="black",
            linewidth=0.6,
            alpha=0.85,
            label=f"{label} (n={len(vals)}, mean={m:.1%})",
        )
        ax.scatter(
            [i] * len(vals),
            [v * 100 for v in vals],
            color="black",
            s=18,
            zorder=5,
            alpha=0.55,
        )
    ax.set_xticks(range(len(crash_data)))
    ax.set_xticklabels([d[0] for d in crash_data])
    ax.set_ylabel("collision rate (%) at baseline")
    ax.legend(fontsize=8)
    ax.set_ylim(bottom=0)
    ax.grid(axis="y", alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / "crash_by_condition.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("saved crash_by_condition.png")
else:
    print("crash_by_condition skipped - fewer than 2 conditions found in results/")


# bimodal vs no_signal
BIMODAL_NO_SIGNAL = [
    ("bimodal", "bimodal", "#2c7fb8"),
    # ("same_reward", "same_reward", "#fd8d3c"),
    # ("random", "random", "#74c476"),
    ("no_signal", "no_signal", "#c7c6c1"),
]
crash_data = []
for label, prefix, color in BIMODAL_NO_SIGNAL:
    crash_vals = load_condition_crashes(prefix)
    if crash_vals:
        crash_data.append((label, crash_vals, color))

if len(crash_data) >= 2:
    fig, ax = plt.subplots(figsize=(6, 4))
    for i, (label, vals, color) in enumerate(crash_data):
        m, lo, hi = ci95(vals)
        ax.bar(
            i,
            m * 100,
            yerr=[[(m - lo) * 100], [(hi - m) * 100]],
            capsize=6,
            color=color,
            edgecolor="black",
            linewidth=0.6,
            alpha=0.85,
            label=f"{label} (n={len(vals)}, mean={m:.1%})",
        )
        ax.scatter(
            [i] * len(vals),
            [v * 100 for v in vals],
            color="black",
            s=18,
            zorder=5,
            alpha=0.55,
        )
    ax.set_xticks(range(len(crash_data)))
    ax.set_xticklabels([d[0] for d in crash_data])
    ax.set_ylabel("collision rate (%) at baseline")
    ax.legend(fontsize=8)
    ax.set_ylim(bottom=0)
    ax.grid(axis="y", alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / "crash_bimodal_no_signal.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("saved crash_bimodal_no_signal.png")
else:
    print("crash_bimodal_no_signal skipped - fewer than 2 conditions found in results/")


ret_data = []
for label, prefix, color in BIMODAL_NO_SIGNAL:
    ret_vals = load_condition_returns(prefix)
    if ret_vals:
        ret_data.append((label, ret_vals, color))

if len(ret_data) >= 2:
    fig, ax = plt.subplots(figsize=(6, 4))
    for i, (label, vals, color) in enumerate(ret_data):
        m, lo, hi = ci95(vals)
        ax.bar(
            i,
            m,
            yerr=[[m - lo], [hi - m]],
            capsize=6,
            color=color,
            edgecolor="black",
            linewidth=0.6,
            alpha=0.85,
            label=f"{label} (n={len(vals)}, mean={m:.1f})",
        )
        ax.scatter([i] * len(vals), vals, color="black", s=18, zorder=5, alpha=0.55)
    ax.set_xticks(range(len(ret_data)))
    ax.set_xticklabels([d[0] for d in ret_data])
    ax.set_ylabel("mean episode return")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / "return_bimodal_no_signal.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("saved return_bimodal_no_signal.png")
else:
    print(
        "return_bimodal_no_signal skipped - fewer than 2 conditions found in results/"
    )


# CIC by condition
CIC_CONDITIONS = [
    ("bimodal", "bimodal", "#2c7fb8"),
    ("same_reward", "same_reward", "#fd8d3c"),
    ("random", "random", "#74c476"),
]


def load_condition_cics(prefix: str) -> list:
    p = RESULTS / f"cic_{prefix}.json"
    if not p.exists():
        return []
    return [r["cic"] for r in json.loads(p.read_text())]


def plot_cic_by_condition():
    cond_data = []
    for label, prefix, color in CIC_CONDITIONS:
        vals = load_condition_cics(prefix)
        if vals:
            cond_data.append((label, vals, color))

    if not cond_data:
        print(
            "cic_by_condition skipped - no cic_*.json found "
            "(run: python scripts/compute_cic.py)"
        )
        return

    fig, ax = plt.subplots(figsize=(6, 4))
    for i, (label, vals, color) in enumerate(cond_data):
        if len(vals) >= 2:
            m, lo, hi = ci95(vals)
        else:
            m = float(np.mean(vals))
            lo = hi = m
        ax.bar(
            i,
            m,
            yerr=[[m - lo], [hi - m]],
            capsize=6,
            color=color,
            edgecolor="black",
            linewidth=0.6,
            alpha=0.85,
            label=f"{label} (n={len(vals)}, mean={m:.3f})",
        )
        ax.scatter([i] * len(vals), vals, color="black", s=18, zorder=5, alpha=0.55)
    ax.axhline(0, color="red", linestyle="--", linewidth=0.8, label="no influence (0)")
    ax.set_xticks(range(len(cond_data)))
    ax.set_xticklabels([d[0] for d in cond_data])
    ax.set_ylabel("CIC (nats)")
    ax.legend(fontsize=8)
    ax.set_ylim(bottom=0)
    ax.grid(axis="y", alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / "cic_by_condition.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("saved cic_by_condition.png")


plot_cic_by_condition()


# Counterfactual Decoding Effect (CDE) by condition
def load_cde(prefix: str, mode: str, key: str) -> list:
    """Load a CDE metric for all seeds of a condition prefix from results/cde.json."""
    p = RESULTS / "cde.json"
    if not p.exists():
        return []
    recs = json.loads(p.read_text())
    return [
        r[mode][key]
        for r in recs
        if r["seed"].startswith(prefix) and r[mode].get("n_valid", 0) > 0
    ]


def plot_cde_by_condition():
    if not (RESULTS / "cde.json").exists():
        print(
            "cde_by_condition skipped - results/cde.json not found "
            "(run: python scripts/compute_cde.py)"
        )
        return

    cond_data = []
    for label, prefix, color in CIC_CONDITIONS:
        vals = load_cde(prefix, "randomise", "cde_l2")
        if vals:
            cond_data.append((label, vals, color))
    if not cond_data:
        print("cde_by_condition skipped - no usable records in cde.json")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))

    for i, (label, vals, color) in enumerate(cond_data):
        if len(vals) >= 2:
            m, lo, hi = ci95(vals)
        else:
            m = float(np.mean(vals))
            lo = hi = m
        ax1.bar(
            i,
            m,
            yerr=[[m - lo], [hi - m]],
            capsize=6,
            color=color,
            edgecolor="black",
            linewidth=0.6,
            alpha=0.85,
            label=f"{label} (n={len(vals)}, mean={m:.3f})",
        )
        ax1.scatter([i] * len(vals), vals, color="black", s=18, zorder=5, alpha=0.55)
    ax1.set_xticks(range(len(cond_data)))
    ax1.set_xticklabels([d[0] for d in cond_data])
    ax1.set_ylabel(r"$|\Delta \mathrm{action}|$  (L2, token randomised)")
    ax1.legend(fontsize=8)
    ax1.set_ylim(bottom=0)
    ax1.grid(axis="y", alpha=0.3)
    ax1.spines[["top", "right"]].set_visible(False)

    # Right: bimodal per-dim |Δ| (longitudinal vs lateral), mean ± 95% CI.
    accel = load_cde("bimodal", "randomise", "d_accel")
    steer = load_cde("bimodal", "randomise", "d_steer")
    if accel and steer:
        for i, (vals, lab, col) in enumerate(
            [(accel, "accel", "#2c7fb8"), (steer, "steer", "#6baed6")]
        ):
            m, lo, hi = ci95(vals) if len(vals) >= 2 else (np.mean(vals),) * 3
            ax2.bar(
                i,
                m,
                yerr=[[m - lo], [hi - m]],
                capsize=6,
                color=col,
                edgecolor="black",
                linewidth=0.6,
                alpha=0.85,
                label=f"{lab} (mean={m:.3f})",
            )
            ax2.scatter(
                [i] * len(vals), vals, color="black", s=18, zorder=5, alpha=0.55
            )
        ax2.set_xticks([0, 1])
        ax2.set_xticklabels(["longitudinal", "lateral"])
        ax2.set_ylabel(r"$|\Delta|$ per action dim  (bimodal)")
        ax2.legend(fontsize=8)
        ax2.set_ylim(bottom=0)
        ax2.grid(axis="y", alpha=0.3)
        ax2.spines[["top", "right"]].set_visible(False)

    fig.tight_layout()
    fig.savefig(OUT / "cde_by_condition.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("saved cde_by_condition.png")


plot_cde_by_condition()


# signal heatmaps  p(m|θ) and p(θ|m)
psm_stack, pts_stack = [], []
for s in seeds:
    rec = next((r for r in data[s] if r["condition"] == "baseline"), None)
    if rec and rec.get("p_sig_given_type") and rec.get("p_type_given_sig"):
        psm_stack.append(np.array(rec["p_sig_given_type"]))
        pts_stack.append(np.array(rec["p_type_given_sig"]))

if psm_stack:
    psm = np.stack(psm_stack)  # (n_seeds, n_types, n_signals)
    pts = np.stack(pts_stack)  # (n_seeds, n_signals, n_types)
    n = len(psm_stack)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    im1 = ax1.imshow(psm.mean(0), cmap="Blues", vmin=0, vmax=1, aspect="auto")
    ax1.set_xticks([0, 1, 2])
    ax1.set_xticklabels(["s0", "s1", "s2"])
    ax1.set_yticks([0, 1])
    ax1.set_yticklabels(["cautious", "assertive"])
    ax1.set_title(r"$p(m \mid \theta)$")
    ax1.set_xlabel("signal m")
    ax1.set_ylabel(r"type $\theta$")
    for i in range(psm.shape[1]):
        for j in range(psm.shape[2]):
            mv, lo_v, hi_v = ci95(psm[:, i, j])
            ax1.text(
                j,
                i,
                f"{mv:.2f}\n±{(hi_v - lo_v) / 2:.2f}",
                ha="center",
                va="center",
                color="white" if mv > 0.6 else "black",
                fontsize=9,
            )
    fig.colorbar(im1, ax=ax1)

    im2 = ax2.imshow(pts.mean(0), cmap="Oranges", vmin=0, vmax=1, aspect="auto")
    ax2.set_xticks([0, 1])
    ax2.set_xticklabels(["t0", "t1"])
    ax2.set_yticks([0, 1, 2])
    ax2.set_yticklabels(["s0", "s1", "s2"])
    ax2.set_title(r"$p(\theta \mid m)$")
    ax2.set_xlabel(r"type $\theta$")
    ax2.set_ylabel("signal m")
    for i in range(pts.shape[1]):
        for j in range(pts.shape[2]):
            mv, lo_v, hi_v = ci95(pts[:, i, j])
            ax2.text(
                j,
                i,
                f"{mv:.2f}\n±{(hi_v - lo_v) / 2:.2f}",
                ha="center",
                va="center",
                color="white" if mv > 0.6 else "black",
                fontsize=9,
            )
    fig.colorbar(im2, ax=ax2)

    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(OUT / "signal_heatmaps.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("saved signal_heatmaps.png")
else:
    print("signal heatmaps skipped - p_sig_given_type missing from results")


# per-seed signal heatmaps  p(m|θ), sorted by return

per_seed = []
for s in seeds:
    rec = next((r for r in data[s] if r["condition"] == "baseline"), None)
    if rec and rec.get("p_sig_given_type") and rec.get("mean_return") is not None:
        per_seed.append((s, np.array(rec["p_sig_given_type"]), rec["mean_return"]))

if per_seed:
    per_seed.sort(key=lambda x: -x[2])  # high return first
    n = len(per_seed)
    ncol = 5
    nrow = int(np.ceil(n / ncol))
    n_types, n_sig = per_seed[0][1].shape
    fig, axes = plt.subplots(nrow, ncol, figsize=(2.2 * ncol, 2.0 * nrow))
    axes = np.atleast_1d(axes).ravel()
    for ax, (s, p, ret) in zip(axes, per_seed):
        ax.imshow(p, cmap="Blues", vmin=0, vmax=1, aspect="auto")
        ax.set_xticks(range(n_sig))
        ax.set_xticklabels([f"s{j}" for j in range(n_sig)], fontsize=7)
        ax.set_yticks(range(n_types))
        ax.set_yticklabels([f"t{i}" for i in range(n_types)], fontsize=7)
        for i in range(n_types):
            for j in range(n_sig):
                ax.text(
                    j,
                    i,
                    f"{p[i, j]:.2f}",
                    ha="center",
                    va="center",
                    color="white" if p[i, j] > 0.6 else "black",
                    fontsize=6.5,
                )
        m0 = int(p[0].argmax())
        m1 = int(p[1].argmax()) if n_types > 1 else m0
        ax.set_title(
            f"seed {s} | ret {ret:.0f}\nconv s{m0}/s{m1}", fontsize=7.5, color="black"
        )
        # for sp in ax.spines.values():
        #     sp.set_edgecolor("0.6" if good else "firebrick")
        #     sp.set_linewidth(0.6 if good else 1.8)
    for ax in axes[n:]:
        ax.axis("off")
    # fig.suptitle(
    #     r"Per-seed emission $p(m\mid\theta)$ - "
    #     f"{CONDITION}, sorted by return\n"
    #     "high-return seeds pool on different tokens: the label is arbitrary "
    #     "(red = low-return / failed seeds)",
    #     fontsize=10,
    # )
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(OUT / "signal_heatmaps_per_seed.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("saved signal_heatmaps_per_seed.png")
else:
    print("per-seed heatmaps skipped - p_sig_given_type/mean_return missing")


# intervention effects  crash% vs none


def load_condition_iv_deltas(prefix):
    """Load per-seed intervention deltas for a condition prefix. Returns None if missing."""
    files = sorted(RESULTS.glob(f"{prefix}_seed*.json"))
    if not files:
        return None
    base_crashes, iv_crash = [], {iv: [] for iv in INTERVENTIONS}
    for f in files:
        records = json.loads(f.read_text())
        base = get(records, "baseline", "collision_rate")
        if base is None:
            continue
        base_crashes.append(base)
        for iv in INTERVENTIONS:
            v = get(records, iv, "collision_rate")
            iv_crash[iv].append((v - base) if v is not None else None)
    if not base_crashes:
        return None
    return {iv: [v for v in vals if v is not None] for iv, vals in iv_crash.items()}


IV_CONDITIONS = [
    ("bimodal", iv_deltas, "#2c7fb8"),
    ("same_reward", load_condition_iv_deltas("same_reward"), "#fd8d3c"),
    ("random", load_condition_iv_deltas("random"), "#74c476"),
]
IV_CONDITIONS = [(lbl, d, c) for lbl, d, c in IV_CONDITIONS if d]

n_conds = len(IV_CONDITIONS)
width = 0.8 / max(n_conds, 1)
x = np.arange(len(INTERVENTIONS))

fig, ax = plt.subplots(figsize=(10, 4))
for ci, (cond_label, cond_deltas, color) in enumerate(IV_CONDITIONS):
    means, los, his = [], [], []
    for iv in INTERVENTIONS:
        vals = [d * 100 for d in cond_deltas.get(iv, []) if d is not None]
        if vals:
            m, lo, hi = ci95(vals)
        else:
            m, lo, hi = 0.0, 0.0, 0.0
        means.append(m)
        los.append(lo)
        his.append(hi)
    offset = (ci - (n_conds - 1) / 2) * width
    err = [np.array(means) - np.array(los), np.array(his) - np.array(means)]
    ax.bar(
        x + offset,
        means,
        width=width * 0.9,
        yerr=err,
        capsize=4,
        color=color,
        edgecolor="black",
        linewidth=0.5,
        alpha=0.85,
        label=cond_label,
    )

ax.axhline(0, color="black", linewidth=0.8)
ax.set_xticks(x)
ax.set_xticklabels(INTERVENTIONS)
ax.set_ylabel("Δ collision rate vs baseline (pp)")
ax.legend(fontsize=8)
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig(OUT / "interventions.png", dpi=300, bbox_inches="tight")
plt.close()
print("saved interventions.png")


# intervention effects on return  Δreturn vs none --------------------------


def load_condition_iv_return_deltas(prefix):
    """Per-seed Δ mean_return (intervention - none) for one condition. None if missing."""
    files = sorted(RESULTS.glob(f"{prefix}_seed*.json"))
    if not files:
        return None
    iv_delta = {iv: [] for iv in INTERVENTIONS}
    any_base = False
    for f in files:
        records = json.loads(f.read_text())
        base = get(records, "baseline", "mean_return")
        if base is None:
            continue
        any_base = True
        for iv in INTERVENTIONS:
            v = get(records, iv, "mean_return")
            if v is not None:
                iv_delta[iv].append(v - base)
    if not any_base:
        return None
    return iv_delta


IV_RETURN_CONDITIONS = [
    ("bimodal", load_condition_iv_return_deltas("bimodal"), "#2c7fb8"),
    ("same_reward", load_condition_iv_return_deltas("same_reward"), "#fd8d3c"),
    ("random", load_condition_iv_return_deltas("random"), "#74c476"),
]
IV_RETURN_CONDITIONS = [(lbl, d, c) for lbl, d, c in IV_RETURN_CONDITIONS if d]

n_conds = len(IV_RETURN_CONDITIONS)
width = 0.8 / max(n_conds, 1)
x = np.arange(len(INTERVENTIONS))

fig, ax = plt.subplots(figsize=(10, 4))
for ci, (cond_label, cond_deltas, color) in enumerate(IV_RETURN_CONDITIONS):
    means, los, his = [], [], []
    for iv in INTERVENTIONS:
        vals = cond_deltas.get(iv, [])
        if vals:
            m, lo, hi = ci95(vals)
        else:
            m, lo, hi = 0.0, 0.0, 0.0
        means.append(m)
        los.append(lo)
        his.append(hi)
    offset = (ci - (n_conds - 1) / 2) * width
    err = [np.array(means) - np.array(los), np.array(his) - np.array(means)]
    ax.bar(
        x + offset,
        means,
        width=width * 0.9,
        yerr=err,
        capsize=4,
        color=color,
        edgecolor="black",
        linewidth=0.5,
        alpha=0.85,
        label=cond_label,
    )

ax.axhline(0, color="black", linewidth=0.8)
ax.set_xticks(x)
ax.set_xticklabels(INTERVENTIONS)
ax.set_ylabel("Δ mean return vs baseline")
ax.legend(fontsize=8)
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig(OUT / "interventions_reward.png", dpi=300, bbox_inches="tight")
plt.close()
print("saved interventions_reward.png")


#  per-condition intervention figures - each vs its OWN baseline.
for cond_label, prefix, color in COMPARE_CONDITIONS:
    crash_d = load_condition_iv_deltas(prefix)
    ret_d = load_condition_iv_return_deltas(prefix)
    if not crash_d or not ret_d:
        continue
    n_seed = len(load_condition_returns(prefix))
    fig, (axc, axr) = plt.subplots(1, 2, figsize=(12, 4))
    for ax, deltas, scale, ylab in [
        (axc, crash_d, 100.0, "Δ collision rate vs baseline (pp)"),
        (axr, ret_d, 1.0, "Δ mean return vs baseline"),
    ]:
        means, los, his = [], [], []
        for iv in INTERVENTIONS:
            vals = [d * scale for d in deltas.get(iv, []) if d is not None]
            if vals:
                m, lo, hi = ci95(vals)
            else:
                m, lo, hi = 0.0, 0.0, 0.0
            means.append(m)
            los.append(lo)
            his.append(hi)
        err = [np.array(means) - np.array(los), np.array(his) - np.array(means)]
        ax.bar(
            np.arange(len(INTERVENTIONS)),
            means,
            yerr=err,
            capsize=4,
            color=color,
            edgecolor="black",
            linewidth=0.5,
            alpha=0.85,
        )
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_xticks(np.arange(len(INTERVENTIONS)))
        ax.set_xticklabels(INTERVENTIONS, rotation=30, ha="right")
        ax.set_ylabel(ylab)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle(f"Intervention effects - {cond_label} (n={n_seed}) vs baseline")
    fig.tight_layout()
    fig.savefig(OUT / f"interventions_{prefix}.png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"saved interventions_{prefix}.png")


# 6. correlation scatter  rho vs return, rho vs crash, return vs crash

rho_arr = np.array(rhos)
ret_arr = np.array(returns)
crash_arr = np.array(crashes)


def pearson_label(x, y):
    r, p = stats.pearsonr(x, y)
    sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."
    return f"r={r:+.2f} {sig}"


panels = [
    (rho_arr, ret_arr, r"$\rho$", "mean return", r"$\rho$ vs Return"),
    (rho_arr, crash_arr, r"$\rho$", "collision rate", r"$\rho$ vs Collision Rate"),
    (ret_arr, crash_arr, "mean return", "collision rate", "Return vs Collision Rate"),
]

fig, axes = plt.subplots(1, 3, figsize=(13, 4))
for ax, (x, y, xlabel, ylabel, title) in zip(axes, panels):
    ax.scatter(
        x, y, color="#2166ac", edgecolors="black", linewidths=0.5, s=60, alpha=0.85
    )
    m_fit, b_fit = np.polyfit(x, y, 1)
    xr = np.linspace(x.min(), x.max(), 100)
    ax.plot(
        xr, m_fit * xr + b_fit, color="#555", linewidth=1.2, linestyle="--", alpha=0.7
    )
    ax.text(
        0.04,
        0.96,
        pearson_label(x, y),
        transform=ax.transAxes,
        fontsize=8,
        va="top",
        bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.8),
    )
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, linewidth=0.4, alpha=0.4)
    ax.spines[["top", "right"]].set_visible(False)

fig.suptitle(
    f"{CONDITION} baseline (none condition), n={len(seeds)} seeds", fontsize=10
)
fig.tight_layout(rect=[0, 0, 1, 0.94])
fig.savefig(OUT / "correlation.png", dpi=300, bbox_inches="tight")
plt.close()
print("saved correlation.png")


# speaker consistency by condition

ALL_CONDITIONS = ["baseline"] + INTERVENTIONS
sc_by_cond = {
    cond: [get(data[s], cond, "sc") for s in seeds] for cond in ALL_CONDITIONS
}
sc_by_cond = {
    cond: [v for v in vals if v is not None] for cond, vals in sc_by_cond.items()
}

if any(sc_by_cond.values()):
    fig, ax = plt.subplots(figsize=(9, 4))
    cond_labels = list(sc_by_cond.keys())
    vals_list = [sc_by_cond[c] for c in cond_labels]

    bp = ax.boxplot(
        vals_list,
        tick_labels=cond_labels,
        patch_artist=True,
        medianprops=dict(color="black", linewidth=2),
    )
    palette = ["#2c7fb8"] + [
        "#d7191c" if c == "randomise" else "#41ab5d" for c in INTERVENTIONS
    ]
    for patch, color in zip(bp["boxes"], palette):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.axhline(
        1 / 3,
        color="grey",
        linestyle="--",
        linewidth=1,
        label="uniform baseline (1/n_signals)",
    )
    ax.set_ylabel("Speaker Consistency (SC)")
    ax.legend(fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", alpha=0.3)
    plt.xticks(rotation=20, ha="right")
    fig.tight_layout()
    fig.savefig(OUT / "sc_by_condition.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("saved sc_by_condition.png")
else:
    print("sc_by_condition skipped - sc missing from results")


# 8. rho-over-training curves

if not TRAINING:
    print("\nSkipping training curves (pass --training to enable)")
else:
    import torch

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from src.config import N_SIGNALS, N_TYPES, resolve_device
    from src.env import SignallingHighwayEnv
    from src.metrics import full_rho_report
    from src.rollout import self_obs_from_obs, nbr_tensors_from_info
    from src.train import load_checkpoint

    CHECKPOINTS = Path("checkpoints")
    N_EPISODES = 20
    DEVICE = resolve_device("auto")

    def rho_from_checkpoint(ckpt_path):
        policy, meta = load_checkpoint(ckpt_path, DEVICE)
        policy.eval()
        env = SignallingHighwayEnv(config=dict(meta["env_config"]))
        all_types, all_sigs = [], []
        for ep in range(N_EPISODES):
            obs, info = env.reset(seed=9999 + ep)
            done = False
            while not done:
                so = self_obs_from_obs(obs).to(DEVICE)
                no, ns, nm = nbr_tensors_from_info(info)
                ot = torch.from_numpy(info["types"]).long().to(DEVICE)
                with torch.no_grad():
                    out = policy(so, ot, no.to(DEVICE), ns.to(DEVICE), nm.to(DEVICE))
                sig = out["signal"].cpu().numpy()
                all_types.append(info["types"].copy())
                all_sigs.append(sig.copy())
                N = len(env.controlled_vehicles)
                obs, _, terminated, truncated, info = env.step(
                    [(out["action"].cpu().numpy()[i], int(sig[i])) for i in range(N)]
                )
                done = bool(terminated or truncated)
        t = np.concatenate(all_types)
        s = np.concatenate(all_sigs)
        return float(full_rho_report(t, s, N_TYPES, N_SIGNALS)["rho"])

    fig, ax = plt.subplots(figsize=(11, 4))
    colours = plt.cm.tab20.colors

    for ci, seed_dir in enumerate(sorted(CHECKPOINTS.glob(f"{CONDITION}_seed*"))):
        seed_label = seed_dir.name.replace(f"{CONDITION}_seed", "seed")
        ckpts = sorted(seed_dir.glob("iter_*.pt"))
        iters = [int(c.stem.split("_")[1]) for c in ckpts]
        final = seed_dir / "final.pt"
        if final.exists():
            last_iter = iters[-1] + 50 if iters else 50
            ckpts = list(ckpts) + [final]
            iters = iters + [last_iter]

        rho_curve = []
        print(f"evaluating {seed_label} ({len(ckpts)} checkpoints)...")
        for it, ckpt in zip(iters, ckpts):
            try:
                r = rho_from_checkpoint(ckpt)
                rho_curve.append((it, r))
                print(f"  iter {it:05d}: rho={r:.3f}")
            except Exception as e:
                print(f"  iter {it}: skipped ({e})")

        if rho_curve:
            xs, ys = zip(*rho_curve)
            ax.plot(
                xs,
                ys,
                marker=".",
                markersize=4,
                linewidth=1,
                color=colours[ci % len(colours)],
                label=seed_label,
            )

    ax.axhline(
        0.1, color="red", linestyle="--", linewidth=0.8, label="pooling threshold"
    )
    ax.set_xlabel("Training iteration")
    ax.set_ylabel(r"$\rho$")
    ax.legend(fontsize=7, ncol=4)
    ax.set_ylim(bottom=0)
    fig.tight_layout()
    fig.savefig(OUT / "rho_training.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("saved rho_training.png")


def plot_rho_over_training():
    import pandas as pd

    csvs = sorted(Path("checkpoints").glob(f"{CONDITION}_seed*/metrics.csv"))
    if not csvs:
        print("rho_over_training skipped - no checkpoints/*/metrics.csv found")
        return
    series = {}
    for c in csvs:
        df = pd.read_csv(c)
        if "rho" in df and "iter" in df:
            series[c.parent.name] = df.set_index("iter")["rho"]
    if not series:
        print("rho_over_training skipped - no rho column in metrics.csv")
        return
    M = pd.DataFrame(series)
    n = M.notna().sum(axis=1)
    mean = M.mean(axis=1)
    h = 1.96 * M.std(axis=1, ddof=1) / np.sqrt(n.clip(lower=1))

    fig, ax = plt.subplots(figsize=(7, 4))
    for col in M.columns:  # faint per-seed curves
        ax.plot(M.index, M[col], color="#9bb8d3", lw=0.5, alpha=0.35)
    ax.plot(mean.index, mean, color="#c0504d", lw=2, label="mean over seeds")
    ax.fill_between(
        mean.index, mean - h, mean + h, color="#c0504d", alpha=0.2, label="95% CI"
    )
    ax.axhline(0.1, ls="--", color="gray", lw=1, label="pooling threshold (0.1)")
    ax.text(
        mean.index.max(),
        0.1,
        "",
        va="bottom",
        ha="right",
        fontsize=8,
        color="gray",
    )
    ax.set_xlabel("training iteration")
    ax.set_ylabel(r"$\rho$")
    ax.set_ylim(bottom=0)
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / "rho_over_training.png", dpi=300, bbox_inches="tight")
    print("saved rho_over_training.png")


plot_rho_over_training()


def plot_rho_cic_outcomes():
    import re

    cic_path = RESULTS / f"cic_{CONDITION}.json"
    if not cic_path.exists():
        print(f"rho_cic_outcomes skipped - {cic_path} not found")
        return
    cic_by_seed = {}
    for r in json.loads(cic_path.read_text()):
        mm = re.search(r"(\d+)", str(r.get("seed", "")))
        if mm:
            cic_by_seed[int(mm.group(1))] = r["cic"]

    rows = []
    for s in seeds:
        num = int(re.search(r"(\d+)", s).group(1))
        rho = get(data[s], "baseline", "rho")
        coll = get(data[s], "baseline", "collision_rate")
        ret = get(data[s], "baseline", "mean_return")
        if num in cic_by_seed and None not in (rho, coll, ret):
            rows.append((rho, coll * 100.0, ret, cic_by_seed[num]))
    if len(rows) < 3:
        print("rho_cic_outcomes skipped - not enough aligned seeds")
        return
    rho_a, coll_a, ret_a, cic_a = (np.array(x) for x in zip(*rows))

    def panel(ax, x, y, xl, yl):
        ax.scatter(
            x, y, s=28, color="#3b6ea5", edgecolor="k", linewidth=0.4, alpha=0.85
        )
        m_fit, b_fit = np.polyfit(x, y, 1)
        xr = np.linspace(x.min(), x.max(), 100)
        ax.plot(xr, m_fit * xr + b_fit, color="#c0392b", linewidth=1.1, alpha=0.85)

        r_p, p_p = stats.pearsonr(x, y)
        r_s, p_s = stats.spearmanr(x, y)

        def ptxt(p):
            return "p < .001" if p < 0.001 else f"p = {p:.3f}".replace("0.", ".")

        ax.text(
            0.03,
            0.97,
            f"Pearson $r$ = {r_p:+.2f} ({ptxt(p_p)})\n"
            f"Spearman $r_s$ = {r_s:+.2f} ({ptxt(p_s)})",
            transform=ax.transAxes,
            fontsize=7.5,
            va="top",
            ha="left",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#bbbbbb", alpha=0.9),
        )
        ax.set_xlabel(xl)
        ax.set_ylabel(yl)

    fig, ax = plt.subplots(2, 2, figsize=(7.2, 6))
    panel(ax[0, 0], rho_a, coll_a, r"$\rho$", "collision rate (%)")
    panel(ax[0, 1], rho_a, ret_a, r"$\rho$", "mean return")
    panel(ax[1, 0], cic_a, coll_a, "CIC (nats)", "collision rate (%)")
    panel(ax[1, 1], cic_a, ret_a, "CIC (nats)", "mean return")
    fig.tight_layout()
    fig.savefig(OUT / "rho_cic_outcomes.png", dpi=300, bbox_inches="tight")
    print("saved rho_cic_outcomes.png")


plot_rho_cic_outcomes()


def plot_token_corruption_by_family():
    """Collision-rate change under the three token-corrupting interventions,
    for each training family, with 95% seed-level CIs. Shows that the penalty
    tracks whether a family learned to use the token."""
    FAMILIES = [
        ("bimodal", "bimodal", "#2c7fb8"),
        ("same_reward", "same-reward", "#fd8d3c"),
        ("random", "random-signal", "#74c476"),
    ]
    IVS = ["permute", "randomise", "desync"]

    fig, ax = plt.subplots(figsize=(7.2, 4))
    width = 0.26
    for k, (prefix, label, colour) in enumerate(FAMILIES):
        files = sorted(RESULTS.glob(f"{prefix}_seed*.json"))
        recs = [{r["condition"]: r for r in json.loads(f.read_text())} for f in files]
        means, errs = [], []
        for iv in IVS:
            d = (
                np.array(
                    [
                        r[iv]["collision_rate"] - r["baseline"]["collision_rate"]
                        for r in recs
                    ]
                )
                * 100.0
            )
            m, lo, hi = ci95(d)
            means.append(m)
            errs.append(m - lo)
        xs = np.arange(len(IVS)) + (k - 1) * width
        ax.bar(
            xs,
            means,
            width,
            yerr=errs,
            capsize=4,
            color=colour,
            edgecolor="black",
            linewidth=0.6,
            alpha=0.9,
            label=f"{label} (n={len(recs)})",
        )

    ax.axhline(0, color="black", linewidth=0.9)
    ax.set_xticks(np.arange(len(IVS)))
    ax.set_xticklabels(IVS)
    ax.set_ylabel("change in collision rate (percentage points)")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / "token_corruption_by_family.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("saved token_corruption_by_family.png")


plot_token_corruption_by_family()
