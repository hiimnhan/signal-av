import glob
import json

import numpy as np
from scipy import stats
from statsmodels.stats.multitest import multipletests

RESULTS = "results"
COND = "bimodal"
INTERV = ["hide", "randomise", "permute", "desync", "hide_kin", "hide_all"]
METRICS = {"collision_rate": 100.0, "mean_return": 1.0}  # metric -> display scale


def _rows():
    return [
        {r["condition"]: r for r in json.loads(open(fn).read())}
        for fn in sorted(glob.glob(f"{RESULTS}/{COND}_seed*.json"))
    ]


def _ci95(a):
    a = np.asarray(a, dtype=float)
    h = stats.t.ppf(0.975, df=len(a) - 1) * a.std(ddof=1) / np.sqrt(len(a))
    return float(a.mean()), float(a.mean() - h), float(a.mean() + h)


def paired_stats():
    rows = _rows()

    def col(cond, key):
        return np.array([r[cond][key] for r in rows], dtype=float)

    out = {}
    for metric, scale in METRICS.items():
        base = col("baseline", metric)
        res = {"baseline": {"mean": float(base.mean() * scale)}}
        pvals = []
        for iv in INTERV:
            x = col(iv, metric)
            dm, dlo, dhi = _ci95((x - base) * scale)
            try:
                p = float(stats.wilcoxon(x, base).pvalue)
            except ValueError:  # all differences zero
                p = 1.0
            pvals.append(p)
            res[iv] = {
                "mean": float(x.mean() * scale),
                "delta": dm,
                "ci": [dlo, dhi],
                "p": p,
            }
        for iv, pa in zip(INTERV, multipletests(pvals, method="holm")[1]):
            res[iv]["p_holm"] = float(pa)
        out[metric] = res
    return out


if __name__ == "__main__":
    print(json.dumps(paired_stats(), indent=2))
