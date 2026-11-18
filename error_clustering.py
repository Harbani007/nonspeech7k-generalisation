"""Do test errors cluster by recording? Consequences for evaluation (source-disjoint test set).

For each system (one row per training run), using its test_predictions.csv:
  * ICC of the binary error indicator within source groups (one-way ANOVA estimator), with a permutation p-value
    (errors shuffled across clips WITHIN each class, which keeps every class's error rate; 2000 permutations);
  * design effect DE = 1 + (m0 - 1) * ICC (m0 = ANOVA-adjusted mean group size) and effective test size n / DE;
  * concentration: share of all errors that falls in the 10% of recordings with most errors, vs their clip share;
  * share of recordings that are either all-correct or all-wrong (among recordings with >= 2 clips);
  * macro-F1 95% CI width from a clip bootstrap (treats clips as independent) vs a source-group bootstrap.

  python error_clustering.py --runs ../results/v3/runs --out ../results/v3/error_clustering.csv
"""
import argparse
import glob
import os

import numpy as np
import pandas as pd

from dl_metrics import macro_f1

K = 7


def icc_binary(e, g):
    """One-way random-effects ICC(1) for a 0/1 outcome; returns (icc, m0)."""
    df = pd.DataFrame({"e": e.astype(float), "g": g})
    st = df.groupby("g").e.agg(["sum", "count", "mean"])
    n, k = len(df), len(st)
    grand = df.e.mean()
    ssb = (st["count"] * (st["mean"] - grand) ** 2).sum()
    ssw = ((df.e - df.g.map(st["mean"])) ** 2).sum()
    msb, msw = ssb / (k - 1), ssw / (n - k)
    m0 = (n - (st["count"] ** 2).sum() / n) / (k - 1)
    icc = (msb - msw) / (msb + (m0 - 1) * msw) if (msb + (m0 - 1) * msw) > 0 else 0.0
    return float(icc), float(m0)


def boot_ci_width(y, p, g, n=1000, seed=0, grouped=True):
    rng = np.random.default_rng(seed)
    if grouped:
        uniq = np.unique(g)
        pos = {x: np.where(g == x)[0] for x in uniq}
        draws = [np.concatenate([pos[x] for x in rng.choice(uniq, len(uniq))]) for _ in range(n)]
    else:
        draws = [rng.integers(0, len(y), len(y)) for _ in range(n)]
    v = [macro_f1(y[i], p[i], K) for i in draws]
    lo, hi = np.percentile(v, [2.5, 97.5])
    return float(hi - lo)


def analyse(df, perms, seed):
    y, p, g = df.y.values, df.pred.values, df.group.values.astype(str)
    e = (y != p).astype(int)
    icc, m0 = icc_binary(e, g)
    rng = np.random.default_rng(seed)
    null = []
    for _ in range(perms):
        ep = e.copy()
        for c in np.unique(y):
            ix = np.where(y == c)[0]
            ep[ix] = rng.permutation(ep[ix])
        null.append(icc_binary(ep, g)[0])
    pval = (1 + np.sum(np.array(null) >= icc)) / (1 + perms)
    per = pd.DataFrame({"e": e, "g": g}).groupby("g").e.agg(["sum", "count", "mean"])
    top = per.sort_values("sum", ascending=False).head(max(1, int(round(0.1 * len(per)))))
    multi = per[per["count"] >= 2]
    de = 1 + (m0 - 1) * icc
    return dict(test_clips=len(y), test_sources=len(per), errors=int(e.sum()), macro_f1=macro_f1(y, p, K),
                error_icc=icc, icc_perm_p=pval, icc_null_mean=float(np.mean(null)), m0=m0, design_effect=de,
                effective_n=len(y) / de,
                top10pct_recordings_share_of_errors=float(top["sum"].sum() / max(e.sum(), 1)),
                top10pct_recordings_share_of_clips=float(top["count"].sum() / len(y)),
                multi_clip_recordings=len(multi),
                share_all_correct_or_all_wrong=float(((multi["mean"] == 0) | (multi["mean"] == 1)).mean()),
                ci_width_clip_bootstrap=boot_ci_width(y, p, g, grouped=False),
                ci_width_source_bootstrap=boot_ci_width(y, p, g, grouped=True))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--perms", type=int, default=2000)
    ap.add_argument("--glob", default="*source_disjoint*")
    a = ap.parse_args()
    rows = []
    for d in sorted(glob.glob(os.path.join(a.runs, a.glob))):
        f = os.path.join(d, "test_predictions.csv")
        if not os.path.exists(f):
            continue
        r = analyse(pd.read_csv(f), a.perms, 0)
        r["run"] = os.path.basename(d)
        rows.append(r)
        print(f"{r['run']:45s} F1 {r['macro_f1']:.3f} errors {r['errors']:4d} ICC {r['error_icc']:.3f} "
              f"(p {r['icc_perm_p']:.4f}, null {r['icc_null_mean']:.3f}) DE {r['design_effect']:.2f} "
              f"n_eff {r['effective_n']:.0f}/{r['test_clips']} top10% recs hold {r['top10pct_recordings_share_of_errors']:.2f} "
              f"of errors ({r['top10pct_recordings_share_of_clips']:.2f} of clips); CI width clip {r['ci_width_clip_bootstrap']:.3f} "
              f"vs source {r['ci_width_source_bootstrap']:.3f}", flush=True)
    pd.DataFrame(rows).to_csv(a.out, index=False)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
