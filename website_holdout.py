"""Leave-one-website-out test: does accuracy drop when a website is absent from training?

For each held-out website W, its recordings are split into 5 folds grouped by File_ID. Each test fold is
classified twice:
  holdout : trained on the OTHER websites only
  control : trained on the other websites AND the remaining recordings of W (source-matched)
Neither sees the test recordings, so the gap (control - holdout) is the cost of never having heard website W.

Matching (--match):
  clips      : same number of clips per class in both conditions (the holdout's count, capped at --cap)
  recordings : same number of distinct recordings per class in both conditions, one clip per recording, so
               training size, class composition and recording diversity are identical; only the website mix
               differs

Only classes with at least --min-recs recordings in W (and in the other websites) are scored; the rest are
listed as untestable. Intervals: one bootstrap over test recordings (the unit of independence), with all
seeds averaged inside each resample, paired between conditions and between feature sets.

  python website_holdout.py --meta-train ... --meta-test ... --feats feats --sets spec struct-nf ast --match recordings
"""
import argparse
import itertools
import warnings

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning

from common import load_meta
from run_experiments import fit_predict, get_blocks, prep

warnings.filterwarnings("ignore", category=ConvergenceWarning)


def macro_f1(yt, yp, labels):
    f = []
    for c in labels:
        tp = np.sum((yp == c) & (yt == c))
        p, t = np.sum(yp == c), np.sum(yt == c)
        f.append(2 * tp / (p + t) if p + t else 0.0)
    return float(np.mean(f))


def sample_clips(pool, y, cap, rng, quota=None):
    out = []
    for c in np.unique(y[pool]):
        ic = pool[y[pool] == c]
        n = min(cap if quota is None else quota.get(c, 0), len(ic))
        out.extend(rng.choice(ic, n, replace=False))
    return np.array(out, dtype=int)


def sample_recordings(pool, y, g, cap, rng, quota=None):
    """One random clip from each of R randomly chosen recordings per class."""
    out = []
    for c in np.unique(y[pool]):
        ic = pool[y[pool] == c]
        recs = np.unique(g[ic])
        n = min(cap if quota is None else quota.get(c, 0), len(recs))
        for r in rng.choice(recs, n, replace=False):
            out.append(rng.choice(ic[g[ic] == r]))
    return np.array(out, dtype=int)


def merge_groups(meta, pairs):
    """Union the groups of clips that are the same audio, so a clip and its re-upload never straddle a split."""
    parent = {g: g for g in meta["group"]}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    grp = dict(zip(meta["filename"], meta["group"]))
    for f1, f2 in zip(pairs["a"], pairs["b"]):
        parent[find(grp[f1])] = find(grp[f2])
    merged = meta["group"].map(find)
    print(f"merged duplicate audio: {meta['group'].nunique()} -> {merged.nunique()} groups")
    return merged


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--meta-train", required=True)
    ap.add_argument("--meta-test", required=True)
    ap.add_argument("--feats", default="feats")
    ap.add_argument("--sets", nargs="+", default=["spec", "struct-nf"])
    ap.add_argument("--match", choices=["clips", "recordings"], default="recordings")
    ap.add_argument("--cap", type=int, default=100, help="max training clips (or recordings) per class")
    ap.add_argument("--min-recs", type=int, default=5)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--out", default="website_holdout.csv")
    ap.add_argument("--duplicates", help="CSV of clip pairs (columns a, b, same_audio) whose groups are merged")
    ap.add_argument("--save-preds", action="store_true",
                    help="also write <out>_predictions.csv: every test clip x seed x condition x feature set")
    a = ap.parse_args()

    meta = load_meta(a.meta_train, a.meta_test)  # pooled: the official split plays no role here
    if a.duplicates:
        meta["group"] = merge_groups(meta, pd.read_csv(a.duplicates).query("same_audio"))
    y, g, src = meta["y"].values, meta["group"].values, meta["source"].values
    names = meta.groupby("y")["cls"].first()
    recs = meta.groupby(["source", "y"])["group"].nunique()
    blocks = {s: get_blocks(s, meta, a.feats) for s in a.sets}

    def draw(pool, rng, quota=None):
        if a.match == "recordings":
            return sample_recordings(pool, y, g, a.cap, rng, quota)
        return sample_clips(pool, y, a.cap, rng, quota)

    def counts(tr):
        if a.match == "recordings":
            return {c: len(np.unique(g[tr][y[tr] == c])) for c in np.unique(y[tr])}
        return {c: int((y[tr] == c).sum()) for c in np.unique(y[tr])}

    summary, per_class, diffs, all_preds = [], [], [], []
    for w in sorted(set(src)):
        other = np.where(src != w)[0]
        ok = [c for c in names.index if recs.get((w, c), 0) >= a.min_recs
              and meta.iloc[other].query("y == @c")["group"].nunique() >= a.min_recs]
        print(f"\nheld-out website {w}: scored {[names[c] for c in ok]}; "
              f"untestable {[names[c] for c in names.index if c not in ok]}")
        if not ok:
            continue
        test_pool = np.where((src == w) & np.isin(y, ok))[0]
        yt, rt = y[test_pool], g[test_pool]
        P = {(s, k): np.empty((a.seeds, len(test_pool)), int) for s in a.sets for k in ("control", "holdout")}
        log, fold_store = [], []
        for seed in range(a.seeds):
            rng = np.random.default_rng(seed)
            fold_of = {r: i % 5 for i, r in enumerate(rng.permutation(np.unique(rt)))}
            folds = np.array([fold_of[r] for r in rt])
            fold_store.append(folds)
            for f in range(5):
                te = test_pool[folds == f]
                keep = ~np.isin(g, g[te])  # never train on a test recording, even across websites
                tr_h = draw(np.where(keep & (src != w))[0], rng)
                tr_c = draw(np.where(keep)[0], rng, counts(tr_h))
                assert counts(tr_c) == counts(tr_h), "conditions not matched"
                log.append(dict(n=len(tr_h), recs_h=len(np.unique(g[tr_h])), recs_c=len(np.unique(g[tr_c])),
                                w_share_control=float((src[tr_c] == w).mean())))
                for s in a.sets:
                    for k, tr in (("holdout", tr_h), ("control", tr_c)):
                        Xtr, Xte = prep([b[tr] for b in blocks[s]], [b[te] for b in blocks[s]])
                        P[(s, k)][seed, folds == f] = fit_predict(Xtr, y[tr], g[tr], Xte, True)
            print(f"  seed {seed} done", flush=True)
        if a.save_preds:
            for (s_, k_), arr in P.items():
                for seed in range(a.seeds):
                    all_preds.append(pd.DataFrame(dict(
                        heldout=w, featureset=s_, condition=k_, seed=seed, fold=fold_store[seed],
                        filename=meta["filename"].values[test_pool], group=rt, y=yt, pred=arr[seed],
                        scored_class=True)))
        lg = pd.DataFrame(log).mean()
        print(f"  training size {lg.n:.0f}; distinct recordings holdout {lg.recs_h:.0f} / control {lg.recs_c:.0f}; "
              f"share of {w} clips in control training {lg.w_share_control:.2f}")

        # paired bootstrap over test recordings; every seed scored on the same resample
        uniq = np.unique(rt)
        pos = {r: np.where(rt == r)[0] for r in uniq}
        brng = np.random.default_rng(12345)

        def gap(i, s):
            return np.mean([macro_f1(yt[i], P[(s, "control")][k][i], ok) - macro_f1(yt[i], P[(s, "holdout")][k][i], ok)
                            for k in range(a.seeds)])

        boots = {s: [] for s in a.sets}
        for _ in range(a.boot):
            i = np.concatenate([pos[r] for r in brng.choice(uniq, len(uniq))])
            for s in a.sets:
                boots[s].append(gap(i, s))
        allidx = np.arange(len(yt))
        for s in a.sets:
            c = np.mean([macro_f1(yt, P[(s, "control")][k], ok) for k in range(a.seeds)])
            h = np.mean([macro_f1(yt, P[(s, "holdout")][k], ok) for k in range(a.seeds)])
            lo, hi = np.percentile(boots[s], [2.5, 97.5])
            summary.append(dict(heldout=w, featureset=s, match=a.match, classes=len(ok),
                                test_clips=len(yt), test_recordings=len(uniq), train_size=lg.n,
                                control=c, holdout=h, gap=gap(allidx, s), ci_lo=lo, ci_hi=hi))
            for cl in ok:
                m = yt == cl
                per_class.append(dict(heldout=w, featureset=s, cls=names[cl], test_recordings=len(np.unique(rt[m])),
                                      control_recall=(P[(s, "control")][:, m] == cl).mean(),
                                      holdout_recall=(P[(s, "holdout")][:, m] == cl).mean()))
        for s1, s2 in itertools.combinations(a.sets, 2):
            d = np.array(boots[s1]) - np.array(boots[s2])
            diffs.append(dict(heldout=w, pair=f"{s1} - {s2}", gap_diff=gap(allidx, s1) - gap(allidx, s2),
                              ci_lo=np.percentile(d, 2.5), ci_hi=np.percentile(d, 97.5)))

    out = pd.DataFrame(summary)
    out.to_csv(a.out, index=False)
    pc = pd.DataFrame(per_class)
    pc["drop"] = pc["control_recall"] - pc["holdout_recall"]
    pc.to_csv(a.out.replace(".csv", "_per_class.csv"), index=False)
    dd = pd.DataFrame(diffs)
    dd.to_csv(a.out.replace(".csv", "_gap_diffs.csv"), index=False)
    if a.save_preds:
        pd.concat(all_preds).to_csv(a.out.replace(".csv", "_predictions.csv.gz"), index=False)
    pd.set_option("display.width", 200)
    print(f"\nmacro-F1 over scored classes ({a.match} matched), mean over {a.seeds} seeds; "
          f"95% CI from a bootstrap over test recordings")
    print(out.set_index(["heldout", "featureset"])[["classes", "test_recordings", "train_size", "control",
                                                    "holdout", "gap", "ci_lo", "ci_hi"]].round(3))
    print("\ndifference between feature sets' gaps (positive = first loses more)")
    print(dd.set_index(["heldout", "pair"]).round(3))
    print("\nper-class recall drop (control - holdout)")
    print(pc.pivot_table(index=["heldout", "cls", "test_recordings"], columns="featureset", values="drop").round(2))


if __name__ == "__main__":
    main()
