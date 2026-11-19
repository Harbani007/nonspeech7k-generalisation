"""Tables and figures for the deep-learning experiments, built only from saved run folders.

Every figure is written as PNG + SVG with the CSV it was drawn from. Uncertainty: paired bootstrap over SOURCE
GROUPS of the test set (2000 resamples), all available training seeds averaged inside each resample.

  python dl_report.py --data results/v3/dl --runs results/v3/runs --audit results/v3 --out results/v3/report
"""
import argparse
import glob
import json
import os
import shutil

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from dl_data import CLASSES  # noqa: E402
from dl_metrics import confusion, grouped_bootstrap, metrics, source_nn_rate  # noqa: E402

K = len(CLASSES)
SYSTEMS = [  # (label, run-folder glob relative to runs, role)
    ("Spectral + LR", "ml_spec_{p}", "traditional ML"),
    ("Structure + LR", "ml_struct-nf_{p}", "traditional ML"),
    ("CNN (scratch)", "cnn_{p}_standard_s*", "deep learning, quick control"),
    ("CNN (scratch), source-balanced", "cnn_{p}_source_s*", "deep learning, quick control"),
    ("Frozen AST (10 s) + LR", "ml_ast_{p}", "pretrained representation"),
    ("AST frozen + linear head", "ast_frozen_{p}_standard_s*", "pretrained representation"),
    ("AST-FT ordinary", "ast_ft_{p}_standard_s*", "fine-tuned baseline (CE)"),
    ("AST-FT source-balanced", "ast_ft_{p}_source_s*", "CE + source-balanced sampling"),
    ("AST-FT cross-source contrastive", "ast_ftsc_{p}_standard_s*", "CE + cross-source SupCon"),
    ("AST-FT combined", "ast_ftsc_{p}_source_s*", "CE + SupCon + source-balanced sampling"),
    ("AST-FT recording-adversarial", "ast_ftadv_{p}_standard_s*", "EXPLORATORY: CE + gradient-reversal recording adversary"),
    ("AST-FT ordinary (waveform input)", "ast_wv_{p}_standard_s*", "control for background swap"),
    ("AST-FT background-swap", "ast_bgs_{p}_standard_s*", "CE + background-swap augmentation"),
]
AST4 = ["AST-FT ordinary", "AST-FT source-balanced", "AST-FT cross-source contrastive", "AST-FT combined"]


def save(fig, out, name, df=None):
    for ext in ("png", "svg"):
        fig.savefig(os.path.join(out, f"{name}.{ext}"), dpi=160, bbox_inches="tight")
    plt.close(fig)
    if df is not None:
        df.to_csv(os.path.join(out, f"{name}.csv"), index=False)


def load_runs(runs, pattern):
    dirs = sorted(d for d in glob.glob(os.path.join(runs, pattern)) if os.path.exists(os.path.join(d, "test_predictions.csv")))
    return dirs, [pd.read_csv(os.path.join(d, "test_predictions.csv")) for d in dirs]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--runs", required=True)
    ap.add_argument("--audit", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--boot", type=int, default=2000)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    sp = pd.read_csv(os.path.join(a.data, "splits.csv"))
    dedup_ok = set(sp.filename[sp.official_dedup == "test"])

    # ---- main table: every system x evaluation ------------------------------------------------------------
    rows, preds_cache = [], {}
    for label, pat, role in SYSTEMS:
        for proto, evalname in [("source_disjoint", "source-disjoint test"), ("official", "official test"),
                                ("official", "official test, de-duplicated")]:
            dirs, P = load_runs(a.runs, pat.format(p=proto))
            if not P:
                continue
            if evalname.endswith("de-duplicated"):
                P = [p[p.filename.isin(dedup_ok)].reset_index(drop=True) for p in P]
            y, g = P[0].y.values, P[0].group.values
            assert all((p.filename.values == P[0].filename.values).all() for p in P)
            preds = [p.pred.values for p in P]
            preds_cache[(label, evalname)] = (y, g, preds, P[0])
            pt, _, _, ci, _, _ = grouped_bootstrap(y, preds, None, g, K, n=a.boot)
            ms = [metrics(y, q, K) for q in preds]
            rows.append(dict(system=label, role=role, evaluation=evalname, seeds=len(P), test_clips=len(y),
                             test_sources=len(np.unique(g)), macro_f1=pt, ci_lo=ci[0], ci_hi=ci[1],
                             macro_f1_seed_sd=float(np.std([m["macro_f1"] for m in ms])) if len(ms) > 1 else np.nan,
                             accuracy=float(np.mean([m["accuracy"] for m in ms])),
                             balanced_accuracy=float(np.mean([m["balanced_accuracy"] for m in ms])),
                             **{f"recall_{c}": float(np.mean([m["per_class_recall"][i] for m in ms])) for i, c in enumerate(CLASSES)}))
    tab = pd.DataFrame(rows)
    tab.to_csv(os.path.join(a.out, "system_comparison.csv"), index=False)
    pd.set_option("display.width", 220)
    print(tab[["system", "evaluation", "seeds", "test_clips", "test_sources", "macro_f1", "ci_lo", "ci_hi",
               "macro_f1_seed_sd", "accuracy", "balanced_accuracy"]].round(3).to_string(index=False))

    # ---- paired differences ---------------------------------------------------------------------------------
    pairs = [("AST-FT ordinary", "AST-FT source-balanced", "balanced - ordinary"),
             ("AST-FT ordinary", "AST-FT cross-source contrastive", "contrastive - ordinary"),
             ("AST-FT ordinary", "AST-FT combined", "combined - ordinary"),
             ("AST-FT source-balanced", "AST-FT combined", "combined - balanced"),
             ("AST-FT cross-source contrastive", "AST-FT combined", "combined - contrastive"),
             ("AST frozen + linear head", "AST-FT ordinary", "fine-tuned - frozen (AST, same 4 s input)"),
             ("AST-FT ordinary", "AST-FT recording-adversarial", "EXPLORATORY adversarial - ordinary"),
             ("AST-FT ordinary (waveform input)", "AST-FT background-swap", "background-swap - ordinary (same waveform inputs)"),
             ("CNN (scratch)", "AST-FT ordinary", "AST-FT - CNN"),
             ("CNN (scratch)", "CNN (scratch), source-balanced", "source-balanced - standard (CNN)"),
             ("Structure + LR", "CNN (scratch)", "CNN - structure LR")]
    drows = []
    for A, B, name in pairs:
        for evalname in ("source-disjoint test", "official test"):
            if (A, evalname) in preds_cache and (B, evalname) in preds_cache:
                ya, ga, pa, fa = preds_cache[(A, evalname)]
                yb, gb, pb, fb = preds_cache[(B, evalname)]
                if not (fa.filename.values == fb.filename.values).all():
                    continue
                sa, sb, d, _, _, cd = grouped_bootstrap(ya, pa, pb, ga, K, n=a.boot)
                drows.append(dict(comparison=name, evaluation=evalname, a=A, b=B, macro_f1_a=sa, macro_f1_b=sb,
                                  diff=d, ci_lo=cd[0], ci_hi=cd[1], seeds_a=len(pa), seeds_b=len(pb),
                                  ci_excludes_zero=bool(cd[0] > 0 or cd[1] < 0)))
    # original vs corrected evaluation of the same models: official-test score minus de-duplicated score
    for label, _, _ in SYSTEMS:
        k1, k2 = (label, "official test"), (label, "official test, de-duplicated")
        if k1 in preds_cache and k2 in preds_cache:
            y1, g1, p1, f1 = preds_cache[k1]
            keep = f1.filename.isin(dedup_ok).values
            uniq = np.unique(g1)
            rng = np.random.default_rng(7)
            posg = {x: np.where(g1 == x)[0] for x in uniq}
            dif = []
            for _ in range(a.boot):
                ix = np.concatenate([posg[x] for x in rng.choice(uniq, len(uniq))])
                ixk = ix[keep[ix]]
                dif.append(np.mean([metrics(y1[ix], q[ix], K)["macro_f1"] - metrics(y1[ixk], q[ixk], K)["macro_f1"] for q in p1]))
            pt = np.mean([metrics(y1, q, K)["macro_f1"] - metrics(y1[keep], q[keep], K)["macro_f1"] for q in p1])
            lo, hi = np.percentile(dif, [2.5, 97.5])
            drows.append(dict(comparison="official - de-duplicated official (same model)", evaluation="official test",
                              a=label, b=label, diff=pt, ci_lo=lo, ci_hi=hi, seeds_a=len(p1), ci_excludes_zero=bool(lo > 0 or hi < 0)))
    diffs = pd.DataFrame(drows)
    diffs.to_csv(os.path.join(a.out, "paired_differences.csv"), index=False)
    print("\n", diffs.round(3).to_string(index=False))

    sd = tab[tab.evaluation == "source-disjoint test"].set_index("system")

    # G1/G2 learning curves
    hist = []
    for d in sorted(glob.glob(os.path.join(a.runs, "*_s*"))):
        h = os.path.join(d, "history.csv")
        if os.path.exists(h):
            hist.append(pd.read_csv(h).assign(run=os.path.basename(d)))
    if hist:
        H = pd.concat(hist)
        for fam, title in [("cnn_source_disjoint", "CNN (scratch), source-disjoint protocol")]:
            h = H[H.run.str.startswith(fam)]
            if h.empty:
                continue
            fig, ax = plt.subplots(1, 2, figsize=(12, 4))
            for run, d in h.groupby("run"):
                st = "--" if "_source_" in run else "-"
                ax[0].plot(d.epoch, d.train_loss, st, label=f"{run} train")
                ax[0].plot(d.epoch, d.val_loss, st, alpha=.6, label=f"{run} val")
                ax[1].plot(d.epoch, d.val_macro_f1, st, label=f"{run} macro-F1")
                ax[1].plot(d.epoch, d.val_acc, st, alpha=.4, label=f"{run} acc")
            ax[0].set(title=f"{title}: loss", xlabel="epoch", ylabel="cross-entropy")
            ax[1].set(title="validation accuracy / macro-F1", xlabel="epoch", ylim=(0, 1))
            for x in ax:
                x.legend(fontsize=6)
            save(fig, a.out, f"G1_G2_learning_curves_{fam}", h)

    # G3 confusion matrices (source-disjoint, summed over seeds), row-normalised
    cms = [(s, preds_cache[(s, "source-disjoint test")]) for s in ["AST frozen + linear head"] + AST4
           if (s, "source-disjoint test") in preds_cache]
    if cms:
        nc = min(3, len(cms))
        nr = int(np.ceil(len(cms) / nc))
        fig, axs = plt.subplots(nr, nc, figsize=(5.2 * nc, 4.6 * nr), squeeze=False)
        for ax in axs.ravel()[len(cms):]:
            ax.axis("off")
        axs = axs.ravel()
        recs = []
        for ax, (s, (y, g, preds, _)) in zip(axs, cms):
            c = sum(confusion(y, q, K) for q in preds)
            cn = c / c.sum(1, keepdims=True)
            ax.imshow(cn, cmap="Blues", vmin=0, vmax=1)
            for i in range(K):
                for j in range(K):
                    ax.text(j, i, f"{cn[i, j]:.2f}", ha="center", va="center", fontsize=7, color="white" if cn[i, j] > .5 else "black")
            ax.set_xticks(range(K), CLASSES, rotation=45, fontsize=7)
            ax.set_yticks(range(K), CLASSES, fontsize=7)
            ax.set(title=f"{s}\n({len(preds)} seed(s))", xlabel="predicted", ylabel="true")
            for i in range(K):
                for j in range(K):
                    recs.append(dict(system=s, true=CLASSES[i], pred=CLASSES[j], count=int(c[i, j]), row_share=cn[i, j]))
        save(fig, a.out, "G3_confusion_matrices", pd.DataFrame(recs))

    # G4 per-class recall
    if len(sd):
        rc = sd[[f"recall_{c}" for c in CLASSES]]
        fig, ax = plt.subplots(figsize=(11, 4))
        w = 0.8 / len(rc)
        for i, (s, r) in enumerate(rc.iterrows()):
            ax.bar(np.arange(K) + i * w - 0.4 + w / 2, r.values, w, label=s)
        ax.set_xticks(range(K), CLASSES)
        ax.set(ylim=(0, 1), ylabel="recall (mean over seeds)", title="Per-class recall, source-disjoint test")
        ax.legend(fontsize=7, ncol=2)
        save(fig, a.out, "G4_per_class_recall", rc.reset_index())

    def bars(systems, name, title, evals=("source-disjoint test",)):
        t = tab[tab.system.isin(systems) & tab.evaluation.isin(evals)].copy()
        if t.empty:
            return
        t["order"] = t.system.map({s: i for i, s in enumerate(systems)})
        t = t.sort_values(["order", "evaluation"])
        fig, ax = plt.subplots(figsize=(max(6, 1.6 * len(t)), 4))
        cols = {"source-disjoint test": "#55A868", "official test": "#C44E52", "official test, de-duplicated": "#4C72B0"}
        xs = np.arange(len(t))
        ax.bar(xs, t.macro_f1, color=[cols[e] for e in t.evaluation])
        ax.errorbar(xs, t.macro_f1, yerr=[t.macro_f1 - t.ci_lo, t.ci_hi - t.macro_f1], fmt="none", color="k", capsize=3)
        ax.set_xticks(xs, [f"{s}\n[{e}]" if len(evals) > 1 else s for s, e in zip(t.system, t.evaluation)], fontsize=7, rotation=20, ha="right")
        ax.set(ylim=(0, 1), ylabel="macro-F1 (95% CI, source bootstrap)", title=title)
        save(fig, a.out, name, t.drop(columns="order"))

    bars([s for s, _, _ in SYSTEMS], "G0_all_systems", "All systems, source-disjoint test")
    bars(["AST frozen + linear head", "AST-FT ordinary"], "G5_frozen_vs_finetuned_ast",
         "Frozen vs fine-tuned AST (same 4 s input, same head)")
    bars(AST4, "G6_ast_2x2_source_aware_training",
         "AST fine-tuning: ordinary / source-balanced / cross-source contrastive / combined")
    bars(["Spectral + LR", "Structure + LR", "CNN (scratch)", "Frozen AST (10 s) + LR"],
         "G7_original_vs_corrected_evaluation", "Official vs de-duplicated official vs source-disjoint test",
         evals=("official test", "official test, de-duplicated", "source-disjoint test"))

    # G8 source concentration (from source_audit.py)
    src = os.path.join(a.audit, "figures", "A2_source_concentration.png")
    if os.path.exists(src):
        for ext in ("png", "svg"):
            shutil.copy(os.path.join(a.audit, "figures", f"A2_source_concentration.{ext}"), os.path.join(a.out, f"G8_source_concentration.{ext}"))
        shutil.copy(os.path.join(a.audit, "source_concentration_class.csv"), os.path.join(a.out, "G8_source_concentration.csv"))
    ast_2x2(a, preds_cache)
    print("wrote", a.out)


def fast_f1(y, p):
    c = np.bincount(y * K + p, minlength=K * K).reshape(K, K)
    tp, sup, pred = np.diag(c), c.sum(1), c.sum(0)
    ok = sup > 0
    return float((2 * tp[ok] / np.maximum(sup[ok] + pred[ok], 1)).mean())


def ast_2x2(a, preds_cache):
    """Factorial analysis of the controlled AST experiment, learning curves, source-clustering diagnostic."""
    def key(s):
        return (s, "source-disjoint test")
    if not all(key(s) in preds_cache for s in AST4):
        print("2x2 AST analysis skipped: not all four conditions have finished runs")
        return
    y, g, _, f0 = preds_cache[key(AST4[0])]
    for s in AST4:
        assert (preds_cache[key(s)][3].filename.values == f0.filename.values).all()
    P = {s: preds_cache[key(s)][2] for s in AST4}
    o, b, c, m = AST4

    def effects(ix):
        sc = {s: np.mean([fast_f1(y[ix], q[ix]) for q in P[s]]) for s in AST4}
        out = {f"macro-F1 {s}": sc[s] for s in AST4}
        out.update(balanced_main_effect=(sc[b] + sc[m] - sc[o] - sc[c]) / 2,
                   contrastive_main_effect=(sc[c] + sc[m] - sc[o] - sc[b]) / 2,
                   interaction=(sc[m] - sc[b]) - (sc[c] - sc[o]),
                   combined_minus_ordinary=sc[m] - sc[o])
        return out
    uniq = np.unique(g)
    posg = {x: np.where(g == x)[0] for x in uniq}
    rng = np.random.default_rng(2024)
    pt = effects(np.arange(len(y)))
    boot = pd.DataFrame([effects(np.concatenate([posg[x] for x in rng.choice(uniq, len(uniq))])) for _ in range(a.boot)])
    eff = pd.DataFrame([dict(quantity=k, estimate=v, ci_lo=np.percentile(boot[k], 2.5), ci_hi=np.percentile(boot[k], 97.5))
                        for k, v in pt.items()])
    eff["ci_excludes_zero"] = (eff.ci_lo > 0) | (eff.ci_hi < 0)
    eff["seeds_per_condition"] = min(len(P[s]) for s in AST4)
    eff["test_sources"] = len(uniq)
    eff.to_csv(os.path.join(a.out, "ast_2x2_effects.csv"), index=False)
    print("\n2x2 AST experiment (source-disjoint test, macro-F1, source-grouped bootstrap):")
    print(eff.round(4).to_string(index=False))

    rows = []
    for s in AST4:
        for w in sorted(f0.source.unique()):
            ix = np.where(f0.source.values == w)[0]
            rows.append(dict(system=s, website=w, clips=len(ix), sources=len(np.unique(g[ix])),
                             classes=len(np.unique(y[ix])), macro_f1=np.mean([fast_f1(y[ix], q[ix]) for q in P[s]])))
    pw = pd.DataFrame(rows)
    pw.to_csv(os.path.join(a.out, "ast_2x2_per_website.csv"), index=False)
    print("\nper-website macro-F1 (source-disjoint test):\n", pw.pivot(index="website", columns="system", values="macro_f1")[AST4].round(3).to_string())

    cols = dict(zip(AST4, ["#4C72B0", "#55A868", "#C44E52", "#8172B2"]))
    pats = dict(zip(AST4, ["ast_ft_source_disjoint_standard_s*", "ast_ft_source_disjoint_source_s*",
                           "ast_ftsc_source_disjoint_standard_s*", "ast_ftsc_source_disjoint_source_s*"]))
    hs = []
    for s in AST4:
        for d in sorted(glob.glob(os.path.join(a.runs, pats[s]))):
            if os.path.exists(os.path.join(d, "test_metrics.json")):
                hs.append(pd.read_csv(os.path.join(d, "history.csv")).assign(system=s, run=os.path.basename(d)))
    H = pd.concat(hs)
    fig, ax = plt.subplots(1, 4, figsize=(21, 4.2))
    for (s, run), d in H.groupby(["system", "run"]):
        lab = s if run.endswith("_s0") else None
        ax[0].plot(d.epoch, d.train_ce, color=cols[s], label=lab)
        ax[0].plot(d.epoch, d.val_loss, "--", color=cols[s])
        ax[1].plot(d.epoch, d.val_macro_f1, color=cols[s], label=lab)
        ax[1].plot(d.epoch, d.val_acc, ":", color=cols[s])
        if d.train_supcon.abs().sum() > 0:
            ax[2].plot(d.epoch, d.train_supcon, color=cols[s], label=lab)
        ax[3].plot(d.epoch, d.anchors_with_positive, color=cols[s], label=lab)
    ax[0].set(title="cross-entropy: train (solid), val (dashed)", xlabel="epoch")
    ax[1].set(title="validation macro-F1 (solid), accuracy (dotted)", xlabel="epoch")
    ax[2].set(title="cross-source SupCon loss (train)", xlabel="epoch")
    ax[3].set(title="batch anchors with a cross-source positive", xlabel="epoch", ylim=(0, 1))
    for x in ax:
        x.legend(fontsize=7)
    save(fig, a.out, "G1_G2_learning_curves_ast_2x2", H)

    # source-clustering diagnostic (dl_metrics.source_nn_rate): test partition only, same-class reference pool,
    # self excluded, duplicates count as the same source. Frozen AST (pretrained encoder, untouched) is the reference.
    rows = []
    systems9 = ["AST frozen + linear head"] + AST4 + ["AST-FT recording-adversarial"]
    pats9 = dict(pats, **{"AST frozen + linear head": "ast_frozen_source_disjoint_standard_s*",
                          "AST-FT recording-adversarial": "ast_ftadv_source_disjoint_standard_s*"})
    cols9 = dict(cols, **{"AST frozen + linear head": "#999999", "AST-FT recording-adversarial": "#DD8452"})
    for s in systems9:
        for d in sorted(glob.glob(os.path.join(a.runs, pats9[s]))):
            e = os.path.join(d, "test_embeddings.npy")
            if not os.path.exists(e) or not os.path.exists(os.path.join(d, "test_metrics.json")):
                continue
            tp = pd.read_csv(os.path.join(d, "test_predictions.csv"))
            assert (tp.filename.values == f0.filename.values).all()
            rows.append(dict(system=s, run=os.path.basename(d), **source_nn_rate(np.load(e), y, g)))
    if rows:
        R = pd.DataFrame(rows)
        systems9 = [s for s in systems9 if s in set(R.system)]
        agg = R.groupby("system", sort=False)[["rate", "rate_eligible"]].agg(["mean", "std", "count"]).reindex(systems9)
        agg.columns = ["_".join(c) for c in agg.columns]
        fig, ax = plt.subplots(figsize=(9, 4))
        ax.bar(range(len(systems9)), agg["rate_mean"], yerr=agg["rate_std"].fillna(0), color=[cols9[s] for s in systems9], capsize=3)
        ax.axhline(R.chance.iloc[0], color="k", ls="--", lw=1, label="chance (random same-class neighbour)")
        ax.set_xticks(range(len(systems9)), [x.replace("AST-FT ", "").replace(" + linear head", "") for x in systems9])
        agg = agg.rename(columns={"rate_mean": "mean", "rate_std": "std"})
        ax.set(ylabel="within-class nearest neighbour\nfrom the same source", ylim=(0, 1),
               title="Source clustering of test embeddings (lower = less source-specific)")
        ax.legend(fontsize=8)
        save(fig, a.out, "G9_source_clustering_of_embeddings", R)
        print("\nsource clustering (same-source within-class NN rate; chance %.3f, eligible-clip chance %.3f, "
              "%d of %d test clips eligible):\n" % (R.chance.iloc[0], R.chance_eligible.iloc[0], R.n_eligible.iloc[0], R.n.iloc[0]),
              agg.round(3).to_string())


if __name__ == "__main__":
    main()
