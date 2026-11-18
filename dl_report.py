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
from dl_metrics import confusion, grouped_bootstrap, metrics  # noqa: E402

K = len(CLASSES)
SYSTEMS = [  # (label, run-folder glob relative to runs, role)
    ("Spectral + LR", "ml_spec_{p}", "traditional ML"),
    ("Structure + LR", "ml_struct-nf_{p}", "traditional ML"),
    ("CNN (scratch)", "cnn_{p}_standard_s*", "deep learning"),
    ("CNN (scratch), source-balanced", "cnn_{p}_source_s*", "deep learning, proposed sampling"),
    ("Frozen AST (10 s) + LR", "ml_ast_{p}", "pretrained representation"),
    ("Frozen AST (4 s) + linear head", "ast_frozen_{p}_standard_s*", "pretrained representation"),
    ("AST, last block fine-tuned", "ast_ft_{p}_standard_s*", "transfer learning"),
    ("AST, last block fine-tuned, source-balanced", "ast_ft_{p}_source_s*", "transfer learning, proposed sampling"),
]


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
    pairs = [("CNN (scratch)", "CNN (scratch), source-balanced", "source-balanced - standard (CNN)"),
             ("AST, last block fine-tuned", "AST, last block fine-tuned, source-balanced", "source-balanced - standard (AST FT)"),
             ("Frozen AST (4 s) + linear head", "AST, last block fine-tuned", "fine-tuned - frozen (AST, same 4 s input)"),
             ("CNN (scratch)", "AST, last block fine-tuned", "AST FT - CNN"),
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
        for fam, title in [("cnn_source_disjoint", "CNN (scratch), source-disjoint protocol"),
                           ("ast_ft_source_disjoint", "AST last-block fine-tuning, source-disjoint protocol")]:
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
    cms = [(s, preds_cache[(s, "source-disjoint test")]) for s in ("CNN (scratch)", "Frozen AST (10 s) + LR", "AST, last block fine-tuned")
           if (s, "source-disjoint test") in preds_cache]
    if cms:
        fig, axs = plt.subplots(1, len(cms), figsize=(5.2 * len(cms), 4.6))
        axs = np.atleast_1d(axs)
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
    bars(["Frozen AST (4 s) + linear head", "AST, last block fine-tuned"], "G5_frozen_vs_finetuned_ast",
         "Frozen vs fine-tuned AST (same 4 s input, same head)")
    bars(["CNN (scratch)", "CNN (scratch), source-balanced", "AST, last block fine-tuned",
          "AST, last block fine-tuned, source-balanced"], "G6_standard_vs_source_balanced",
         "Standard vs source-balanced training (source-disjoint test)")
    bars(["Spectral + LR", "Structure + LR", "CNN (scratch)", "Frozen AST (10 s) + LR", "AST, last block fine-tuned"],
         "G7_original_vs_corrected_evaluation", "Official vs de-duplicated official vs source-disjoint test",
         evals=("official test", "official test, de-duplicated", "source-disjoint test"))

    # G8 source concentration (from source_audit.py)
    src = os.path.join(a.audit, "figures", "A2_source_concentration.png")
    if os.path.exists(src):
        for ext in ("png", "svg"):
            shutil.copy(os.path.join(a.audit, "figures", f"A2_source_concentration.{ext}"), os.path.join(a.out, f"G8_source_concentration.{ext}"))
        shutil.copy(os.path.join(a.audit, "source_concentration_class.csv"), os.path.join(a.out, "G8_source_concentration.csv"))
    print("wrote", a.out)


if __name__ == "__main__":
    main()
