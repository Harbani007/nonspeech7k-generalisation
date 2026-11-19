"""What are vocalisation classifiers listening to? Foreground / background ablation on the source-disjoint test set.

Hypothesis (H-bg): trained models classify a substantial share of clips ABOVE CHANCE even when the vocalisation is
removed (background only), including for recordings never seen in training. Counter-hypothesis: they rely on the
vocalisation, so background-only performance falls to chance.

For every selected clip (the same clip selection as the AST experiments: clips.csv of the fbank/cache dir):
  foreground mask = frames (25 ms, hop 10 ms) whose RMS level is within --fg-db dB of the clip's loudest frame,
                    dilated by --dilate-ms on each side, edges smoothed with a 10 ms raised-cosine ramp
  variants:
    full        the original 4 s clip
    background  vocalisation removed (foreground samples faded to zero)
    foreground  only the vocalisation kept (background faded to zero)
    random      CONTROL: the same mask circularly shifted to a random other position, so the same amount and
                shape of audio is removed but not where the vocalisation is
  Clips whose foreground covers < 5% or > 80% of the clip are excluded (reported).

Evaluated:
  * every fine-tuned AST checkpoint matching --run-globs (no retraining, test partition only);
  * a frozen-AST probe: logistic regression on pretrained AST embeddings, trained on the TRAIN partition of a
    variant and tested on the TEST partition of the same variant (is class information present in the background
    at all?), plus trained on full / tested on background.
Statistics: macro-F1, 95% CI by bootstrap over test source groups (seeds averaged inside each resample), paired
differences between variants, and a chance level from 1000 permutations of the predictions.

  python background_probe.py --data <dir with splits.csv> --clips <fbank or cache dir>/clips.csv --audio-root <audio> \
      --runs <runs dir> --out <out dir> [--device cuda] [--limit 50]
"""
import argparse
import glob
import json
import os
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.signal import resample_poly

from ast_finetune import ASTTop, embed, load_ast
from common import index_audio
from dl_data import CLASSES
from dl_metrics import grouped_bootstrap, metrics
from features import load_mono

K = len(CLASSES)
SR = 16000
VARIANTS = ["full", "background", "foreground", "random"]


def masks_for(x, fg_db, dilate_ms, rng, edge_hold=False):
    """Return (fg_gain, random_gain, fg_fraction) for a 16 kHz waveform.
    edge_hold=False (used for all results up to 19 Nov 2026): the 10 ms edge ramp pads the clip ends with zeros, so a
    vocalisation that touches the start/end of the clip gets a gain of only about 0.5 in its first/last ~10 ms.
    edge_hold=True pads with the mask's edge value instead, so the gain stays 1 there (used by swap_probe.py)."""
    win, hop = 400, 160
    n = len(x)
    nfr = max(1, 1 + (n - win) // hop)
    rms = np.array([np.sqrt(np.mean(x[i * hop:i * hop + win] ** 2) + 1e-12) for i in range(nfr)])
    db = 20 * np.log10(rms + 1e-12)
    fr = db >= db.max() - fg_db
    d = int(round(dilate_ms / 10))
    if d > 0:
        fr = np.convolve(fr.astype(float), np.ones(2 * d + 1), mode="same") > 0
    m = np.zeros(n)
    for i in np.where(fr)[0]:
        m[i * hop:min(n, i * hop + win)] = 1.0
    ramp = np.hanning(2 * 160 + 1)
    ramp /= ramp.sum()
    if edge_hold:
        end = (nfr - 1) * hop + win  # samples after the last full frame belong to no frame: give them its state
        if end < n:
            m[end:] = float(fr[-1])
        h = len(ramp) // 2
        g = np.clip(np.convolve(np.pad(m, h, mode="edge"), ramp, mode="valid"), 0, 1)
    else:
        g = np.clip(np.convolve(m, ramp, mode="same"), 0, 1)
    shift = int(rng.integers(n // 4, 3 * n // 4 + 1))
    return g, np.roll(g, shift), float(m.mean())


def make_variants(x, g, gr):
    return {"full": x, "background": x * (1 - g), "foreground": x * g, "random": x * (1 - gr)}


@torch.no_grad()
def run_model(ast, top, k, fbanks, dev, bs=32):
    probs, embs = [], []
    for i in range(0, len(fbanks), bs):
        xb = torch.from_numpy(fbanks[i:i + bs].astype(np.float32)).to(dev)
        with torch.autocast(dev.type, dtype=torch.float16, enabled=dev.type == "cuda"):
            h = embed(ast, xb)
            for layer in ast.layers[:k]:
                h = layer(h, None)
            e = top.pooled(h).float()
        embs.append(e.cpu())
        probs.append(F.softmax(top.head(e.to(dev)), 1).cpu() if hasattr(top, "head") else None)
    return torch.cat(probs).numpy(), torch.cat(embs).numpy()


def chance_f1(y, preds, n=1000, seed=0):
    rng = np.random.default_rng(seed)
    v = [np.mean([metrics(y, rng.permutation(p), K)["macro_f1"] for p in preds]) for _ in range(n)]
    return float(np.mean(v)), float(np.percentile(v, 95))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="dir with splits.csv")
    ap.add_argument("--clips", required=True, help="clips.csv of the fbank/cache dir (same clip selection)")
    ap.add_argument("--audio-root", required=True)
    ap.add_argument("--runs", required=True)
    ap.add_argument("--run-globs", nargs="+", default=["ast_ft_source_disjoint_standard_s*", "ast_ftsc_source_disjoint_source_s*",
                                                       "ast_ft_source_disjoint_source_s*", "ast_ftsc_source_disjoint_standard_s*",
                                                       "ast_ftadv_source_disjoint_standard_s*"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--fg-db", type=float, default=20.0)
    ap.add_argument("--dilate-ms", type=float, default=50.0)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--limit", type=int, default=0, help="debug: only the first N clips of each partition")
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--examples", type=int, default=2, help="example wavs per class saved for the demo")
    ap.add_argument("--compare", nargs=2, metavar=("SYSTEM_A", "SYSTEM_B"),
                    help="also report paired differences B - A between two systems on every variant")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    dev = torch.device("cuda" if (a.device == "auto" and torch.cuda.is_available()) or a.device == "cuda" else "cpu")
    t0 = time.time()

    sp = pd.read_csv(os.path.join(a.data, "splits.csv"))
    clips = pd.read_csv(a.clips)
    info = sp.iloc[clips.row.values].reset_index(drop=True)
    info["part"] = clips.part.values
    if a.limit:  # debug: first N clips per (partition, class)
        info = info.groupby(["part", "y"], group_keys=False).head(a.limit).reset_index(drop=True)
    idx = index_audio(a.audio_root)
    fe, _ = load_ast()

    # ---- waveforms, masks, variant filterbanks -------------------------------------------------------------
    rng = np.random.default_rng(0)
    waves, fgfrac = {v: [] for v in VARIANTS}, []
    for f in info.filename.values:
        x = resample_poly(load_mono(idx[f])[0], 1, 2).astype(np.float32)
        g, gr, frac = masks_for(x, a.fg_db, a.dilate_ms, rng)
        fgfrac.append(frac)
        for v, w in make_variants(x, g, gr).items():
            waves[v].append(w.astype(np.float32))
    info["fg_fraction"] = fgfrac
    info["eligible"] = (info.fg_fraction >= 0.05) & (info.fg_fraction <= 0.80)
    summ = info.groupby(["part", "cls"]).agg(clips=("filename", "size"), eligible=("eligible", "sum"),
                                             median_fg_fraction=("fg_fraction", "median")).reset_index()
    summ.to_csv(os.path.join(a.out, "mask_summary.csv"), index=False)
    print(summ[summ.part == "test"].to_string(index=False), flush=True)
    fb = {}
    for v in VARIANTS:
        out = []
        for i in range(0, len(info), 64):
            out.append(fe(waves[v][i:i + 64], sampling_rate=SR, return_tensors="np")["input_values"].astype(np.float16))
        fb[v] = np.concatenate(out)
    print(f"filterbanks for {len(info)} clips x {len(VARIANTS)} variants ({time.time() - t0:.0f}s)", flush=True)

    te = np.where((info.part == "test") & info.eligible)[0]
    tr = np.where((info.part == "train") & info.eligible)[0]
    yte, gte = info.y.values[te], info.group.values[te]

    # ---- fine-tuned checkpoints --------------------------------------------------------------------------
    preds, rows_pred = {}, []
    for pat in a.run_globs:
        for d in sorted(glob.glob(os.path.join(a.runs, pat))):
            ck_path = os.path.join(d, "best.pt")
            if not os.path.exists(os.path.join(d, "test_metrics.json")) or not os.path.exists(ck_path):
                continue
            ck = torch.load(ck_path, weights_only=False, map_location="cpu")
            _, ast = load_ast()  # fresh pretrained weights for every checkpoint (ASTTop shares modules with ast)
            k = ck["freeze_k"]
            top = ASTTop(ast, k, K)
            top.load_state_dict(ck["state_dict"])
            ast.to(dev).eval(); top.to(dev).eval()
            system = pat.split("_s*")[0]
            for v in VARIANTS:
                p, _ = run_model(ast, top, k, fb[v][te], dev)
                preds.setdefault((system, v), []).append(p.argmax(1))
                rows_pred.append(pd.DataFrame({"run": os.path.basename(d), "system": system, "variant": v,
                                               "filename": info.filename.values[te], "y": yte, "group": gte,
                                               "pred": p.argmax(1), "p_max": p.max(1)}))
            print(f"{os.path.basename(d)} (freeze_k {k}) scored on {len(te)} test clips x 4 variants ({time.time() - t0:.0f}s)", flush=True)
            del ast, top
            if dev.type == "cuda":
                torch.cuda.empty_cache()

    # ---- frozen-AST probe: is class information in the background at all? ----------------------------------
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    _, ast = load_ast()
    top = ASTTop(ast, 12, K)  # no transformer layers in the top: pooled output of the pretrained encoder
    ast.to(dev).eval(); top.to(dev).eval()
    E = {v: run_model(ast, top, 12, fb[v], dev)[1] for v in VARIANTS}
    ytr = info.y.values[tr]
    for train_v, test_v in [(v, v) for v in VARIANTS] + [("full", "background"), ("full", "foreground")]:
        clf = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=3000, class_weight="balanced"))
        clf.fit(E[train_v][tr], ytr)
        system = f"frozen-AST probe (train {train_v})"
        preds.setdefault((system, test_v), []).append(clf.predict(E[test_v][te]))
    print(f"frozen probe done ({time.time() - t0:.0f}s)", flush=True)

    # ---- statistics ----------------------------------------------------------------------------------------
    res = []
    for (system, v), P in preds.items():
        pt, _, _, ci, _, _ = grouped_bootstrap(yte, P, None, gte, K, n=a.boot)
        ch, ch95 = chance_f1(yte, P, n=200)
        ms = [metrics(yte, q, K) for q in P]
        res.append(dict(system=system, variant=v, seeds=len(P), test_clips=len(te), test_sources=len(np.unique(gte)),
                        macro_f1=pt, ci_lo=ci[0], ci_hi=ci[1], chance_mean=ch, chance_95=ch95,
                        above_chance=bool(ci[0] > ch95),
                        **{f"recall_{c}": float(np.mean([m["per_class_recall"][i] for m in ms])) for i, c in enumerate(CLASSES)}))
    R = pd.DataFrame(res)
    R.to_csv(os.path.join(a.out, "background_probe_results.csv"), index=False)
    pd.set_option("display.width", 250)
    print(R[["system", "variant", "seeds", "macro_f1", "ci_lo", "ci_hi", "chance_95", "above_chance"]].round(3).to_string(index=False))

    diffs = []
    for system in sorted({s for s, _ in preds}):
        for va, vb in [("full", "background"), ("background", "random"), ("full", "foreground"), ("foreground", "background")]:
            if (system, va) in preds and (system, vb) in preds:
                A, B = preds[(system, va)], preds[(system, vb)]
                if len(A) != len(B):
                    continue
                sa, sb, dd, _, _, cd = grouped_bootstrap(yte, A, B, gte, K, n=a.boot)
                diffs.append(dict(system=system, a=va, b=vb, macro_f1_a=sa, macro_f1_b=sb, b_minus_a=dd,
                                  ci_lo=cd[0], ci_hi=cd[1], ci_excludes_zero=bool(cd[0] > 0 or cd[1] < 0)))
    if a.compare:
        sa_, sb_ = a.compare
        for v in VARIANTS:
            if (sa_, v) in preds and (sb_, v) in preds:
                A, Bp = preds[(sa_, v)], preds[(sb_, v)]
                ma, mb, dd, _, _, cd = grouped_bootstrap(yte, A, Bp, gte, K, n=a.boot)
                diffs.append(dict(system=f"{sb_} - {sa_}", a=v, b=v, macro_f1_a=ma, macro_f1_b=mb, b_minus_a=dd,
                                  ci_lo=cd[0], ci_hi=cd[1], ci_excludes_zero=bool(cd[0] > 0 or cd[1] < 0)))
    D = pd.DataFrame(diffs)
    D.to_csv(os.path.join(a.out, "background_probe_paired_differences.csv"), index=False)
    print("\n", D.round(3).to_string(index=False))
    if rows_pred:
        pd.concat(rows_pred).to_csv(os.path.join(a.out, "background_probe_predictions.csv.gz"), index=False)

    # ---- figure ---------------------------------------------------------------------------------------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    systems = list(dict.fromkeys(R.system))
    fig, ax = plt.subplots(figsize=(max(8, 2.2 * len(systems)), 4.5))
    w = 0.2
    for j, v in enumerate(VARIANTS):
        t = R[R.variant == v].set_index("system").reindex(systems)
        x = np.arange(len(systems)) + (j - 1.5) * w
        ax.bar(x, t.macro_f1, w, label=v)
        ax.errorbar(x, t.macro_f1, yerr=[t.macro_f1 - t.ci_lo, t.ci_hi - t.macro_f1], fmt="none", color="k", capsize=2)
    ax.axhline(R.chance_95.max(), color="k", ls="--", lw=1, label="chance (95th pct of permutations)")
    ax.set_xticks(range(len(systems)), [s.replace("_source_disjoint", "").replace("frozen-AST probe ", "probe\n") for s in systems],
                  fontsize=7, rotation=15)
    ax.set(ylim=(0, 1), ylabel="macro-F1 (95% CI, source bootstrap)",
           title="Source-disjoint test: full clip vs vocalisation removed (background) vs vocalisation only vs random removal")
    ax.legend(fontsize=7, ncol=5)
    for ext in ("png", "svg"):
        fig.savefig(os.path.join(a.out, f"G10_background_probe.{ext}"), dpi=160, bbox_inches="tight")

    # ---- demo examples: background-only clips the first fine-tuned model still classifies correctly --------
    if rows_pred and a.examples:
        import soundfile as sf
        P0 = pd.concat(rows_pred)
        first = P0.run.iloc[0]
        bg = P0[(P0.run == first) & (P0.variant == "background") & (P0.pred == P0.y)].sort_values("p_max", ascending=False)
        ex_dir = os.path.join(a.out, "examples")
        os.makedirs(ex_dir, exist_ok=True)
        pos_of = {f: i for i, f in enumerate(info.filename.values)}
        kept = []
        for c in range(K):
            for f in bg[bg.y == c].filename.head(a.examples):
                i = pos_of[f]
                stem = f"{CLASSES[c]}__{os.path.splitext(f)[0]}"
                sf.write(os.path.join(ex_dir, stem + "__full.wav"), waves["full"][i], SR)
                sf.write(os.path.join(ex_dir, stem + "__background_only.wav"), waves["background"][i], SR)
                kept.append(dict(file=f, cls=CLASSES[c], model=first, fg_fraction=info.fg_fraction.values[i]))
        pd.DataFrame(kept).to_csv(os.path.join(ex_dir, "examples.csv"), index=False)
    json.dump(vars(a), open(os.path.join(a.out, "args.json"), "w"), indent=1)
    print(f"done in {time.time() - t0:.0f}s -> {a.out}")


if __name__ == "__main__":
    main()
