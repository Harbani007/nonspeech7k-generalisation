"""Test-time background swap: does a trained classifier change its decision when ONLY the background behind an
unchanged vocalisation is replaced? Pre-specified in docs/PREREGISTRATION_SWAP_PROBE.md (read that first).

No training. Existing AST checkpoints only; test partition only (recordings never seen in training).

For every eligible test clip x with vocalisation mask g (20 dB rule of background_probe.masks_for):
    swapped = x * g + s * d * (1 - g_d)
x * g is identical in every condition; d * (1 - g_d) is the background of a donor test clip from ANOTHER recording,
with the donor's own vocalisation removed by its 20 dB or 30 dB mask; s matches the donor background level to the
target's own background level. Donors: one same-class (control) and six different-class (one per other class),
the different-class donors matched to the same-class donor on background fraction, level, spectral centroid, duration.

Outputs (one directory; refuses to overwrite a finished one):
  config.json, reproduce.txt         arguments, git commit, exact command
  donors.csv, donor_balance.csv      the pairing and its balance check (standardised mean differences)
  runs/<run>/predictions.csv.gz      every prediction of every checkpoint (one folder per checkpoint)
  reproduction_check.csv             'orig' predictions vs the run's stored test_predictions.csv
  condition_scores.csv               macro-recall / accuracy per system x condition, 95% CI (recording bootstrap)
  swap_effects.csv                   same - different (PRIMARY), orig - same, attraction, flip rates, with CIs
  attraction_by_class.csv            donor attraction per donor class
  residue_audit.csv                  donor backgrounds alone; loudest remaining frame relative to the donor's peak
  decision.json                      the pre-registered decision for the primary system
  G11_swap_probe.png/svg, examples/  figure; listening set (wav) + waveform/spectrogram sheets

  python swap_probe.py --data <dir with splits.csv> --clips <cache dir>/clips.csv --audio-root <audio> \
      --runs <runs dir> --out <out dir> [--device cuda] [--limit 3] [--check-only]
"""
import argparse
import glob
import json
import os
import re
import subprocess
import sys
import time
from math import gcd

import numpy as np
import pandas as pd

from dl_data import CLASSES

K = len(CLASSES)
SR = 16000
MAX_S = 4.0
PRIMARY = "ast_ft_source_disjoint_standard"
DEFAULT_GLOBS = ["ast_ft_source_disjoint_standard_s*", "ast_ft_source_disjoint_source_s*",
                 "ast_ftsc_source_disjoint_standard_s*", "ast_ftsc_source_disjoint_source_s*"]
ARMS = [("m20", 20, True), ("m30", 30, True), ("raw20", 20, False)]  # (name, donor mask dB, level matched)
SCALE_LIMITS = (0.01, 100.0)
MIN_BG_DBFS = -70.0
FEATURES = ["bg_fraction", "bg_log_rms", "bg_centroid", "duration"]


# ---- pure signal functions (unit-tested in tests/test_swap_probe.py) ----------------------------------------
def fit_length(d, n):
    """Crop or zero-pad d to n samples (zeros = no background where the donor is shorter)."""
    d = np.asarray(d, np.float32)
    return d[:n] if len(d) >= n else np.pad(d, (0, n - len(d)))


def bg_level(x, g):
    """RMS of x over its background region (samples with 1 - g > 0.5); nan if there is no background."""
    m = (1.0 - np.asarray(g)) > 0.5
    return float(np.sqrt(np.mean(np.asarray(x, np.float64)[m] ** 2))) if m.any() else float("nan")


def match_scale(x, g, d, gd, limits=SCALE_LIMITS):
    """Gain s so that the donor background has the target background's RMS. Returns (s, was_clipped)."""
    t, b = bg_level(x, g), bg_level(d, gd)
    if not np.isfinite(t) or not np.isfinite(b) or b <= 0:
        return 1.0, True
    s = t / b
    c = float(np.clip(s, *limits))
    return c, c != s


def compose(x, g, d, gd, scale):
    """Target vocalisation x * g plus the donor background scale * d * (1 - gd); donor fitted to len(x)."""
    n = len(x)
    d, gd = fit_length(d, n), fit_length(gd, n)
    return (np.asarray(x, np.float32) * np.asarray(g, np.float32) + np.float32(scale) * d * (1 - gd)).astype(np.float32)


def frame_db(x, win=400, hop=160):
    n = len(x)
    nfr = max(1, 1 + (n - win) // hop)
    x = np.asarray(x, np.float64)
    return np.array([10 * np.log10(np.mean(x[i * hop:i * hop + win] ** 2) + 1e-20) for i in range(nfr)])


def bg_features(d, gd):
    """Donor-background descriptors used for matching."""
    b = np.asarray(d, np.float64) * (1 - np.asarray(gd))
    spec = np.abs(np.fft.rfft(b))
    f = np.fft.rfftfreq(len(b), 1 / SR)
    lev = bg_level(d, gd)
    return dict(bg_fraction=float(np.mean((1 - np.asarray(gd)) > 0.5)),
                bg_log_rms=float(20 * np.log10(lev + 1e-12)) if np.isfinite(lev) else float("nan"),
                bg_centroid=float((f * spec).sum() / (spec.sum() + 1e-12)), duration=len(d) / SR)


# ---- pure pairing and statistics ------------------------------------------------------------------------------
def pair_donors(targets, pool, k=K, seed=0, features=FEATURES):
    """targets: DataFrame(idx, y, group, n); pool: DataFrame(idx, y, group, n, *features).
    For each target: a random same-class donor from another group with n >= target n (else the longest), then for
    every other class the nearest pool clip (standardised features) to that same-class donor, with the same group
    and length restrictions. Returns DataFrame(target, same, diff_0..diff_{k-1}); diff_y = -1."""
    rng = np.random.default_rng(seed)
    F = pool[features].to_numpy(float)
    F = (F - F.mean(0)) / (F.std(0) + 1e-12)
    py, pg, pn, pidx = pool.y.to_numpy(), pool.group.to_numpy(), pool.n.to_numpy(), pool.idx.to_numpy()
    rows = []
    for t in targets.itertuples():
        def cand(c):
            ok = (py == c) & (pg != t.group)
            long_ = ok & (pn >= t.n)
            if long_.any():
                return np.where(long_)[0]
            ix = np.where(ok)[0]
            return ix[pn[ix] == pn[ix].max()] if len(ix) else ix
        cs = cand(t.y)
        if len(cs) == 0:
            continue
        s = int(rng.choice(cs))
        r = dict(target=int(t.idx), same=int(pidx[s]))
        for c in range(k):
            if c == t.y:
                r[f"diff_{c}"] = -1
                continue
            cc = cand(c)
            r[f"diff_{c}"] = int(pidx[cc[np.argmin(((F[cc] - F[s]) ** 2).sum(1))]]) if len(cc) else -1
        rows.append(r)
    return pd.DataFrame(rows)


def smd(a, b):
    """Standardised mean difference (pooled SD)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    sd = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2) if len(a) > 1 and len(b) > 1 else np.nan
    return float((a.mean() - b.mean()) / sd) if sd and sd > 0 else 0.0


def macro_recall(y, correct, k=K):
    """Mean over classes present of the mean (possibly fractional) correctness."""
    y, correct = np.asarray(y), np.asarray(correct, float)
    r = [correct[y == c].mean() for c in range(k) if (y == c).any()]
    return float(np.mean(r))


def attraction(y, pdiff, k=K):
    """pdiff: (n, k) predictions; column c = prediction when the donor is of class c (column y_i ignored).
    Returns per-target arrays (excess, donor_rate, null_rate):
      donor_rate_i = mean_c 1[pdiff[i, c] == c]
      null_rate_i  = mean_c mean_{c' not in (y_i, c)} 1[pdiff[i, c'] == c]
    excess = donor_rate - null_rate (0 when the donor's class does not pull the prediction)."""
    y, pdiff = np.asarray(y), np.asarray(pdiff)
    n = len(y)
    ex, dr, nr = np.zeros(n), np.zeros(n), np.zeros(n)
    for i in range(n):
        wrong = [c for c in range(k) if c != y[i]]
        d = [float(pdiff[i, c] == c) for c in wrong]
        nl = [np.mean([float(pdiff[i, c2] == c) for c2 in wrong if c2 != c]) for c in wrong]
        dr[i], nr[i] = np.mean(d), np.mean(nl)
        ex[i] = dr[i] - nr[i]
    return ex, dr, nr


def group_bootstrap(stat, groups, n=2000, seed=12345):
    """stat(ix) -> 1-D array of statistics on target indices ix. Resamples groups with replacement.
    Returns (point, lo, hi) arrays (95% percentile interval)."""
    groups = np.asarray(groups)
    uniq = np.unique(groups)
    pos = [np.where(groups == g)[0] for g in uniq]
    rng = np.random.default_rng(seed)
    point = np.asarray(stat(np.arange(len(groups))), float)
    B = np.array([stat(np.concatenate([pos[j] for j in rng.integers(0, len(uniq), len(uniq))])) for _ in range(n)], float)
    return point, np.percentile(B, 2.5, axis=0), np.percentile(B, 97.5, axis=0)


def decide(d20, lo20, hi20, d30, lo30, hi30, big=0.05, small=0.02):
    """Pre-registered decision (docs/PREREGISTRATION_SWAP_PROBE.md)."""
    if d20 >= big and lo20 > 0 and d30 >= big and lo30 > 0:
        return "GO"
    if hi20 < small:
        return "NEGLIGIBLE"
    if lo20 > 0 or hi20 < 0:
        return "DETECTED BUT SMALL" if hi20 >= 0 else "REVERSED (different-class donors help)"
    return "INCONCLUSIVE"


# ---- models ---------------------------------------------------------------------------------------------------
class Bank:
    """One pretrained AST supplies the (frozen, pretrained) patch embeddings and lower layers; each checkpoint
    supplies its own fine-tuned top (layers k..11 + LayerNorm + head) on a separate copy of the weights."""

    def __init__(self, run_dirs, dev):
        import torch
        from ast_finetune import ASTTop, load_ast
        self.torch, self.dev = torch, dev
        self.fe, self.base = load_ast()
        self.base.to(dev).eval()
        self.tops = []
        for d in run_dirs:
            ck = torch.load(os.path.join(d, "best.pt"), weights_only=False, map_location="cpu")
            _, a2 = load_ast()
            top = ASTTop(a2, ck["freeze_k"], K)
            top.load_state_dict(ck["state_dict"])
            self.tops.append((os.path.basename(d), ck["freeze_k"], top.to(dev).eval()))
            del a2
        self.ks = sorted({k for _, k, _ in self.tops})

    def fbank(self, waves):
        return self.fe(list(waves), sampling_rate=SR, return_tensors="np")["input_values"].astype(np.float32)

    def probs(self, waves):
        """dict run -> (n, K) probabilities."""
        torch = self.torch
        from ast_finetune import embed
        x = torch.from_numpy(self.fbank(waves)).to(self.dev)
        out = {}
        with torch.no_grad(), torch.autocast(self.dev.type, dtype=torch.float16, enabled=self.dev.type == "cuda"):
            h, done = embed(self.base, x), 0
            for k in self.ks:
                for layer in self.base.layers[done:k]:
                    h = layer(h, None)
                done = k
                for name, kk, top in self.tops:
                    if kk == k:
                        out[name] = torch.softmax(top.head(top.pooled(h).float()), 1).float().cpu().numpy()
        return out


def find_runs(runs, globs, expect_k):
    dirs = []
    for pat in globs:
        for d in sorted(glob.glob(os.path.join(runs, pat))):
            if os.path.exists(os.path.join(d, "test_metrics.json")) and os.path.exists(os.path.join(d, "best.pt")):
                dirs.append(d)
    names = [os.path.basename(d) for d in dirs]
    assert len(set(names)) == len(names), f"duplicate run names: {names}"
    if expect_k is not None:
        bad = [(n, json.load(open(os.path.join(d, "test_metrics.json"))).get("freeze_k")) for n, d in zip(names, dirs)]
        bad = [b for b in bad if b[1] != expect_k]
        assert not bad, f"checkpoints with freeze_k != {expect_k}: {bad}"
    return dirs


def system_of(run):
    return re.sub(r"_s\d+$", "", run)


def git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=os.path.dirname(os.path.abspath(__file__)),
                                       text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "unknown"


# ---- main -----------------------------------------------------------------------------------------------------
def load_test(a):
    from scipy.signal import resample_poly
    from background_probe import masks_for
    from common import index_audio
    from features import load_mono
    sp = pd.read_csv(os.path.join(a.data, "splits.csv"))
    clips = pd.read_csv(a.clips)
    info = sp.iloc[clips.row.values[clips.part.values == "test"]].reset_index(drop=True)
    idx = index_audio(a.audio_root)
    rng = np.random.default_rng(0)  # random-shift masks of masks_for are not used here
    W, G20, G30, rows = [], [], [], []
    for f in info.filename.values:
        x, sr = load_mono(idx[f])
        q = gcd(SR, sr)
        x = resample_poly(x, SR // q, sr // q).astype(np.float32)[:int(MAX_S * SR)]
        g20, _, fr20 = masks_for(x, 20.0, a.dilate_ms, rng, edge_hold=True)
        g30, _, fr30 = masks_for(x, 30.0, a.dilate_ms, rng, edge_hold=True)
        W.append(x); G20.append(g20.astype(np.float32)); G30.append(g30.astype(np.float32))
        tl = bg_level(x, g20)
        rows.append(dict(n=len(x), fg20=fr20, fg30=fr30, target_bg_dbfs=20 * np.log10(tl + 1e-12) if np.isfinite(tl) else np.nan,
                         **bg_features(x, g30)))
    info = pd.concat([info, pd.DataFrame(rows)], axis=1)
    info["idx"] = np.arange(len(info))
    return info, W, G20, G30


def build_inputs(t, info, P, W, G20, G30):
    """All condition waveforms for one target row t of the pairing P. Returns list of (condition, donor idx, wave, scale, clipped)."""
    i = int(t.target)
    x, g = W[i], G20[i]
    out = [("orig", -1, x, 1.0, False), ("fg", -1, (x * g).astype(np.float32), 1.0, False)]
    for arm, db, matched in ARMS:
        Gd = G20 if db == 20 else G30
        for c in ["same"] + [f"diff_{c}" for c in range(K)]:
            j = int(getattr(t, c))
            if j < 0:
                continue
            d, gd = fit_length(W[j], len(x)), fit_length(Gd[j], len(x))
            s, clipped = match_scale(x, g, d, gd) if matched else (1.0, False)
            out.append((f"{c}_{arm}", j, compose(x, g, d, gd, s), s, clipped))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--clips", required=True, help="clips.csv of the fbank/cache/waves dir (same clip selection)")
    ap.add_argument("--audio-root", required=True)
    ap.add_argument("--runs", required=True)
    ap.add_argument("--run-globs", nargs="+", default=DEFAULT_GLOBS)
    ap.add_argument("--expect-freeze-k", type=int, default=None, help="refuse checkpoints with another freeze_k")
    ap.add_argument("--primary", default=PRIMARY)
    ap.add_argument("--out", required=True)
    ap.add_argument("--dilate-ms", type=float, default=50.0)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0, help="donor draw (fixed a priori: 0)")
    ap.add_argument("--limit", type=int, default=0, help="smoke test: first N targets per class")
    ap.add_argument("--check-only", action="store_true", help="load every checkpoint, check reproduction on --check-n clips, exit")
    ap.add_argument("--check-n", type=int, default=64)
    ap.add_argument("--examples", type=int, default=1, help="listening examples per class (+ hard cases)")
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args()
    import torch
    dev = torch.device("cuda" if (a.device == "auto" and torch.cuda.is_available()) or a.device == "cuda" else "cpu")
    if os.path.exists(os.path.join(a.out, "decision.json")) and not a.overwrite and not a.check_only:
        raise SystemExit(f"{a.out} already holds a finished swap probe; use a new --out (or --overwrite)")
    os.makedirs(a.out, exist_ok=True)
    t0 = time.time()
    if not a.check_only:
        json.dump(dict(args=vars(a), git_commit=git_commit(), arms=ARMS, scale_limits=SCALE_LIMITS, min_bg_dbfs=MIN_BG_DBFS,
                       match_features=FEATURES, preregistration="docs/PREREGISTRATION_SWAP_PROBE.md", device=str(dev),
                       torch=torch.__version__), open(os.path.join(a.out, "config.json"), "w"), indent=1, default=str)
        open(os.path.join(a.out, "reproduce.txt"), "w").write(
            f"git checkout {git_commit()}\npython {' '.join(sys.argv)}\n")
    run_dirs = find_runs(a.runs, a.run_globs, a.expect_freeze_k)
    assert run_dirs, f"no finished checkpoints under {a.runs} matching {a.run_globs}"
    print(f"{len(run_dirs)} checkpoints: " + ", ".join(os.path.basename(d) for d in run_dirs), flush=True)
    bank = Bank(run_dirs, dev)
    print(f"loaded on {dev}; freeze_k values {bank.ks} ({time.time() - t0:.0f}s)", flush=True)

    info, W, G20, G30 = load_test(a)
    print(f"test clips {len(info)} loaded with masks ({time.time() - t0:.0f}s)", flush=True)

    # ---- reproduction check: 'orig' must reproduce each run's stored test predictions -------------------------
    chk_ix = np.arange(len(info)) if not a.check_only else np.arange(min(a.check_n, len(info)))
    if a.limit:
        chk_ix = chk_ix[:a.check_n]
    pr = {}
    for s in range(0, len(chk_ix), a.batch):
        for name, p in bank.probs([W[i] for i in chk_ix[s:s + a.batch]]).items():
            pr.setdefault(name, []).append(p.argmax(1))
    rep = []
    for d in run_dirs:
        name = os.path.basename(d)
        stored = pd.read_csv(os.path.join(d, "test_predictions.csv")).set_index("filename")["pred"]
        mine = np.concatenate(pr[name])
        ref = stored.reindex(info.filename.values[chk_ix]).to_numpy()
        rep.append(dict(run=name, freeze_k=dict((n, k) for n, k, _ in bank.tops)[name], clips=len(chk_ix),
                        agreement=float(np.mean(mine == ref))))
    R = pd.DataFrame(rep)
    R.to_csv(os.path.join(a.out, "reproduction_check.csv"), index=False)
    print(R.to_string(index=False), flush=True)
    if a.check_only:
        print(f"check-only done ({time.time() - t0:.0f}s)")
        return

    # ---- targets, donor pool, pairing ----------------------------------------------------------------------
    elig_t = info[(info.fg20 >= 0.05) & (info.fg20 <= 0.80)]
    pool = info[(info.fg20 >= 0.05) & (info.fg20 <= 0.80) & (info.fg30 >= 0.05) & (info.fg30 <= 0.80)
                & (info.bg_log_rms > MIN_BG_DBFS)]
    targets = elig_t.groupby("y", group_keys=False).head(a.limit) if a.limit else elig_t
    P = pair_donors(targets, pool, seed=a.seed)
    dropped = len(targets) - len(P)
    print(f"targets {len(P)} (of {len(elig_t)} eligible; {dropped} without a same-class donor); donor pool {len(pool)} "
          f"clips; per class {pool.groupby('cls').size().to_dict()}", flush=True)
    Pd = P.copy()
    Pd.insert(1, "target_file", info.filename.values[P.target])
    Pd.insert(2, "target_cls", info.cls.values[P.target])
    Pd.to_csv(os.path.join(a.out, "donors.csv"), index=False)
    same_f = info.iloc[P.same][FEATURES]
    diff_ix = P[[f"diff_{c}" for c in range(K)]].to_numpy().ravel()
    diff_f = info.iloc[diff_ix[diff_ix >= 0]][FEATURES]
    bal = pd.DataFrame([dict(feature=f, same_mean=same_f[f].mean(), diff_mean=diff_f[f].mean(), smd=smd(same_f[f], diff_f[f]))
                        for f in FEATURES])
    reuse = pd.Series(np.concatenate([P.same.to_numpy(), diff_ix[diff_ix >= 0]])).value_counts()
    bal.to_csv(os.path.join(a.out, "donor_balance.csv"), index=False)
    print(bal.round(3).to_string(index=False), f"\nmost-reused donor used {reuse.max()} times; distinct donors {len(reuse)}", flush=True)

    # ---- run every condition through every checkpoint --------------------------------------------------------
    recs, buf = [], []

    def flush():
        if not buf:
            return
        pb = bank.probs([b[3] for b in buf])
        for name, p in pb.items():
            for (ti, cond, j, _, s, cl), q in zip(buf, p):
                recs.append((name, ti, cond, j, s, cl, int(q.argmax()), *np.round(q, 4)))
        buf.clear()

    n_checked = 0
    for r, t in enumerate(P.itertuples()):
        for cond, j, w, s, cl in build_inputs(t, info, P, W, G20, G30):
            if n_checked < 50 and j >= 0:  # runtime check: the target vocalisation is untouched
                i = int(t.target)
                gd = fit_length((G20 if cond.endswith("20") else G30)[j], len(w))
                resid = w - np.float32(s) * fit_length(W[j], len(w)) * (1 - gd) - W[i] * G20[i]
                assert np.abs(resid).max() < 1e-5, f"target vocalisation changed ({cond})"
                n_checked += 1
            buf.append((int(t.target), cond, j, w, s, cl))
            if len(buf) >= a.batch:
                flush()
        if r % 25 == 0:
            print(f"  target {r + 1}/{len(P)} ({time.time() - t0:.0f}s)", flush=True)
    # donor backgrounds alone (residue audit), at natural level
    for j in sorted(set(P.same) | set(diff_ix[diff_ix >= 0])):
        for db, Gd in ((20, G20), (30, G30)):
            buf.append((int(j), f"donorbg_m{db}", int(j), (W[j] * (1 - Gd[j])).astype(np.float32), 1.0, False))
            if len(buf) >= a.batch:
                flush()
    flush()
    cols = ["run", "target", "condition", "donor", "scale", "scale_clipped", "pred"] + [f"p_{c}" for c in CLASSES]
    D = pd.DataFrame(recs, columns=cols)
    for name in D.run.unique():
        od = os.path.join(a.out, "runs", name)
        os.makedirs(od, exist_ok=True)
        x = D[D.run == name].copy()
        x["target_file"] = info.filename.values[x.target]
        x["y"] = info.y.values[x.target]
        x["donor_cls"] = np.where(x.donor >= 0, info.cls.values[x.donor.clip(lower=0)], "")
        x.to_csv(os.path.join(od, "predictions.csv.gz"), index=False)
    print(f"all predictions done ({time.time() - t0:.0f}s)", flush=True)

    analyse(a, info, P, D, bank, W, G20, G30, t0)


def analyse(a, info, P, D, bank, W, G20, G30, t0):
    y = info.y.values[P.target]
    grp = info.group.values[P.target]
    tix = {t: r for r, t in enumerate(P.target)}
    systems = list(dict.fromkeys(system_of(n) for n, _, _ in bank.tops))
    rows_c, rows_e, rows_a = [], [], []
    for system in systems:
        runs = [n for n, _, _ in bank.tops if system_of(n) == system]
        S = D[D.run.isin(runs) & D.target.isin(tix)]
        # pred[run][cond] -> (n,) ; diff conds -> (n, K)
        pred = {}
        for (run, cond), g in S.groupby(["run", "condition"]):
            v = np.full(len(P), -1)
            v[[tix[t] for t in g.target]] = g.pred.to_numpy()
            pred.setdefault(run, {})[cond] = v

        def corr(cond):
            return np.mean([(pred[r][cond] == y).astype(float) for r in runs], 0)

        def diffmat(r, arm):
            return np.stack([pred[r].get(f"diff_{c}_{arm}", np.full(len(P), -1)) for c in range(K)], 1)

        def corr_diff(arm):
            out = []
            for r in runs:
                M = diffmat(r, arm)
                ok = np.arange(K)[None, :] != y[:, None]
                out.append(((M == y[:, None]) & ok).sum(1) / ok.sum(1))
            return np.mean(out, 0)

        C = {"orig": corr("orig"), "fg": corr("fg")}
        att = {}
        for arm, _, _ in ARMS:
            C[f"same_{arm}"] = corr(f"same_{arm}")
            C[f"diff_{arm}"] = corr_diff(arm)
            ex = [attraction(y, diffmat(r, arm)) for r in runs]
            att[arm] = tuple(np.mean([e[i] for e in ex], 0) for i in range(3))
            flips = []
            for r in runs:
                M, s_ = diffmat(r, arm), pred[r][f"same_{arm}"]
                ok = np.arange(K)[None, :] != y[:, None]
                flips.append((((M != s_[:, None]) & ok).sum(1) / ok.sum(1),
                              ((M == np.arange(K)[None, :]) & ok).sum(1) / ok.sum(1)))
            att[arm] = att[arm] + (np.mean([f[0] for f in flips], 0), np.mean([f[1] for f in flips], 0))
        names = list(C)
        pt, lo, hi = group_bootstrap(lambda ix: [macro_recall(y[ix], C[c][ix]) for c in names], grp, a.boot)
        for c, p_, l_, h_ in zip(names, pt, lo, hi):
            rows_c.append(dict(system=system, condition=c, seeds=len(runs), targets=len(P), recordings=len(np.unique(grp)),
                               macro_recall=p_, ci_lo=l_, ci_hi=h_, accuracy=float(C[c].mean())))
        effects = [("swap effect: same - different", f"same_{arm}", f"diff_{arm}", arm) for arm, _, _ in ARMS] + \
                  [("manipulation: orig - same", "orig", f"same_{arm}", arm) for arm, _, _ in ARMS] + \
                  [("fg - same", "fg", "same_m20", "m20")]
        st = lambda ix: [macro_recall(y[ix], C[u][ix]) - macro_recall(y[ix], C[v][ix]) for _, u, v, _ in effects] + \
            [att[arm][k][ix].mean() for arm, _, _ in ARMS for k in range(5)]  # noqa: E731
        pt, lo, hi = group_bootstrap(st, grp, a.boot)
        labels = [(e[0], e[3]) for e in effects] + [(m, arm) for arm, _, _ in ARMS for m in
                                                     ("attraction excess", "donor-class rate", "null rate", "flip rate", "flip-to-donor rate")]
        for (lab, arm), p_, l_, h_ in zip(labels, pt, lo, hi):
            rows_e.append(dict(system=system, effect=lab, arm=arm, estimate=p_, ci_lo=l_, ci_hi=h_,
                               ci_excludes_zero=bool(l_ > 0 or h_ < 0)))
        # attraction per donor class (point estimates, 20 dB matched arm)
        for c in range(K):
            m = y != c
            dr = np.mean([(diffmat(r, "m20")[m, c] == c).mean() for r in runs])
            nl = np.mean([np.mean([(diffmat(r, "m20")[m][:, c2][y[m] != c2] == c).mean()
                                   for c2 in range(K) if c2 != c]) for r in runs])
            rows_a.append(dict(system=system, donor_cls=CLASSES[c], donor_class_rate=dr, null_rate=nl, excess=dr - nl))
    Cs, Es, As = pd.DataFrame(rows_c), pd.DataFrame(rows_e), pd.DataFrame(rows_a)
    Cs.to_csv(os.path.join(a.out, "condition_scores.csv"), index=False)
    Es.to_csv(os.path.join(a.out, "swap_effects.csv"), index=False)
    As.to_csv(os.path.join(a.out, "attraction_by_class.csv"), index=False)
    pd.set_option("display.width", 250)
    print("\n", Cs.round(3).to_string(index=False))
    print("\n", Es.round(3).to_string(index=False))

    # ---- residue audit ----------------------------------------------------------------------------------------
    from dl_metrics import metrics
    first = [n for n, _, _ in bank.tops if system_of(n) == a.primary] or [bank.tops[0][0]]
    res = []
    for db, Gd in ((20, G20), (30, G30)):
        B = D[D.run.isin(first) & (D.condition == f"donorbg_m{db}")]
        donors = B.donor.unique()
        # loudest remaining frame of the donor background relative to the donor's loudest frame (mask check):
        # expected near or below -db, except where the dilation and edge ramp keep some vocalisation onset
        rel = np.array([frame_db(W[j] * (1 - Gd[j])).max() - frame_db(W[j]).max() for j in donors])
        ms = [metrics(info.y.values[B[B.run == r].donor], B[B.run == r].pred.to_numpy(), K) for r in first]
        res.append(dict(mask_db=db, model=system_of(first[0]), seeds=len(first), donor_clips=len(donors),
                        macro_f1_donor_class=float(np.mean([m["macro_f1"] for m in ms])),
                        balanced_accuracy=float(np.mean([m["balanced_accuracy"] for m in ms])),
                        loudest_remaining_frame_rel_peak_db_median=float(np.median(rel)),
                        loudest_remaining_frame_rel_peak_db_p95=float(np.percentile(rel, 95)),
                        loudest_remaining_frame_rel_peak_db_max=float(rel.max()),
                        share_donors_with_frame_above_minus_db=float(np.mean(rel > -db))))
    RA = pd.DataFrame(res)
    RA.to_csv(os.path.join(a.out, "residue_audit.csv"), index=False)
    print("\nresidue audit (donor background alone; model = primary system):\n", RA.round(3).T.to_string(), flush=True)
    figure_and_examples(a, info, P, D, W, G20, G30, Cs, first)

    # ---- decision ------------------------------------------------------------------------------------------
    sysname = a.primary if a.primary in systems else systems[0]
    e = Es[(Es.system == sysname) & (Es.effect == "swap effect: same - different")].set_index("arm")
    dec = decide(e.estimate["m20"], e.ci_lo["m20"], e.ci_hi["m20"], e.estimate["m30"], e.ci_lo["m30"], e.ci_hi["m30"])
    out = dict(primary_system=sysname, is_preregistered_primary=sysname == PRIMARY,
               freeze_k=sorted({k for n, k, _ in bank.tops if system_of(n) == sysname}), decision=dec,
               delta_m20=[e.estimate["m20"], e.ci_lo["m20"], e.ci_hi["m20"]],
               delta_m30=[e.estimate["m30"], e.ci_lo["m30"], e.ci_hi["m30"]], targets=len(P), limit=a.limit,
               note="smoke test (--limit): NOT a result" if a.limit else "full run")
    json.dump(out, open(os.path.join(a.out, "decision.json"), "w"), indent=1)
    print("\nDECISION", json.dumps(out), flush=True)


def spec_db_max(x):
    """Largest value (dB) of the matplotlib specgram of x, used as a shared colour scale."""
    from matplotlib import mlab
    p, _, _ = mlab.specgram(np.asarray(x, np.float64) + 1e-9, NFFT=512, Fs=SR, noverlap=352)
    return float(10 * np.log10(p.max() + 1e-20))


def figure_and_examples(a, info, P, D, W, G20, G30, Cs, first):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import soundfile as sf
    conds = ["orig", "fg"] + [f"{k}_{arm}" for arm, _, _ in ARMS for k in ("same", "diff")]
    systems = list(dict.fromkeys(Cs.system))
    fig, ax = plt.subplots(figsize=(max(8, 2.6 * len(systems)), 4.5))
    w = 0.8 / len(conds)
    for j, c in enumerate(conds):
        t = Cs[Cs.condition == c].set_index("system").reindex(systems)
        x = np.arange(len(systems)) + (j - (len(conds) - 1) / 2) * w
        ax.bar(x, t.macro_recall, w, label=c, color=plt.cm.tab20(j))
        ax.errorbar(x, t.macro_recall, yerr=[t.macro_recall - t.ci_lo, t.ci_hi - t.macro_recall], fmt="none",
                    color="k", capsize=1.5, lw=0.8)
    ax.set_xticks(range(len(systems)), [s.replace("_source_disjoint", "") for s in systems], fontsize=8)
    ax.set(ylim=(0, 1), ylabel="macro-recall of the vocalisation's class (95% CI)",
           title="Background swap behind an unchanged vocalisation (same-class donor = control)")
    ax.legend(fontsize=7, ncol=4)
    for ext in ("png", "svg"):
        fig.savefig(os.path.join(a.out, f"G11_swap_probe.{ext}"), dpi=160, bbox_inches="tight")
    plt.close(fig)

    # listening set: random targets per class + hard cases + the most suspicious donor backgrounds
    ex = os.path.join(a.out, "examples")
    os.makedirs(ex, exist_ok=True)
    rng = np.random.default_rng(0)
    Q = D[D.run == first[0]]
    y = info.y.values[P.target]
    pick = []
    for c in range(K):
        r = np.where(y == c)[0]
        pick += [(int(i), f"random_{CLASSES[c]}") for i in rng.choice(r, min(a.examples, len(r)), replace=False)]
    tinfo = info.iloc[P.target].reset_index(drop=True)
    pick.append((int(tinfo.fg20.idxmax()), "hard_largest_vocalisation"))
    pick.append((int(tinfo.target_bg_dbfs.idxmin()), "hard_quietest_background"))
    sus = Q[Q.condition == "donorbg_m30"].copy()
    sus["p_true"] = [getattr(r, f"p_{info.cls.values[r.donor]}") for r in sus.itertuples()]
    sus_donors = sus.sort_values("p_true", ascending=False).donor.head(3).tolist()
    rows = []
    for r, tag in pick:
        t = P.iloc[r]
        i = int(t.target)
        q = Q[Q.target == i].set_index("condition")
        # different-class donor shown: one whose 20 dB swap changed the prediction, else the first
        dc = [c for c in range(K) if c != y[r] and f"diff_{c}_m20" in q.index]
        ch = [c for c in dc if q.pred[f"diff_{c}_m20"] != q.pred["same_m20"]]
        c = (ch or dc)[0]
        show = [("orig", -1, "orig"), ("fg", -1, "fg"), ("same_m20", int(t.same), "same_m20"),
                (f"diff_{c}_m20", int(t[f"diff_{c}"]), "diff_m20"), ("same_m30", int(t.same), "same_m30"),
                (f"diff_{c}_m30", int(t[f"diff_{c}"]), "diff_m30")]
        built = {cond: wv for cond, j, wv, s, cl in build_inputs(t, info, P, W, G20, G30)}
        stem = f"{tag}__{os.path.splitext(info.filename.values[i])[0]}"
        vmax = spec_db_max(W[i])  # one colour scale per sheet (dB), so quiet backgrounds look quiet
        fig, axs = plt.subplots(len(show), 2, figsize=(12, 1.6 * len(show)), squeeze=False)
        for k_, (cond, j, lab) in enumerate(show):
            wv = built[cond]
            sf.write(os.path.join(ex, f"{stem}__{lab}.wav"), wv, SR)
            pred = CLASSES[int(q.pred[cond])]
            rows.append(dict(file=f"{stem}__{lab}.wav", tag=tag, target=info.filename.values[i], target_cls=CLASSES[y[r]],
                             condition=cond, donor=info.filename.values[j] if j >= 0 else "",
                             donor_cls=info.cls.values[j] if j >= 0 else "", model=first[0], pred=pred))
            tt = np.arange(len(wv)) / SR
            axs[k_, 0].plot(tt, wv, lw=0.4, color="k")
            axs[k_, 0].plot(tt, G20[i] * (np.abs(wv).max() + 1e-9), lw=0.8, color="tab:red", alpha=0.7)
            axs[k_, 0].set_ylabel(lab, fontsize=7)
            axs[k_, 0].set_title(f"pred: {pred}" + (f" | donor {info.cls.values[j]}" if j >= 0 else ""), fontsize=7, loc="left")
            axs[k_, 1].specgram(wv + 1e-9, NFFT=512, Fs=SR, noverlap=352, cmap="magma", vmin=vmax - 80, vmax=vmax)
            axs[k_, 1].set_ylim(0, 8000)
        fig.suptitle(f"{tag}: {info.filename.values[i]} ({CLASSES[y[r]]}); red = target vocalisation mask (20 dB); spectrogram colours: one 80 dB scale per sheet", fontsize=9)
        fig.tight_layout()
        fig.savefig(os.path.join(ex, f"{stem}.png"), dpi=110)
        plt.close(fig)
    for j in sus_donors:
        stem = f"suspicious_donor_bg__{os.path.splitext(info.filename.values[j])[0]}"
        for db, Gd in ((20, G20), (30, G30)):
            sf.write(os.path.join(ex, f"{stem}__bg_m{db}.wav"), (W[j] * (1 - Gd[j])).astype(np.float32), SR)
            pq = Q[(Q.condition == f"donorbg_m{db}") & (Q.donor == j)]
            rows.append(dict(file=f"{stem}__bg_m{db}.wav", tag="suspicious_donor_background", target="", target_cls="",
                             condition=f"donorbg_m{db}", donor=info.filename.values[j], donor_cls=info.cls.values[j],
                             model=first[0], pred=CLASSES[int(pq.pred.iloc[0])]))
        sf.write(os.path.join(ex, f"{stem}__full.wav"), W[j], SR)
        vmax = spec_db_max(W[j])
        fig, axs = plt.subplots(3, 1, figsize=(12, 5))
        for k_, (lab, wv) in enumerate([("full", W[j]), ("bg 20 dB", W[j] * (1 - G20[j])), ("bg 30 dB", W[j] * (1 - G30[j]))]):
            axs[k_].specgram(wv + 1e-9, NFFT=512, Fs=SR, noverlap=352, cmap="magma", vmin=vmax - 80, vmax=vmax)
            axs[k_].set_ylim(0, 8000)
            axs[k_].set_ylabel(lab, fontsize=7)
        fig.suptitle(f"donor background with the highest donor-class probability: {info.filename.values[j]} ({info.cls.values[j]})", fontsize=9)
        fig.tight_layout()
        fig.savefig(os.path.join(ex, f"{stem}.png"), dpi=110)
        plt.close(fig)
    pd.DataFrame(rows).to_csv(os.path.join(ex, "examples.csv"), index=False)


if __name__ == "__main__":
    main()
