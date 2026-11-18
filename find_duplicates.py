"""Find clips that are the same audio but belong to different source groups (re-uploads within or across websites).

1. Exact copies: hash of the decoded PCM samples of every clip (catches bit-identical audio regardless of score).
2. Candidates: for every clip, its --topk nearest neighbours from OTHER source groups in embedding space
   (default AST), kept if cosine >= --screen.
3. Confirmation on waveforms (all clips are 32 kHz mono, checked): peak normalised cross-correlation over all
   lags, so time shifts, different clip lengths and gain changes do not matter.
     confirmed : xcorr >= --xcorr
     uncertain : --uncertain <= xcorr < --xcorr (kept apart, never merged)
     rejected  : below
4. Control: the same xcorr on random different-source pairs of the same class, to estimate how often unrelated
   audio would pass the confirmation threshold.
Also writes a sensitivity table: confirmed pairs by (screening cosine, xcorr) threshold.

  python find_duplicates.py --meta-train ... --meta-test ... --audio-root <audio> --feats feats --out duplicates.csv
"""
import argparse
import hashlib

import numpy as np
import pandas as pd
from scipy.signal import fftconvolve

from common import index_audio, load_feats, load_meta
from features import load_mono


def peak_xcorr(a, b):
    """Max |normalised cross-correlation| over all lags; overlap-normalised by the shorter clip."""
    a = (a - a.mean()) / (a.std() + 1e-9)
    b = (b - b.mean()) / (b.std() + 1e-9)
    c = np.abs(fftconvolve(a, b[::-1], mode="full"))
    k = int(c.argmax())
    return float(c[k] / min(len(a), len(b))), k - (len(b) - 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--meta-train", required=True)
    ap.add_argument("--meta-test", required=True)
    ap.add_argument("--audio-root", required=True)
    ap.add_argument("--feats", default="feats")
    ap.add_argument("--set", default="ast")
    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--screen", type=float, default=0.95)
    ap.add_argument("--xcorr", type=float, default=0.9)
    ap.add_argument("--uncertain", type=float, default=0.6)
    ap.add_argument("--n-control", type=int, default=500)
    ap.add_argument("--out", default="duplicates.csv")
    a = ap.parse_args()

    meta = load_meta(a.meta_train, a.meta_test)
    idx = index_audio(a.audio_root)
    cache = {}

    def wav(f):
        if f not in cache:
            x, sr = load_mono(idx[f])
            assert sr == 32000, (f, sr)
            cache[f] = x
        return cache[f]

    # 1. exact PCM copies
    h = pd.Series({f: hashlib.sha1(wav(f).tobytes()).hexdigest() for f in meta.filename})
    meta["pcm_sha1"] = meta.filename.map(h)
    exact = meta[meta.duplicated("pcm_sha1", keep=False)].sort_values("pcm_sha1")
    print(f"bit-identical decoded audio: {len(exact)} clips in {exact.pcm_sha1.nunique()} sets")

    # 2. embedding candidates
    X, _ = load_feats(a.feats, meta, a.set)
    X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)
    g = meta["group"].to_numpy(dtype=str)
    cand = {}
    for i in range(0, len(meta), 1000):
        S = X[i:i + 1000] @ X.T
        S[g[i:i + 1000, None] == g[None, :]] = -1
        top = np.argsort(-S, axis=1)[:, :a.topk]
        for r, row in enumerate(top):
            for j in row:
                if S[r, j] >= a.screen:
                    key = tuple(sorted((i + r, int(j))))
                    cand[key] = max(cand.get(key, 0), float(S[r, j]))
    for f1, f2 in zip(exact.filename[:-1], exact.filename[1:]):  # exact copies are always candidates
        i, j = meta.index[meta.filename == f1][0], meta.index[meta.filename == f2][0]
        if g[i] != g[j]:
            cand.setdefault(tuple(sorted((i, j))), float(X[i] @ X[j]))
    print(f"{len(cand)} candidate pairs (top-{a.topk} neighbours, cosine >= {a.screen})", flush=True)

    rows = []
    for (i, j), cos in sorted(cand.items()):
        r1, r2 = meta.iloc[i], meta.iloc[j]
        xc, lag = peak_xcorr(wav(r1.filename), wav(r2.filename))
        rows.append(dict(a=r1.filename, b=r2.filename, source_a=r1.group, source_b=r2.group,
                         file_id_a=r1.file_id, file_id_b=r2.file_id, site_a=r1.source, site_b=r2.source,
                         split_a=r1.split, split_b=r2.split, cls_a=r1.cls, cls_b=r2.cls, cosine=cos, xcorr=xc,
                         lag_ms=lag / 32.0, dur_a_s=len(wav(r1.filename)) / 32000, dur_b_s=len(wav(r2.filename)) / 32000,
                         bit_identical=r1.pcm_sha1 == r2.pcm_sha1))
    df = pd.DataFrame(rows)
    df["status"] = np.where(df.xcorr >= a.xcorr, "confirmed", np.where(df.xcorr >= a.uncertain, "uncertain", "rejected"))
    df["same_audio"] = df.status == "confirmed"
    df["cross_website"] = df.site_a != df.site_b
    df["cross_official_split"] = df.split_a != df.split_b
    df["label_conflict"] = df.cls_a.str.replace("yawm", "yawn") != df.cls_b.str.replace("yawm", "yawn")
    df.sort_values(["status", "xcorr"], ascending=[True, False]).to_csv(a.out, index=False)

    # 4. control: random different-source same-class pairs
    rng = np.random.default_rng(0)
    ctrl = []
    while len(ctrl) < a.n_control:
        i, j = rng.choice(len(meta), 2, replace=False)
        if g[i] != g[j] and meta.y.iat[i] == meta.y.iat[j]:
            ctrl.append(dict(a=meta.filename.iat[i], b=meta.filename.iat[j], cosine=float(X[i] @ X[j]),
                             xcorr=peak_xcorr(wav(meta.filename.iat[i]), wav(meta.filename.iat[j]))[0]))
    ctrl = pd.DataFrame(ctrl)
    ctrl.to_csv(a.out.replace(".csv", "_control.csv"), index=False)

    sens = [dict(screen=s, xcorr=x, confirmed=int(((df.cosine >= s) & (df.xcorr >= x)).sum()),
                 cross_split=int(((df.cosine >= s) & (df.xcorr >= x) & df.cross_official_split).sum()),
                 cross_website=int(((df.cosine >= s) & (df.xcorr >= x) & df.cross_website).sum()))
            for s in (0.95, 0.97, 0.98, 0.99) for x in (0.7, 0.8, 0.9, 0.95)]
    pd.DataFrame(sens).to_csv(a.out.replace(".csv", "_sensitivity.csv"), index=False)

    s = df[df.same_audio]
    print(df.status.value_counts().to_string())
    print(f"confirmed: across websites {s.cross_website.sum()}, across official train/test {s.cross_official_split.sum()}, "
          f"conflicting labels {s.label_conflict.sum()}, bit-identical {s.bit_identical.sum()}")
    print(f"control (random same-class different-source pairs, n={len(ctrl)}): xcorr max {ctrl.xcorr.max():.3f}, "
          f"99th pct {ctrl.xcorr.quantile(.99):.3f}, share >= {a.xcorr}: {(ctrl.xcorr >= a.xcorr).mean():.4f}")
    print(pd.DataFrame(sens).pivot(index="screen", columns="xcorr", values="confirmed"))


if __name__ == "__main__":
    main()
