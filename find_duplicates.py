"""Find clips that are the same audio but carry different groups (re-uploads across or within websites).

Candidates: for every clip, its nearest neighbour from ANOTHER group in embedding space (default AST, cosine
above --threshold). Each candidate pair is then checked on the waveforms: peak normalised cross-correlation
above --xcorr counts as the same audio. Writes a CSV (columns a, b, same_audio, ...) for
website_holdout.py --duplicates.

  python find_duplicates.py --meta-train ... --meta-test ... --audio-root <audio> --feats feats --out duplicates.csv
"""
import argparse

import numpy as np
import pandas as pd
from scipy.signal import fftconvolve

from common import index_audio, load_feats, load_meta
from features import load_mono


def peak_xcorr(a, b):
    a = (a - a.mean()) / (a.std() + 1e-9)
    b = (b - b.mean()) / (b.std() + 1e-9)
    return float(np.abs(fftconvolve(a, b[::-1], mode="full")).max() / min(len(a), len(b)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--meta-train", required=True)
    ap.add_argument("--meta-test", required=True)
    ap.add_argument("--audio-root", required=True)
    ap.add_argument("--feats", default="feats")
    ap.add_argument("--set", default="ast")
    ap.add_argument("--threshold", type=float, default=0.98)
    ap.add_argument("--xcorr", type=float, default=0.9)
    ap.add_argument("--out", default="duplicates.csv")
    a = ap.parse_args()

    meta = load_meta(a.meta_train, a.meta_test)
    X, _ = load_feats(a.feats, meta, a.set)
    X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)
    g = np.asarray(meta["group"].values, dtype=str)
    best, who = np.zeros(len(meta)), np.zeros(len(meta), int)
    for i in range(0, len(meta), 1000):
        S = X[i:i + 1000] @ X.T
        S[g[i:i + 1000, None] == g[None, :]] = -1  # the neighbour must come from another group
        who[i:i + 1000], best[i:i + 1000] = S.argmax(1), S.max(1)
    pairs = sorted({tuple(sorted((i, int(who[i])))) for i in np.where(best > a.threshold)[0]})

    idx = index_audio(a.audio_root)
    rows = []
    for i, j in pairs:
        r1, r2 = meta.iloc[i], meta.iloc[j]
        rows.append(dict(a=r1.filename, b=r2.filename, site_a=r1.source, site_b=r2.source, split_a=r1.split,
                         split_b=r2.split, cls_a=r1.cls, cls_b=r2.cls, cosine=max(best[i], best[j]),
                         xcorr=peak_xcorr(load_mono(idx[r1.filename])[0], load_mono(idx[r2.filename])[0])))
    df = pd.DataFrame(rows)
    df["same_audio"] = df["xcorr"] > a.xcorr
    df.to_csv(a.out, index=False)
    s = df[df.same_audio]
    print(f"{len(df)} candidate pairs (cosine > {a.threshold}), {len(s)} confirmed same audio (xcorr > {a.xcorr})")
    print(f"  across websites {(s.site_a != s.site_b).sum()}, across official train/test {(s.split_a != s.split_b).sum()}, "
          f"different labels {(s.cls_a != s.cls_b).sum()}")
    print(s.groupby([s.site_a + "/" + s.site_b, s.split_a + "/" + s.split_b]).size().rename("pairs"))


if __name__ == "__main__":
    main()
