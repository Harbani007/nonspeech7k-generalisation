"""Re-check duplicate candidates with a properly bounded similarity (robustness check of find_duplicates.py).

find_duplicates.peak_xcorr standardises each whole clip and divides the peak cross-correlation by the shorter
clip's length. When clips differ in length or loudness profile this is not bounded by 1 (33 of 137 confirmed pairs
scored > 1). Here the score at every lag is the cosine similarity of the OVERLAPPING segments,
    sum(a_seg * b_seg) / sqrt(sum(a_seg^2) * sum(b_seg^2))    (<= 1 by Cauchy-Schwarz),
maximised over lags whose overlap covers at least half of the shorter clip and >= 10% of each clip's energy.

  python recheck_duplicates.py --dups ../results/v3/duplicates.csv --control ../results/v3/duplicates_control.csv \
      --audio-root ../data/audio --out ../results/v3/duplicates_recheck.csv
"""
import argparse

import numpy as np
import pandas as pd
from scipy.signal import fftconvolve, resample_poly

from common import index_audio
from features import load_mono


def bounded_xcorr(a, b, min_overlap=0.5):
    a = a - a.mean()
    b = b - b.mean()
    la, lb = len(a), len(b)
    c = fftconvolve(a, b[::-1], mode="full")
    s = np.arange(-(lb - 1), la)
    n0, n1 = np.maximum(0, s), np.minimum(la, lb + s)
    ca, cb = np.concatenate([[0], np.cumsum(a * a)]), np.concatenate([[0], np.cumsum(b * b)])
    ea, eb = ca[n1] - ca[n0], cb[n1 - s] - cb[n0 - s]
    # the overlap must cover half the shorter clip AND hold >= 10% of each clip's energy (avoids near-silent overlaps,
    # where the cumulative-sum energies suffer cancellation and the ratio is numerically meaningless)
    ok = ((n1 - n0) >= min_overlap * min(la, lb)) & (ea >= 0.1 * ca[-1]) & (eb >= 0.1 * cb[-1])
    score = np.where(ok, np.abs(c) / np.sqrt(np.maximum(ea * eb, 1e-30)), 0)
    score = np.minimum(score, 1.0 + 1e-6)
    k = int(score.argmax())
    return float(score[k]), int(s[k])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dups", required=True)
    ap.add_argument("--control", required=True)
    ap.add_argument("--audio-root", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    idx = index_audio(a.audio_root)
    cache = {}

    def wav(f):
        if f not in cache:
            cache[f] = resample_poly(load_mono(idx[f])[0], 1, 2)
        return cache[f]

    d = pd.read_csv(a.dups)
    d[["xcorr_bounded", "lag_samples_16k"]] = [bounded_xcorr(wav(x), wav(y)) for x, y in zip(d.a, d.b)]
    ctl = pd.read_csv(a.control)
    ctl["xcorr_bounded"] = [bounded_xcorr(wav(x), wav(y))[0] for x, y in zip(ctl.a, ctl.b)]
    d.to_csv(a.out, index=False)
    ctl.to_csv(a.out.replace(".csv", "_control.csv"), index=False)

    conf = d.status == "confirmed"
    print(f"control pairs: max bounded score {ctl.xcorr_bounded.max():.3f}, 99th pct {ctl.xcorr_bounded.quantile(.99):.3f}")
    print(f"original confirmed pairs: {conf.sum()}; bounded score >= 0.9: {(d.xcorr_bounded[conf] >= 0.9).sum()}, "
          f">= 0.8: {(d.xcorr_bounded[conf] >= 0.8).sum()}, min {d.xcorr_bounded[conf].min():.3f}")
    for t in (0.9, 0.8, 0.7):
        new = d.xcorr_bounded >= t
        print(f"threshold {t}: confirmed {new.sum()} (of 277 candidates); cross-website {(new & d.cross_website).sum()}, "
              f"cross official split {(new & d.cross_official_split).sum()}, label conflicts {(new & d.label_conflict).sum()}; "
              f"agree with original status {(new == conf).sum()}/277")
    print(d[conf].sort_values("xcorr_bounded")[["a", "b", "xcorr", "xcorr_bounded"]].head(8).to_string(index=False))


if __name__ == "__main__":
    main()
