"""Data for the deep-learning experiments: splits, log-mel cache, samplers.

Splits (all written to <out>/splits.csv with one column per protocol, values train/val/test):
  source_disjoint : pooled clips, partitioned by merged source group (reconstructed source + confirmed
                    duplicate components); per class ~20% of sources -> test, ~15% -> val, rest -> train, with
                    sources of >= 100 clips kept in train (see source_split). No source and no duplicated audio
                    appears in two partitions. This is the primary protocol.
  official        : the official test set as test; the official train set split by merged source group into
                    train (~85%) and val (~15%). Known contamination: some official test clips share a source or
                    duplicated audio with official train (see source_audit.py); reported, not removed.
  official_dedup  : as official, but test clips whose merged group also occurs in official train are dropped.

Log-mel: 32 kHz -> 16 kHz, 25 ms Hann window, 10 ms hop, 64 mel bands (20-8000 Hz), log(1e-6 + power),
fixed 4.0 s (400 frames): shorter clips zero-padded at the end, longer clips centre-cropped. Cached as float16.

  python dl_data.py --meta-train ... --meta-test ... --audio-root <audio> --duplicates duplicates.csv --out <dir>
"""
import argparse
import os

import numpy as np
import pandas as pd
from scipy.signal import resample_poly

SR = 16000
N_FFT, HOP, WIN = 512, 160, 400
N_MELS, FRAMES = 64, 400
CLASSES = ["breath", "cough", "crying", "laugh", "screaming", "sneeze", "yawn"]


def load_meta_merged(meta_train, meta_test, duplicates):
    from common import load_meta
    from sources import merge_duplicates
    meta = load_meta(meta_train, meta_test)
    meta["cls"] = meta["cls"].replace("yawm", "yawn")
    dup = pd.read_csv(duplicates)
    meta["group"] = merge_duplicates(meta["group"], meta["filename"], dup[dup.status == "confirmed"])
    return meta


def source_split(idx, y, g, fracs, seed, big=100):
    """Assign whole groups to partitions, apportioning SOURCES per class (not clips): each group goes to the
    class it has most clips of; within a class, groups are shuffled and the first fracs[0] share of sources go
    to test, the next fracs[1] to val, the rest to train. Groups with >= `big` clips always go to train, so no
    test or val class is a handful of giant recordings. Returns {group: partition}."""
    rng = np.random.default_rng(seed)
    d = pd.DataFrame({"g": g[idx], "y": y[idx]})
    size = d.groupby("g").size()
    major = d.groupby("g").y.agg(lambda s: s.value_counts().index[0])
    part = {}
    for c in sorted(major.unique()):
        gs = np.array(sorted(major.index[major == c]))
        gs = gs[rng.permutation(len(gs))]
        small = [x for x in gs if size[x] < big]
        n_te, n_va = int(round(fracs[0] * len(gs))), int(round(fracs[1] * len(gs)))
        for i, x in enumerate(small):
            part[x] = "test" if i < n_te else "val" if i < n_te + n_va else "train"
        for x in gs:
            part.setdefault(x, "train")
    return part


def make_splits(meta, seed=0):
    y, g = meta["y"].values, meta["group"].values
    out = pd.DataFrame({"filename": meta["filename"]})
    part = source_split(np.arange(len(meta)), y, g, (0.2, 0.15), seed)
    out["source_disjoint"] = [part[x] for x in g]

    tr = np.where(meta["split"] == "train")[0]
    part_off = source_split(tr, y, g, (0.0, 0.15), seed)
    off = np.where(meta["split"] == "test", "test", [part_off.get(x, "test") for x in g])
    out["official"] = off
    leaked = meta["split"].eq("test") & meta["group"].isin(set(g[tr]))
    out["official_dedup"] = np.where(leaked, "excluded", off)
    return out


def check_disjoint(meta, col):
    """Raise if any group occurs in more than one of train/val/test."""
    m = meta[meta[col].isin(["train", "val", "test"])]
    n = m.groupby("group")[col].nunique()
    bad = n[n > 1]
    if len(bad):
        raise AssertionError(f"{col}: {len(bad)} groups straddle partitions, e.g. {list(bad.index[:3])}")


def mel_filters():
    from transformers.audio_utils import mel_filter_bank
    return mel_filter_bank(num_frequency_bins=N_FFT // 2 + 1, num_mel_filters=N_MELS, min_frequency=20.0,
                           max_frequency=8000.0, sampling_rate=SR, norm=None, mel_scale="htk").astype(np.float32)


def logmel(x32k, fb):
    x = resample_poly(x32k, 1, 2).astype(np.float32)
    n = HOP * (FRAMES - 1) + WIN
    if len(x) < n:
        x = np.pad(x, (0, n - len(x)))
    else:
        s = (len(x) - n) // 2
        x = x[s:s + n]
    w = np.hanning(WIN + 1)[:-1].astype(np.float32)
    idx = np.arange(WIN)[None, :] + HOP * np.arange(FRAMES)[:, None]
    spec = np.abs(np.fft.rfft(x[idx] * w, n=N_FFT, axis=1)) ** 2  # frames x bins
    return np.log(spec @ fb + 1e-6).T.astype(np.float32)  # mels x frames


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--meta-train", required=True)
    ap.add_argument("--meta-test", required=True)
    ap.add_argument("--audio-root", required=True)
    ap.add_argument("--duplicates", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    from common import index_audio
    from features import load_mono

    meta = load_meta_merged(a.meta_train, a.meta_test, a.duplicates)
    sp = make_splits(meta)
    full = meta.merge(sp, on="filename")
    for col in ("source_disjoint", "official_dedup"):
        check_disjoint(full, col)
    full[["filename", "y", "cls", "group", "source", "split", "source_disjoint", "official", "official_dedup"]] \
        .to_csv(os.path.join(a.out, "splits.csv"), index=False)
    for col in ("source_disjoint", "official", "official_dedup"):
        t = full.groupby([col, "cls"]).agg(clips=("filename", "size"), sources=("group", "nunique")).unstack(0)
        print(f"\n{col}\n{t}")
    print(f"\nofficial: {(full.official_dedup == 'excluded').sum()} official test clips share a merged source "
          "group with official train (excluded in official_dedup)")

    cache = os.path.join(a.out, "logmel.npy")
    if os.path.exists(cache):
        print("log-mel cache exists:", cache)
        return
    fb = mel_filters()
    idx = index_audio(a.audio_root)
    M = np.empty((len(full), N_MELS, FRAMES), np.float16)
    for i, f in enumerate(full.filename):
        x, sr = load_mono(idx[f])
        assert sr == 32000
        M[i] = logmel(x, fb)
        if i % 1000 == 0:
            print("log-mel", i, flush=True)
    np.save(cache, M)
    print("wrote", cache, M.shape)


if __name__ == "__main__":
    main()
