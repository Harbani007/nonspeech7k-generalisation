"""Extract features / frozen embeddings for every clip and cache them as .npz files.

  python extract.py --meta-train "metadata of train set .csv" --meta-test "metadata of test set.csv" \
      --audio-root <folder containing the unzipped train/ and test/> --out feats --sets struct spec ast clap

struct/spec need only numpy/scipy. ast/clap need torch + transformers and download model weights
from the Hugging Face hub on first use (not exercised in this repo's smoke test).
"""
import argparse
import os
from multiprocessing import Pool

import numpy as np

from common import index_audio, load_meta, save_feats
from features import COMPUTERS, load_mono


def _hand(args):
    path, name = args
    x, sr = load_mono(path)
    return COMPUTERS[name](x, sr)


def run_handcrafted(name, paths, out, workers):
    with Pool(workers) as pool:
        rows = pool.map(_hand, [(p, name) for p in paths], chunksize=16)
    cols = list(rows[0].keys())
    return np.array([[r[c] for c in cols] for r in rows]), cols


def _resample(x, sr, target):
    from scipy.signal import resample_poly
    from math import gcd
    g = gcd(int(sr), int(target))
    return resample_poly(x, target // g, int(sr) // g).astype(np.float32)


def run_ast(paths, batch=16):
    import torch
    from transformers import ASTFeatureExtractor, ASTModel
    ck = "MIT/ast-finetuned-audioset-10-10-0.4593"
    fe, model = ASTFeatureExtractor.from_pretrained(ck), ASTModel.from_pretrained(ck).eval()
    embs = []
    for i in range(0, len(paths), batch):
        waves = []
        for p in paths[i:i + batch]:
            x, sr = load_mono(p)
            waves.append(_resample(x, sr, 16000))
        with torch.no_grad():
            inp = fe(waves, sampling_rate=16000, return_tensors="pt")
            embs.append(model(**inp).pooler_output.numpy())
        print(f"ast {i + len(waves)}/{len(paths)}", flush=True)
    X = np.concatenate(embs)
    return X, [f"ast{i}" for i in range(X.shape[1])]


def run_clap(paths, batch=16):
    import torch
    from transformers import ClapModel, ClapProcessor
    ck = "laion/clap-htsat-unfused"
    proc, model = ClapProcessor.from_pretrained(ck), ClapModel.from_pretrained(ck).eval()
    embs = []
    for i in range(0, len(paths), batch):
        waves = []
        for p in paths[i:i + batch]:
            x, sr = load_mono(p)
            waves.append(_resample(x, sr, 48000))
        try:
            inp = proc(audios=waves, sampling_rate=48000, return_tensors="pt")
        except TypeError:  # newer transformers renamed the kwarg
            inp = proc(audio=waves, sampling_rate=48000, return_tensors="pt")
        with torch.no_grad():
            embs.append(model.get_audio_features(**inp).numpy())
        print(f"clap {i + len(waves)}/{len(paths)}", flush=True)
    X = np.concatenate(embs)
    return X, [f"clap{i}" for i in range(X.shape[1])]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--meta-train", required=True)
    ap.add_argument("--meta-test", required=True)
    ap.add_argument("--audio-root", required=True)
    ap.add_argument("--out", default="feats")
    ap.add_argument("--sets", nargs="+", default=["struct", "spec"])
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--limit", type=int, default=0, help="debug: only first N clips")
    a = ap.parse_args()

    meta = load_meta(a.meta_train, a.meta_test)
    if a.limit:
        meta = meta.iloc[:a.limit]
    idx = index_audio(a.audio_root)
    missing = [f for f in meta["filename"] if f not in idx]
    if missing:
        raise SystemExit(f"{len(missing)} metadata files not found under audio root, e.g. {missing[:3]}")
    paths = [idx[f] for f in meta["filename"]]
    os.makedirs(a.out, exist_ok=True)
    for name in a.sets:
        if name in COMPUTERS:
            X, cols = run_handcrafted(name, paths, a.out, a.workers)
        elif name == "ast":
            X, cols = run_ast(paths)
        elif name == "clap":
            X, cols = run_clap(paths)
        else:
            raise SystemExit(f"unknown set {name}")
        save_feats(os.path.join(a.out, f"{name}.npz"), meta["filename"], X, cols)
        print(f"saved {name}: {X.shape}")


if __name__ == "__main__":
    main()
