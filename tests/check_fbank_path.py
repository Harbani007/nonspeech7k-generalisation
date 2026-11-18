"""Equivalence check of the GPU input path (filterbanks + live frozen layers) against the CPU token cache.

1. For N real clips, embed(fbank) -> AST layers 0..10 must reproduce the cached layer-10 outputs (float16 cache).
2. Builds a mini fbank directory (a few train/val/test clips) and runs ast_finetune.py train --inputs fbank end to end
   with --freeze-k 10 (two pretrained layers trained), to exercise the full code path.

  python tests/check_fbank_path.py --data ../results/v3/dl --cache ../results/v3/ast_cache --audio-root ../data/audio --work <tmp dir>
"""
import argparse
import os
import subprocess
import sys

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ast_finetune import MAX_LEN, embed, load_ast  # noqa: E402
from common import index_audio  # noqa: E402
from features import load_mono  # noqa: E402
from scipy.signal import resample_poly  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--cache", required=True)
    ap.add_argument("--audio-root", required=True)
    ap.add_argument("--work", required=True)
    ap.add_argument("--n", type=int, default=6)
    a = ap.parse_args()
    torch.set_grad_enabled(False)
    clips = pd.read_csv(os.path.join(a.cache, "clips.csv"))
    H = np.load(os.path.join(a.cache, "tokens_k11.f16"), mmap_mode="r")
    fe, ast = load_ast()
    idx = index_audio(a.audio_root)
    rows = np.linspace(0, len(clips) - 1, a.n).astype(int)
    waves = [resample_poly(load_mono(idx[f])[0], 1, 2).astype(np.float32) for f in clips.filename.values[rows]]
    fb = fe(waves, sampling_rate=16000, return_tensors="np")["input_values"].astype(np.float16)
    h = embed(ast, torch.from_numpy(fb.astype(np.float32)))
    for layer in ast.layers[:11]:
        h = layer(h, None)
    ref = torch.from_numpy(np.asarray(H[rows], np.float32))
    rel = float((h - ref).norm() / ref.norm())
    print(f"[1] live layers 0-10 from float16 filterbanks vs cached tokens, {a.n} clips: relative difference {rel:.2e}")
    assert rel < 1e-2, "fbank path does not reproduce the cached tokens"

    # mini fbank dir: 32 train, 16 val, 16 test clips from the real selection
    os.makedirs(a.work, exist_ok=True)
    pick = np.concatenate([np.where(clips.part.values == p)[0][:n] for p, n in (("train", 32), ("val", 16), ("test", 16))])
    sub = clips.iloc[pick].reset_index(drop=True)
    d = os.path.join(a.work, "mini_fbank")
    os.makedirs(d, exist_ok=True)
    sub.to_csv(os.path.join(d, "clips.csv"), index=False)
    M = np.lib.format.open_memmap(os.path.join(d, "fbank.f16"), mode="w+", dtype=np.float16, shape=(len(sub), MAX_LEN, 128))
    for i in range(0, len(sub), 16):
        w = [resample_poly(load_mono(idx[f])[0], 1, 2).astype(np.float32) for f in sub.filename.values[i:i + 16]]
        M[i:i + len(w)] = fe(w, sampling_rate=16000, return_tensors="np")["input_values"].astype(np.float16)
    M.flush()
    open(os.path.join(d, "fbank.f16.done"), "w").write(str(len(sub)))
    out = os.path.join(a.work, "mini_run")
    cmd = [sys.executable, "ast_finetune.py", "train", "--data", a.data, "--cache", d, "--inputs", "fbank", "--freeze-k", "10",
           "--mode", "finetune", "--sampler", "source", "--supcon", "0.5", "--epochs", "1", "--samples-per-epoch", "32",
           "--batch", "16", "--out", out]
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    print("\n".join(x for x in r.stdout.splitlines() if x.startswith(("device", "mode", "epoch", "TEST"))))
    assert r.returncode == 0, r.stderr[-2000:]
    drift = pd.read_csv(os.path.join(out, "weight_drift.csv"), index_col=0).relative_change
    layers = sorted({n.split(".")[0] for n in drift.index if n.startswith("ast_layer")})
    print(f"[2] end-to-end fbank run OK; pretrained layers updated: {layers}; max relative change {drift.max():.2e}")
    assert layers == ["ast_layer_10", "ast_layer_11"] and drift.max() > 0
    print("FBANK PATH CHECK PASSED")


if __name__ == "__main__":
    main()
