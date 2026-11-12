"""Smoke-test data only: tiny synthetic stand-in for Nonspeech7k with the same metadata layout.

Each 'recording' has its own gain, noise floor and spectral colour; clips are segments drawn from it,
so recording identity leaks across a random clip split exactly as it can in the real data.
Results on this data say nothing about Nonspeech7k.
"""
import argparse
import os

import numpy as np
import pandas as pd
import soundfile as sf
from scipy import signal

SR = 16000
CLASSES = ["breath", "cough", "crying", "laugh", "screaming", "sneeze", "yawm"]


def _noise(rng, n):
    return rng.standard_normal(n)


def _tone(t, f0, vib=0.0, vib_hz=5.0):
    ph = 2 * np.pi * np.cumsum(f0 * (1 + vib * np.sin(2 * np.pi * vib_hz * t))) / SR
    return sum(np.sin(h * ph) / h for h in (1, 2, 3))


def make_clip(cls, rng):
    dur = rng.uniform(2.0, 4.0)
    n = int(dur * SR)
    t = np.arange(n) / SR
    if cls == 0:  # breath: slow noise swells
        env = 0.5 * (1 - np.cos(2 * np.pi * rng.uniform(0.25, 0.5) * t + rng.uniform(0, 6)))
        x = _noise(rng, n) * env
    elif cls == 1:  # cough: 1-3 short decaying bursts
        x = np.zeros(n)
        for s in np.cumsum(rng.uniform(0.3, 0.6, rng.integers(1, 4))) - 0.3:
            i = int((0.2 + s) * SR)
            m = min(int(0.15 * SR), n - i)
            if m > 0:
                x[i:i + m] += _noise(rng, m) * np.exp(-np.arange(m) / (0.04 * SR))
    elif cls == 2:  # crying: wobbling voiced tone with ~2 Hz AM
        x = _tone(t, rng.uniform(350, 500), 0.05, 2.0) * (0.6 + 0.4 * np.sin(2 * np.pi * rng.uniform(1.5, 2.5) * t))
    elif cls == 3:  # laugh: train of voiced bursts at 4-6 Hz
        r = rng.uniform(4, 6)
        env = np.clip(np.sin(2 * np.pi * r * t), 0, None) ** 2 * np.linspace(1, 0.4, n)
        x = _tone(t, rng.uniform(220, 320)) * env
    elif cls == 4:  # scream: sustained loud voiced tone with vibrato
        x = _tone(t, rng.uniform(650, 900), 0.01, 6.0) * (0.8 + 0.2 * rng.random())
    elif cls == 5:  # sneeze: build-up hiss then one burst
        k = int(0.4 * n)
        env = np.concatenate([np.linspace(0, 0.3, k), np.exp(-np.arange(n - k) / (0.06 * SR))])
        x = _noise(rng, n) * env
    else:  # yawn: long voiced swell with falling pitch
        env = np.sin(np.pi * t / dur) ** 2
        x = _tone(t, np.linspace(280, 180, n)) * env
    return x / (np.abs(x).max() + 1e-9)


def apply_recording(x, rec, rng):
    sos = signal.butter(2, rec["cutoff"] / (SR / 2), btype="low", output="sos")
    x = signal.sosfilt(sos, x) if rec["cutoff"] < SR / 2 * 0.95 else x
    noise = signal.sosfilt(signal.butter(1, rec["noise_hp"] / (SR / 2), btype="high", output="sos"),
                           rng.standard_normal(len(x)))
    return (rec["gain"] * x + rec["noise"] * noise).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="synthetic")
    ap.add_argument("--recs-per-class", type=int, default=16)
    ap.add_argument("--clips", type=int, default=12)
    ap.add_argument("--test-recs", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    rows = {"train": [], "test": []}
    for cls, name in enumerate(CLASSES):
        for r in range(a.recs_per_class):
            split = "test" if r < a.test_recs else "train"
            rid = f"rec{cls}{r:02d}"
            rec = dict(gain=rng.uniform(0.2, 1.0), noise=10 ** rng.uniform(-3, -1.5),
                       cutoff=rng.uniform(3000, 7500), noise_hp=rng.uniform(50, 2000))
            os.makedirs(os.path.join(a.out, "audio", split), exist_ok=True)
            for k in range(a.clips):
                fn = f"{rid}-{k}_0_0.wav"
                x = apply_recording(make_clip(cls, rng), rec, rng)
                sf.write(os.path.join(a.out, "audio", split, fn), x, SR)
                rows[split].append({"Filename": fn, "File_ID": rid, "Durationin ms": int(len(x) / SR * 1000),
                                    "Class_id": cls, "Classname": name, "Augment Id": 0,
                                    "Augmentation type": "Original", "source": "synthetic"})
    for split, r in rows.items():
        pd.DataFrame(r).to_csv(os.path.join(a.out, f"metadata_{split}.csv"), index=False)
    print({k: len(v) for k, v in rows.items()})


if __name__ == "__main__":
    main()
