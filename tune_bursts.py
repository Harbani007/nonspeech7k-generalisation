"""Choose the burst-detection thresholds (peak prominence, minimum gap) using the official TRAIN set only.

  python tune_bursts.py --meta-train "metadata of train set .csv" --meta-test "metadata of test set.csv" \
      --audio-root <audio folder> --out burst_params.json

Run this once, before looking at any group-split result, then pass the JSON to extract.py --burst-params.
Each threshold pair is scored by the macro-F1 of a logistic regression on the 7 burst_* features alone, under
5-fold cross-validation grouped by source recording (File_ID), inside the train set. The test set is never
loaded. Rule: keep the default (features.py) unless another pair beats it by more than one standard error
across folds, so the choice is not driven by fold noise.
"""
import argparse
import json
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from common import index_audio, load_meta
from features import (BURST_MIN_GAP_S, BURST_PROM_DB, _stft_power, burst_features, envelope_db,
                      load_mono)

PROMS_DB = [3.0, 4.5, 6.0, 9.0, 12.0]
GAPS_S = [0.05, 0.08, 0.12, 0.20]


def _envelope(path):
    x, sr = load_mono(path)
    _, env_db = envelope_db(_stft_power(x, sr)[1])
    return env_db.astype(np.float32), len(x) / sr


def score(envs, y, groups, prom, gap, folds, seed):
    X = np.array([list(burst_features(e, d, prom, gap).values()) for e, d in envs])
    cv = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=seed)
    f1s = []
    for tr, va in cv.split(X, y, groups):
        clf = make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=2000))
        clf.fit(X[tr], y[tr])
        f1s.append(f1_score(y[va], clf.predict(X[va]), average="macro"))
    return np.mean(f1s), np.std(f1s) / np.sqrt(len(f1s)), np.mean(X[:, 0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--meta-train", required=True)
    ap.add_argument("--meta-test", required=True)
    ap.add_argument("--audio-root", required=True)
    ap.add_argument("--out", default="burst_params.json")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    meta = load_meta(a.meta_train, a.meta_test)
    meta = meta[meta["split"] == "train"].reset_index(drop=True)
    idx = index_audio(a.audio_root)
    envs = [_envelope(idx[f]) for f in meta["filename"]]  # cheap: about 10 ms a clip
    y, groups = meta["y"].to_numpy(), meta["group"].to_numpy()
    print(f"train only: {len(meta)} clips, {meta['group'].nunique()} recordings, {a.folds}-fold grouped CV\n")

    rows = []
    for prom in PROMS_DB:
        for gap in GAPS_S:
            m, se, nb = score(envs, y, groups, prom, gap, a.folds, a.seed)
            rows.append({"prom_db": prom, "min_gap_s": gap, "macro_f1": m, "se": se, "mean_bursts": nb})
    tab = pd.DataFrame(rows)
    print(tab.pivot(index="prom_db", columns="min_gap_s", values="macro_f1").round(3).to_string())

    default = tab[(tab.prom_db == BURST_PROM_DB) & (tab.min_gap_s == BURST_MIN_GAP_S)].iloc[0]
    best = tab.loc[tab["macro_f1"].idxmax()]
    pick = best if best["macro_f1"] - default["macro_f1"] > best["se"] else default
    why = "beats the default by more than 1 SE" if pick is best else "no pair beats the default by more than 1 SE"
    print(f"\ndefault {BURST_PROM_DB:g} dB / {BURST_MIN_GAP_S:g} s: {default['macro_f1']:.3f}")
    print(f"best    {best['prom_db']:g} dB / {best['min_gap_s']:g} s: {best['macro_f1']:.3f} (SE {best['se']:.3f})")
    print(f"chosen  {pick['prom_db']:g} dB / {pick['min_gap_s']:g} s ({why})")

    with open(a.out, "w", encoding="utf-8") as f:
        json.dump({"prom_db": float(pick["prom_db"]), "min_gap_s": float(pick["min_gap_s"]), "reason": why,
                   "criterion": f"burst-only macro-F1, {a.folds}-fold CV grouped by File_ID, train set only",
                   "grid": tab.round(4).to_dict(orient="records")}, f, indent=2)
    print(f"saved {a.out}")


if __name__ == "__main__":
    main()
