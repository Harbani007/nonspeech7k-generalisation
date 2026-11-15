"""Metadata loading and feature-cache helpers shared by the scripts."""
import os

import numpy as np
import pandas as pd


def _col(df, *candidates):
    norm = {c.strip().lower().replace(" ", "").replace("_", ""): c for c in df.columns}
    for cand in candidates:
        key = cand.lower().replace(" ", "").replace("_", "")
        if key in norm:
            return norm[key]
    raise KeyError(f"none of {candidates} in {list(df.columns)}")


def _site(d):
    try:
        url = d[_col(d, "source")].astype(str)
    except KeyError:
        return "unknown"
    return url.str.extract(r"//(?:www\.)?([a-z0-9-]+)", expand=False).fillna("unknown")


def load_meta(train_csv, test_csv):
    """Pooled metadata: filename, group (= File_ID, the source recording), y, cls, split, source (website)."""
    parts = []
    for path, split in [(train_csv, "train"), (test_csv, "test")]:
        d = pd.read_csv(path)
        d.columns = [c.strip() for c in d.columns]
        parts.append(pd.DataFrame({
            "filename": d[_col(d, "Filename")].astype(str).str.strip(),
            "group": d[_col(d, "File_ID")].astype(str).str.strip(),
            "y": d[_col(d, "Class_id")].astype(int),
            "cls": d[_col(d, "Classname")].astype(str).str.strip(),
            "split": split,
            "source": _site(d),
        }))
    return pd.concat(parts, ignore_index=True)


def index_audio(root):
    idx = {}
    for dirpath, _, files in os.walk(root):
        for fn in files:
            if fn.lower().endswith(".wav"):
                idx[fn] = os.path.join(dirpath, fn)
    return idx


def save_feats(path, names, X, columns):
    np.savez(path, names=np.array(list(names), dtype=str), X=np.asarray(X, dtype=np.float32),
             columns=np.array(list(columns), dtype=str))


def load_feats(feat_dir, meta, name):
    z = np.load(os.path.join(feat_dir, f"{name}.npz"), allow_pickle=False)
    pos = {n: i for i, n in enumerate(z["names"])}
    return z["X"][[pos[f] for f in meta["filename"]]], list(z["columns"])
