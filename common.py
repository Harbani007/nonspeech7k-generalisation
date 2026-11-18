"""Metadata loading and feature-cache helpers shared by the scripts."""
import os
import re

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


def _underlying_source(filename, file_id):
    """Aigei filenames like '142=KKKId0LDwVc-192_2_0.wav' or '148=17933_2_0.wav' give each clip its own File_ID
    but name the video or sound it was cut from after the '='. Clips sharing that key share a source."""
    m = re.match(r"^\d+=([^_]+?)(?:-man crying-aigei)?(?:-\d+)?_\d+_\d+\.wav$", filename)
    return m.group(1) if m else file_id


def load_meta(train_csv, test_csv, groups="source"):
    """Pooled metadata: filename, file_id, group, y, cls, split, source (website).

    groups="source" (default): group = the underlying recording, i.e. File_ID except for Aigei clips whose
    filename names the video/sound they were cut from (see _underlying_source); a numeric key equal to a
    Freesound File_ID is the same sound and shares its group. groups="file_id": group = File_ID as published.
    """
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
    meta = pd.concat(parts, ignore_index=True)
    meta.insert(1, "file_id", meta["group"])
    if groups == "source":
        meta["group"] = [_underlying_source(f, g) for f, g in zip(meta["filename"], meta["file_id"])]
    return meta


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
