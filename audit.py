"""Phase 0 audit: do train and test share source recordings, and how concentrated is each class?

  python audit.py "metadata of train set .csv" "metadata of test set.csv"
"""
import sys

import pandas as pd

from common import load_meta

meta = load_meta(sys.argv[1], sys.argv[2])
tr, te = meta[meta.split == "train"], meta[meta.split == "test"]
print("clips: train", len(tr), "test", len(te))
shared = set(tr.group) & set(te.group)
print("recordings in both splits:", len(shared), "of", te.group.nunique(), "test recordings")
print("test clips from shared recordings: %.1f%%" % (100 * te.group.isin(shared).mean()))
g = tr.groupby("cls").group.agg(clips="size", recordings="nunique")
g["clips_per_recording"] = (g.clips / g.recordings).round(1)
g["largest_recording_share"] = tr.groupby("cls").group.apply(lambda s: s.value_counts(normalize=True).iloc[0]).round(3)
print(g)

# Where the clips come from. Sources unevenly spread over classes are a shortcut a classifier can learn.
names = tr.groupby("y").cls.first()
for part, d in [("train", tr), ("test", te)]:
    t = pd.crosstab(d.y.map(names), d.source, margins=True, margins_name="all")
    print(f"\n{part} clips by website")
    print(t)
    print(f"{part} recordings by website:", d.groupby("source").group.nunique().to_dict())
