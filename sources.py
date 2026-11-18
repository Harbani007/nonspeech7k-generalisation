"""Source-recording identity for Nonspeech7k clips.

The published `File_ID` is a clip-family identifier, not a global recording identifier:

* Freesound: File_ID is the Freesound sound ID (global, trustworthy).
* YouTube:   File_ID is the YouTube video ID (sometimes truncated, e.g. 'W7rJpRqK' for 'W7rJpRqK-8E').
* Aigei:     File_ID is a counter that restarts for every filename family ('79-cough-aigei', '79-sneeze-aigei'
             and Freesound sound 79 all carry File_ID '79'). Crying clips are named '<n>=<key>-<k>_2_0.wav',
             where <key> names the video or sound the clip was cut from (an 11-character YouTube ID, a numeric
             ID, an opaque token, or an Aigei-internal 'N-man crying-aigei' index).

`parse_source` turns (filename, File_ID, website) into a namespaced source key plus the rule that produced it,
so every grouping decision is auditable. Keys never collide across namespaces, so unrelated recordings are not
merged because their numbers happen to match. The only cross-website link made from names alone is a shared
YouTube video ID ('yt:<id>'), which is an exact identifier. Numeric keys found in Aigei names look like
Freesound sound IDs but are kept in their own namespace: that link is a hypothesis, to be confirmed on audio by
the duplicate check (`merge_duplicates`).
"""
from __future__ import annotations

import re
from dataclasses import dataclass

import pandas as pd

# how much we trust the key
KNOWN = "known"        # an exact external identifier (Freesound sound ID, YouTube video ID)
INFERRED = "inferred"  # parsed from a naming convention whose meaning is not documented

_CLIP_SUFFIX = re.compile(r"_\d+_\d+\.wav$")
_YT_ID = re.compile(r"^[A-Za-z0-9_-]{11}$")


@dataclass(frozen=True)
class Source:
    key: str        # namespaced source key, e.g. 'yt:KKKId0LDwVc', 'fs:108160', 'aigei:cough:79'
    rule: str       # which parsing rule fired
    confidence: str  # KNOWN or INFERRED
    freesound_id_candidate: str = ""  # numeric key in an Aigei name that may be a Freesound sound ID


def _stem(filename: str) -> str:
    return _CLIP_SUFFIX.sub("", filename)


def parse_source(filename: str, file_id: str, website: str) -> Source:
    filename, file_id = str(filename).strip(), str(file_id).strip()
    if website == "freesound":
        return Source(f"fs:{file_id}", "freesound_file_id", KNOWN)
    if website == "youtube":
        # the filename carries the full video ID when File_ID is truncated
        m = re.match(r"^([A-Za-z0-9_-]{11})", _stem(filename))
        vid = m.group(1) if m and m.group(1).startswith(file_id) else file_id
        return Source(f"yt:{vid}", "youtube_video_id", KNOWN)
    if website != "aigei":
        return Source(f"{website}:{file_id}", "unknown_website_file_id", INFERRED)

    stem = _stem(filename)
    if "=" in stem:
        rest = stem.split("=", 1)[1]
        m = re.match(r"^(\d+)-man crying-aigei(?:-\d+)?$", rest)
        if m:
            return Source(f"aigei:man crying:{m.group(1)}", "aigei_crying_index", INFERRED)
        m = re.match(r"^(.+)-\d+$", rest)
        base = m.group(1) if m else rest
        if _YT_ID.match(base):
            return Source(f"yt:{base}", "aigei_name_youtube_id", INFERRED)
        if base.isdigit():
            return Source(f"aigei:num:{base}", "aigei_name_numeric_id", INFERRED, freesound_id_candidate=base)
        return Source(f"aigei:tok:{base}", "aigei_name_token", INFERRED)

    m = re.match(r"^(\d+)-(.+?)-aigei(?:-\d+)?$", stem)
    if m:  # '<n>-<family>-aigei[-k]': File_ID n counts within the family
        return Source(f"aigei:{m.group(2)}:{m.group(1)}", "aigei_family_counter", INFERRED)
    m = re.match(r"^(\d+)-\d+$", stem)
    if m:
        return Source(f"aigei:num:{m.group(1)}", "aigei_name_numeric_id", INFERRED,
                      freesound_id_candidate=m.group(1))
    return Source(f"aigei:fileid:{file_id}", "aigei_fallback_file_id", INFERRED)


def add_sources(meta: pd.DataFrame) -> pd.DataFrame:
    """Add source_key / source_rule / source_confidence / freesound_id_candidate columns."""
    parsed = [parse_source(f, i, w) for f, i, w in zip(meta["filename"], meta["file_id"], meta["source"])]
    out = meta.copy()
    out["source_key"] = [p.key for p in parsed]
    out["source_rule"] = [p.rule for p in parsed]
    out["source_confidence"] = [p.confidence for p in parsed]
    out["freesound_id_candidate"] = [p.freesound_id_candidate for p in parsed]
    return out


def namespaced_file_id(meta: pd.DataFrame) -> pd.Series:
    """Published File_ID made collision-free (website, and for Aigei the filename family), but otherwise as
    published: one group per File_ID. This is the fair 'File_ID grouping' baseline."""
    fam = meta["filename"].str.extract(r"^\d+-(.+?)-aigei", expand=False).fillna("")
    eq = meta["filename"].str.contains("=", regex=False)
    fam = fam.where(~eq, "eq")
    key = meta["source"] + ":" + meta["file_id"]
    aigei = meta["source"] == "aigei"
    return key.where(~aigei, "aigei:" + fam + ":" + meta["file_id"])


class _UnionFind:
    def __init__(self, items):
        self.parent = {x: x for x in items}

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:  # the smaller key becomes the root, so component IDs are stable
            if rb < ra:
                ra, rb = rb, ra
            self.parent[rb] = ra


def merge_duplicates(groups: pd.Series, filenames: pd.Series, pairs: pd.DataFrame) -> pd.Series:
    """Connected components of the graph whose nodes are groups and whose edges are confirmed same-audio clip
    pairs (columns a, b: filenames). Returns the component ID per clip: the lexicographically smallest member
    group, so IDs do not depend on pair order. Transitive links are followed (A~B, B~C => A, B, C together)."""
    uf = _UnionFind(sorted(set(groups)))
    grp = dict(zip(filenames, groups))
    missing = [f for f in list(pairs["a"]) + list(pairs["b"]) if f not in grp]
    if missing:
        raise KeyError(f"{len(missing)} duplicate-pair filenames not in metadata, e.g. {missing[:3]}")
    for fa, fb in zip(pairs["a"], pairs["b"]):
        uf.union(grp[fa], grp[fb])
    return groups.map(uf.find)


def leakage(meta: pd.DataFrame, group_col: str = "group", split_col: str = "split") -> pd.DataFrame:
    """Groups that occur in both splits, with their clip counts per split."""
    t = pd.crosstab(meta[group_col], meta[split_col])
    if not {"train", "test"} <= set(t.columns):
        return t.iloc[0:0]
    return t[(t["train"] > 0) & (t["test"] > 0)]


def effective_sources(counts) -> float:
    """Inverse Simpson index of clips over sources: the number of equally sized sources that would give the
    same concentration. 958 clips with one source holding 376 of them gives far fewer than 70."""
    s = pd.Series(counts, dtype=float)
    p = s / s.sum()
    return float(1.0 / (p ** 2).sum())
