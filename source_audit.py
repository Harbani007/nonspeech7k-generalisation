"""Source-identity audit: clip->source manifest, source concentration per class x website, official-split leakage
under three groupings, and figures.

  python source_audit.py --meta-train ... --meta-test ... --duplicates duplicates.csv --out results/v3
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from common import load_meta  # noqa: E402
from sources import effective_sources, leakage, merge_duplicates  # noqa: E402


def concentration(meta, keys):
    rows = []
    for k, d in meta.groupby(keys):
        vc = d.group.value_counts()
        rows.append(dict(zip(keys, k if isinstance(k, tuple) else (k,)), clips=len(d), sources=len(vc),
                         clips_per_source=len(d) / len(vc), largest_source=vc.index[0], largest_clips=int(vc.iloc[0]),
                         largest_share=vc.iloc[0] / len(d), top5_share=vc.iloc[:5].sum() / len(d),
                         effective_sources=effective_sources(vc.values)))
    return pd.DataFrame(rows)


def save(fig, out, name):
    for ext in ("png", "svg"):
        fig.savefig(os.path.join(out, f"{name}.{ext}"), dpi=160, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--meta-train", required=True)
    ap.add_argument("--meta-test", required=True)
    ap.add_argument("--duplicates", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    fig_dir = os.path.join(a.out, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    meta = load_meta(a.meta_train, a.meta_test)
    meta["cls"] = meta["cls"].replace("yawm", "yawn")  # one test label is misspelled; same Class_id
    meta["file_id_ns"] = load_meta(a.meta_train, a.meta_test, groups="file_id")["group"]
    meta["file_id_raw"] = meta["file_id"]
    dup = pd.read_csv(a.duplicates)
    conf = dup[dup.status == "confirmed"]
    meta["source_group"] = meta["group"]
    meta["merged_group"] = merge_duplicates(meta["group"], meta["filename"], conf)
    dup_files = set(conf.a) | set(conf.b)
    meta["in_confirmed_duplicate"] = meta.filename.isin(dup_files)
    meta[["filename", "split", "source", "cls", "y", "file_id_raw", "file_id_ns", "source_group", "source_rule",
          "source_confidence", "freesound_id_candidate", "merged_group", "in_confirmed_duplicate"]] \
        .to_csv(os.path.join(a.out, "source_manifest.csv"), index=False)

    # groupings compared
    lines = []
    for col, label in [("file_id_raw", "File_ID as published"), ("file_id_ns", "File_ID, collision-free"),
                       ("source_group", "reconstructed source"), ("merged_group", "source + confirmed duplicates")]:
        m = meta.assign(group=meta[col])
        lk = leakage(m)
        te = m[m.split == "test"]
        lines.append(dict(grouping=label, groups=m.group.nunique(), test_groups=te.group.nunique(),
                          groups_in_both_splits=len(lk), test_clips_from_shared=int(te.group.isin(lk.index).sum()),
                          test_clips=len(te), multi_class_groups=int((m.groupby("group").cls.nunique() > 1).sum())))
    grp = pd.DataFrame(lines)
    grp["test_share_shared"] = grp.test_clips_from_shared / grp.test_clips
    grp.to_csv(os.path.join(a.out, "groupings_and_leakage.csv"), index=False)
    print(grp.to_string(index=False))

    m = meta.assign(group=meta["merged_group"])
    lk = leakage(m)
    m[m.group.isin(lk.index)].sort_values(["group", "split"]).to_csv(
        os.path.join(a.out, "official_split_leakage_clips.csv"), index=False)

    cw = concentration(meta, ["cls", "source"])
    cw.to_csv(os.path.join(a.out, "source_concentration_class_website.csv"), index=False)
    cc = concentration(meta, ["cls"])
    raw = meta.groupby("cls").file_id_ns.nunique().rename("published_file_ids")
    cc = cc.merge(raw, left_on="cls", right_index=True)
    cc.to_csv(os.path.join(a.out, "source_concentration_class.csv"), index=False)
    print(cw.round(3).to_string(index=False))
    sizes = meta.groupby("group").size().rename("clips").reset_index()
    sizes.sort_values("clips", ascending=False).to_csv(os.path.join(a.out, "source_sizes.csv"), index=False)
    print("\nlargest sources:\n", sizes.sort_values("clips", ascending=False).head(8).to_string(index=False))

    # figures
    classes = sorted(meta.cls.unique())
    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(len(classes))
    c = cc.set_index("cls").loc[classes]
    ax.bar(x - 0.27, c.clips, 0.27, label="clips", color="#4C72B0")
    ax.bar(x, c.published_file_ids, 0.27, label="published File_IDs", color="#DD8452")
    ax.bar(x + 0.27, c.sources, 0.27, label="reconstructed sources", color="#55A868")
    ax.set_xticks(x, classes)
    ax.set_ylabel("count")
    ax.set_title("Clips vs published File_IDs vs independent sources per class")
    ax.legend()
    save(fig, fig_dir, "A1_clips_vs_sources")

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(x - 0.2, c.largest_share, 0.4, label="largest source", color="#C44E52")
    ax.bar(x + 0.2, c.top5_share, 0.4, label="top-5 sources", color="#8172B2")
    ax.set_xticks(x, classes)
    ax.set_ylim(0, 1)
    ax.set_ylabel("share of the class's clips")
    ax.set_title("Source concentration per class")
    ax.legend()
    save(fig, fig_dir, "A2_source_concentration")

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(sizes.clips, bins=np.logspace(0, np.log10(sizes.clips.max() + 1), 30), color="#4C72B0")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("clips per source (log)")
    ax.set_ylabel("number of sources (log)")
    ax.set_title(f"Source-size distribution ({len(sizes)} sources, {len(meta)} clips)")
    save(fig, fig_dir, "A3_source_size_distribution")

    cry = meta[(meta.cls == "crying") & (meta.source == "aigei")].group.value_counts()
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(range(len(cry)), cry.values, color="#C44E52")
    ax.set_xlabel(f"Aigei crying sources, ranked ({len(cry)} sources, {cry.sum()} clips)")
    ax.set_ylabel("clips")
    ax.set_title(f"One source ({cry.index[0]}) supplies {cry.iloc[0]} of {cry.sum()} Aigei crying clips")
    save(fig, fig_dir, "A4_aigei_crying_sources")
    print("wrote", fig_dir)


if __name__ == "__main__":
    main()
