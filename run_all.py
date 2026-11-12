"""Run the whole pipeline: audit -> features -> experiments -> recording probe -> figures.

Demo without the dataset (synthetic audio, results are NOT findings):
  python run_all.py --synthetic --workdir out_demo

Real data (after downloading and unzipping Nonspeech7k from Zenodo):
  python run_all.py --meta-train "metadata of train set .csv" --meta-test "metadata of test set.csv" \
      --audio-root <folder with train/ and test/> --workdir out
Add --embeddings to also extract AST and CLAP embeddings (needs torch + transformers, downloads weights).
"""
import argparse
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def run(script, *args, save=None):
    cmd = [sys.executable, os.path.join(HERE, script), *map(str, args)]
    print(">>", " ".join(cmd), flush=True)
    res = subprocess.run(cmd, check=True, capture_output=bool(save), text=True)
    if save:
        print(res.stdout)
        with open(save, "w", encoding="utf-8") as f:
            f.write(res.stdout)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--meta-train")
    ap.add_argument("--meta-test")
    ap.add_argument("--audio-root")
    ap.add_argument("--workdir", default="out")
    ap.add_argument("--embeddings", action="store_true")
    ap.add_argument("--seeds", type=int)
    ap.add_argument("--budgets", nargs="+", type=int)
    ap.add_argument("--skip-extract", action="store_true", help="reuse cached features in workdir/feats")
    a = ap.parse_args()

    wd = os.path.abspath(a.workdir)
    os.makedirs(wd, exist_ok=True)
    if a.synthetic:
        syn = os.path.join(wd, "synthetic")
        run("make_synthetic.py", "--out", syn)
        a.meta_train, a.meta_test = os.path.join(syn, "metadata_train.csv"), os.path.join(syn, "metadata_test.csv")
        a.audio_root = os.path.join(syn, "audio")
    elif not (a.meta_train and a.meta_test and a.audio_root):
        sys.exit("give --meta-train, --meta-test and --audio-root, or use --synthetic")
    seeds = a.seeds or (3 if a.synthetic else 5)
    budgets = a.budgets or ([300, 500] if a.synthetic else [300, 500, 700])
    feats, figs = os.path.join(wd, "feats"), os.path.join(wd, "figures")
    flag = ["--synthetic"] if a.synthetic else []
    sets = ["struct", "spec"] + (["ast", "clap"] if a.embeddings else [])
    exp_sets = ["spec", "struct", "struct-nf"] + (["ast", "ast+struct"] if a.embeddings else [])

    run("audit.py", a.meta_train, a.meta_test, save=os.path.join(wd, "audit.txt"))
    if not a.skip_extract:
        run("extract.py", "--meta-train", a.meta_train, "--meta-test", a.meta_test, "--audio-root", a.audio_root,
            "--out", feats, "--sets", *sets)
    run("run_experiments.py", "--meta-train", a.meta_train, "--meta-test", a.meta_test, "--feats", feats,
        "--sets", *exp_sets, "--seeds", seeds, "--budgets", *budgets, "--out", os.path.join(wd, "results.csv"),
        save=os.path.join(wd, "summary.txt"))
    run("probe_source.py", "--meta-train", a.meta_train, "--meta-test", a.meta_test, "--feats", feats,
        "--sets", *[s for s in ["spec", "struct", "ast"] if s in sets],
        "--min-clips", 10 if a.synthetic else 20, "--out", os.path.join(wd, "probe.csv"),
        save=os.path.join(wd, "probe.txt"))
    run("plots.py", os.path.join(wd, "results.csv"), "--out", figs, *flag)
    run("class_profile.py", "--meta-train", a.meta_train, "--meta-test", a.meta_test, "--feats", feats, "--out", figs, *flag)
    print(f"\nDone. Outputs in {wd}: audit.txt, summary.txt, probe.txt, results.csv, figures/")


if __name__ == "__main__":
    main()
