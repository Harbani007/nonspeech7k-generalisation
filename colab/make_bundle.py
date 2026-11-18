"""Build the upload bundle for Colab: code (this repository) + splits.csv.

  python colab/make_bundle.py --splits ../results/v3/dl/splits.csv --out ../colab_upload/ns7k_colab_bundle.zip

Inside the zip: ns7k-src/... (code, tests, docs, notebooks) and splits/splits.csv.
"""
import argparse
import os
import zipfile

SKIP_DIRS = {".git", "__pycache__", ".ipynb_checkpoints"}
SKIP_EXT = {".pt", ".npy", ".npz", ".f16", ".zip", ".wav"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    n = 0
    with zipfile.ZipFile(a.out, "w", zipfile.ZIP_DEFLATED) as z:
        for d, dirs, files in os.walk(repo):
            dirs[:] = [x for x in dirs if x not in SKIP_DIRS]
            for f in files:
                if os.path.splitext(f)[1].lower() in SKIP_EXT:
                    continue
                p = os.path.join(d, f)
                z.write(p, os.path.join("ns7k-src", os.path.relpath(p, repo)).replace(os.sep, "/"))
                n += 1
        z.write(a.splits, "splits/splits.csv")
    print(f"wrote {a.out}: {n} code files + splits.csv ({os.path.getsize(a.out) / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
