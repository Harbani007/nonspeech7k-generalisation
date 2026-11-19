"""Accuracy, macro precision/recall/F1 per system on the source-disjoint test set, from saved test_predictions.csv.
  python report/compute_metrics.py --runs ../results/v3/runs --out report/build"""
import argparse, glob, os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dl_data import CLASSES
from dl_metrics import confusion

SYSTEMS = [("Spectral features + LR", "ml_spec_source_disjoint"), ("Structure features + LR", "ml_struct-nf_source_disjoint"),
           ("Frozen AST embedding + LR", "ml_ast_source_disjoint"), ("Log-mel CNN (from scratch)*", "cnn_source_disjoint_standard_s0"),
           ("AST frozen + linear head", "ast_frozen_source_disjoint_standard_s*"), ("AST fine-tuned: ordinary", "ast_ft_source_disjoint_standard_s*"),
           ("AST fine-tuned: source-balanced", "ast_ft_source_disjoint_source_s*"), ("AST fine-tuned: contrastive", "ast_ftsc_source_disjoint_standard_s*"),
           ("AST fine-tuned: combined", "ast_ftsc_source_disjoint_source_s*")]

def prf(y, p):
    c = confusion(y, p, len(CLASSES)); tp = np.diag(c).astype(float)
    prec = np.divide(tp, c.sum(0), out=np.zeros(len(tp)), where=c.sum(0) > 0)
    rec = np.divide(tp, c.sum(1), out=np.zeros(len(tp)), where=c.sum(1) > 0)
    f1 = np.divide(2 * prec * rec, prec + rec, out=np.zeros(len(tp)), where=(prec + rec) > 0)
    return tp.sum() / c.sum(), prec, rec, f1, c

ap = argparse.ArgumentParser(); ap.add_argument("--runs", required=True); ap.add_argument("--out", required=True); a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
rows, per = [], []
for name, pat in SYSTEMS:
    ds = sorted(glob.glob(os.path.join(a.runs, pat)))
    P = [pd.read_csv(os.path.join(d, "test_predictions.csv")) for d in ds if os.path.exists(os.path.join(d, "test_predictions.csv"))]
    if not P: continue
    m = [prf(t.y.values, t.pred.values) for t in P]
    rows.append(dict(system=name, seeds=len(P), test_clips=len(P[0]), accuracy=np.mean([x[0] for x in m]),
                     macro_precision=np.mean([x[1].mean() for x in m]), macro_recall=np.mean([x[2].mean() for x in m]),
                     macro_f1=np.mean([x[3].mean() for x in m]), macro_f1_sd=np.std([x[3].mean() for x in m])))
    for i, c in enumerate(CLASSES):
        per.append(dict(system=name, cls=c, precision=np.mean([x[1][i] for x in m]), recall=np.mean([x[2][i] for x in m]),
                        f1=np.mean([x[3][i] for x in m]), support=int(m[0][4][i].sum())))
R = pd.DataFrame(rows); R.to_csv(os.path.join(a.out, "metrics_summary.csv"), index=False)
pd.DataFrame(per).to_csv(os.path.join(a.out, "metrics_per_class.csv"), index=False)
pd.set_option("display.width", 200); print(R.round(3).to_string(index=False))
