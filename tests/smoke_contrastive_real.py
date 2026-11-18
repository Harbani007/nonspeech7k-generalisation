"""Real-data smoke test of the cross-source contrastive objective (needs the AST token cache; not part of unittest).

On real cached AST tokens from the source-disjoint TRAINING partition only, checks that:
  1. sampled batches contain valid cross-source positive pairs (same class, different source group);
  2. the contrastive loss is finite and non-zero, and its gradient reaches the pretrained AST layer-11 weights;
  3. optimising the contrastive term alone changes those weights and lowers the loss on a fixed batch;
  4. if every clip is relabelled as one source, no anchor is eligible and the loss is exactly 0
     (same-source pairs are never used as positives).

  python tests/smoke_contrastive_real.py --data ../results/v3/dl --cache ../results/v3/ast_cache
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ast_finetune import ASTTop, batch_pair_stats, cross_source_supcon, load_ast  # noqa: E402
from train_cnn import sample_weights  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--cache", required=True)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    torch.manual_seed(0)
    sp = pd.read_csv(os.path.join(a.data, "splits.csv"))
    clips = pd.read_csv(os.path.join(a.cache, "clips.csv"))
    H = np.load(os.path.join(a.cache, "tokens_k11.f16"), mmap_mode="r")
    info = sp.iloc[clips.row.values].reset_index(drop=True)
    tr = np.where(clips.part.values == "train")[0]
    assert int(open(os.path.join(a.cache, "tokens_k11.f16.done")).read()) >= tr.max() + 1, "train tokens not cached yet"
    assert (info.source_disjoint.values[tr] == "train").all()
    y = torch.tensor(info.y.values)
    gid = torch.tensor(pd.factorize(info.group.values)[0])
    lines = []

    def say(s):
        print(s, flush=True)
        lines.append(s)

    _, ast = load_ast()
    model = ASTTop(ast, 11, 7)
    proj = nn.Sequential(nn.Linear(768, 768), nn.ReLU(), nn.Linear(768, 128))
    enc = [p for n, p in model.named_parameters() if not n.startswith("head")]
    gen = torch.Generator().manual_seed(0)

    for sampler in ("standard", "source"):
        w = sample_weights(info.iloc[tr].reset_index(drop=True), sampler)
        stats = []
        for _ in range(50):  # batch composition only, no forward pass
            ib = np.sort(tr[torch.multinomial(w, a.batch, replacement=True, generator=gen).numpy()])
            stats.append(batch_pair_stats(y[ib], gid[ib]))
        st = np.array(stats)
        say(f"[1] sampler {sampler:8s}: cross-source positive pairs per batch of {a.batch}: mean {st[:, 0].mean():.1f} "
            f"(min {st[:, 0].min():.0f}); anchors with a positive: mean {st[:, 1].mean():.2f}")
        assert st[:, 0].min() > 0

    w = sample_weights(info.iloc[tr].reset_index(drop=True), "source")
    ib = np.sort(tr[torch.multinomial(w, a.batch, replacement=True, generator=gen).numpy()])
    x, yb, gb = torch.from_numpy(np.asarray(H[ib], np.float32)), y[ib], gid[ib]
    say(f"    fixed batch: {len(ib)} clips, {len(set(gb.tolist()))} source groups, classes {np.bincount(yb.numpy(), minlength=7).tolist()}")

    z = F.normalize(proj(model.pooled(x)), dim=1)
    loss, frac = cross_source_supcon(z, yb, gb, 0.1)
    g = torch.autograd.grad(loss, enc)
    gn = float(torch.sqrt(sum((t ** 2).sum() for t in g)))
    say(f"[2] contrastive loss {loss.item():.4f}; anchors used {frac:.2f}; grad norm on pretrained AST layer 11 + LayerNorm {gn:.3e}")
    assert torch.isfinite(loss) and loss.item() > 0 and gn > 0

    qname, qparam = next((n, p) for n, p in model.named_parameters() if n.endswith(("query.weight", "q_proj.weight")))
    w0 = qparam.detach().clone()
    opt = torch.optim.AdamW(enc + list(proj.parameters()), lr=1e-4)
    l0 = loss.item()
    for _ in range(5):
        loss, _ = cross_source_supcon(F.normalize(proj(model.pooled(x)), dim=1), yb, gb, 0.1)
        opt.zero_grad(); loss.backward(); opt.step()
    l5, _ = cross_source_supcon(F.normalize(proj(model.pooled(x)), dim=1), yb, gb, 0.1)
    w1 = qparam.detach()
    rel = float((w1 - w0).norm() / w0.norm())
    say(f"[3] 5 contrastive-only steps on the fixed batch: loss {l0:.4f} -> {l5.item():.4f}; "
        f"relative change of AST layer-11 {qname.split('.', 2)[2]} {rel:.2e}")
    assert l5.item() < l0 and rel > 0

    l_one, f_one = cross_source_supcon(z.detach(), yb, torch.zeros_like(gb), 0.1)
    say(f"[4] all clips relabelled as one source: anchors used {f_one:.2f}, loss {l_one.item():.4f} (must be 0, 0)")
    assert f_one == 0.0 and l_one.item() == 0.0
    say("SMOKE TEST PASSED")
    if a.out:
        open(a.out, "w").write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
