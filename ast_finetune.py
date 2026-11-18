"""Partial fine-tuning of the Audio Spectrogram Transformer (AST) on Nonspeech7k, designed for a CPU-only machine.

Checkpoint: MIT/ast-finetuned-audioset-10-10-0.4593 (AST pretrained on ImageNet, then AudioSet; Gong et al. 2021).
The AudioSet classification head is discarded; a new 7-class linear head is trained.

Input: 4.0 s at 16 kHz -> 128-bin Kaldi fbank via ASTFeatureExtractor(max_length=400) with the checkpoint's
normalisation. The checkpoint was trained on 1024 frames (10.24 s); for 400 frames the time axis of the
pretrained position embeddings is truncated to the first 39 patch columns (as in the original AST code for
shorter inputs), giving 12 x 39 patches + CLS + distillation token = 470 tokens.

Compute plan (no GPU): layers 0..K-1 are FROZEN and run once; their output tokens are cached to disk (float16).
Training then runs only layers K..11 + final LayerNorm + new head, so the pretrained weights of layers K..11
receive gradients and are updated by AdamW. Modes:
  frozen   : all 12 layers frozen, only the new linear head is trained (pretrained-representation baseline)
  finetune : layers K..11, final LayerNorm and head trained (partial fine-tuning)
Samplers 'standard' and 'source' are defined exactly as in train_cnn.py.

Cross-source supervised contrastive term (--supcon LAMBDA > 0), added to cross-entropy:
  z = L2-normalised projection (768 -> 768 -> 128 MLP) of the pooled AST embedding; for anchor i in a batch,
  positives are clips of the SAME class from a DIFFERENT source group (sources.py keys with confirmed duplicates
  merged, so a re-upload is never a positive); clips of the same class AND same source are removed from both the
  positives and the denominator (neither pulled together nor pushed apart); all other clips are negatives.
  Anchors with no eligible positive in the batch are skipped. LAMBDA = 0 gives plain cross-entropy with an
  identical initialisation, batch sequence and optimiser (the projection head is always built).

  python ast_finetune.py cache --data results/v3/dl --audio-root data/audio --freeze-k 10 --out results/v3/ast_cache
  python ast_finetune.py train --data results/v3/dl --cache results/v3/ast_cache --mode finetune --sampler source --seed 0 --out runs/ast_ft_source_s0
"""
import argparse
import json
import os
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from dl_data import CLASSES
from dl_metrics import metrics
from train_cnn import sample_weights

CKPT = "MIT/ast-finetuned-audioset-10-10-0.4593"
MAX_LEN = 400


def load_ast():
    from transformers import ASTFeatureExtractor, ASTModel
    fe = ASTFeatureExtractor.from_pretrained(CKPT, max_length=MAX_LEN)
    return fe, ASTModel.from_pretrained(CKPT).eval()


def embed(ast, x):
    """Patch + position embeddings for inputs shorter than the checkpoint's 1024 frames. x: (B, frames, 128)."""
    e = ast.embeddings
    patches = e.patch_embeddings(x)  # (B, 12 * T', 768), frequency-major
    f_dim, t_full = e.get_shape(ast.config)
    t = patches.shape[1] // f_dim
    pos = e.position_embeddings
    pos_patch = pos[:, 2:].reshape(1, f_dim, t_full, -1)[:, :, :t].reshape(1, f_dim * t, -1)
    B = x.shape[0]
    cls = e.cls_token.expand(B, -1, -1) + pos[:, :1]
    dist = e.distillation_token.expand(B, -1, -1) + pos[:, 1:2]
    return torch.cat([cls, dist, patches + pos_patch], dim=1)


class ASTTop(nn.Module):
    """Upper AST layers K..11 + final LayerNorm + new linear head; input = cached output of layer K-1."""

    def __init__(self, ast, k, n_classes=7):
        super().__init__()
        self.layers = nn.ModuleList(ast.layers[k:])
        self.layernorm = ast.layernorm
        self.head = nn.Linear(ast.config.hidden_size, n_classes)

    def pooled(self, h):
        for layer in self.layers:
            h = layer(h, None)
        h = self.layernorm(h)
        return (h[:, 0] + h[:, 1]) / 2  # AST pooling: mean of CLS and distillation tokens

    def forward(self, h):
        return self.head(self.pooled(h))


def select_clips(sp, protocol, per_class_train, seed):
    """Train: up to per_class_train random clips per class (clip-uniform, so big sources keep their natural
    weight in the pool); val and test: all clips."""
    rng = np.random.default_rng(seed)
    part = sp[protocol].values
    tr = []
    for c in sorted(sp.y.unique()):
        ix = np.where((part == "train") & (sp.y.values == c))[0]
        tr.extend(rng.choice(ix, min(per_class_train, len(ix)), replace=False))
    va, te = np.where(part == "val")[0], np.where(part == "test")[0]
    return np.sort(np.array(tr)), va, te


def cmd_cache(a):
    from common import index_audio
    from features import load_mono
    from scipy.signal import resample_poly
    os.makedirs(a.out, exist_ok=True)
    sp = pd.read_csv(os.path.join(a.data, "splits.csv"))
    tr, va, te = select_clips(sp, a.protocol, a.per_class_train, 0)
    rows = np.concatenate([tr, va, te])
    part = np.array(["train"] * len(tr) + ["val"] * len(va) + ["test"] * len(te))
    meta_path = os.path.join(a.out, "clips.csv")
    pd.DataFrame({"row": rows, "part": part, "filename": sp.filename.values[rows]}).to_csv(meta_path, index=False)
    fe, ast = load_ast()
    torch.set_grad_enabled(False)
    n_tok = 2 + 12 * ((MAX_LEN - 16) // 10 + 1)
    path = os.path.join(a.out, f"tokens_k{a.freeze_k}.f16")
    done_path = path + ".done"
    H = np.lib.format.open_memmap(path, mode="r+" if os.path.exists(path) else "w+", dtype=np.float16,
                                  shape=(len(rows), n_tok, 768))
    start = int(open(done_path).read()) if os.path.exists(done_path) else 0  # resumable
    idx = index_audio(a.audio_root)
    t0 = time.time()
    for i in range(start, len(rows), a.batch):
        waves = []
        for f in sp.filename.values[rows[i:i + a.batch]]:
            x, sr = load_mono(idx[f])
            waves.append(resample_poly(x, 1, 2).astype(np.float32))
        x = fe(waves, sampling_rate=16000, return_tensors="pt")["input_values"]
        h = embed(ast, x)
        for layer in ast.layers[:a.freeze_k]:
            h = layer(h, None)
        H[i:i + len(waves)] = h.numpy().astype(np.float16)
        H.flush()
        open(done_path, "w").write(str(i + len(waves)))
        if (i // a.batch) % 10 == 0:
            el = time.time() - t0
            print(f"cached {i + len(waves)}/{len(rows)} ({el:.0f}s, {el / (i + len(waves) - start):.2f}s/clip)", flush=True)
    print("done", path)


def ast_name(n, k):
    """'layers.0.mlp.fc1.weight' inside ASTTop is pretrained AST layer k+0: report it as 'ast_layer_11.mlp...'."""
    if n.startswith("layers."):
        i, rest = n[len("layers."):].split(".", 1)
        return f"ast_layer_{k + int(i)}.{rest}"
    return n


def cross_source_supcon(z, y, g, tau):
    """Class-conditioned cross-source supervised contrastive loss.

    z: (B, d) L2-normalised embeddings; y: (B,) class ids; g: (B,) source-group ids.
    Returns (loss, fraction of anchors that had at least one eligible positive)."""
    B = z.shape[0]
    sim = z @ z.T / tau
    eye = torch.eye(B, dtype=torch.bool, device=z.device)
    same_y = y[:, None] == y[None, :]
    same_g = g[:, None] == g[None, :]
    pos = same_y & ~same_g                     # same class, independent source
    excluded = (same_y & same_g) | eye         # same class & same source, or self
    logits = sim.masked_fill(excluded, float("-inf"))
    log_prob = logits - torch.logsumexp(logits, dim=1, keepdim=True)
    n_pos = pos.sum(1)
    ok = n_pos > 0
    if not ok.any():
        return z.sum() * 0.0, 0.0
    per_anchor = -(log_prob.masked_fill(~pos, 0.0).sum(1)[ok] / n_pos[ok])
    return per_anchor.mean(), float(ok.float().mean())


LOCK_STALE_S = 900  # a run lock not refreshed for 15 min is treated as left behind by a dead process


def atomic_save(obj, path):
    tmp = path + ".tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path)


def acquire_lock(out):
    """One process per run folder. The lock file is refreshed every step; an old one is taken over."""
    lock = os.path.join(out, ".lock")
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        age = time.time() - os.path.getmtime(lock)
        if age < LOCK_STALE_S:
            raise SystemExit(f"{out} is locked by another run (lock refreshed {age:.0f}s ago); refusing to write")
        print(f"taking over stale lock ({age:.0f}s old)", flush=True)
        os.remove(lock)
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    os.write(fd, str(os.getpid()).encode())
    os.close(fd)
    return lock


def batch_pair_stats(yb, gb):
    """Cross-source positive pairs (ordered, i != j) and share of anchors with at least one."""
    pos = (yb[:, None] == yb[None, :]) & (gb[:, None] != gb[None, :])
    return int(pos.sum()), float(pos.any(1).float().mean())


def cmd_train(a):
    os.makedirs(a.out, exist_ok=True)
    if os.path.exists(os.path.join(a.out, "test_metrics.json")):
        raise SystemExit(f"{a.out} already has test_metrics.json; refusing to overwrite a finished run")
    lock = acquire_lock(a.out)
    try:
        _train(a, lock)
    finally:
        if os.path.exists(lock):
            os.remove(lock)


def _train(a, lock):
    torch.manual_seed(a.seed)
    sp = pd.read_csv(os.path.join(a.data, "splits.csv"))
    clips = pd.read_csv(os.path.join(a.cache, "clips.csv"))
    H = np.load(os.path.join(a.cache, f"tokens_k{a.freeze_k}.f16"), mmap_mode="r")
    assert int(open(os.path.join(a.cache, f"tokens_k{a.freeze_k}.f16.done")).read()) == len(clips), "cache incomplete"
    info = sp.iloc[clips.row.values].reset_index(drop=True)
    pos = {p: np.where(clips.part.values == p)[0] for p in ("train", "val", "test")}
    y = torch.tensor(info.y.values, dtype=torch.long)

    _, ast = load_ast()
    torch.manual_seed(a.seed)  # same new-head initialisation for every condition at a given seed
    model = ASTTop(ast, a.freeze_k, len(CLASSES))
    proj = nn.Sequential(nn.Linear(768, 768), nn.ReLU(), nn.Linear(768, a.proj_dim))  # built in every condition
    for p in model.parameters():
        p.requires_grad = a.mode == "finetune"
    for p in model.head.parameters():
        p.requires_grad = True
    enc = [p for n, p in model.named_parameters() if p.requires_grad and not n.startswith("head")]
    n_tr = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_frozen = sum(p.numel() for p in ast.parameters()) + sum(p.numel() for p in model.head.parameters()) - n_tr
    groups = [{"params": enc, "lr": a.lr},
              {"params": list(model.head.parameters()) + (list(proj.parameters()) if a.supcon > 0 else []), "lr": a.head_lr}]
    groups = [g for g in groups if g["params"]]
    opt = torch.optim.AdamW(groups, weight_decay=a.wd)
    steps_per_epoch = a.samples_per_epoch // a.batch
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[g["lr"] for g in groups],
                                                total_steps=a.epochs * steps_per_epoch, pct_start=0.1)
    dtr = info.iloc[pos["train"]].reset_index(drop=True)
    w = sample_weights(dtr, a.sampler)
    gid = torch.tensor(pd.factorize(info.group.values)[0], dtype=torch.long)  # sources.py groups, duplicates merged
    gen = torch.Generator().manual_seed(1000 + a.seed)
    w_before = {n: p.detach().clone() for n, p in model.named_parameters() if not n.startswith("head")}

    # ---- resume -----------------------------------------------------------------------------------------------
    run_args = {k: v for k, v in vars(a).items() if k != "out"}
    hist, best, seen, grad_log, step_log, start_ep = [], -1.0, [], [], [], 1
    last = os.path.join(a.out, "last.pt")
    if os.path.exists(last):
        st = torch.load(last, weights_only=False)
        if st["args"] != run_args:
            raise SystemExit(f"{last} was written with different arguments; refusing to resume:\n{st['args']}\n{run_args}")
        model.load_state_dict(st["model"]); proj.load_state_dict(st["proj"])
        opt.load_state_dict(st["opt"]); sched.load_state_dict(st["sched"])
        gen.set_state(st["gen"]); torch.set_rng_state(st["torch_rng"])
        hist, best, seen, grad_log, step_log = st["hist"], st["best"], st["seen"], st["grad_log"], st["step_log"]
        start_ep = st["epoch"] + 1
        print(f"resuming after epoch {st['epoch']} from {last}", flush=True)
    print(f"mode {a.mode}; trainable params {n_tr:,}; "
          f"frozen params {n_frozen:,}; train pool {len(dtr)} clips / {dtr.group.nunique()} sources; "
          f"val {len(pos['val'])}; test {len(pos['test'])}; sampler {a.sampler}; supcon {a.supcon} (tau {a.tau})", flush=True)

    def tokens(ix):
        return torch.from_numpy(np.asarray(H[ix], np.float32))

    @torch.no_grad()
    def predict(ix, return_emb=False):
        model.eval()
        embs = torch.cat([model.pooled(tokens(ix[i:i + 32])) for i in range(0, len(ix), 32)])
        p = F.softmax(model.head(embs), 1).numpy()
        return (p, embs.numpy()) if return_emb else p

    def gnorm(params):
        g = [p.grad for p in params if p.grad is not None]
        return float(torch.sqrt(sum((x.float() ** 2).sum() for x in g))) if g else 0.0

    t0 = time.time()
    for ep in range(start_ep, a.epochs + 1):
        model.train()
        draw = torch.multinomial(w, steps_per_epoch * a.batch, replacement=True, generator=gen).numpy()
        seen.append(draw)
        tot = ce_tot = sc_tot = 0.0
        ep_steps = []
        for s in range(steps_per_epoch):
            b = draw[s * a.batch:(s + 1) * a.batch]
            ix = pos["train"][b]
            ib = ix[np.argsort(ix)]  # memmap reads want sorted indices
            yb, gb = y[ib], gid[ib]
            emb = model.pooled(tokens(ib))
            ce = F.cross_entropy(model.head(emb), yb)
            n_pairs, frac = batch_pair_stats(yb, gb)
            sc = None
            if a.supcon > 0:
                sc, frac_used = cross_source_supcon(F.normalize(proj(emb), dim=1), yb, gb, a.tau)
                assert abs(frac_used - frac) < 1e-6
            loss = ce if sc is None else ce + a.supcon * sc
            rec = dict(epoch=ep, step=s, ce=ce.item(), supcon=sc.item() if sc is not None else np.nan,
                       loss=loss.item(), cross_source_pairs=n_pairs, anchors_with_positive=frac)
            opt.zero_grad()
            if s == 0 and a.mode == "finetune":
                # separate gradient contributions of the two loss terms on the pretrained encoder weights
                g_ce = torch.autograd.grad(ce, enc, retain_graph=True)
                rec["enc_grad_norm_ce"] = float(torch.sqrt(sum((x ** 2).sum() for x in g_ce)))
                if sc is not None:
                    g_sc = torch.autograd.grad(a.supcon * sc, enc, retain_graph=True)
                    rec["enc_grad_norm_supcon"] = float(torch.sqrt(sum((x ** 2).sum() for x in g_sc)))
                    f1, f2 = torch.cat([x.flatten() for x in g_ce]), torch.cat([x.flatten() for x in g_sc])
                    rec["enc_grad_cosine_ce_supcon"] = float(F.cosine_similarity(f1, f2, dim=0))
            loss.backward()
            rec["enc_grad_norm"] = gnorm(enc)
            rec["head_grad_norm"] = gnorm(model.head.parameters())
            rec["proj_grad_norm"] = gnorm(proj.parameters())
            if s == 0 and a.mode == "finetune":
                gn = {ast_name(n, a.freeze_k): float(p.grad.norm()) for n, p in model.named_parameters() if p.grad is not None}
                grad_log.append(dict(epoch=ep, **{k: v for k, v in gn.items() if "ast_layer" in k and k.endswith(("query.weight", "q_proj.weight", "dense.weight", "fc1.weight", "fc2.weight"))}))
            opt.step()
            sched.step()
            tot += loss.item(); ce_tot += ce.item(); sc_tot += rec["supcon"] if sc is not None else 0.0
            ep_steps.append(rec)
            os.utime(lock)
            if s % 8 == 0:
                print(f"  ep {ep} step {s}/{steps_per_epoch} loss {loss.item():.3f} ce {ce.item():.3f} "
                      f"supcon {rec['supcon']:.3f} pairs {n_pairs} anchors_with_pos {frac:.2f} "
                      f"enc_grad {rec['enc_grad_norm']:.3e} ({time.time() - t0:.0f}s)", flush=True)
        step_log.extend(ep_steps)
        E = pd.DataFrame(ep_steps)
        pv = predict(pos["val"])
        yv = y[pos["val"]].numpy()
        vloss = float(F.nll_loss(torch.log(torch.from_numpy(pv) + 1e-9), torch.from_numpy(yv)))
        m = metrics(yv, pv.argmax(1), len(CLASSES))
        hist.append(dict(epoch=ep, train_loss=tot / steps_per_epoch, train_ce=ce_tot / steps_per_epoch,
                         train_supcon=sc_tot / steps_per_epoch, anchors_with_positive=E.anchors_with_positive.mean(),
                         cross_source_pairs_per_batch=E.cross_source_pairs.mean(), enc_grad_norm=E.enc_grad_norm.mean(),
                         val_loss=vloss, val_acc=m["accuracy"], val_macro_f1=m["macro_f1"],
                         val_bal_acc=m["balanced_accuracy"], elapsed_s=time.time() - t0))
        print(f"epoch {ep} train_loss {tot / steps_per_epoch:.3f} val_loss {vloss:.3f} val_acc {m['accuracy']:.3f} "
              f"val_macroF1 {m['macro_f1']:.3f} pairs/batch {E.cross_source_pairs.mean():.1f} ({time.time() - t0:.0f}s)", flush=True)
        if m["macro_f1"] > best:
            best = m["macro_f1"]
            pd.DataFrame(pv, columns=[f"p_{c}" for c in CLASSES]).assign(y=yv).to_csv(os.path.join(a.out, "val_predictions.csv"), index=False)
            atomic_save(dict(state_dict={k: v for k, v in model.state_dict().items()}, freeze_k=a.freeze_k,
                             mode=a.mode, classes=CLASSES, ckpt=CKPT, max_len=MAX_LEN, epoch=ep, args=vars(a)),
                        os.path.join(a.out, "best.pt"))
        atomic_save(dict(model=model.state_dict(), proj=proj.state_dict(), opt=opt.state_dict(), sched=sched.state_dict(),
                         gen=gen.get_state(), torch_rng=torch.get_rng_state(), hist=hist, best=best, seen=seen,
                         grad_log=grad_log, step_log=step_log, epoch=ep, args=run_args), last)
    pd.DataFrame(hist).to_csv(os.path.join(a.out, "history.csv"), index=False)
    pd.DataFrame(step_log).to_csv(os.path.join(a.out, "step_log.csv"), index=False)
    if grad_log:
        pd.DataFrame(grad_log).to_csv(os.path.join(a.out, "grad_norms.csv"), index=False)

    # how far did the pretrained weights move? (final weights vs the pretrained checkpoint)
    drift = {ast_name(n, a.freeze_k): float((p.detach() - w_before[n]).norm() / (w_before[n].norm() + 1e-12))
             for n, p in model.named_parameters() if n in w_before}
    ck = torch.load(os.path.join(a.out, "best.pt"), weights_only=False)
    model.load_state_dict(ck["state_dict"])
    pt, et = predict(pos["test"], return_emb=True)
    np.save(os.path.join(a.out, "test_embeddings.npy"), et.astype(np.float16))
    res = metrics(y[pos["test"]].numpy(), pt.argmax(1), len(CLASSES))
    seen = np.concatenate(seen)
    res.update(dict(best_epoch=ck["epoch"], best_val_macro_f1=best, trainable_params=n_tr, train_seconds=time.time() - t0,
                    max_relative_weight_change=max(drift.values()) if drift else 0.0,
                    mean_relative_weight_change=float(np.mean(list(drift.values()))) if drift else 0.0,
                    distinct_sources_seen=int(dtr.group.iloc[np.unique(seen)].nunique()), **vars(a)))
    pd.Series(drift).rename("relative_change").to_csv(os.path.join(a.out, "weight_drift.csv"))
    p = info.iloc[pos["test"]][["filename", "y", "cls", "group", "source"]].reset_index(drop=True)
    p["pred"] = pt.argmax(1)
    for i, c in enumerate(CLASSES):
        p[f"p_{c}"] = pt[:, i]
    p.to_csv(os.path.join(a.out, "test_predictions.csv"), index=False)
    # test_metrics.json is written last: its presence marks a complete run
    json.dump(res, open(os.path.join(a.out, "test_metrics.json.tmp"), "w"), indent=1)
    os.replace(os.path.join(a.out, "test_metrics.json.tmp"), os.path.join(a.out, "test_metrics.json"))
    print(f"TEST: acc {res['accuracy']:.3f} macroF1 {res['macro_f1']:.3f} bal_acc {res['balanced_accuracy']:.3f}; "
          f"max relative change of pretrained weights {res['max_relative_weight_change']:.4f}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("cache")
    c.add_argument("--data", required=True)
    c.add_argument("--audio-root", required=True)
    c.add_argument("--protocol", default="source_disjoint")
    c.add_argument("--per-class-train", type=int, default=200)
    c.add_argument("--freeze-k", type=int, default=11)
    c.add_argument("--batch", type=int, default=8)
    c.add_argument("--out", required=True)
    t = sub.add_parser("train")
    t.add_argument("--data", required=True)
    t.add_argument("--cache", required=True)
    t.add_argument("--freeze-k", type=int, default=11)
    t.add_argument("--mode", choices=["frozen", "finetune"], default="finetune")
    t.add_argument("--sampler", choices=["standard", "source"], default="standard")
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--epochs", type=int, default=3)
    t.add_argument("--samples-per-epoch", type=int, default=768)
    t.add_argument("--batch", type=int, default=16)
    t.add_argument("--lr", type=float, default=3e-5)
    t.add_argument("--head-lr", type=float, default=1e-3)
    t.add_argument("--wd", type=float, default=1e-2)
    t.add_argument("--supcon", type=float, default=0.0, help="weight of the cross-source contrastive term (0 = off)")
    t.add_argument("--tau", type=float, default=0.1)
    t.add_argument("--proj-dim", type=int, default=128)
    t.add_argument("--out", required=True)
    a = ap.parse_args()
    cmd_cache(a) if a.cmd == "cache" else cmd_train(a)


if __name__ == "__main__":
    main()
