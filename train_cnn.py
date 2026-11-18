"""Train a log-mel CNN from scratch on Nonspeech7k (PyTorch, CPU or GPU).

Two sampling conditions, everything else identical (architecture, initialisation for a given seed, augmentation,
optimizer, batch size, number of optimizer steps, validation-based checkpoint selection):
  standard : class-balanced sampling, uniform over CLIPS within a class
             (a source with 800 clips is drawn 800x as often as a source with 1 clip)
  source   : class-balanced sampling, uniform over SOURCES within a class, then uniform over that source's clips
             (the proposed source-aware intervention)
Both draw with replacement, so they see the same number of examples; they differ only in which.

  python train_cnn.py --data results/v3/dl --protocol source_disjoint --sampler source --seed 0 --out runs/cnn_source_s0
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


class ConvBlock(nn.Module):
    def __init__(self, cin, cout, convs=2):
        super().__init__()
        layers = []
        for i in range(convs):
            layers += [nn.Conv2d(cin if i == 0 else cout, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout),
                       nn.ReLU(inplace=True)]
        self.net = nn.Sequential(*layers, nn.MaxPool2d(2))

    def forward(self, x):
        return self.net(x)


class LogMelCNN(nn.Module):
    """Input log-mel (64 x 400 frames, 10 ms hop) is average-pooled to 20 ms frames; then 4 conv blocks
    (conv3x3-BN-ReLU, x1 in the first block and x2 after, each followed by 2x2 max-pool), global mean+max pooling
    over time, dropout and a linear head."""

    def __init__(self, n_classes=7, width=16, dropout=0.3):
        super().__init__()
        ch = [1, width, width * 2, width * 4, width * 8]
        self.features = nn.Sequential(*[ConvBlock(ch[i], ch[i + 1], 1 if i == 0 else 2) for i in range(4)])
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(2 * ch[-1], n_classes)

    def forward(self, x):  # x: (B, mels, frames)
        h = self.features(F.avg_pool2d(x.unsqueeze(1), (1, 2)))
        h = h.mean(dim=2)  # pool frequency
        h = torch.cat([h.mean(dim=2), h.amax(dim=2)], dim=1)  # pool time: mean + max
        return self.head(self.drop(h))


def sample_weights(df, mode):
    n_cls = df.groupby("y").filename.transform("size")
    if mode == "standard":
        w = 1.0 / n_cls
    elif mode == "source":
        n_src = df.groupby("y").group.transform("nunique")
        n_clips_in_src = df.groupby(["y", "group"]).filename.transform("size")
        w = 1.0 / (n_src * n_clips_in_src)
    else:
        raise ValueError(mode)
    return torch.tensor((w / w.sum()).values, dtype=torch.double)


def augment(x, gen):
    """Random circular time shift (up to 0.5 s) + one frequency mask (<=8 bands) + one time mask (<=40 frames)."""
    B, M, T = x.shape
    x = x.clone()
    for i in range(B):
        x[i] = torch.roll(x[i], int(torch.randint(-50, 51, (1,), generator=gen)), dims=1)
        f = int(torch.randint(0, 9, (1,), generator=gen)); f0 = int(torch.randint(0, M - f + 1, (1,), generator=gen))
        t = int(torch.randint(0, 41, (1,), generator=gen)); t0 = int(torch.randint(0, T - t + 1, (1,), generator=gen))
        x[i, f0:f0 + f, :] = 0.0
        x[i, :, t0:t0 + t] = 0.0
    return x


@torch.no_grad()
def predict(model, X, batch=128):
    model.eval()
    out = [F.softmax(model(X[i:i + batch]), dim=1) for i in range(0, len(X), batch)]
    return torch.cat(out).numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="folder with splits.csv and logmel.npy (dl_data.py)")
    ap.add_argument("--protocol", default="source_disjoint")
    ap.add_argument("--sampler", choices=["standard", "source"], default="standard")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--samples-per-epoch", type=int, default=2048)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--wd", type=float, default=1e-2)
    ap.add_argument("--width", type=int, default=16)
    ap.add_argument("--patience", type=int, default=0, help="0 = no early stopping (fixed budget)")
    ap.add_argument("--train-limit", type=int, default=0, help="debug: use only N training clips")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)

    sp = pd.read_csv(os.path.join(a.data, "splits.csv"))
    M = np.load(os.path.join(a.data, "logmel.npy"), mmap_mode="r")
    part = sp[a.protocol].values
    tr, va, te = (np.where(part == p)[0] for p in ("train", "val", "test"))
    if a.train_limit:
        tr = np.random.default_rng(a.seed).choice(tr, a.train_limit, replace=False)
    mu = np.asarray(M[tr], np.float32).mean(axis=(0, 2), keepdims=True)[0]  # per-mel, train only
    sd = np.asarray(M[tr], np.float32).std(axis=(0, 2), keepdims=True)[0] + 1e-5

    def tensor(ix):
        return torch.from_numpy((np.asarray(M[ix], np.float32) - mu) / sd)

    Xtr, Xva, Xte = tensor(tr), tensor(va), tensor(te)
    y = sp["y"].values
    ytr, yva, yte = (torch.from_numpy(y[ix]).long() for ix in (tr, va, te))
    dtr = sp.iloc[tr].reset_index(drop=True)

    torch.manual_seed(a.seed)  # identical initial weights for both samplers at a given seed
    model = LogMelCNN(len(CLASSES), a.width)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.wd)
    steps_per_epoch = a.samples_per_epoch // a.batch
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=a.epochs * steps_per_epoch)
    w = sample_weights(dtr, a.sampler)
    gen = torch.Generator().manual_seed(1000 + a.seed)  # same RNG stream for both samplers
    print(model)
    print(f"trainable parameters: {n_params:,}; train {len(tr)} clips / {dtr.group.nunique()} sources; "
          f"val {len(va)}; test {len(te)}; sampler {a.sampler}", flush=True)

    hist, best, bad, seen = [], -1.0, 0, []
    t0 = time.time()
    for ep in range(1, a.epochs + 1):
        model.train()
        draw = torch.multinomial(w, steps_per_epoch * a.batch, replacement=True, generator=gen)
        seen.append(draw.numpy())
        tot, n = 0.0, 0
        for s in range(steps_per_epoch):
            b = draw[s * a.batch:(s + 1) * a.batch]
            xb = augment(Xtr[b], gen)
            loss = F.cross_entropy(model(xb), ytr[b])
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            tot += loss.item() * len(b); n += len(b)
        pv = predict(model, Xva)
        vloss = float(F.nll_loss(torch.log(torch.from_numpy(pv) + 1e-9), yva))
        m = metrics(yva.numpy(), pv.argmax(1), len(CLASSES))
        hist.append(dict(epoch=ep, train_loss=tot / n, val_loss=vloss, val_acc=m["accuracy"],
                         val_macro_f1=m["macro_f1"], val_bal_acc=m["balanced_accuracy"], lr=sched.get_last_lr()[0],
                         elapsed_s=time.time() - t0))
        print(f"epoch {ep:2d} train_loss {tot / n:.3f} val_loss {vloss:.3f} val_acc {m['accuracy']:.3f} "
              f"val_macroF1 {m['macro_f1']:.3f} ({time.time() - t0:.0f}s)", flush=True)
        if m["macro_f1"] > best:
            best, bad = m["macro_f1"], 0
            torch.save(dict(state_dict=model.state_dict(), width=a.width, classes=CLASSES, mu=mu, sd=sd,
                            epoch=ep, args=vars(a)), os.path.join(a.out, "best.pt"))
        else:
            bad += 1
            if a.patience and bad >= a.patience:
                print("early stop", flush=True)
                break
    pd.DataFrame(hist).to_csv(os.path.join(a.out, "history.csv"), index=False)

    ck = torch.load(os.path.join(a.out, "best.pt"), weights_only=False)
    model.load_state_dict(ck["state_dict"])
    pt = predict(model, Xte)
    res = metrics(yte.numpy(), pt.argmax(1), len(CLASSES))
    seen = np.concatenate(seen)
    top5 = dtr.group.value_counts().index[:5]
    exposure = dict(distinct_clips_seen=int(len(np.unique(seen))), distinct_sources_seen=int(dtr.group.iloc[np.unique(seen)].nunique()),
                    share_draws_top5_sources=float(dtr.group.iloc[seen].isin(top5).mean()),
                    share_clips_top5_sources=float(dtr.group.isin(top5).mean()))
    res.update(dict(best_epoch=ck["epoch"], best_val_macro_f1=best, trainable_params=n_params,
                    train_seconds=time.time() - t0, **exposure, **{k: v for k, v in vars(a).items()}))
    json.dump(res, open(os.path.join(a.out, "test_metrics.json"), "w"), indent=1)
    p = sp.iloc[te][["filename", "y", "cls", "group", "source"]].reset_index(drop=True)
    p["pred"] = pt.argmax(1)
    for i, c in enumerate(CLASSES):
        p[f"p_{c}"] = pt[:, i]
    p.to_csv(os.path.join(a.out, "test_predictions.csv"), index=False)
    print(f"TEST ({a.protocol}, best epoch {ck['epoch']}): acc {res['accuracy']:.3f} macroF1 {res['macro_f1']:.3f} "
          f"bal_acc {res['balanced_accuracy']:.3f}; exposure {exposure}", flush=True)


if __name__ == "__main__":
    main()
