"""Generate the two demo notebooks (run from anywhere): python professor_demo/make_notebooks.py"""
import os

import nbformat as nbf

HERE = os.path.dirname(os.path.abspath(__file__))
md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell

SETUP = r'''import os, sys, time
from pathlib import Path
import numpy as np, pandas as pd, torch, matplotlib.pyplot as plt
REPO = Path.cwd().parent if Path.cwd().name == "professor_demo" else Path.cwd()
sys.path.insert(0, str(REPO))
# data and results live next to the repository; override with environment variables if moved
DATA = Path(os.environ.get("NS7K_DATA", REPO.parent / "data"))
RESULTS = Path(os.environ.get("NS7K_RESULTS", REPO.parent / "results" / "v3"))
from common import index_audio
from features import load_mono
from dl_data import CLASSES
print("torch", torch.__version__, "| CUDA available:", torch.cuda.is_available(), "| CPU threads:", torch.get_num_threads())
audio = index_audio(DATA / "audio")
splits = pd.read_csv(RESULTS / "dl" / "splits.csv")
print(len(audio), "audio files indexed;", len(splits), "clips in the split manifest")'''

demo = nbf.v4.new_notebook()
demo.cells = [
    md("# Live deep-learning demo: Nonspeech7k vocalisation recognition\n\n"
       "Every number below is computed live from real audio and saved checkpoints; nothing is hard-coded.\n\n"
       "* **CNN**: log-mel CNN trained from scratch (`train_cnn.py`).\n"
       "* **AST**: pretrained Audio Spectrogram Transformer (`MIT/ast-finetuned-audioset-10-10-0.4593`) with its "
       "last transformer block fine-tuned and a new 7-class head (`ast_finetune.py`).\n\n"
       "Test clips are drawn from the **source-disjoint test partition**: no recording they were cut from "
       "(and no duplicated copy of it) was seen in training or validation.\n\n"
       "Scores shown are **raw softmax outputs, not calibrated probabilities**."),
    code(SETUP),
    md("## 1. Pick a real test recording"),
    code(r'''test = splits[splits.source_disjoint == "test"]
rng = np.random.default_rng(int(os.environ.get("DEMO_SEED", 3)))
row = test.iloc[rng.integers(len(test))]
wave, sr = load_mono(audio[row.filename])
print(f"file {row.filename} | true class: {row.cls} | website: {row.source} | source group: {row.group}")
print(f"{len(wave)/sr:.2f} s at {sr} Hz")'''),
    md("## 2. Play it"),
    code(r'''from IPython.display import Audio, display
display(Audio(wave, rate=sr))'''),
    md("## 3. Waveform and log-mel spectrogram (the CNN's input)"),
    code(r'''from inference import CNNClassifier, ASTClassifier
cnn_ck = RESULTS / "runs" / "cnn_source_disjoint_standard_s0" / "best.pt"
cnn = CNNClassifier(cnn_ck)
M = cnn.features(wave)
fig, ax = plt.subplots(2, 1, figsize=(10, 5))
ax[0].plot(np.arange(len(wave)) / sr, wave, lw=.5); ax[0].set(title=f"waveform: {row.filename} ({row.cls})", xlabel="s")
ax[1].imshow(M, origin="lower", aspect="auto", extent=[0, 4, 0, M.shape[0]]); ax[1].set(title="log-mel (64 bands, 10 ms hop, 4 s window)", xlabel="s", ylabel="mel band")
plt.tight_layout(); plt.show()'''),
    md("## 4. CNN inference"),
    code(r'''print(cnn.model)
print(f"trainable parameters: {cnn.n_params:,}; checkpoint epoch {cnn.info['epoch']} (selected on validation macro-F1)")
s_cnn, t_cnn = cnn.predict(wave)
print(f"CNN prediction: {CLASSES[s_cnn.argmax()]} (true: {row.cls}) | inference {1000*t_cnn:.1f} ms")
pd.Series(s_cnn, index=CLASSES).round(3).rename("raw softmax score").to_frame()'''),
    md("## 5. Fine-tuned AST inference"),
    code(r'''ast_ck = RESULTS / "runs" / "ast_ft_source_disjoint_standard_s0" / "best.pt"
astc = ASTClassifier(ast_ck)
print(f"AST: {astc.n_params:,} parameters; layers 0..{astc.k-1} frozen pretrained, layer {astc.k}..11 fine-tuned, new head")
s_ast, t_ast = astc.predict(wave)
print(f"AST prediction: {CLASSES[s_ast.argmax()]} (true: {row.cls}) | inference {1000*t_ast:.0f} ms on CPU")
fig, ax = plt.subplots(figsize=(8, 3))
x = np.arange(len(CLASSES))
ax.bar(x - .2, s_cnn, .4, label="CNN"); ax.bar(x + .2, s_ast, .4, label="AST fine-tuned")
ax.set_xticks(x, CLASSES); ax.set_ylabel("raw softmax score (not calibrated)"); ax.set_title(f"true class: {row.cls}"); ax.legend(); plt.show()'''),
    md("## 6. Compare both models on several test recordings (one per class)"),
    code(r'''rows = []
for c in CLASSES:
    r = test[test.cls == c].sample(1, random_state=int(os.environ.get("DEMO_SEED", 3))).iloc[0]
    w, _ = load_mono(audio[r.filename])
    a, ta = cnn.predict(w); b, tb = astc.predict(w)
    rows.append(dict(file=r.filename, website=r.source, true=c, cnn=CLASSES[a.argmax()], cnn_score=a.max(),
                     ast=CLASSES[b.argmax()], ast_score=b.max(), cnn_ms=1000*ta, ast_ms=1000*tb))
res = pd.DataFrame(rows)
print("CNN correct:", (res.cnn == res.true).sum(), "/", len(res), "| AST correct:", (res.ast == res.true).sum(), "/", len(res))
res.round(3)'''),
    md("## 7. Saved full-test-set results (computed offline by `dl_report.py`)"),
    code(r'''tab = pd.read_csv(RESULTS / "report" / "system_comparison.csv")
tab[tab.evaluation == "source-disjoint test"][["system", "seeds", "test_clips", "test_sources", "macro_f1", "ci_lo", "ci_hi", "accuracy"]].round(3)'''),
]

train = nbf.v4.new_notebook()
train.cells = [
    md("# Live training demo: a CNN learning from Nonspeech7k\n\n"
       "Shows the full deep-learning loop on a **small subset of the training partition** (no test clips): "
       "initialisation, forward pass, loss, backpropagation, optimizer step, parameter updates, validation. "
       "The real experiments (`train_cnn.py`, `ast_finetune.py`) use the same code at full scale."),
    code(SETUP),
    code(r'''import torch.nn.functional as F
from train_cnn import LogMelCNN, augment
from dl_metrics import metrics
torch.manual_seed(0)
M = np.load(RESULTS / "dl" / "logmel.npy", mmap_mode="r")
tr = splits.index[splits.source_disjoint == "train"].values
va = splits.index[splits.source_disjoint == "val"].values
rng = np.random.default_rng(0)
tr_s = np.sort(np.concatenate([rng.choice(tr[splits.y.values[tr] == c], 40, replace=False) for c in range(7)]))
va_s = np.sort(np.concatenate([rng.choice(va[splits.y.values[va] == c], 15, replace=False) for c in range(7)]))
mu = np.asarray(M[tr_s], np.float32).mean((0, 2), keepdims=True)[0]; sd = np.asarray(M[tr_s], np.float32).std((0, 2), keepdims=True)[0] + 1e-5
Xtr = torch.from_numpy((np.asarray(M[tr_s], np.float32) - mu) / sd); ytr = torch.tensor(splits.y.values[tr_s])
Xva = torch.from_numpy((np.asarray(M[va_s], np.float32) - mu) / sd); yva = torch.tensor(splits.y.values[va_s])
print("train subset", Xtr.shape, "| val subset", Xva.shape, "| test clips used: 0")'''),
    md("## 1. Initialise the network"),
    code(r'''model = LogMelCNN(7, 16)
opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-2)
print(model)
print("trainable parameters:", sum(p.numel() for p in model.parameters() if p.requires_grad))'''),
    md("## 2. One step by hand: forward, loss, backward, optimizer step"),
    code(r'''xb, yb = Xtr[:16], ytr[:16]
w0 = model.head.weight.detach().clone(); c0 = model.features[0].net[0].weight.detach().clone()
logits = model(xb)                                  # forward propagation
loss = F.cross_entropy(logits, yb)                  # loss
opt.zero_grad(); loss.backward()                    # backpropagation
print("logits shape", tuple(logits.shape), "| loss", round(loss.item(), 4))
print("gradient norm, first conv:", model.features[0].net[0].weight.grad.norm().item())
print("gradient norm, head      :", model.head.weight.grad.norm().item())
opt.step()                                          # optimizer step
print("first-conv weights changed by", (model.features[0].net[0].weight - c0).norm().item())
print("head weights changed by      ", (model.head.weight - w0).norm().item())'''),
    md("## 3. Short training run with validation after every epoch"),
    code(r'''hist = []; gen = torch.Generator().manual_seed(0)
t0 = time.time()
for ep in range(1, 9):
    model.train(); perm = torch.randperm(len(Xtr), generator=gen); tot = 0
    for i in range(0, len(perm), 16):
        b = perm[i:i+16]
        loss = F.cross_entropy(model(augment(Xtr[b], gen)), ytr[b])
        opt.zero_grad(); loss.backward(); opt.step(); tot += loss.item() * len(b)
    model.eval()
    with torch.no_grad():
        lv = model(Xva); vl = F.cross_entropy(lv, yva).item()
    m = metrics(yva.numpy(), lv.argmax(1).numpy(), 7)
    hist.append(dict(epoch=ep, train_loss=tot/len(Xtr), val_loss=vl, val_acc=m["accuracy"], val_macro_f1=m["macro_f1"]))
    print(f"epoch {ep} train_loss {tot/len(Xtr):.3f} val_loss {vl:.3f} val_acc {m['accuracy']:.3f} val_macroF1 {m['macro_f1']:.3f} ({time.time()-t0:.0f}s)")
h = pd.DataFrame(hist)
fig, ax = plt.subplots(1, 2, figsize=(10, 3.5))
ax[0].plot(h.epoch, h.train_loss, label="train"); ax[0].plot(h.epoch, h.val_loss, label="val"); ax[0].set(title="loss", xlabel="epoch"); ax[0].legend()
ax[1].plot(h.epoch, h.val_acc, label="val accuracy"); ax[1].plot(h.epoch, h.val_macro_f1, label="val macro-F1"); ax[1].set(title="validation", xlabel="epoch", ylim=(0, 1)); ax[1].legend()
plt.show()'''),
    md("A 280-clip, 8-epoch run is only a demonstration of the training loop; the full experiment's learning "
       "curves are in `results/v3/report/G1_G2_learning_curves_*.png`.\n\n"
       "## 4. Evidence that the pretrained AST was actually fine-tuned\n"
       "Gradient norms on pretrained transformer weights during training, and how far those weights moved."),
    code(r'''run = RESULTS / "runs" / "ast_ft_source_disjoint_standard_s0"
print(open(run / "train.log").read().splitlines()[-1])
display(pd.read_csv(run / "grad_norms.csv").T.head(12))
drift = pd.read_csv(run / "weight_drift.csv", index_col=0)
drift.sort_values("relative_change", ascending=False).head(10)'''),
]

for nb, name in [(demo, "LIVE_DL_DEMO.ipynb"), (train, "LIVE_TRAINING_DEMO.ipynb")]:
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    nbf.write(nb, os.path.join(HERE, name))
    print("wrote", os.path.join(HERE, name))
