"""Load trained checkpoints and classify a raw Nonspeech7k waveform (used by the demo notebook).

Scores returned are RAW SOFTMAX outputs of each network. They are not calibrated probabilities: no temperature
scaling or other calibration was fitted, so a 0.9 score does not mean the model is right 90% of the time.
"""
import time

import numpy as np
import torch
import torch.nn.functional as F
from scipy.signal import resample_poly

from dl_data import CLASSES, logmel, mel_filters


class CNNClassifier:
    def __init__(self, ckpt_path):
        from train_cnn import LogMelCNN
        ck = torch.load(ckpt_path, weights_only=False, map_location="cpu")
        self.model = LogMelCNN(len(ck["classes"]), ck["width"]).eval()
        self.model.load_state_dict(ck["state_dict"])
        self.mu, self.sd, self.fb, self.info = ck["mu"], ck["sd"], mel_filters(), ck
        self.n_params = sum(p.numel() for p in self.model.parameters())

    def features(self, wave32k):
        return logmel(wave32k, self.fb)

    @torch.no_grad()
    def predict(self, wave32k):
        t = time.perf_counter()
        x = torch.from_numpy((self.features(wave32k) - self.mu) / self.sd)[None]
        s = F.softmax(self.model(x), 1)[0].numpy()
        return s, time.perf_counter() - t


class ASTClassifier:
    """Pretrained AST layers 0..K-1 (frozen) + the trained top (layers K..11, LayerNorm, head) from a run."""

    def __init__(self, ckpt_path):
        from ast_finetune import ASTTop, load_ast
        ck = torch.load(ckpt_path, weights_only=False, map_location="cpu")
        self.fe, self.ast = load_ast()
        self.k = ck["freeze_k"]
        self.top = ASTTop(self.ast, self.k, len(ck["classes"])).eval()
        self.top.load_state_dict(ck["state_dict"])
        self.info = ck
        self.n_params = sum(p.numel() for p in self.ast.parameters()) + sum(p.numel() for p in self.top.head.parameters())

    @torch.no_grad()
    def predict(self, wave32k):
        from ast_finetune import embed
        t = time.perf_counter()
        x = self.fe([resample_poly(wave32k, 1, 2).astype(np.float32)], sampling_rate=16000, return_tensors="pt")["input_values"]
        h = embed(self.ast, x)
        for layer in self.ast.layers[:self.k]:
            h = layer(h, None)
        s = F.softmax(self.top(h), 1)[0].numpy()
        return s, time.perf_counter() - t


__all__ = ["CNNClassifier", "ASTClassifier", "CLASSES"]
