"""Temporal-structure features (burst count/rhythm, modulation spectrum, envelope shape) and a
spectral control set (MFCC + spectral stats). numpy/scipy only.

Structure feature names carry a prefix so groups can be ablated:
  dur_   duration / activity
  nf_    noise-floor-sensitive (dynamic range, mean level) -- may leak recording conditions
  env_   envelope shape
  burst_ burst counting, inter-burst interval, attack/decay
  acf_   envelope periodicity
  mod_   modulation spectrum (4 bands x 5 modulation ranges)
"""
import numpy as np
import soundfile as sf
from scipy import signal
from scipy.fft import dct
from scipy.stats import kurtosis, skew

HOP_S, WIN_S, EPS = 0.010, 0.025, 1e-10
BANDS_HZ = [(100, 500), (500, 1500), (1500, 4000), (4000, 10000)]
MOD_EDGES = [0.5, 1.0, 2.0, 4.0, 8.0, 16.0]


def load_mono(path):
    x, sr = sf.read(path, dtype="float32", always_2d=True)
    return x.mean(axis=1), sr


def _smooth(a, k=5):
    return np.convolve(a, np.ones(k) / k, mode="same")


def _stft_power(x, sr):
    nper, hop = int(WIN_S * sr), int(HOP_S * sr)
    if len(x) < nper + hop:
        x = np.pad(x, (0, nper + hop - len(x)))
    f, _, Z = signal.stft(x, fs=sr, nperseg=nper, noverlap=nper - hop, window="hann",
                          boundary=None, padded=False)
    return f, np.abs(Z) ** 2


def _rise_decay(env_db, pk, drop=10.0, max_frames=100):
    top, i, j = env_db[pk], pk, pk
    while i > 0 and pk - i < max_frames and env_db[i] > top - drop:
        i -= 1
    while j < len(env_db) - 1 and j - pk < max_frames and env_db[j] > top - drop:
        j += 1
    return (pk - i) * HOP_S, (j - pk) * HOP_S


def struct_features(x, sr):
    f, P = _stft_power(x, sr)
    n = P.shape[1]
    frame_p = _smooth(P.sum(0))
    amp = np.sqrt(frame_p)
    env_db = 10 * np.log10(frame_p + EPS)
    env_db = env_db - env_db.max()  # level-invariant: relative to the loudest frame
    out = {}

    active = env_db > -20.0
    out["dur_s"] = len(x) / sr
    out["dur_active_s"] = active.sum() * HOP_S
    out["dur_active_frac"] = active.mean()
    out["nf_dyn_range_db"] = np.percentile(env_db, 95) - np.percentile(env_db, 5)
    out["nf_mean_db"] = env_db.mean()

    out["env_skew"] = float(skew(amp)) if amp.std() > 0 else 0.0
    out["env_kurt"] = float(kurtosis(amp)) if amp.std() > 0 else 0.0
    out["env_tcentroid"] = (np.arange(n) * amp).sum() / (amp.sum() + EPS) / n
    p = frame_p / (frame_p.max() + EPS)
    out["env_flatness"] = np.exp(np.mean(np.log(p + EPS))) / (p.mean() + EPS)

    pk, props = signal.find_peaks(env_db, prominence=6.0, distance=8)
    dur = len(x) / sr
    out["burst_n"] = len(pk)
    out["burst_per_s"] = len(pk) / dur
    ibi = np.diff(pk) * HOP_S
    out["burst_ibi_mean"] = ibi.mean() if len(ibi) else dur
    out["burst_ibi_std"] = ibi.std() if len(ibi) > 1 else 0.0
    out["burst_ibi_cv"] = ibi.std() / (ibi.mean() + EPS) if len(ibi) > 1 else 0.0
    if len(pk):
        top = pk[np.argsort(props["prominences"])[::-1][:5]]
        rd = np.array([_rise_decay(env_db, q) for q in top])
        out["burst_attack_s"], out["burst_decay_s"] = rd.mean(axis=0)
    else:
        out["burst_attack_s"] = out["burst_decay_s"] = 0.0

    e = amp - amp.mean()
    ac = np.correlate(e, e, "full")[len(e) - 1:]
    ac = ac / (ac[0] + EPS)
    lo, hi = 10, min(100, len(ac) - 2)
    acf_peak, acf_lag = 0.0, 0.0
    if hi > lo:
        cand, _ = signal.find_peaks(ac[lo:hi + 1])
        if len(cand):
            best = cand[np.argmax(ac[lo:hi + 1][cand])]
            acf_peak, acf_lag = float(ac[lo + best]), (lo + best) * HOP_S
    out["acf_peak"], out["acf_lag_s"] = acf_peak, acf_lag

    nyq = sr / 2 * 0.95
    mod_f = np.fft.rfftfreq(512, HOP_S)
    win = np.hanning(min(n, 512))
    for b, (blo, bhi) in enumerate(BANDS_HZ):
        bhi = min(bhi, nyq)
        mask = (f >= blo) & (f < bhi)
        if blo >= bhi or not mask.any():
            vals = np.zeros(len(MOD_EDGES) - 1)
        else:
            be = np.sqrt(P[mask].sum(0))[:512]
            be = be / (be.mean() + EPS) - 1.0
            S = np.abs(np.fft.rfft(be * win[:len(be)], 512)) ** 2
            tot = S[(mod_f >= MOD_EDGES[0]) & (mod_f < MOD_EDGES[-1])].sum() + EPS
            vals = np.array([S[(mod_f >= a) & (mod_f < c)].sum() / tot
                             for a, c in zip(MOD_EDGES[:-1], MOD_EDGES[1:])])
        for k, v in enumerate(vals):
            out[f"mod_b{b}_{MOD_EDGES[k]:g}-{MOD_EDGES[k + 1]:g}hz"] = v
    return {k: float(np.nan_to_num(v)) for k, v in out.items()}


def _mel_fb(f, n_mels=40, fmin=50.0, fmax=8000.0):
    fmax = min(fmax, f[-1])
    hz2mel = lambda h: 2595 * np.log10(1 + h / 700)
    mel2hz = lambda m: 700 * (10 ** (m / 2595) - 1)
    pts = mel2hz(np.linspace(hz2mel(fmin), hz2mel(fmax), n_mels + 2))
    fb = np.zeros((n_mels, len(f)))
    for i in range(n_mels):
        l, c, r = pts[i:i + 3]
        fb[i] = np.clip(np.minimum((f - l) / (c - l), (r - f) / (r - c)), 0, None)
    return fb


def spec_features(x, sr):
    """Spectral control set: MFCC 1-13 (c0 dropped, gain-invariant) mean/std + spectral stats."""
    f, P = _stft_power(x, sr)
    env_db = 10 * np.log10(_smooth(P.sum(0)) + EPS)
    keep = env_db > env_db.max() - 30.0
    P = P[:, keep] if keep.sum() >= 3 else P
    mfcc = dct(np.log(_mel_fb(f) @ P + EPS), axis=0, norm="ortho")[1:14]
    tot = P.sum(0) + EPS
    cen = (f[:, None] * P).sum(0) / tot
    bw = np.sqrt((((f[:, None] - cen) ** 2) * P).sum(0) / tot)
    roll = f[np.argmax(np.cumsum(P, 0) >= 0.85 * tot, axis=0)]
    flat = np.exp(np.mean(np.log(P + EPS), 0)) / (P.mean(0) + EPS)
    out = {}
    for i in range(13):
        out[f"mfcc{i + 1}_mean"], out[f"mfcc{i + 1}_std"] = mfcc[i].mean(), mfcc[i].std()
    for name, v in [("centroid", cen), ("bandwidth", bw), ("rolloff", roll), ("flatness", flat)]:
        out[f"{name}_mean"], out[f"{name}_std"] = v.mean(), v.std()
    out["zcr"] = float(np.mean(np.abs(np.diff(np.sign(x))) > 0))
    return {k: float(np.nan_to_num(v)) for k, v in out.items()}


COMPUTERS = {"struct": struct_features, "spec": spec_features}
