"""Classification metrics and a source-grouped paired bootstrap shared by the deep-learning scripts."""
import numpy as np


def confusion(y, p, k):
    c = np.zeros((k, k), int)
    np.add.at(c, (y, p), 1)
    return c


def metrics(y, p, k):
    c = confusion(np.asarray(y), np.asarray(p), k)
    tp = np.diag(c).astype(float)
    sup, pred = c.sum(1), c.sum(0)
    rec = np.divide(tp, sup, out=np.zeros(k), where=sup > 0)
    f1 = np.divide(2 * tp, sup + pred, out=np.zeros(k), where=(sup + pred) > 0)
    present = sup > 0
    return dict(accuracy=float(tp.sum() / c.sum()), macro_f1=float(f1[present].mean()),
                balanced_accuracy=float(rec[present].mean()), per_class_recall=rec.tolist(),
                per_class_f1=f1.tolist(), confusion=c.tolist())


def macro_f1(y, p, k):
    return metrics(y, p, k)["macro_f1"]


def grouped_bootstrap(y, preds_a, preds_b, groups, k, n=2000, seed=12345):
    """Paired bootstrap over source groups (the unit of independence). preds_a / preds_b: lists of prediction
    arrays (one per training seed); each resample scores every seed on the same resampled clips and averages
    over seeds. Returns (point_a, point_b, diff, ci_a, ci_b, ci_diff)."""
    y, groups = np.asarray(y), np.asarray(groups)
    uniq = np.unique(groups)
    pos = {g: np.where(groups == g)[0] for g in uniq}
    rng = np.random.default_rng(seed)

    def score(ix, P):
        return float(np.mean([macro_f1(y[ix], p[ix], k) for p in P]))

    allix = np.arange(len(y))
    A, B = score(allix, preds_a), score(allix, preds_b) if preds_b else np.nan
    ba, bb = [], []
    for _ in range(n):
        ix = np.concatenate([pos[g] for g in rng.choice(uniq, len(uniq))])
        ba.append(score(ix, preds_a))
        if preds_b:
            bb.append(score(ix, preds_b))
    ba = np.array(ba)
    ci = lambda v: tuple(np.percentile(v, [2.5, 97.5]))  # noqa: E731
    if not preds_b:
        return A, None, None, ci(ba), None, None
    bb = np.array(bb)
    return A, B, B - A, ci(ba), ci(bb), ci(bb - ba)


def source_nn_rate(E, y, g):
    """Source-clustering diagnostic for embeddings of ONE evaluation partition (here the source-disjoint test set).

    For every clip, its nearest neighbour by cosine similarity is searched among the OTHER clips of the same
    partition with the SAME true class (reference pool = same-class clips of that partition; the clip itself is
    excluded). The diagnostic is the share of clips whose nearest neighbour comes from the same source group.
    Source groups are sources.py keys with confirmed duplicates merged, so a re-upload counts as the same source.

    chance = expected share if the neighbour were drawn uniformly from the reference pool
           = mean over clips of (same-class clips in its group - 1) / (same-class clips - 1).
    'eligible' restricts both numbers to clips that have at least one same-source AND at least one other-source
    clip in their reference pool (only for these can the neighbour go either way).
    Returns dict(rate, chance, rate_eligible, chance_eligible, n, n_eligible)."""
    E = np.asarray(E, np.float64)
    E = E / np.linalg.norm(E, axis=1, keepdims=True)
    y, g = np.asarray(y), np.asarray(g)
    same, chance, elig = [], [], []
    for k in np.unique(y):
        ix = np.where(y == k)[0]
        if len(ix) < 2:
            continue
        S = E[ix] @ E[ix].T
        np.fill_diagonal(S, -np.inf)  # self-match excluded
        gg = g[ix]
        n_same = np.array([(gg == x).sum() - 1 for x in gg])
        same.append(gg[S.argmax(1)] == gg)
        chance.append(n_same / (len(ix) - 1))
        elig.append((n_same > 0) & (n_same < len(ix) - 1))
    same, chance, elig = map(np.concatenate, (same, chance, elig))
    return dict(rate=float(same.mean()), chance=float(chance.mean()),
                rate_eligible=float(same[elig].mean()) if elig.any() else np.nan,
                chance_eligible=float(chance[elig].mean()) if elig.any() else np.nan,
                n=int(len(same)), n_eligible=int(elig.sum()))
