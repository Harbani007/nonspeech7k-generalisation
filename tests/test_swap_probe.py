import os
import sys
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from swap_probe import (attraction, bg_level, compose, decide, fit_length, group_bootstrap,  # noqa: E402
                        macro_recall, match_scale, pair_donors)


def _mask(n, a, b):
    g = np.zeros(n, np.float32)
    g[a:b] = 1
    return g


class SignalTest(unittest.TestCase):
    def setUp(self):
        r = np.random.default_rng(0)
        self.x, self.d = r.normal(0, 0.1, 1600).astype(np.float32), r.normal(0, 0.5, 2000).astype(np.float32)
        self.g, self.gd = _mask(1600, 400, 800), _mask(2000, 100, 600)

    def test_fit_length(self):
        self.assertEqual(len(fit_length(self.d, 1600)), 1600)
        p = fit_length(self.d[:1000], 1600)
        self.assertEqual(len(p), 1600)
        self.assertTrue(np.all(p[1000:] == 0))

    def test_target_vocalisation_is_unchanged(self):
        s, _ = match_scale(self.x, self.g, fit_length(self.d, 1600), fit_length(self.gd, 1600))
        w = compose(self.x, self.g, self.d, self.gd, s)
        donor_part = np.float32(s) * fit_length(self.d, 1600) * (1 - fit_length(self.gd, 1600))
        np.testing.assert_allclose(w - donor_part, self.x * self.g, atol=1e-6)
        # inside the vocalisation, where the donor is masked too, the waveform is exactly the target's
        np.testing.assert_array_equal(w[400:600], self.x[400:600])

    def test_level_matching(self):
        d, gd = fit_length(self.d, 1600), fit_length(self.gd, 1600)
        s, clipped = match_scale(self.x, self.g, d, gd)
        self.assertFalse(clipped)
        self.assertAlmostEqual(bg_level(s * d, gd), bg_level(self.x, self.g), places=5)

    def test_scale_is_clipped_for_near_silent_donor(self):
        d = np.full(1600, 1e-7, np.float32)
        s, clipped = match_scale(self.x, self.g, d, np.zeros(1600))
        self.assertTrue(clipped)
        self.assertEqual(s, 100.0)

    def test_no_background_gives_unit_scale(self):
        s, clipped = match_scale(self.x, np.ones(1600), self.d[:1600], self.gd[:1600])
        self.assertEqual((s, clipped), (1.0, True))


class MaskEdgeTest(unittest.TestCase):
    def test_edge_hold_keeps_full_gain_at_clip_start(self):
        from background_probe import masks_for
        r = np.random.default_rng(0)
        x = r.normal(0, 1e-3, 16000)
        x[:1600] = r.normal(0, 0.5, 1600)  # loud vocalisation touching the clip start
        g_old, _, _ = masks_for(x, 20, 50, np.random.default_rng(0))
        g_new, _, f_new = masks_for(x, 20, 50, np.random.default_rng(0), edge_hold=True)
        self.assertLess(g_old[0], 0.6)           # the old behaviour leaks about half the amplitude
        self.assertAlmostEqual(g_new[0], 1.0)
        self.assertEqual(len(g_new), len(x))
        np.testing.assert_allclose(g_new[4000:12000], g_old[4000:12000])  # identical away from the clip ends

    def test_edge_hold_keeps_full_gain_at_clip_end(self):
        from background_probe import masks_for
        r = np.random.default_rng(1)
        x = r.normal(0, 1e-3, 16100)  # 100 samples after the last full frame
        x[-1600:] = r.normal(0, 0.5, 1600)
        g, _, _ = masks_for(x, 20, 50, np.random.default_rng(0), edge_hold=True)
        self.assertAlmostEqual(g[-1], 1.0)
        self.assertGreater(g[-200:].min(), 0.999)


class PairingTest(unittest.TestCase):
    def setUp(self):
        rows = []
        for i in range(30):
            rows.append(dict(idx=i, y=i % 3, group=i // 2, n=4000 if i % 5 else 3000,
                             bg_fraction=0.5, bg_log_rms=-30 + i, bg_centroid=1000.0, duration=1.0))
        self.pool = pd.DataFrame(rows)
        self.targets = self.pool[["idx", "y", "group", "n"]]

    def test_constraints(self):
        P = pair_donors(self.targets, self.pool, k=3, seed=0)
        pool = self.pool.set_index("idx")
        for r in P.itertuples():
            t = pool.loc[r.target]
            self.assertEqual(pool.loc[r.same].y, t.y)
            self.assertNotEqual(pool.loc[r.same].group, t.group)
            self.assertGreaterEqual(pool.loc[r.same].n, t.n)
            for c in range(3):
                j = getattr(r, f"diff_{c}")
                if c == t.y:
                    self.assertEqual(j, -1)
                else:
                    self.assertEqual(pool.loc[j].y, c)
                    self.assertNotEqual(pool.loc[j].group, t.group)

    def test_deterministic_and_matched(self):
        P1, P2 = pair_donors(self.targets, self.pool, k=3, seed=0), pair_donors(self.targets, self.pool, k=3, seed=0)
        pd.testing.assert_frame_equal(P1, P2)
        # only bg_log_rms varies: the different-class donor is the eligible one closest in level to the same-class donor
        pool = self.pool.set_index("idx")
        r = P1.iloc[0]
        t = pool.loc[r.target]
        c = [c for c in range(3) if c != t.y][0]
        cand = pool[(pool.y == c) & (pool.group != t.group) & (pool.n >= t.n)]
        best = (cand.bg_log_rms - pool.loc[r.same].bg_log_rms).abs().idxmin()
        self.assertEqual(r[f"diff_{c}"], best)


class StatsTest(unittest.TestCase):
    def test_macro_recall_fractional(self):
        self.assertAlmostEqual(macro_recall([0, 0, 1], [1.0, 0.0, 0.5], 2), (0.5 + 0.5) / 2)

    def test_attraction_null_cases(self):
        y = np.array([0, 1, 2])
        always_correct = np.tile(y[:, None], (1, 7))
        ex, dr, nr = attraction(y, always_correct)
        np.testing.assert_array_equal(ex, 0)
        biased = np.full((3, 7), 3)  # always predicts class 3, whatever the donor
        ex, dr, nr = attraction(y, biased)
        np.testing.assert_allclose(ex, 0, atol=1e-12)
        self.assertAlmostEqual(dr[0], 1 / 6)

    def test_attraction_full(self):
        y = np.array([0, 4])
        follows_donor = np.tile(np.arange(7)[None, :], (2, 1))
        ex, dr, nr = attraction(y, follows_donor)
        np.testing.assert_allclose(dr, 1)
        np.testing.assert_allclose(nr, 0)
        np.testing.assert_allclose(ex, 1)

    def test_group_bootstrap(self):
        v = np.array([1.0, 1.0, 3.0, 3.0])
        g = np.array([0, 0, 1, 1])
        pt, lo, hi = group_bootstrap(lambda ix: [v[ix].mean(), 2.0], g, n=500)
        self.assertAlmostEqual(pt[0], 2.0)
        self.assertTrue(1.0 <= lo[0] <= 2.0 <= hi[0] <= 3.0)
        self.assertEqual((lo[1], hi[1]), (2.0, 2.0))
        pt2, lo2, hi2 = group_bootstrap(lambda ix: [v[ix].mean()], g, n=500)
        self.assertEqual((lo[0], hi[0]), (lo2[0], hi2[0]))  # same seed -> same resamples

    def test_decide(self):
        self.assertEqual(decide(0.08, 0.03, 0.12, 0.06, 0.01, 0.10), "GO")
        self.assertEqual(decide(0.08, 0.03, 0.12, 0.03, -0.01, 0.07), "DETECTED BUT SMALL")
        self.assertEqual(decide(0.003, -0.01, 0.015, 0.0, -0.02, 0.02), "NEGLIGIBLE")
        self.assertEqual(decide(0.01, -0.02, 0.04, 0.0, -0.02, 0.03), "INCONCLUSIVE")
        self.assertEqual(decide(0.03, 0.01, 0.05, 0.0, -0.02, 0.03), "DETECTED BUT SMALL")


if __name__ == "__main__":
    unittest.main()
