import os
import sys
import unittest

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ast_finetune import cross_source_supcon  # noqa: E402


class CrossSourceSupConTest(unittest.TestCase):
    def test_same_source_pairs_are_not_positives(self):
        z = F.normalize(torch.randn(4, 8), dim=1)
        y = torch.tensor([0, 0, 1, 1])
        g = torch.tensor([0, 1, 2, 2])  # class 1 has a single source in the batch
        _, frac = cross_source_supcon(z, y, g, 0.1)
        self.assertAlmostEqual(frac, 0.5)

    def test_no_eligible_anchor_gives_zero_loss(self):
        z = F.normalize(torch.randn(4, 8), dim=1)
        loss, frac = cross_source_supcon(z, torch.tensor([0, 0, 1, 1]), torch.tensor([5, 5, 7, 7]), 0.1)
        self.assertEqual(frac, 0.0)
        self.assertEqual(float(loss), 0.0)

    def test_aligned_cross_source_positives_lower_the_loss(self):
        a, b = F.normalize(torch.tensor([[1.0, 0.0], [0.0, 1.0]]), dim=1)
        y, g = torch.tensor([0, 0, 1, 1]), torch.tensor([0, 1, 2, 3])
        aligned = torch.stack([a, a, b, b])
        mixed = torch.stack([a, b, a, b])
        self.assertLess(float(cross_source_supcon(aligned, y, g, 0.1)[0]),
                        float(cross_source_supcon(mixed, y, g, 0.1)[0]))

    def test_same_source_same_class_partner_is_not_a_negative(self):
        # anchor 0 (class 0, source 0); positive 1 (class 0, source 1); partner 2 (class 0, source 0); negative 3.
        y, g = torch.tensor([0, 0, 0, 1]), torch.tensor([0, 1, 0, 2])
        base = F.normalize(torch.randn(4, 8, generator=torch.Generator().manual_seed(0)), dim=1)
        moved = base.clone()
        moved[2] = -base[0]  # moving the same-source partner must not change anchor 0's term,
        # but it does change anchor 1's and 2's terms, so compare anchor 0 alone via a 3-clip batch without 1's anchoring
        def anchor0(z):
            sim = z @ z.T / 0.1
            return -(sim[0, 1] - torch.logsumexp(sim[0, [1, 3]], 0))
        loss_b, loss_m = anchor0(base), anchor0(moved)
        self.assertAlmostEqual(float(loss_b), float(loss_m), places=5)
        # and the implementation's anchor-0 term equals that reference
        full = cross_source_supcon(base, y, g, 0.1)[0]
        sim = base @ base.T / 0.1
        ref1 = -(sim[1, [0, 2]] - torch.logsumexp(sim[1, [0, 2, 3]], 0)).mean()
        ref2 = -(sim[2, 1] - torch.logsumexp(sim[2, [1, 3]], 0))
        self.assertAlmostEqual(float(full), float((anchor0(base) + ref1 + ref2) / 3), places=4)

    def test_gradients_flow(self):
        z = torch.randn(6, 8, requires_grad=True)
        loss, _ = cross_source_supcon(F.normalize(z, dim=1), torch.tensor([0, 0, 1, 1, 2, 2]), torch.arange(6), 0.1)
        loss.backward()
        self.assertGreater(float(z.grad.norm()), 0)


if __name__ == "__main__":
    unittest.main()
