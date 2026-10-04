"""CUT-Down project inputs and support library (library tests need torch; skipped otherwise)."""

import importlib.util
import json
from pathlib import Path
import sys
import unittest

from autoresearch.config import ROOT, load_run_config

PROJECT = ROOT / "projects" / "cut-down"
HAS_TORCH = importlib.util.find_spec("torch") is not None


@unittest.skipUnless(PROJECT.is_dir(), "projects/ is git-ignored; study absent in clean clones")
class ProjectInputTests(unittest.TestCase):
    def test_ideas_and_config_load(self):
        ideas = json.loads((PROJECT / "ideas.json").read_text())
        self.assertEqual(len(ideas), 1)
        for key in ("Title", "Abstract", "Short Hypothesis", "Experiments", "Risk Factors and Limitations"):
            self.assertIn(key, ideas[0])
        self.assertEqual(set(ideas[0]["Stage Goals"]), {"1", "2", "3", "4"})
        cfg = load_run_config(PROJECT / "config.yaml")
        self.assertEqual(cfg["exp_name"], "run")
        self.assertIn("cutdownlib.py", [Path(p).name for p in cfg["exec"]["support_files"]])


@unittest.skipUnless(HAS_TORCH and PROJECT.is_dir(), "needs torch and the local cut-down study")
class LibraryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        sys.path.insert(0, str(PROJECT))
        import cutdownlib
        cls.lib = cutdownlib

    def toy(self):
        """Planted utility: detail features carry the target only in the left half."""
        import torch
        torch.manual_seed(0)
        b, c, h, w = 4, 3, 16, 16
        f_s = torch.zeros(b, c, h, w)
        target = torch.randn(b, c, h, w)
        f_d = torch.zeros_like(target)
        f_d[..., : w // 2] = target[..., : w // 2]   # detail helps on the left
        f_s[..., w // 2:] = target[..., w // 2:] * 1.0  # stable helps on the right
        support = torch.ones(b, 1, h, w, dtype=torch.bool)

        def probe(f):
            err = ((f - target) ** 2).mean(1)  # (B,h,w)
            return self.lib.ProbeLoss(fg=err.mean((1, 2))[:, None], bg=torch.zeros(b))
        return f_s, f_d, support, probe

    def test_scup_recovers_planted_utility_and_predicts_heldout(self):
        import torch
        f_s, f_d, support, probe = self.toy()
        g = torch.Generator().manual_seed(1)
        util = self.lib.scup_utility(f_s, f_d, probe, support, k=64, block=2, lambda_bg=0.0, generator=g)
        left, right = util.u[..., :8].mean(), util.u[..., 8:].mean()
        self.assertGreater(float(left), 0)
        self.assertLess(float(right), 0)
        r = self.lib.heldout_agreement(f_s, f_d, probe, util, support, k=16, block=2,
                                       generator=torch.Generator().manual_seed(2))
        self.assertGreater(r, 0.5)
        q, c = self.lib.utility_targets(util)
        self.assertTrue(((q >= 0) & (q <= 1) & (c >= 0) & (c < 1)).all())
        self.assertFalse(util.u.requires_grad)

    def test_model_policies_and_losses(self):
        import torch
        m = self.lib.CUTDown(8, 16)
        x = torch.randn(2, 8, 16, 16)
        out, g = m(x)
        self.assertEqual(out.shape, (2, 16, 8, 8))
        self.assertTrue(torch.allclose(g, torch.full_like(g, 0.5)))
        fs, fd = m.experts(x)
        self.assertTrue(torch.allclose(m(x, "stable")[0], fs, atol=1e-5))
        self.assertTrue(torch.allclose(m(x, "detail")[0], fd, atol=1e-5))
        q, c = torch.rand(2, 1, 8, 8), torch.rand(2, 1, 8, 8)
        loss = self.lib.total_loss(torch.tensor(1.0), g, q, c)
        loss.backward()
        self.assertIsNotNone(m.router.net[0].weight.grad)
        self.assertFalse(torch.isnan(self.lib.ranking_loss(g, q, c)))
        for cond in self.lib.SUPERVISION:
            self.lib.supervision_loss(cond, g, q=q, c=c, target=q)

    def test_gradient_baseline_split_half_and_mask_mix(self):
        import torch
        f_s, f_d, support, probe = self.toy()
        gu = self.lib.gradient_utility(f_s, f_d, probe, support, lambda_bg=0.0)
        self.assertGreater(float(gu.u[..., :8].mean()), 0)
        self.assertLess(float(gu.u[..., 8:].mean()), 0)
        rels = [self.lib.split_half_utility(f_s, f_d, probe, support, k=k, block=2, lambda_bg=0.0)[2] for k in (8, 256)]
        self.assertGreater(rels[1], rels[0])  # reliability grows with K (variance falls)
        self.assertGreater(rels[1], 0.5)
        out, m = self.lib.random_mask_mix(f_s, f_d, block=4)
        self.assertEqual(out.shape, f_s.shape)
        self.assertTrue(((m >= 0) & (m <= 1)).all())
        self.assertTrue(((m > 0.01) & (m < 0.99)).any())  # soft blends are covered, not just {0,1}

    def test_box_support_matches_dense_and_heldout_ignores_image_scale(self):
        import torch
        torch.manual_seed(3)
        b, n, h, w = 2, 5, 16, 16
        x0, y0 = torch.randint(0, 12, (b, n)), torch.randint(0, 12, (b, n))
        cells = torch.stack([x0, y0, x0 + torch.randint(1, 5, (b, n)), y0 + torch.randint(1, 5, (b, n))], -1)
        valid = torch.rand(b, n) > 0.2
        dense = torch.zeros(b, n, h, w, dtype=torch.bool)
        for i in range(b):
            for o in range(n):
                cx0, cy0, cx1, cy1 = cells[i, o].tolist()
                dense[i, o, cy0:min(cy1, h), cx0:min(cx1, w)] = bool(valid[i, o])
        box = self.lib.BoxSupport(cells, valid, h, w)
        v = torch.randn(b, n)
        self.assertTrue(torch.allclose(box.paint(v), self.lib.DenseSupport(dense).paint(v), atol=1e-5))
        self.assertTrue(torch.allclose(box.overlap, self.lib.DenseSupport(dense).overlap, atol=1e-5))

        # heldout_agreement is per image: a 1000x loss-scale difference between images must not matter
        f_s, f_d, support, probe = self.toy()
        scale = torch.tensor([1.0, 1000.0, 1.0, 1000.0])

        def scaled(f):
            l = probe(f)
            return self.lib.ProbeLoss(fg=l.fg * scale[:, None], bg=l.bg + 0.3 * scale)
        util = self.lib.scup_utility(f_s, f_d, scaled, support, k=128, block=2, lambda_bg=0.0,
                                     generator=torch.Generator().manual_seed(5))
        r = self.lib.heldout_agreement(f_s, f_d, scaled, util, support, k=32, block=2,
                                       generator=torch.Generator().manual_seed(6))
        self.assertGreater(r, 0.5)

    def test_losses_are_safe_under_autocast(self):
        import torch
        g, q, c = torch.rand(2, 1, 4, 4), torch.rand(2, 1, 4, 4), torch.rand(2, 1, 4, 4)
        with torch.autocast("cpu", dtype=torch.bfloat16):
            self.assertTrue(torch.isfinite(self.lib.utility_loss(g.bfloat16(), q, c)))
            self.assertTrue(torch.isfinite(self.lib.supervision_loss("snr", g, target=q)))

    def test_recovery_and_cache(self):
        import torch
        self.assertAlmostEqual(self.lib.routing_recovery(0.30, 0.40, 0.35), 0.5)
        self.assertTrue(self.lib.routing_recovery(0.30, 0.30, 0.35) != self.lib.routing_recovery(0.30, 0.30, 0.35))
        cache = self.lib.UtilityCache(capacity=2)
        for i in range(3):
            cache.put(i, torch.rand(1, 4, 4), torch.rand(1, 4, 4), {"step": i})
        self.assertEqual(len(cache), 2)
        self.assertEqual(cache.stale_keys(0.5, 5), [1])


if __name__ == "__main__":
    unittest.main()
