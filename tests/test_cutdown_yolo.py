"""CUT-Down + Ultralytics glue: fixed-assignment probe (skipped without torch/ultralytics)."""

import importlib.util
import os
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT / "projects" / "cut-down"
HAVE = all(importlib.util.find_spec(m) for m in ("torch", "ultralytics")) and PROJECT.is_dir()


@unittest.skipUnless(HAVE, "needs torch, ultralytics and the local cut-down study")
class YoloProbeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        from ultralytics.cfg import get_cfg
        from ultralytics.nn.tasks import DetectionModel
        from ultralytics.utils.loss import v8DetectionLoss
        sys.path.insert(0, str(PROJECT))
        import cutdown_yolo, cutdownlib
        cls.cy, cls.cdl, cls.torch = cutdown_yolo, cutdownlib, torch
        torch.manual_seed(0)
        cls.model = DetectionModel("yolo11n.yaml", nc=3, verbose=False)
        cls.model.args = get_cfg()
        cls.layer = cutdown_yolo.install_cutdown(cls.model)
        for mod in cls.model.modules():  # random init is near-constant at the head; make the output depend on the input
            if isinstance(mod, torch.nn.Conv2d):
                torch.nn.init.kaiming_normal_(mod.weight)
        cls.model.eval()
        for p in cls.model.parameters():
            p.requires_grad_(False)
        cls.crit = v8DetectionLoss(cls.model)
        cls.imgs = torch.rand(2, 3, 128, 128)
        cls.batch = {"batch_idx": torch.tensor([0, 0, 1.0]), "cls": torch.tensor([[0], [1], [2.0]]),
                     "bboxes": torch.tensor([[.3, .3, .1, .1], [.7, .6, .15, .1], [.5, .5, .2, .2]])}
        cls.pr = cutdown_yolo.FixedAssignmentProbe(cls.model, cls.crit, cls.imgs, cls.batch)

    def half(self):
        return self.cdl.intervene(self.pr.f_s, self.pr.f_d, self.torch.full_like(self.pr.f_s[:, :1], 0.5))

    def test_shapes_and_full_model_still_runs(self):
        self.assertEqual(self.pr.f_s.shape, (2, 32, 32, 32))
        self.assertEqual(self.pr.support.overlap.shape, (2, 1, 32, 32))
        self.assertGreater(float(self.pr.support.overlap.max()), 0)
        self.layer.mode = "half"
        self.model(self.imgs)  # whole model forward with the swapped layer
        self.layer.mode = "router"

    def test_probe_tracks_ultralytics_cls_and_box_loss(self):
        l = self.pr.probe(self.half())
        preds = self.crit.parse_output(self.cy.forward_head(self.model, self.half()))
        _, loss, _ = self.crit.get_assigned_targets_and_loss(preds, self.batch)
        ref = float(loss[0] + loss[1])
        got = float(l.fg.sum() + l.bg.sum()) / 1.0
        # probe uses the fixed reference assignment, so at M0 it must agree closely
        self.assertAlmostEqual(got, ref, delta=0.05 * max(ref, 1e-3) + 1e-3 * 2)

    def test_scup_and_gradient_run_on_probe(self):
        torch = self.torch
        u = self.cdl.scup_utility(self.pr.f_s, self.pr.f_d, self.pr.probe, self.pr.support, k=4)
        g = self.cdl.gradient_utility(self.pr.f_s, self.pr.f_d, self.pr.probe, self.pr.support)
        self.assertEqual(u.u.shape, g.u.shape)
        self.assertTrue(torch.isfinite(u.u).all() and torch.isfinite(g.u).all())

    def test_fixed_assignment_ignores_current_predictions(self):
        a = self.pr.probe(self.cdl.intervene(self.pr.f_s, self.pr.f_d, self.torch.zeros_like(self.pr.f_s[:, :1])))
        b = self.pr.probe(self.cdl.intervene(self.pr.f_s, self.pr.f_d, self.torch.ones_like(self.pr.f_s[:, :1])))
        self.assertGreater(float((a.fg - b.fg).abs().max()), 0.0)  # loss responds to M while assignment stays fixed


@unittest.skipUnless(HAVE and os.environ.get("CUTDOWN_SMOKE") == "1", "set CUTDOWN_SMOKE=1 (downloads coco8, trains on CPU)")
class TrainerSmokeTests(unittest.TestCase):
    def test_router_only_distillation_updates_only_the_router(self):
        import torch
        from ultralytics import settings
        sys.path.insert(0, str(PROJECT))
        import cutdown_yolo as cy
        data_dir = PROJECT / "data"
        data_dir.mkdir(exist_ok=True)
        settings.update({"datasets_dir": str(data_dir)})
        names = {0: "a", 1: "b", 2: "c"}
        model = cy.build_detector(80, {i: str(i) for i in range(80)}, weights=None)
        trainer, last = cy.train_detector(
            model, dict(data="coco8.yaml", imgsz=64, epochs=1, batch=4, workers=0, device="cpu", plots=False, val=False,
                        amp=False, warmup_epochs=0, nbs=4, optimizer="SGD", lr0=0.1, project=str(PROJECT / "runs"),
                        name="smoke", exist_ok=True, verbose=False, **cy.NO_GEOMETRY_AUG),
            cut_mode="router", router_mode="only", extra_loss=lambda batch, layer: ((layer.last_g - 0.8) ** 2).mean())
        layer = cy.get_layer(trainer.model)
        self.assertGreater(float(layer.cut.router.net[-1].weight.abs().sum()), 0.0)  # was zero-initialised
        self.assertFalse(any(p.requires_grad for n, p in trainer.model.named_parameters() if "router" not in n))
        reloaded = cy.load_detector(last)  # checkpoint round-trips (no unpicklable hook)
        self.assertGreater(float(cy.get_layer(reloaded).cut.router.net[-1].weight.abs().sum()), 0.0)


if __name__ == "__main__":
    unittest.main()
