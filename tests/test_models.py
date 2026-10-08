"""Small CPU model contract and optional frozen-source equivalence checks.

Run with ``python -m unittest discover -s tests``. For a local extraction audit,
run ``python tests/test_models.py --reference PATH_TO_MODELS_PY`` or set
``GIFT_REFERENCE_MODELS``. The optional reference snapshot is not distributed.
All inputs and optimizer updates here are synthetic and disposable.
"""
from __future__ import annotations

import argparse
import importlib.util
import os
from pathlib import Path
import sys
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gift import models


def _input(horizon):
    generator = torch.Generator().manual_seed(617006 + horizon)
    data = torch.randn(2, 12, 5, 3, generator=generator)
    data[..., 0] *= torch.linspace(.2, 1.8, 5)[None, None, :]
    data[..., 0] += torch.linspace(-2, 2, 5)[None, None, :]
    target = torch.randn(2, 5, horizon, generator=generator)
    valid = torch.ones(2, 12, 5, dtype=torch.bool)
    valid[:, 4, 2] = False
    return data, target, valid


def _graph():
    graph = torch.eye(5)
    for row in range(5):
        graph[row, (row + 1) % 5] = 1
        if row % 2 == 0:
            graph[row, (row + 3) % 5] = 1
    return graph


def _attach_graph(model):
    if model.formal_variant == "GRAPH_CONSTRAINED":
        model.set_graph(_graph())


def _exercise_new_paths(model):
    # Zero initialization initially hides state/deviation paths. Fixed test-only
    # values make their forward and gradient equivalence observable as well.
    with torch.no_grad():
        if hasattr(model, "statistics_encoder"):
            model.statistics_encoder[-1].weight.fill_(.017)
            model.statistics_encoder[-1].bias.fill_(.011)
        if hasattr(model, "deviation_projection"):
            model.deviation_projection.weight.copy_(torch.eye(model.width) * .03)
            model.deviation_projection.bias.fill_(.007)


def _load_reference():
    configured = os.environ.get("GIFT_REFERENCE_MODELS")
    if not configured:
        return None
    path = Path(configured).expanduser().resolve()
    if path.is_dir():
        path /= "models.py"
    if not path.is_file():
        raise ValueError(f"Reference models.py does not exist: {path}")
    spec = importlib.util.spec_from_file_location("_gift_frozen_reference", path)
    module = importlib.util.module_from_spec(spec)
    # The frozen source adjusts sys.path for its local legacy dependencies.
    # Restore the caller's search path after importing that optional snapshot.
    previous_path = sys.path[:]
    try:
        sys.path.insert(0, str(path.parent))
        spec.loader.exec_module(module)
    finally:
        sys.path[:] = previous_path
    return module


class ModelContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_shapes_auxiliary_losses_and_count_anchors(self):
        for variant in models.VARIANTS:
            for horizon in (6, 12):
                with self.subTest(variant=variant, horizon=horizon):
                    model = models.build(variant, horizon=horizon).eval()
                    _attach_graph(model)
                    data, target, valid = _input(horizon)
                    model.history_valid_mask = valid
                    with torch.no_grad():
                        prediction = model(data)
                    self.assertEqual(prediction.shape, (2, 5, horizon))
                    self.assertTrue(torch.isfinite(prediction).all())
                    auxiliary = model.auxiliary_losses(data, target, valid)
                    self.assertEqual(bool(auxiliary), hasattr(model, "history_predictor"))
                    if auxiliary:
                        self.assertTrue(torch.isfinite(auxiliary["history"]))
                    audit = models.configure_policy(model)
                    if variant == "GIFT_CE":
                        expected = (333981, 206010) if horizon == 6 else (334755, 206784)
                        self.assertEqual(
                            (audit["total_parameters"], audit["trainable_parameters"]), expected,
                        )

    def test_seeded_build_preserves_cpu_rng_and_aliases(self):
        torch.manual_seed(739)
        before = torch.get_rng_state().clone()
        first = models.build("CE", seed=1)
        self.assertTrue(torch.equal(before, torch.get_rng_state()))
        second = models.build("GIFT_CE", seed=1)
        self.assertEqual(list(first.state_dict()), list(second.state_dict()))
        for name, value in first.state_dict().items():
            self.assertTrue(torch.equal(value, second.state_dict()[name]), name)


class FrozenModelEquivalence(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.reference = _load_reference()
        if cls.reference is None:
            raise unittest.SkipTest("Supply --reference or GIFT_REFERENCE_MODELS for extraction audit")
        torch.set_num_threads(1)

    def _assert_state_equal(self, actual, expected, label):
        left, right = actual.state_dict(), expected.state_dict()
        self.assertEqual(list(left), list(right), label + " keys/order")
        for name in left:
            self.assertTrue(torch.equal(left[name], right[name]), f"{label}: {name}")

    def _assert_close(self, actual, expected, label):
        try:
            torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-7)
        except AssertionError as error:
            self.fail(f"{label}: {error}")

    def _assert_gradients_equal(self, actual, expected):
        left, right = dict(actual.named_parameters()), dict(expected.named_parameters())
        self.assertEqual(list(left), list(right))
        for name in left:
            self.assertEqual(left[name].grad is None, right[name].grad is None, name)
            if left[name].grad is not None:
                self._assert_close(left[name].grad, right[name].grad, "gradient " + name)

    def test_all_variants_initialization_forward_auxiliary_gradients_and_update(self):
        for variant in models.VARIANTS:
            for horizon in (6, 12):
                for seed in (0, 1):
                    with self.subTest(variant=variant, horizon=horizon, seed=seed):
                        actual = models.build(variant, horizon=horizon, seed=seed)
                        expected = self.reference.build(variant, horizon=horizon, seed=seed)
                        self._assert_state_equal(actual, expected, "initialization")
                        for model in (actual, expected):
                            _attach_graph(model)
                            model.eval()
                        data, target, valid = _input(horizon)
                        for model in (actual, expected):
                            model.history_valid_mask = valid
                        with torch.no_grad():
                            self._assert_close(actual(data), expected(data), "initial prediction")
                        for model in (actual, expected):
                            _exercise_new_paths(model)
                        # A full-policy step exercises gradients through the
                        # communication paths; a default step also checks freezing.
                        for policy in ("FULL", "DEFAULT_CE"):
                            self.assertEqual(
                                models.configure_policy(actual, policy, seed),
                                self.reference.configure_policy(expected, policy, seed),
                            )
                            frozen = {name: value.detach().clone()
                                      for name, value in actual.named_parameters()
                                      if not value.requires_grad}
                            optimizers = [torch.optim.AdamW(
                                [p for p in model.parameters() if p.requires_grad],
                                lr=.001, weight_decay=.0001,
                            ) for model in (actual, expected)]
                            predictions, auxiliaries = [], []
                            for model in (actual, expected):
                                model.train()
                                model.zero_grad(set_to_none=True)
                                torch.manual_seed(418000 + seed)
                                prediction = model(data)
                                auxiliary = model.auxiliary_losses(data, target, valid)
                                loss = (prediction - target).abs().mean()
                                if "history" in auxiliary:
                                    loss = loss + .05 * auxiliary["history"]
                                loss.backward()
                                predictions.append(prediction.detach())
                                auxiliaries.append(auxiliary)
                            self._assert_close(predictions[0], predictions[1], "training prediction")
                            self.assertEqual(set(auxiliaries[0]), set(auxiliaries[1]))
                            for key in auxiliaries[0]:
                                self._assert_close(auxiliaries[0][key], auxiliaries[1][key], "auxiliary " + key)
                            self._assert_gradients_equal(actual, expected)
                            for optimizer in optimizers:
                                optimizer.step()
                            self._assert_state_equal(actual, expected, policy + " optimizer update")
                            for name, parameter in actual.named_parameters():
                                if name in frozen:
                                    self.assertTrue(torch.equal(parameter, frozen[name]), "frozen " + name)

    def test_all_variant_policy_audits(self):
        policies = ("FULL", "HEAD_ONLY", "INTERACTION_ONLY", "DEFAULT_CE",
                    "FROZEN_TWO_DAY", "ZERO_DAY", "RESET_QK")
        for variant in models.VARIANTS:
            for horizon in (6, 12):
                for seed in (0, 1):
                    actual = models.build(variant, horizon=horizon, seed=seed)
                    expected = self.reference.build(variant, horizon=horizon, seed=seed)
                    for policy in policies:
                        # RESET_QK is defined only for variants retaining Q/K.
                        if policy == "RESET_QK" and not hasattr(actual.blocks[0], "kernel_query"):
                            continue
                        with self.subTest(variant=variant, horizon=horizon, seed=seed, policy=policy):
                            left = models.configure_policy(actual, policy, seed)
                            right = self.reference.configure_policy(expected, policy, seed)
                            self.assertEqual(left, right)
                            self.assertEqual(
                                [(n, p.requires_grad) for n, p in actual.named_parameters()],
                                [(n, p.requires_grad) for n, p in expected.named_parameters()],
                            )
                            self._assert_state_equal(actual, expected, policy + " policy state")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--reference", help="Optional frozen models.py path or snapshot directory")
    arguments, remaining = parser.parse_known_args()
    if arguments.reference:
        os.environ["GIFT_REFERENCE_MODELS"] = arguments.reference
    unittest.main(argv=[sys.argv[0], *remaining])
