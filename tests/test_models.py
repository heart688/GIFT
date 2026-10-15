"""Fast synthetic checks for the released GIFT model."""
import unittest
import torch
from gift import GIFT


class ModelContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_seeded_initialization_and_parameter_counts(self):
        for horizon, total, trainable in ((6, 333981, 206010), (12, 334755, 206784)):
            torch.manual_seed(739)
            before = torch.get_rng_state().clone()
            first = GIFT(horizon=horizon, seed=1)
            self.assertTrue(torch.equal(before, torch.get_rng_state()))
            second = GIFT(horizon=horizon, seed=1)
            for name, value in first.state_dict().items():
                self.assertTrue(torch.equal(value, second.state_dict()[name]), name)
            audit = first.freeze_communication()
            self.assertEqual((audit['total'], audit['trainable']), (total, trainable))

    def test_forward_auxiliary_and_node_permutation(self):
        model = GIFT(horizon=12, seed=0).eval()
        x = torch.randn(2, 12, 5, 3, generator=torch.Generator().manual_seed(4))
        valid = torch.ones(2, 5, 12, dtype=torch.bool)
        valid[:, 2, 4] = False
        model.history_valid_mask = valid
        with torch.no_grad():
            prediction = model(x)
            auxiliary = model.auxiliary_loss(valid)
            permutation = torch.tensor([3, 0, 4, 1, 2])
            model.history_valid_mask = valid[:, permutation]
            permuted = model(x[:, :, permutation])
        self.assertEqual(prediction.shape, (2, 5, 12))
        self.assertTrue(torch.isfinite(auxiliary))
        torch.testing.assert_close(permuted, prediction[:, permutation], rtol=1e-5, atol=1e-6)


if __name__ == '__main__':
    unittest.main()
