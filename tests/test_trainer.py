import sys

sys.path.extend(["./", "../"])


import unittest
import torch
import numpy as np

from nas_lib.eigen.trainer_narformer import NarFormerPredictorTrainer
import logging


OPNAMES = [
    "input",
    "none",
    "max_pool_3x3",
    "avg_pool_3x3",
    "skip_connect",
    "sep_conv_3x3",
    "sep_conv_5x5",
    "dil_conv_3x3",
    "dil_conv_5x5",
    "concat",
    "output",
]

logger = logging.getLogger()


class TestTrainer(unittest.TestCase):
    def test_single_target(self):
        data = [[np.random.randint(0, 2, [11, 11]).astype(np.int32), OPNAMES]]
        val_acc = [10.0]
        trainer = NarFormerPredictorTrainer(epochs=3)
        trainer.fit(data, val_acc, logger)
        trainer.pred(data)

    def test_multiple_target(self):
        data = [[np.random.randint(0, 2, [11, 11]).astype(np.int32), OPNAMES]]
        target = np.array([[10.0, 20.0]])
        rate = torch.tensor([20.0, 1.0])
        trainer = NarFormerPredictorTrainer(epochs=3, rate=rate, out_dim=2)
        trainer.fit(data, target, logger)
        trainer.pred(data)


if __name__ == "__main__":
    unittest.main()
