import sys

sys.path.extend(["./", "../"])


import unittest
from nas_lib.models.narformer import Encoder, NARFormer, tokenizer
import torch
import numpy as np


class TestNARFormer(unittest.TestCase):
    def test_tokenizer(self):
        ops = torch.tensor([0, 1, 2, 3, 3, 4, 5, 6])
        adj_mat = np.array(
            [
                [0, 1, 1, 0, 1, 0, 0, 0],
                [0, 0, 0, 1, 0, 1, 0, 0],
                [0, 0, 0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 0, 0, 1],
                [0, 0, 0, 0, 0, 0, 0, 1],
                [0, 0, 0, 0, 0, 0, 0, 1],
                [0, 0, 0, 0, 0, 0, 0, 0],
            ]
        )
        code, rel_pos, c_adj_d, code_depth = tokenizer(ops, adj_mat, 32, 32, 32)
        self.assertTrue(rel_pos.dtype, torch.int)
        self.assertTrue(c_adj_d.dtype, torch.float)

        true_rel_pos = [
            [9, 9, 9, 9, 9, 9, 9, 9, 9, 9],
            [9, 0, 1, 1, 2, 1, 2, 2, 2, 9],
            [9, 8, 0, 8, 1, 8, 1, 2, 2, 9],
            [9, 8, 8, 0, 8, 8, 8, 1, 2, 9],
            [9, 8, 8, 8, 0, 8, 8, 1, 2, 9],
            [9, 8, 8, 8, 8, 0, 8, 8, 1, 9],
            [9, 8, 8, 8, 8, 8, 0, 8, 1, 9],
            [9, 8, 8, 8, 8, 8, 8, 0, 1, 9],
            [9, 8, 8, 8, 8, 8, 8, 8, 0, 9],
            [9, 9, 9, 9, 9, 9, 9, 9, 9, 9],
        ]

        true_c_adj_d = [
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 1, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        ]
        self.assertEqual(list(code.shape), [9, 192])
        self.assertEqual(rel_pos.tolist(), true_rel_pos)
        self.assertEqual(c_adj_d.tolist(), true_c_adj_d)
        self.assertEqual(list(code_depth.shape), [1, 192])
        self.assertAlmostEqual(code[0, :64].tolist(), code[0, 64:128].tolist())
        self.assertAlmostEqual(code[0, :64].tolist(), code[0, 128:].tolist())
        # print(torch.round(code[0, 128:], decimals=4))
        # print(torch.round(code[5, 64:128], decimals=4))

    def test_encoder(self):
        net = Encoder()
        x = torch.rand(2, 10, 192)
        adj = torch.randint(0, 2, [2, 10, 10])
        rel_pos = torch.randint(0, 4, [2, 10, 10])
        y = net(x, rel_pos, adj.float())
        self.assertEqual(list(y.shape), [2, 1, 192])

    def test_narformer(self):
        net = NARFormer()
        ops = torch.tensor([0, 1, 2, 3, 3, 4, 5, 6])
        adj_mat = np.array(
            [
                [0, 1, 1, 0, 1, 0, 0, 0],
                [0, 0, 0, 1, 0, 1, 0, 0],
                [0, 0, 0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 0, 0, 1],
                [0, 0, 0, 0, 0, 0, 0, 1],
                [0, 0, 0, 0, 0, 0, 0, 1],
                [0, 0, 0, 0, 0, 0, 0, 0],
            ]
        )
        code, rel_pos, c_adj_d, code_depth = tokenizer(ops, adj_mat, 32, 32, 32)
        y = net(
            code.unsqueeze(0),
            rel_pos.unsqueeze(0),
            c_adj_d.unsqueeze(0),
            code_depth.unsqueeze(0),
        )
        self.assertEqual(list(y.shape), [1, 1])


if __name__ == "__main__":
    unittest.main()
