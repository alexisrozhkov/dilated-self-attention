import unittest

import torch

from torch.nn import functional as F

from dilated_self_attention.softmax_with_denom import softmax_with_denom


class TestSoftmax(unittest.TestCase):
    def test_softmax(self):
        max_len = 64
        att = torch.normal(0, 1, (1, max_len, max_len))

        att_custom, att_denom = softmax_with_denom(att, dim=-1)
        att_native = F.softmax(att, dim=-1)

        self.assertTrue(torch.allclose(att_custom, att_native))
