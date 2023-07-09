import unittest

import torch

from dilated_self_attention import DilatedCausalSelfAttention


class TestDilatedCausalSelfAttention(unittest.TestCase):
    def test_single_head_noop(self):
        max_len = 64
        emb_dim = 48

        attn = torch.nn.MultiheadAttention(emb_dim, 1, batch_first=True)
        d_attn = DilatedCausalSelfAttention(w=max_len, r=1, offset=0, attn_module=attn)

        x = torch.normal(0, 1, (1, max_len, emb_dim))
        attn_mask = torch.tril(torch.ones(max_len, max_len))

        out = attn(x, x, x, is_causal=True, attn_mask=attn_mask)[0]
        d_out = d_attn(x)

        self.assertTrue(torch.allclose(out, d_out))
