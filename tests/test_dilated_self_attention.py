import unittest

import torch
from nose2.tools import params

from dilated_self_attention.dilated_self_attention import DilatedSelfAttention
from dilated_self_attention.self_attention import CausalSelfAttention


class TestDilatedCausalSelfAttention(unittest.TestCase):
    def test_noop(self):
        max_len = 64
        emb_dim = 48

        attn = CausalSelfAttention(emb_dim, emb_dim, max_len)
        d_attn = DilatedSelfAttention([max_len], [1], 0, attn)

        x = torch.normal(0, 1, (1, max_len, emb_dim))

        out = attn(x)[0]
        d_out = d_attn(x)

        self.assertTrue(torch.allclose(out, d_out))

    @params(0, 1)
    def test_single_head_r2(self, offset):
        max_len = 64
        emb_dim = 48

        attn = CausalSelfAttention(emb_dim, emb_dim, max_len)
        d_attn = DilatedSelfAttention([max_len], [2], offset, attn)

        x = torch.normal(0, 1, (1, max_len, emb_dim))

        # strided dilated attention should be equivalent to a regular one
        # executed over a subset of input elements, with outputs scattered to
        # the corresponding positions and the rest filled with zeros
        out = torch.zeros_like(x)
        out[:, offset::2] = attn(x[:, offset::2])[0]

        d_out = d_attn(x)

        # had to reduce absolute tolerance a bit to avoid false assertions
        self.assertTrue(torch.allclose(out, d_out, atol=1e-6))

    def test_single_head_w_half(self):
        max_len = 64
        emb_dim = 48

        attn = CausalSelfAttention(emb_dim, emb_dim, max_len)
        d_attn = DilatedSelfAttention([max_len // 2], [1], 0, attn)

        x = torch.normal(0, 1, (1, max_len, emb_dim))

        # dilated attention with w == half sequence length should be equivalent
        # to a regular one executed over 2 halves of the input sequence
        # individually with the outputs concatenated
        out = torch.cat(
            (
                attn(x[:, : max_len // 2])[0],
                attn(x[:, max_len // 2 :])[0],
            ),
            dim=1,
        )

        d_out = d_attn(x)

        self.assertTrue(torch.allclose(out, d_out, atol=1e-7))

    def test_multi_k(self):
        max_len = 64
        emb_dim = 48

        attn = CausalSelfAttention(emb_dim, emb_dim, max_len // 2)
        d_attn = DilatedSelfAttention([max_len // 2, max_len], [1, 2], 0, attn)

        x = torch.normal(0, 1, (1, max_len, emb_dim))

        # similar to the 2 tests above, but this time expected output is a
        # weighted sum of 2 partial attention outputs, where weights are
        # calculated based on the softmax denominators of attention matrices
        out1 = torch.zeros_like(x)
        out1_w = torch.zeros((1, max_len))
        out1[:, ::2], out1_w[:, ::2] = attn(x[:, ::2])

        out2a = attn(x[:, : max_len // 2])
        out2b = attn(x[:, max_len // 2 :])
        out2 = torch.cat(
            (
                out2a[0],
                out2b[0],
            ),
            dim=1,
        )
        out2_w = torch.cat(
            (
                out2a[1],
                out2b[1],
            ),
            dim=1,
        )

        out2_alpha = torch.divide(out2_w, out1_w + out2_w)[:, :, None]

        # out = out1*(1-out2_alpha) + out2*out2_alpha
        out = torch.lerp(out1, out2, out2_alpha)

        d_out = d_attn(x)

        self.assertTrue(torch.allclose(out, d_out, atol=1e-6))

    def test_multi_k_half(self):
        max_len = 16
        emb_dim = 6

        attn = CausalSelfAttention(emb_dim, emb_dim, max_len//2)
        d_attn = DilatedSelfAttention([max_len // 2, max_len], [1, 2], 0, attn)

        x = torch.normal(0, 1, (1, max_len, emb_dim))

        d_out = d_attn(x)
        d_out_half = d_attn(x[:, :max_len//2, :])

        # todo: check whether it should be true 🤔
        self.assertTrue(torch.allclose(d_out[:, :max_len//2, :], d_out_half[:, :, :], atol=1e-6))

    def test_unbatchable(self):
        max_len = 64
        emb_dim = 48

        attn = CausalSelfAttention(emb_dim, emb_dim, max_len // 2)

        with self.assertRaises(AssertionError):
            DilatedSelfAttention([max_len // 2, max_len], [2, 2], 0, attn)
