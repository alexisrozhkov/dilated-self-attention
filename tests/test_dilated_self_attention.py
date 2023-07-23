import unittest

import torch
from nose2.tools import params

from dilated_self_attention.dilated_self_attention import DilatedSelfAttention
from dilated_self_attention.self_attention import SelfAttention

MAX_LEN = 32


class TestDilatedCausalSelfAttention(unittest.TestCase):
    @params(*range(1, 1 + MAX_LEN))
    def test_noop(self, input_len):
        emb_dim = 48

        attn = SelfAttention(emb_dim, emb_dim, MAX_LEN)
        d_attn = DilatedSelfAttention([MAX_LEN], [1], 0, attn)

        x = torch.normal(0, 1, (1, input_len, emb_dim))

        out = attn(x)[0]
        d_out = d_attn(x)

        self.assertTrue(torch.allclose(out, d_out))

    def test_exceed_max_len(self):
        emb_dim = 48

        attn = SelfAttention(emb_dim, emb_dim, MAX_LEN)

        x = torch.normal(0, 1, (1, MAX_LEN + 1, emb_dim))

        with self.assertRaises(AssertionError):
            attn(x)

    @params(0, 1)
    def test_single_head_r2(self, offset):
        emb_dim = 48

        attn = SelfAttention(emb_dim, emb_dim, MAX_LEN)
        d_attn = DilatedSelfAttention([MAX_LEN], [2], offset, attn)

        x = torch.normal(0, 1, (1, MAX_LEN, emb_dim))

        # strided dilated attention should be equivalent to a regular one
        # executed over a subset of input elements, with outputs scattered to
        # the corresponding positions and the rest filled with zeros
        out = torch.zeros_like(x)
        out[:, offset::2] = attn(x[:, offset::2])[0]

        d_out = d_attn(x)

        # had to reduce absolute tolerance a bit to avoid false assertions
        self.assertTrue(torch.allclose(out, d_out, atol=1e-6))

    @params(*range(1, 1 + MAX_LEN))
    def test_single_head_w_half(self, input_len):
        emb_dim = 48

        attn = SelfAttention(emb_dim, emb_dim, MAX_LEN)
        d_attn = DilatedSelfAttention([MAX_LEN // 2], [1], 0, attn)

        x = torch.normal(0, 1, (1, input_len, emb_dim))

        # dilated attention with w == half sequence length should be equivalent
        # to a regular one executed over 2 halves of the input sequence
        # individually with the outputs concatenated
        out = torch.cat(
            (
                attn(x[:, : MAX_LEN // 2])[0],
                attn(x[:, MAX_LEN // 2:])[0],
            ),
            dim=1,
        )

        d_out = d_attn(x)

        self.assertTrue(torch.allclose(out, d_out, atol=1e-6))

    @params(*range(1, 1 + MAX_LEN))
    def test_multi_k(self, input_len):
        emb_dim = 48

        attn = SelfAttention(emb_dim, emb_dim, MAX_LEN // 2)
        d_attn = DilatedSelfAttention([MAX_LEN // 2, MAX_LEN], [1, 2], 0, attn)

        x = torch.normal(0, 1, (1, input_len, emb_dim))

        # similar to the 2 tests above, but this time expected output is a
        # weighted sum of 2 partial attention outputs, where weights are
        # calculated based on the softmax denominators of attention matrices
        out1 = torch.zeros_like(x)
        out1_w = torch.zeros((1, input_len))
        out1[:, ::2], out1_w[:, ::2] = attn(x[:, ::2])

        out2a = attn(x[:, : MAX_LEN // 2])
        out2b = attn(x[:, MAX_LEN // 2:])
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

    @params(*range(1, 1 + MAX_LEN))
    def test_different_wr_ratios(self, input_len):
        emb_dim = 48

        attn = SelfAttention(emb_dim, emb_dim, MAX_LEN // 2)
        d_attn = DilatedSelfAttention([MAX_LEN // 4, MAX_LEN], [1, 2], 0, attn)

        x = torch.normal(0, 1, (1, input_len, emb_dim))

        out1 = torch.zeros_like(x)
        out1_w = torch.zeros((1, input_len))
        out1[:, ::2], out1_w[:, ::2] = attn(x[:, ::2])

        out2a = attn(x[:, :MAX_LEN // 4])
        out2b = attn(x[:, MAX_LEN // 4:MAX_LEN // 2])
        out2c = attn(x[:, MAX_LEN // 2:MAX_LEN * 3 // 4])
        out2d = attn(x[:, MAX_LEN * 3 // 4:])

        out2 = torch.cat(
            (
                out2a[0],
                out2b[0],
                out2c[0],
                out2d[0],
            ),
            dim=1,
        )
        out2_w = torch.cat(
            (
                out2a[1],
                out2b[1],
                out2c[1],
                out2d[1],
            ),
            dim=1,
        )

        out2_alpha = torch.divide(out2_w, out1_w + out2_w)[:, :, None]

        out = torch.lerp(out1, out2, out2_alpha)

        d_out = d_attn(x)

        self.assertTrue(torch.allclose(out, d_out, atol=1e-6))

    @params(*range(1, 1 + MAX_LEN))
    def test_different_wr_ratios_missing_outputs(self, input_len):
        emb_dim = 48

        attn = SelfAttention(emb_dim, emb_dim, MAX_LEN // 2)
        d_attn = DilatedSelfAttention([MAX_LEN // 2, MAX_LEN], [2, 2], 0, attn)

        x = torch.normal(0, 1, (1, input_len, emb_dim))

        out1 = torch.zeros_like(x)
        out1_w = torch.zeros((1, input_len))
        out1[:, ::2], out1_w[:, ::2] = attn(x[:, ::2])

        out2a = attn(x[:, :MAX_LEN // 2:2])
        out2b = attn(x[:, MAX_LEN // 2::2])
        out2 = torch.zeros_like(x)
        out2_w = torch.zeros((1, input_len))

        out2[:, ::2] = torch.cat(
            (
                out2a[0],
                out2b[0],
            ),
            dim=1,
        )
        out2_w[:, ::2] = torch.cat(
            (
                out2a[1],
                out2b[1],
            ),
            dim=1,
        )

        out2_alpha = torch.divide(out2_w, out1_w + out2_w)[:, :, None]
        out2_alpha[:, 1::2] = 0  # replace NaNs with zeros

        out = torch.lerp(out1, out2, out2_alpha)

        d_out = d_attn(x)

        self.assertTrue(torch.allclose(out, d_out, atol=1e-6))
