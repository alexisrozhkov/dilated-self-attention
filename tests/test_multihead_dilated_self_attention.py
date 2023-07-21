import unittest

import torch
from nose2.tools import params

from dilated_self_attention.dilated_self_attention import DilatedSelfAttention
from dilated_self_attention.multihead_dilated_self_attention import (
    MultiheadDilatedSelfAttention,
)

MAX_LEN = 32


class TestDilatedCausalSelfAttention(unittest.TestCase):
    @params(*range(1, 1 + MAX_LEN))
    def test_noop(self, input_len):
        emb_dim = 48

        ws = [MAX_LEN]
        rs = [1]

        mh_d_attn = MultiheadDilatedSelfAttention(ws, rs, emb_dim, 1)

        # make sure that the underlying attention has the same weights
        d_attn = DilatedSelfAttention(ws, rs, 0, mh_d_attn.dsas[0].attn)

        x = torch.normal(0, 1, (1, input_len, emb_dim))

        mh_d_out = mh_d_attn(x)

        # project the dilated self attention output using the same mapping as
        # in multihead version
        d_out = mh_d_attn.o_proj(d_attn(x))

        self.assertTrue(torch.allclose(d_out, mh_d_out))

    @params(*range(1, 1 + MAX_LEN))
    def test_2head(self, input_len):
        emb_dim = 48

        ws = [MAX_LEN]
        rs = [1]

        mh_d_attn = MultiheadDilatedSelfAttention(ws, rs, emb_dim, 2)

        # make sure that the underlying attention has the same weights
        d_attn1 = DilatedSelfAttention(ws, rs, 0, mh_d_attn.dsas[0].attn)
        d_attn2 = DilatedSelfAttention(ws, rs, 1, mh_d_attn.dsas[1].attn)

        x = torch.normal(0, 1, (1, input_len, emb_dim))

        mh_d_out = mh_d_attn(x)

        d_out = mh_d_attn.o_proj(torch.cat((d_attn1(x), d_attn2(x)), dim=-1))

        self.assertTrue(torch.allclose(d_out, mh_d_out))

    @params(*range(1, 1 + MAX_LEN))
    def test_2head_3batch(self, input_len):
        emb_dim = 48

        ws = [MAX_LEN]
        rs = [1]

        mh_d_attn = MultiheadDilatedSelfAttention(ws, rs, emb_dim, 2)

        # make sure that the underlying attention has the same weights
        d_attn1 = DilatedSelfAttention(ws, rs, 0, mh_d_attn.dsas[0].attn)
        d_attn2 = DilatedSelfAttention(ws, rs, 1, mh_d_attn.dsas[1].attn)

        x = torch.normal(0, 1, (3, input_len, emb_dim))

        mh_d_out = mh_d_attn(x)

        d_out = mh_d_attn.o_proj(torch.cat((d_attn1(x), d_attn2(x)), dim=-1))

        self.assertTrue(torch.allclose(d_out, mh_d_out))
