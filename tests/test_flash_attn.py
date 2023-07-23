import unittest

import torch
from nose2.tools import params

from dilated_self_attention.dilated_self_attention import DilatedSelfAttention
from dilated_self_attention.self_attention import SelfAttention

MAX_LEN = 32
DEVICE = 'cuda:0'


class TestFlashAttn(unittest.TestCase):
    @params(*range(1, 1 + MAX_LEN*2))
    def test_flash_attn_noop(self, input_len):
        emb_dim = 48

        attn1 = SelfAttention(emb_dim, emb_dim, MAX_LEN, flash=False)
        d_attn1 = DilatedSelfAttention([MAX_LEN], [1], 0, attn1).to(DEVICE)

        attn2 = SelfAttention(emb_dim, emb_dim, MAX_LEN, flash=True)
        attn2.qkv_proj = attn1.qkv_proj
        d_attn2 = DilatedSelfAttention([MAX_LEN], [1], 0, attn2).to(DEVICE)

        x = torch.normal(0, 1, (1, input_len, emb_dim), device=DEVICE)

        d_out1 = d_attn1(x)
        d_out2 = d_attn2(x)

        self.assertTrue(torch.allclose(d_out1, d_out2, atol=1e-3))

    test_flash_attn_noop.gpu = True
    test_flash_attn_noop.flash = True

    @params(*range(1, 1 + MAX_LEN * 2))
    def test_different_wr_ratios(self, input_len):
        emb_dim = 48

        attn1 = SelfAttention(emb_dim, emb_dim, MAX_LEN // 2, flash=False)
        d_attn1 = DilatedSelfAttention([MAX_LEN // 4, MAX_LEN], [1, 2], 0, attn1).to(DEVICE)

        attn2 = SelfAttention(emb_dim, emb_dim, MAX_LEN // 2, flash=True)
        attn2.qkv_proj = attn1.qkv_proj
        d_attn2 = DilatedSelfAttention([MAX_LEN // 4, MAX_LEN], [1, 2], 0, attn2).to(DEVICE)

        x = torch.normal(0, 1, (1, input_len, emb_dim), device=DEVICE)

        d_out1 = d_attn1(x)
        d_out2 = d_attn2(x)

        self.assertTrue(torch.allclose(d_out1, d_out2, atol=1e-3))

    test_different_wr_ratios.gpu = True
    test_different_wr_ratios.flash = True
