import math
from typing import Tuple

import torch

from dilated_self_attention.softmax_with_denom import softmax_with_denom


def _flash_self_attention(qkv_: torch.Tensor, causal: bool):
    from einops import rearrange
    from flash_attn.flash_attn_interface import flash_attn_unpadded_qkvpacked_func

    b, n, c3 = qkv_.size()
    out_dim = c3 // 3

    if n == 0:
        return qkv_[:, :, :out_dim], qkv_[:, :, 0]

    qkv = rearrange(qkv_.half(), "b s (three h d) -> (b s) three h d", three=3, h=1)
    softmax_scale = 1.0 / math.sqrt(out_dim)
    dropout_p = 0.0

    cu_seqlens = torch.arange(
        0, (b + 1) * n, step=n, dtype=torch.int32, device=qkv.device
    )
    output, att_probs, _ = flash_attn_unpadded_qkvpacked_func(
        qkv,
        cu_seqlens,
        n,
        dropout_p,
        softmax_scale=softmax_scale,
        causal=causal,
        return_attn_probs=True,
    )

    output = output.view((b, n, out_dim))

    return output.float(), att_probs[:, 0, :n].exp().float()


def _vanilla_self_attention(qkv: torch.Tensor, out_dim: int, mask: torch.Tensor):
    q, k, v = qkv.split(out_dim, dim=2)
    n = qkv.shape[1]

    # (b, n, c) x (b, c, n) -> (b, n, n)
    att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))

    att = att.masked_fill(mask[:, :n, :n] == 0, float("-inf"))
    att, att_denom = softmax_with_denom(att, dim=-1)

    # (b, n, n) x (b, n, c) -> (b, n, c)
    y = att @ v
    return y, att_denom


class SelfAttention(torch.nn.Module):
    """
    Stripped down version of Andrej Karpathy's implementation from
    https://github.com/karpathy/minGPT modified to return softmax denominator
    """

    def __init__(
        self, embedding_dim: int, out_dim: int, max_n: int, flash: bool = False, causal: bool = True
    ):
        super().__init__()

        self.emb_dim = embedding_dim
        self.out_dim = out_dim
        self.qkv_proj = torch.nn.Linear(self.emb_dim, 3 * self.out_dim)
        self.flash = flash
        self.causal = causal

        if not self.flash:
            mask = torch.ones(max_n, max_n)

            if self.causal:
                mask = torch.tril(mask)

            self.register_buffer(
                "mask", mask.view(1, max_n, max_n)
            )

    def forward(self, x: torch.Tensor, padding_mask: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
        qkv = self.qkv_proj(x)

        if self.flash:
            return _flash_self_attention(qkv, self.causal)

        else:
            assert x.shape[1] <= self.mask.shape[1], \
              "sequence length (dimension 1 of the input tensor) must be smaller" \
              " than max_n provided during construction"

            if padding_mask is not None:
                mask = torch.multiply(
                    self.mask[:, :x.shape[1], :x.shape[1]],
                    torch.multiply(padding_mask[:, None, :],
                                   padding_mask[:, :, None])
                ).long()
            else:
                mask = self.mask

            return _vanilla_self_attention(qkv, self.out_dim, mask)
