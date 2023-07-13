import math
from typing import Tuple

import torch
from einops import rearrange
from flash_attn.flash_attn_interface import flash_attn_unpadded_qkvpacked_func


class CausalSelfAttention(torch.nn.Module):
    """
    Stripped down version of Andrej Karpathy's implementation from
    https://github.com/karpathy/minGPT modified to return softmax denominator
    """

    def __init__(self, embedding_dim: int, out_dim: int, max_n: int):
        super().__init__()

        self.emb_dim = embedding_dim
        self.out_dim = out_dim
        self.qkv_proj = torch.nn.Linear(self.emb_dim, 3 * self.out_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        b, n, c = x.size()
        qkv = rearrange(
            self.qkv_proj(x), "b s (three h d) -> (b s) three h d", three=3, h=1
        )
        softmax_scale = 1.0 / math.sqrt(self.out_dim)
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
            causal=True,
            return_attn_probs=True,
        )

        output = output.view((b, n, self.out_dim))

        return output, att_probs[:, 0, :].exp()
