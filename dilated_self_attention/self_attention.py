import math
from typing import Tuple

import torch

from dilated_self_attention.softmax_with_denom import softmax_with_denom


class CausalSelfAttention(torch.nn.Module):
    """
    Stripped down version of Andrej Karpathy's implementation from
    https://github.com/karpathy/minGPT modified to return softmax denominator
    """

    def __init__(self, embedding_dim: int, max_n: int):
        super().__init__()

        self.emb_dim = embedding_dim

        self.qkv_proj = torch.nn.Linear(self.emb_dim, 3 * self.emb_dim)
        self.o_proj = torch.nn.Linear(self.emb_dim, self.emb_dim)

        self.register_buffer(
            "mask", torch.tril(torch.ones(max_n, max_n)).view(1, max_n, max_n)
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        b, n, c = x.size()
        q, k, v = self.qkv_proj(x).split(self.emb_dim, dim=2)

        # (b, n, c) x (b, c, n) -> (b, n, n)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))

        att = att.masked_fill(self.mask[:, :n, :n] == 0, float("-inf"))
        att, att_denom = softmax_with_denom(att, dim=-1)

        # (b, n, n) x (b, n, c) -> (b, n, c)
        y = att @ v

        return self.o_proj(y), att_denom
