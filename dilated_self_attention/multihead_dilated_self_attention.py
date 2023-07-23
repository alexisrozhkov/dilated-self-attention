from typing import List

import torch

from dilated_self_attention.dilated_self_attention import DilatedSelfAttention
from dilated_self_attention.self_attention import CausalSelfAttention


class MultiheadDilatedSelfAttention(torch.nn.Module):
    def __init__(
        self, ws: List[int], rs: List[int], embedding_dim: int, num_heads: int, flash: bool = False
    ):
        """
        https://arxiv.org/pdf/2307.02486.pdf

        :param ws: a list of segment lengths, trades the globality of attention for efficiency
        :param rs: a list of dilation rates, reduces the computation cost by approximating the attention matrix
        """
        super().__init__()

        assert len(ws) == len(rs)
        assert len(ws) > 0
        assert embedding_dim % num_heads == 0

        self.emb_dim = embedding_dim
        self.n_heads = num_heads

        max_n = max([w // r for w, r in zip(ws, rs)])

        dsas = []
        for head_idx in range(num_heads):
            attn = CausalSelfAttention(
                self.emb_dim, self.emb_dim // self.n_heads, max_n, flash
            )
            dsa = DilatedSelfAttention(ws, rs, head_idx, attn)
            dsas.append(dsa)

        self.dsas = torch.nn.ModuleList(dsas)
        self.o_proj = torch.nn.Linear(self.emb_dim, self.emb_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = torch.cat([dsa(x) for dsa in self.dsas], dim=-1)
        return self.o_proj(y)
