from typing import List

import torch


class DilatedSelfAttention(torch.nn.Module):
    def __init__(
        self, ws: List[int], rs: List[int], offset: int, attn_module: torch.nn.Module
    ):
        """
        https://arxiv.org/pdf/2307.02486.pdf

        :param ws: a list of segment lengths, trades the globality of attention for efficiency
        :param rs: a list of dilation rates, reduces the computation cost by approximating the attention matrix
        :param offset: offset of sparse indices (will be used in multi-head implementation)
        :param attn_module: attention module instance to use under the hood
        """
        super().__init__()
        assert len(ws) == len(rs)
        assert len(ws) > 0
        assert offset < min(rs)

        self.ws = ws
        self.rs = rs
        self.offset = offset
        self.attn = attn_module

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, n, c = x.size()
        x_indices = torch.arange(0, n, dtype=torch.long)[None, :, None].repeat(b, 1, c)

        all_sparse_indices = []
        sparse_os = []
        sparse_att_denoms = []

        out_att_denom_sums = torch.zeros((b, n))
        for w, r in zip(self.ws, self.rs):
            for segment_indices in torch.split(x_indices, w, 1):
                sparse_indices = segment_indices[:, self.offset :: r, :]
                x_i = torch.gather(x, 1, sparse_indices)

                # todo: rearrange to put chunks along the batch dimension
                # then a single attention forward will do the trick
                o_i, o_att_denom_i = self.attn(x_i)

                out_att_denom_sums.scatter_add_(
                    1, sparse_indices[:, :, 0], o_att_denom_i
                )

                all_sparse_indices.append(sparse_indices)
                sparse_os.append(o_i)
                sparse_att_denoms.append(o_att_denom_i)

        out = torch.zeros_like(x)
        for sparse_att_denom, sparse_o, sparse_indices in zip(
            sparse_att_denoms, sparse_os, all_sparse_indices
        ):
            # select attention softmax denominator sums for current sparse indices
            sparse_att_denom_sum = torch.gather(
                out_att_denom_sums, 1, sparse_indices[:, :, 0]
            )

            # compute alphas
            alphas = torch.divide(sparse_att_denom, sparse_att_denom_sum)[:, :, None]

            # scatter and sum alpha-weighted sparse outputs
            out.scatter_add_(1, sparse_indices, torch.multiply(sparse_o, alphas))

        return out
