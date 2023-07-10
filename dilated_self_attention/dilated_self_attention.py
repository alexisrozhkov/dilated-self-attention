from typing import List

import torch


class DilatedSelfAttention(torch.nn.Module):
    def __init__(
        self, ws: List[int], rs: List[int], head_idx: int, attn_module: torch.nn.Module
    ):
        super().__init__()
        assert len(ws) == len(rs)
        assert len(ws) > 0

        self.ws = ws
        self.rs = rs
        self.head_idx = head_idx
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
                offset = self.head_idx % r
                sparse_indices = segment_indices[:, offset::r, :]
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

        head_c = self.attn.out_dim
        out = torch.zeros_like(x)[:, :, :head_c]
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
            out.scatter_add_(
                1, sparse_indices[:, :, :head_c], torch.multiply(sparse_o, alphas)
            )

        return out
