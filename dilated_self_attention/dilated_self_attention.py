from typing import List, Tuple, Union

import torch


def _segment_and_sparsify(
    x: torch.Tensor, ws: List[int], rs: List[int], head_idx: int
) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    b, n, c = x.size()

    sparse_indices = []
    sparse_values = []

    x_indices = torch.arange(0, n, dtype=torch.long, device=x.device)[
        None, :, None
    ].repeat(b, 1, c)

    for w, r in zip(ws, rs):
        for segment_indices in torch.split(x_indices, w, 1):
            offset = head_idx % r
            cur_sparse_indices = segment_indices[:, offset::r, :]
            sparse_indices.append(cur_sparse_indices)
            sparse_values.append(torch.gather(x, 1, cur_sparse_indices))

    return sparse_indices, sparse_values


def _aggregate_denom_sums(
    b: int,
    n: int,
    device: Union[torch.device, str],
    sparse_att_denoms: List[torch.Tensor],
    sparse_indices: List[torch.Tensor],
) -> torch.Tensor:
    assert len(sparse_att_denoms) == len(sparse_indices)
    att_denom_sums = torch.zeros((b, n), device=device)

    for cur_denoms, cur_indices in zip(sparse_att_denoms, sparse_indices):
        att_denom_sums.scatter_add_(1, cur_indices[:, :, 0], cur_denoms)

    return att_denom_sums


def _mix_outputs(
    out_shape: Tuple[int, int, int],
    out_dtype: torch.dtype,
    out_device: Union[torch.device, str],
    sparse_os: List[torch.Tensor],
    sparse_att_denoms: List[torch.Tensor],
    sparse_indices: List[torch.Tensor],
) -> torch.Tensor:
    # calculate sums of softmax denominators
    out_att_denom_sums = _aggregate_denom_sums(
        out_shape[0], out_shape[1], out_device, sparse_att_denoms, sparse_indices
    )

    out = torch.zeros(out_shape, dtype=out_dtype, device=out_device)
    for sparse_att_denom, sparse_o, cur_sparse_indices in zip(
        sparse_att_denoms, sparse_os, sparse_indices
    ):
        # select attention softmax denominator sums for current sparse indices
        sparse_att_denom_sum = torch.gather(
            out_att_denom_sums, 1, cur_sparse_indices[:, :, 0]
        )

        # compute alphas
        alphas = torch.divide(sparse_att_denom, sparse_att_denom_sum)[:, :, None]

        # scatter and sum alpha-weighted sparse outputs
        out.scatter_add_(
            1,
            cur_sparse_indices[:, :, : out.shape[2]],
            torch.multiply(sparse_o, alphas),
        )

    return out


class DilatedSelfAttention(torch.nn.Module):
    def __init__(
        self, ws: List[int], rs: List[int], head_idx: int, attn_module: torch.nn.Module
    ):
        super().__init__()
        assert len(ws) > 0
        assert len(ws) == len(rs)
        assert (
            len(set([w // r for w, r in zip(ws, rs)])) == 1
        ), "for now w/r ratios should be identical to simplify batching"

        self.ws = ws
        self.rs = rs
        self.head_idx = head_idx
        self.attn = attn_module

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, n, c = x.size()

        sparse_indices, sparse_values = _segment_and_sparsify(
            x, self.ws, self.rs, self.head_idx
        )

        # batch values, compute attention in parallel and "unbatch" outputs
        batched_values = torch.cat(sparse_values, dim=0)
        batched_os, batched_att_denoms = self.attn(batched_values)
        sparse_os = torch.split(batched_os, b, 0)
        sparse_att_denoms = torch.split(batched_att_denoms, b, 0)

        return _mix_outputs(
            (b, n, self.attn.out_dim),  # todo: specify out c explicitly
            x.dtype,
            x.device,
            sparse_os,
            sparse_att_denoms,
            sparse_indices,
        )
