import math
from typing import List, Tuple, Union

import torch


def _segment_and_sparsify(
    x: torch.Tensor, ws: List[int], rs: List[int], head_idx: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    b, n, c = x.size()

    x_indices = torch.arange(0, n, dtype=torch.long, device=x.device)[None, :, None]

    num_subatt = sum([int(math.ceil(n / w)) for w in ws])
    max_subatt_n = ws[0] // rs[0]
    sparse_indices = torch.zeros(
        (b, num_subatt * max_subatt_n, c), device=x.device, dtype=torch.int64
    )

    subatt_idx = 0
    for w, r in zip(ws, rs):
        for segment_indices in torch.split(x_indices, w, 1):
            offset = head_idx % r
            sparse_indices[
                :, subatt_idx * max_subatt_n : (subatt_idx + 1) * max_subatt_n
            ] = segment_indices[:, offset::r, :]
            subatt_idx += 1

    sparse_values = torch.gather(x, 1, sparse_indices).view(
        (b * num_subatt, max_subatt_n, c)
    )

    return sparse_indices, sparse_values


def _mix_outputs(
    out_shape: Tuple[int, int, int],
    out_dtype: torch.dtype,
    out_device: Union[torch.device, str],
    a_os: torch.Tensor,
    a_denoms: torch.Tensor,
    a_indices: torch.Tensor,
) -> torch.Tensor:
    # calculate sums of softmax denominators
    att_denom_sums = torch.zeros((out_shape[0], out_shape[1]), device=out_device)
    att_denom_sums.scatter_add_(1, a_indices[:, :, 0], a_denoms)

    out = torch.zeros(out_shape, dtype=out_dtype, device=out_device)

    # select attention softmax denominator sums for current sparse indices
    sparse_att_denom_sum = torch.gather(att_denom_sums, 1, a_indices[:, :, 0])

    # compute alphas
    alphas = torch.divide(a_denoms, sparse_att_denom_sum)[:, :, None]

    out.scatter_add_(
        1,
        a_indices[:, :, : out.shape[2]],
        torch.multiply(a_os, alphas),
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

        a_indices, batched_values = _segment_and_sparsify(
            x, self.ws, self.rs, self.head_idx
        )

        # batch values, compute attention in parallel and "unbatch" outputs
        batched_os, batched_att_denoms = self.attn(batched_values)

        a_denoms = torch.cat(torch.split(batched_att_denoms, b, 0), dim=1)
        a_os = torch.cat(torch.split(batched_os, b, 0), dim=1)

        return _mix_outputs(
            (b, n, self.attn.out_dim),  # todo: specify out c explicitly
            x.dtype,
            x.device,
            a_os,
            a_denoms,
            a_indices,
        )
