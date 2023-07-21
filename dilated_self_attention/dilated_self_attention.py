import math
from typing import List, Tuple, Union, Optional

import torch


def _prepare_sparse_indices(
    x: torch.Tensor, ws: List[int], rs: List[int], head_idx: int
) -> Tuple[int, torch.Tensor, Optional[torch.Tensor]]:
    b, n, c = x.size()

    x_indices = torch.arange(0, n, dtype=torch.long, device=x.device)[None, :, None]

    num_subatt = sum([int(math.ceil(n / w)) for w in ws])
    max_subatt_n = min(n, max([w // r for w, r in zip(ws, rs)]))

    sparse_indices = -1*torch.ones((b, num_subatt * max_subatt_n, c), device=x.device, dtype=torch.int64)

    subatt_idx = 0
    for w, r in zip(ws, rs):
        for segment_indices in torch.split(x_indices, w, 1):
            offset = head_idx % r
            cur_sparse_indices = segment_indices[:, offset::r, :]
            start_idx = subatt_idx*max_subatt_n
            end_idx = start_idx+cur_sparse_indices.shape[1]
            sparse_indices[:, start_idx:end_idx] = cur_sparse_indices
            subatt_idx += 1

    if -1 in sparse_indices:
        padding_mask = sparse_indices[:, :, 0] != -1

        # to allow gather work for batching
        sparse_indices[~padding_mask] = 0

        # combine batch and subattention dims
        padding_mask = padding_mask.view((-1, max_subatt_n))
    else:
        padding_mask = None

    return max_subatt_n, sparse_indices, padding_mask


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

    # select attention softmax denominator sums for current sparse indices
    sparse_att_denom_sum = torch.gather(att_denom_sums, 1, a_indices[:, :, 0])

    # compute alphas
    alphas = torch.divide(a_denoms, sparse_att_denom_sum)[:, :, None]

    out = torch.zeros(out_shape, dtype=out_dtype, device=out_device)

    out.scatter_add_(
        1,
        a_indices[:, :, :out.shape[2]],
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

        # todo: deprecate this requirement
        assert (
            len(set([w // r for w, r in zip(ws, rs)])) == 1
        ), "for now w/r ratios should be identical to simplify batching"

        self.ws = ws
        self.rs = rs
        self.head_idx = head_idx
        self.attn = attn_module
        self.indices = None
        self.indices_shape = None
        self.max_subatt_n = None
        self.padding_mask = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, n, in_c = x.size()

        if self.indices is None or self.indices_shape != x.shape:
            self.max_subatt_n, self.indices, self.padding_mask = _prepare_sparse_indices(
                x, self.ws, self.rs, self.head_idx
            )
            self.indices_shape = x.shape

        # extract subsets for each "subattention" from x elements
        sparse_x = torch.gather(x, 1, self.indices)

        # batch "subattention" values and process in parallel
        batched_x = sparse_x.view((-1, self.max_subatt_n, in_c))
        batched_os, batched_att_denoms = self.attn(batched_x, self.padding_mask)

        # "unbatch" attention outputs
        out_c = batched_os.shape[-1]
        sparse_os = batched_os.view((b, -1, out_c))
        sparse_denoms = batched_att_denoms.view((b, -1))
        sparse_indices = self.indices

        # drop padded elements
        if self.padding_mask is not None:
            padding_mask_flat = self.padding_mask.view((b, -1))
            sparse_os = sparse_os[padding_mask_flat].view((b, -1, out_c))
            sparse_denoms = sparse_denoms[padding_mask_flat].view((b, -1))
            sparse_indices = sparse_indices[padding_mask_flat].view((b, -1, in_c))

        return _mix_outputs(
            (b, n, out_c),
            x.dtype,
            x.device,
            sparse_os,
            sparse_denoms,
            sparse_indices,
        )
