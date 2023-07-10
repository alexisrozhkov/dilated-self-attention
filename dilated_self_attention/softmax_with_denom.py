from typing import Tuple

import torch


def softmax_with_denom(x: torch.Tensor, dim: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """not numerically stable"""
    numerator = torch.exp(x)
    denominator = torch.sum(numerator, dim=dim, keepdim=True)
    return numerator / denominator, denominator.squeeze(dim=dim)
