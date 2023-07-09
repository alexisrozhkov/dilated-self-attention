import torch


class DilatedCausalSelfAttention(torch.nn.Module):
    def __init__(self, w: int, r: int, offset: int, attn_module: torch.nn.Module):
        """
        https://arxiv.org/pdf/2307.02486.pdf

        :param w: segment length, trades the globality of attention for efficiency
        :param r: dilated rate, reduces the computation cost by approximating the attention matrix
        :param offset: offset of sparse indices, must be less than r
        :param attn_module: attention module instance to use under the hood
        """
        super().__init__()
        # assert w % r == 0
        assert offset < r

        self.w = w
        self.r = r
        self.offset = offset
        self.attn = attn_module
        max_len = self.w // self.r
        self.attn_mask = torch.tril(torch.ones(max_len, max_len))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, n, c = x.size()
        x_indices = torch.arange(0, n, dtype=torch.long)[None, :, None].repeat(b, 1, c)

        out = torch.zeros_like(x)

        # todo: rearrange to put chunks along the batch dimension
        # then a single attention forward will do the trick
        for segment_indices in torch.split(x_indices, self.w, 1):
            sparse_indices = segment_indices[:, self.offset::self.r, :]
            x_i = torch.gather(x, 1, sparse_indices)
            o_i = self.attn(x_i, x_i, x_i, is_causal=True, attn_mask=self.attn_mask)[0]
            alpha_i = 1.0  # todo: proper alpha calculation
            out.scatter_add_(1, sparse_indices, alpha_i * o_i)

        return out
