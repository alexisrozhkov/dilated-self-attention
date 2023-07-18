import math
import time

import torch

from dilated_self_attention.multihead_dilated_self_attention import MultiheadDilatedSelfAttention


class MultiheadCausalSelfAttention(torch.nn.Module):
    """
    Stripped down version of Andrej Karpathy's implementation from
    https://github.com/karpathy/minGPT
    """
    def __init__(self, embedding_dim: int, num_head: int, max_n: int):
        super().__init__()
        assert embedding_dim % num_head == 0
        self.c_attn = torch.nn.Linear(embedding_dim, 3 * embedding_dim)
        self.c_proj = torch.nn.Linear(embedding_dim, embedding_dim)
        self.n_head = num_head
        self.n_embd = embedding_dim
        self.register_buffer("bias", torch.tril(torch.ones(max_n, max_n)).view(1, 1, max_n, max_n))

    def forward(self, x):
        B, T, C = x.size()
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)

        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
        att = torch.nn.functional.softmax(att, dim=-1)
        y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C)  # re-assemble all head outputs side by side

        return self.c_proj(y)


def benchmark_single_config(model: torch.nn.Module, seq_len: int, batch_size: int, num_iter: int, device: str) -> float:
    """Run inference num_iter times and return average time per iteration in milliseconds"""
    start_t = time.time()

    for i in range(num_iter):
        x = torch.normal(0, 1, (batch_size, seq_len, emb_dim), device=device)
        attn_output = model(x)

    duration_ms = (time.time() - start_t) * 1000 / num_iter
    return duration_ms


def benchmark_single_model(model: torch.nn.Module, max_seq_len: int, num_seq_lens: int, num_iter: int, device: str):
    seq_lens = reversed([int(max_seq_len/(2**i)) for i in range(num_seq_lens)])

    for seq_len in seq_lens:
        # keep number of tokens per batch fixed
        batch_size = max_seq_len//seq_len
        print(f"{batch_size} x {seq_len}:")

        time_per_batch_ms = benchmark_single_config(model, seq_len, batch_size, num_iter, device)

        # normalise the time by the batch size
        time_per_seq_ms = time_per_batch_ms/batch_size
        print(f"{time_per_seq_ms:.1f} ms")


def main(is_dilated: bool, max_seq_len: int, num_seq_lens: int, num_iter: int, num_heads: int, emb_dim: int, device: str):
    if is_dilated:
        model = MultiheadDilatedSelfAttention(
            [512, 2048],
            [1, 4],
            emb_dim,
            num_heads
        ).to(device)

    else:
        model = MultiheadCausalSelfAttention(
            emb_dim,
            num_heads,
            max_seq_len
        ).to(device)

    benchmark_single_model(model, max_seq_len, num_seq_lens, num_iter, device)


if __name__ == "__main__":
    max_seq_len = 1024 * 8
    num_seq_lens = 4
    num_heads = 6
    emb_dim = 64 * num_heads
    device = 'cuda:0'
    num_iter = 100

    main(False, max_seq_len, num_seq_lens, num_iter, num_heads, emb_dim, device)
