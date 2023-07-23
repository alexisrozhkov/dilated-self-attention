import argparse
import math
import time

import torch

from dilated_self_attention.multihead_dilated_self_attention import MultiheadDilatedSelfAttention


class MultiheadCausalSelfAttention(torch.nn.Module):
    """
    Stripped down version of Andrej Karpathy's implementation from
    https://github.com/karpathy/minGPT
    """
    def __init__(self, embedding_dim: int, num_head: int, max_n: int, flash: bool):
        super().__init__()
        assert embedding_dim % num_head == 0
        self.c_attn = torch.nn.Linear(embedding_dim, 3 * embedding_dim)
        self.c_proj = torch.nn.Linear(embedding_dim, embedding_dim)
        self.n_head = num_head
        self.n_embd = embedding_dim
        self.flash = flash

        if not self.flash:
            self.register_buffer("bias", torch.tril(torch.ones(max_n, max_n)).view(1, 1, max_n, max_n))

    def forward(self, x):
        B, T, C = x.size()
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)

        if self.flash:
            y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=0, is_causal=True)

        else:
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
            att = torch.nn.functional.softmax(att, dim=-1)
            y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)

        y = y.transpose(1, 2).contiguous().view(B, T, C)  # re-assemble all head outputs side by side

        return self.c_proj(y)


def benchmark_single_config(model: torch.nn.Module, seq_len: int, batch_size: int, emb_dim: int, num_iter: int, device: str) -> float:
    """Run inference num_iter times and return average time per iteration in milliseconds"""
    start_t = time.time()

    for i in range(num_iter):
        x = torch.normal(0, 1, (batch_size, seq_len, emb_dim), device=device)
        attn_output = model(x)

    duration_ms = (time.time() - start_t) * 1000 / num_iter
    return duration_ms


def benchmark_single_model(model: torch.nn.Module, max_seq_len: int, num_seq_lens: int, emb_dim: int, num_iter: int, device: str):
    seq_lens = reversed([int(max_seq_len/(2**i)) for i in range(num_seq_lens)])

    for seq_len in seq_lens:
        # keep number of tokens per batch fixed
        batch_size = max_seq_len//seq_len
        print(f"{batch_size} x {seq_len}:")

        time_per_batch_ms = benchmark_single_config(model, seq_len, batch_size, emb_dim, num_iter, device)

        # normalise the time by the batch size
        time_per_seq_ms = time_per_batch_ms/batch_size
        print(f"{time_per_seq_ms:.1f} ms")


def main(is_dilated: bool, max_seq_len: int, num_seq_lens: int, num_iter: int, num_heads: int, emb_dim: int, device: str, flash: bool):
    if is_dilated:
        model = MultiheadDilatedSelfAttention(
            [1024, 4096, 16384],
            [1, 4, 16],
            emb_dim,
            num_heads,
            flash
        ).to(device)

    else:
        model = MultiheadCausalSelfAttention(
            emb_dim,
            num_heads,
            max_seq_len,
            flash
        ).to(device)

    benchmark_single_model(model, max_seq_len, num_seq_lens, emb_dim, num_iter, device)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "is_dilated",
        help="Whether to benchmark a dilated or vanilla self-attention",
        type=int,
    )

    parser.add_argument(
        "max_seq_len",
        help="Maximum sequence length to benchmark",
        type=int,
    )
    parser.add_argument(
        "--num_seq_lens",
        help="Number of sequence length to evaluate (each is 2x larger than the previous one)",
        default=4,
        type=int,
    )

    parser.add_argument(
        "--num_iter",
        help="Number of iterations to repeat the time measurement for (using new random input each time)",
        default=200,
        type=int,
    )

    parser.add_argument(
        "--num_heads",
        help="Number of heads for multi-head self-attention",
        default=3,
        type=int,
    )

    parser.add_argument(
        "--emb_dim",
        help="Embedding dimensionality",
        default=384,
        type=int,
    )

    parser.add_argument(
        "--device",
        help="Device to put the model and input on",
        default="cuda:0",
        type=str,
    )

    parser.add_argument(
        "--flash",
        help="Whether to use optimised self-attention implementation from flash-attn",
        default=0,
        type=int,
    )

    args = parser.parse_args()

    main(
        bool(args.is_dilated),
        args.max_seq_len,
        args.num_seq_lens,
        args.num_iter,
        args.num_heads,
        args.emb_dim,
        args.device,
        bool(args.flash)
    )
