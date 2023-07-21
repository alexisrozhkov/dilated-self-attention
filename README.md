# Dilated Self Attention
This is an attempt to implement the dilated self attention as described in 
[LongNet: Scaling Transformers to 1,000,000,000 Tokens](https://arxiv.org/abs/2307.02486) by Jiayu Ding et al.

## Benchmark results
![Benchmark results](assets/benchmark.svg)

## Installation
### Basic
```shell
virtualenv -p python3.8 .venv
source .venv/bin/activate

# 2 steps below are optional, use to regenerate requirements.txt for your platform
pip install pip-tools
pip-compile

pip install -r requirements.txt
```

### Optimised self-attention implementation
After installing the basic dependencies you can install flash-attn module.  
It is optional, since currently it is not fully integrated, some lower-tier GPUs aren't supported and the installation takes a while.
```shell
pip install flash-attn==1.0.5
```

## Usage
### Run tests
```shell
nose2
```

### Run benchmark
CLI interface:
```shell
usage: benchmark.py [-h] [--num_seq_lens NUM_SEQ_LENS] [--num_iter NUM_ITER] [--num_heads NUM_HEADS] [--emb_dim EMB_DIM] [--device DEVICE] is_dilated max_seq_len

positional arguments:
  is_dilated            Whether to benchmark a dilated or vanilla self-attention
  max_seq_len           Maximum sequence length to benchmark

optional arguments:
  -h, --help            show this help message and exit
  --num_seq_lens NUM_SEQ_LENS
                        Number of sequence length to evaluate (each is 2x larger than the previous one) (default: 4)
  --num_iter NUM_ITER   Number of iterations to repeat the time measurement for (using new random input each time) (default: 200)
  --num_heads NUM_HEADS
                        Number of heads for multi-head self-attention (default: 3)
  --emb_dim EMB_DIM     Embedding dimensionality (default: 384)
  --device DEVICE       Device to put the model and input on (default: cuda:0)
```

Example benchmark output (on Google Colab instance with T4 GPU):
```shell
> python benchmark.py 0 16384
8 x 2048:
2.4 ms
4 x 4096:
11.1 ms
2 x 8192:
45.5 ms
1 x 16384:
208.6 ms

> python python benchmark.py 1 16384
8 x 2048:
7.0 ms
4 x 4096:
10.8 ms
2 x 8192:
20.2 ms
1 x 16384:
26.1 ms
```
Output format:
```shell
{batch size} x {sequence length}:
{sequence inference time}  
```


## Current status
Baseline training and inference is supported, although with some restrictions on the dilated attention configuration and sequence lengths.  
Attempts to use an optimised self-attention implementation were made, but when a softmax denominators are requested the memory usage and inference speed degrade substantially. Authors of the paper mention:

> All of our implementations of attention variants are based on FlashAttention for training efficiency. We customize the flash attention kernels for both sparse
attention and dilated attention.

So perhaps something similar would be necessary, unless another optimised implementation that exposes softmax denominator efficiently would become available.  

If the implementation in this repo is benchmarked vs a vanilla self-attention implementation, the losses seems to be similar to the ones obtained with vanilla implementation, and the GPU RAM and inference speeds scale linearly with the sequence length as expected. 


## To Do
- [x] Benchmarking code and reports for dilated self-attention vs vanilla one
- [ ] Support different w to r ratios for multi-k attention
- [ ] Add dropout(s)
- [ ] Support optimised self-attention implementation (needs to expose softmax denominators efficiently)
- [ ] Distributed training using multiple GPUs handling parts of the sequence
- [ ] Make sure torch.compile works properly (currently I get NaNs at the first iteration of training)
