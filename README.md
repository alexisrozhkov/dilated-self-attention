# Dilated Self Attention
This is an attempt to implement the dilated self attention as described in 
[LongNet: Scaling Transformers to 1,000,000,000 Tokens](https://arxiv.org/abs/2307.02486) by Jiayu Ding et al.


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

## Current status
Baseline training and inference is supported, although with some restrictions on the dilated attention configuration and sequence lengths.  
Attempts to use an optimised self-attention implementation were made, but when a softmax denominators are requested the memory usage and inference speed degrade substantially. Authors of the paper mention:

> All of our implementations of attention variants are based on FlashAttention for training efficiency. We customize the flash attention kernels for both sparse
attention and dilated attention.

So perhaps something similar would be necessary, unless another optimised implementation that exposes softmax denominator efficiently would become available.  

If the implementation in this repo is benchmarked vs a vanilla self-attention implementation, the losses seems to be similar to the ones obtained with vanilla implementation, and the GPU RAM and inference speeds scale linearly with the sequence length as expected. 


## To Do
- [ ] Benchmarking code and reports for dilated self-attention vs vanilla one
- [ ] Support different w to r ratios for multi-k attention
- [ ] Support optimised self-attention implementation (needs to expose softmax denominators efficiently)
- [ ] Distributed training using multiple GPUs handling parts of the sequence
- [ ] Make sure torch.compile works properly (currently I get NaNs at the first iteration of training)
