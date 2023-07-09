# Dilated Self Attention
This is an attempt to implement the dilated self attention as described in 
[LongNet: Scaling Transformers to 1,000,000,000 Tokens](https://arxiv.org/abs/2307.02486) by Jiayu Ding et al.


## Installation
```shell
virtualenv -p python3.8 .venv
source .venv/bin/activate

# 2 steps below are optional, use to regenerate requirements.txt for your platform
pip install pip-tools
pip-compile

pip install -r requirements.txt
```

## Usage
### Run tests
```shell
nose2
```