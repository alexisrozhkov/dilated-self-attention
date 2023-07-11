#!/usr/bin/env python
from distutils.core import setup


install_requires = []

with open("requirements.in") as f:
    for line in f:
        install_requires.append(line.strip())

setup(
    name="dilated-self-attention",
    version="0.1.0",
    description="Implementation of dilated self-attetion as described in "
    "LongNet: Scaling Transformers to 1,000,000,000 Tokens by "
    "Jiayu Ding et al.",
    author="Alexey Rozhkov",
    author_email="alexisrozhkov@gmail.com",
    license="MIT",
    packages=["dilated_self_attention"],
    install_requires=install_requires,
)
