#! /usr/bin/env python3
from setuptools import setup

setup(
    name="dltk",
    version="0.1.0",
    license="MIT",
    packages=["dltk"],
    install_requires=[
        "torch",
        "torchmetrics",
        "tqdm"
    ],
    scripts=[
        "bin/make-mwoz23-archive",
        "bin/launch-workers",
        "bin/train-shda"
    ]
)
