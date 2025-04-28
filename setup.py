#!/usr/bin/env python

from setuptools import find_packages, setup

setup(
    name="llm-adsorbent-search",
    version="0.0.1",
    description="Use an LLM to find useful adsorbent materials",
    author="Adam Wentz",
    author_email="adam@adamwentz.com",
    url="https://github.com/awentzonline/llm-adsorbent-search",
    install_requires=[
        "ase",
        "pydantic-ai[logfire]",
        "numpy",
        "torch",
        "fairchem-core",
        "wandb",
    ],
    packages=find_packages(),
)
