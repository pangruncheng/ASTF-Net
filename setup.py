#!/usr/bin/env python3
"""
Setup script for ASTF-net: A deep learning framework for seismic source time function inversion.
"""

from setuptools import setup, find_packages


# Read the README file
def read_readme():
    with open("README.md", "r", encoding="utf-8") as fh:
        return fh.read()


setup(
    name="astf-net",
    version="0.1.0",
    description="A deep learning framework for seismic source time function inversion",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/ASTF-net",
    packages=find_packages(),
    classifiers=[
        "Intended Audience :: Science/Research/Education/Commercial",
        "License :: OSI Approved :: Apache Software License",
        "Topic :: Scientific/Engineering :: Geophysics",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.10",
    install_requires=[
        "torch>=1.9.0",
        "torchvision>=0.10.0",
        "numpy>=1.21.0",
        "scipy>=1.7.0",
        "obspy>=1.2.0",
        "h5py>=3.1.0",
        "pandas>=1.3.0",
        "matplotlib>=3.4.0",
        "tqdm>=4.62.0",
        "scikit-learn>=1.0.0",
        "jupyter>=1.0.0",
        "hydra-core>=1.3.0",
        "omegaconf>=2.3.0",
        "speechbrain",
        "pytorch-lightning",
    ],
    extras_require={
        "dev": [
            "pre-commit",
            "pytest",
            "ruff",
            "ty",
        ],
        "docs": [
            "sphinx",
        ],
    },
    entry_points={
        "console_scripts": [
            "astf-train=astf.cli.train:main",
            "astf-predict=astf.cli.predict:main",
        ],
    },
    include_package_data=True,
    package_data={
        "astf": ["configs/*.yaml", "configs/*.json"],
    },
    zip_safe=False,
)
