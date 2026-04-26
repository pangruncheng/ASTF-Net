# Overview

ASTF-net is an open-source deep learning framework designed for apparent seismic source time function inversion.

# Data

The dataset used for ASTF-net is **HiASTF**, available on Hugging Face at
[https://huggingface.co/datasets/pangruncheng/HiASTF](https://huggingface.co/datasets/pangruncheng/HiASTF).

> **Note:** This is a gated dataset. You must log in to Hugging Face and accept the dataset conditions before downloading.

Install the Hugging Face CLI if it is not already available:
```bash
pip install huggingface_hub
```

Log in with your Hugging Face account and download the dataset to a local directory:
```bash
huggingface-cli login
huggingface-cli download pangruncheng/HiASTF --repo-type dataset --local-dir /path/to/data/
```

![ASTF-net Logo](https://github.com/user-attachments/assets/c7c4a6e9-006a-447b-b697-31ed34e42424)

# Dataset Setup

After downloading, your data directory should have the following structure:

```
/path/to/data/
├── Train_new_3_pairs_Samezero256_normalize_vr_ratio2_min0_2.h5
├── Validation_new_3_pairs_Samezero256_normalize_vr_ratio2_min0_2.h5
└── test_set/
    ├── Test_level1_new_3_pairs_Samezero256_normalize_vr_1.h5
    ├── Test_level2_new_3_pairs_Samezero256_normalize_vr_1.h5
    └── Test_level3_new_3_pairs_Samezero256_normalize_vr_1.h5
```

These canonical filenames are defined in [`astfnet/constants.py`](astfnet/constants.py):

| Constant | File |
|---|---|
| `TRAIN_FILENAME` | `Train_new_3_pairs_Samezero256_normalize_vr_ratio2_min0_2.h5` |
| `VAL_FILENAME` | `Validation_new_3_pairs_Samezero256_normalize_vr_ratio2_min0_2.h5` |
| `TEST_FILENAMES` | Three `Test_level{1,2,3}_*` files inside `test_set/` |

Pass the root data directory to CLI commands via the `--data` flag (see [Usage](#usage)).

### Requirements
Ensure that [Python](https://www.python.org/downloads/) and [Git](https://git-scm.com/downloads) are configured on your machine with:
```bash
python3 --version
git --version
```

It is recommended to use [uv](https://docs.astral.sh/uv/) for dependency management. uv can be installed with:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Other [uv installation methods](https://docs.astral.sh/uv/getting-started/installation/) can be found on the uv website.

### Installation
Download the repository and navigate to its root directory with:
```bash
git clone https://github.com/pangruncheng/ASTF-net.git
cd ASTF-net
```

Create a virtual environment and activate it with:
```bash
uv venv
source .venv/bin/activate
```

Install the `astfnet` package. Choose the mode that fits your use case:

```bash
# Standard install — core dependencies only
uv sync

# Development install — adds pre-commit, pytest, pytest-cov, ruff, and ty
uv sync --extra dev

# Full install — development and documentation dependencies
uv sync --all-extras
```

By default, `uv sync` installs the package in editable mode, which means that the package does not need to be reinstalled for modifications to the source code to take effect. Running `uv sync` also exposes the CLI entry points listed under `[project.scripts]` in `pyproject.toml`.

Install pre-commit hooks with:
```bash
pre-commit install
```
Test
```bash
pytest tests/
```
### Usage
Installing the application with uv sync exposes the following CLI entry points:
```
astfnet-train               # train a model
```
The entry points can be called directly. For example:
```bash
astfnet-train --config config/config-simplecnn.yaml --data /path/to/data/
```

More information on the usage and the options available for these entry points can be obtained by calling the --help option. For example:
```bash
astfnet-train --help
```
