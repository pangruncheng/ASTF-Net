# Overview

ASTF-net is an open-source deep learning framework designed for apparent seismic source time function inversion.

# Data
![ASTF-net Logo](https://github.com/user-attachments/assets/c7c4a6e9-006a-447b-b697-31ed34e42424)


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

Install the `astfnet` package with:
```bash
uv sync --all-extras
```

This installs the Python version and dependencies specified in `pyproject.toml`. The --all-extra flag ensures that the development and documentation dependencies are also installed. By default, uv sync installs the package in editable mode, which means that the package does not need to be reinstalled for modifications to the source code to take effect. Running uv sync also exposes the CLI entry points listed under `[project.scripts]` in `pyproject.toml`.

Install pre-commit hooks with:
```bash
pre-commit install
```
Test
```bash
pytest tests/
```
