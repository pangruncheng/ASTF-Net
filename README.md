# Overview

ASTF-net is an open-source deep learning framework designed for apparent seismic source time function inversion.

# Data
![ASTF-net Logo](https://github.com/user-attachments/assets/c7c4a6e9-006a-447b-b697-31ed34e42424)


## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/ASTF-net.git
cd ASTF-net

# Create a conda environment
conda create -n astfnet python=3.10
conda activate astfnet

# Install the package
pip install -e .

# Install development dependencies (optional)
pip install -e ".[dev]"
```

## Development

### Setting up development environment

```bash
# Install development dependencies
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install

# Run tests
pytest tests/
```

### Code Quality

The project uses:
- **Ruff**: Fast Python linter and formatter
- **pre-commit**: Git hooks for code quality
