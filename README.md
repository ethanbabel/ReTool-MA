# ReTool-MA

## Installation

### Prerequisites

- Python 3.11
- uv package manager

### Setup

1. In the desired location, clone the repository:
```bash
git clone https://github.com/ethanbabel/ReTool-MA.git
cd ReTool-MA
```

2. Create and activate a virtual environment:
```bash
uv venv
source .venv/bin/activate
```

3. Install MARTI:

Add personal fork of MARTI repository as a subtree in project root (if needed).
```bash
git subtree add --prefix=MARTI git@github.com:ethanbabel/MARTI.git main --squash
```
Install MARTI dependencies.
```bash
cd MARTI
uv pip install -r requirements.txt
```
