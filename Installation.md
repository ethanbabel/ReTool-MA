# ReTool-MA

## Installation

### Note

Personal installation instructions for local development on Apple silicon. 

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

3. Install vLLM for CPU (build wheel from source):

Add vLLM repository as a submodule (if needed).
```bash
cd /Users/Ethan/Downloads
git submodule add https://github.com/vllm-project/vllm.git
```

Install vLLM dependencies.
```bash
cd vllm
uv pip install -r requirements/cpu.txt --index-strategy unsafe-best-match
uv pip install -e .
```
In some cases `uv pip install -e .` will error if done in a VSCode terminal.

4. Install MARTI:

Add personal fork of MARTI repository as a subtree in project root (if needed).
```bash
git subtree add --prefix=MARTI git@github.com:ethanbabel/MARTI.git main --squash
```
Install MARTI dependencies.
```bash
cd MARTI
uv pip install -r requirements.txt
```
