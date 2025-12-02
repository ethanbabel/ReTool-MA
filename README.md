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

3. Install MARTI dependencies:
```bash
cd MARTI
uv pip install -r requirements.txt
```

4. Install huggingface_hub:
```bash
uv pip install huggingface_hub
```

5. Download desired model for local use:

For example with Qwen2.5-3B-Instruct.
```bash
mkdir -p ReTool-MA/models/Qwen2.5-3B-Instruct
hf download Qwen/Qwen2.5-3B-Instruct --local-dir ReTool-MA/models/Qwen2.5-3B-Instruct
```

## Running the ReTool-MA workflow

### Quick sample run
Once dependencies are installed you can exercise a small multi-agent Planner/Executor/Verifier workflow:

```bash
cd MARTI
source ../.venv/bin/activate
export MODEL_PATH=/workspace/ReTool-MA/models/Qwen2.5-3B-Instruct

python -m marti.cli.commands.test_new \
  --config-name ma_retool_ma \
  default_agent.pretrain=$MODEL_PATH \
  default_agent.vllm_num_engines=1 \
  default_agent.vllm_tensor_parallel_size=1 \
  default_agent.prompt_max_len=2048 \
  default_agent.generate_max_len=1024 \
  default_agent.rollout_batch_size=2 \
  default_agent.micro_rollout_batch_size=1 \
  default_agent.temperature=0.2 \
  default_agent.max_samples=4 \
  packing_samples=false \
  prompt_data=data/AMC/test.jsonl \
  prompt_split=train \
  input_key=problem \
  label_key=answer \
  add_prompt_suffix=""
```

- `ma_retool_ma` loads the custom workflow (`marti/worlds/workflows/retool_ma_workflow.py`) plus a `tools_config` that uses the local Python executor, so no external sandbox service is required.
- Results for each problem land in `MARTI/ckpt/results.json` and `MARTI/ckpt/summary.json`. The summary aggregates final accuracy, code execution pass rates, the conditional pass rates for correct vs. incorrect trajectories, and the per-role shaping rewards.

### Full training run
For a longer RL run on the same workflow, first install flash-attn (ensure torch is installed first):
```bash
uv pip install --no-build-isolation flash_attn==2.7.0.post2
```

Then to start training:
```bash
cd MARTI
source ../.venv/bin/activate
export MODEL_PATH=/workspace/ReTool-MA/models/Qwen2.5-3B-Instruct
export WANDB_API_KEY=<YOUR_WANDB_KEY>

ray start --head

bash scripts/run_train_retool_ma.sh $MODEL_PATH $WANDB_API_KEY
```

The script wraps `python -m marti.cli.commands.train --config-name ma_retool_ma` with resource settings that fit a single RTX 5090 (one vLLM engine, small rollout batches) and streams logs/checkpoints to `MARTI/outputs` plus W&B/TensorBoard.

## Model size guidance

The RTX 5090 used in development exposes ~32 GB of VRAM. That budget easily fits three concurrent agents when each runs a `Qwen2.5-3B-Instruct` checkpoint under vLLM. Larger models (e.g., 7B) leave little headroom for executor buffers and RL rollout batches, so start with 3B for all roles. Once the pipeline is stable you can selectively scale the Planner to a larger checkpoint if desired.
