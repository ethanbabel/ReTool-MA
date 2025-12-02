#!/bin/bash
set -x

MODEL_DIR=${1}
WANDB_KEY=${2}
ROOT_DIR=$(pwd)

DATE=$(date +%m%d)
ADVANTAGE="reinforce"
SHORT_NAME="Qwen2.5-3B-Instruct"
TASK="AMC"
ALGO="retool-ma"
PRETRAIN="${MODEL_DIR}"
EXP="${DATE}-${TASK}-${SHORT_NAME}-${ADVANTAGE}-${ALGO}"

SAVE_PATH="${ROOT_DIR}/outputs/${ADVANTAGE}-${ALGO}/${DATE}/${SHORT_NAME}/model"
PROMPT_DATA="json@${ROOT_DIR}/data/${TASK}"
TENSORBOARD="${ROOT_DIR}/logs/tensorboard/${ADVANTAGE}-${ALGO}-${DATE}-${SHORT_NAME}"
CKPT_PATH="${ROOT_DIR}/outputs/${ADVANTAGE}-${ALGO}/${DATE}/${SHORT_NAME}/ckpt"

mkdir -p "${ROOT_DIR}/logs"
mkdir -p "${ROOT_DIR}/logs/std"
mkdir -p "${ROOT_DIR}/logs/tensorboard"
mkdir -p "${ROOT_DIR}/outputs"

PROMPT_MAX_LEN=2048
GENERATE_MAX_LEN=1024

export PYTORCH_NVML_BASED_CUDA_CHECK=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_ENABLE_V1_MULTIPROCESSING=1
export HYDRA_FULL_ERROR=1
export CUDA_LAUNCH_BLOCKING=1
export NCCL_DEBUG=WARN
export RAY_PICKLE_VERBOSE_DEBUG=1
export WANDB_API_KEY="${WANDB_KEY}"

ENV_JSON=$(cat <<EOF
{
  "working_dir": "${ROOT_DIR}",
  "excludes": ["data/", "outputs/", ".git/", "local/", "logs/"],
  "pip": ["hydra-core", "antlr4-python3-runtime==4.9.3", "shortuuid", "class_registry", "json5", "mcp[cli]"]
}
EOF
)

ray job submit --address="http://localhost:8265" \
    --runtime-env-json="${ENV_JSON}" \
    -- python -m marti.cli.commands.train --config-name "ma_retool_ma" \
    credit_ref_num_gpus_per_node=0 \
    credit_num_gpus_per_node=0 \
    parallel_loading=False \
    default_agent.is_reasoning_model=False \
    default_agent.ref_num_nodes=1 \
    ++default_agent.ref_num_gpus_per_node=0 \
    default_agent.critic_num_nodes=1 \
    ++default_agent.critic_num_gpus_per_node=0 \
    ++default_agent.reward_num_gpus_per_node=0 \
    default_agent.actor_num_nodes=1 \
    ++default_agent.actor_num_gpus_per_node=1 \
    default_agent.vllm_num_engines=0 \
    default_agent.vllm_tensor_parallel_size=1 \
    default_agent.vllm_gpu_memory_utilization=0.6 \
    default_agent.pretrain="${PRETRAIN}" \
    default_agent.save_path="${SAVE_PATH}" \
    default_agent.micro_train_batch_size=2 \
    default_agent.train_batch_size=32 \
    default_agent.num_episodes=5 \
    default_agent.save_steps=500 \
    default_agent.eval_steps=2 \
    default_agent.logging_steps=1 \
    default_agent.max_samples=200000 \
    default_agent.micro_rollout_batch_size=1 \
    default_agent.rollout_batch_size=8 \
    default_agent.training_mode="rl" \
    default_agent.n_samples_per_prompt=2 \
    default_agent.max_epochs=1 \
    default_agent.prompt_max_len=${PROMPT_MAX_LEN} \
    default_agent.generate_max_len=${GENERATE_MAX_LEN} \
    default_agent.advantage_estimator=${ADVANTAGE} \
    default_agent.temperature=0.8 \
    default_agent.zero_stage=2 \
    default_agent.bf16=True \
    default_agent.actor_learning_rate=1e-6 \
    default_agent.critic_learning_rate=9e-6 \
    default_agent.init_kl_coef=0.00 \
    default_agent.max_ckpt_num=20 \
    default_agent.normalize_reward=True \
    default_agent.ckpt_path="${CKPT_PATH}" \
    workflow_args.num_rounds=1 \
    workflow_func_path="marti/worlds/workflows/retool_ma_workflow.py" \
    processor_func_path="marti/worlds/workflows/default_processor.py" \
    tools_config.num_workers=8 \
    eval_before_training=False \
    prompt_data="${PROMPT_DATA}" \
    input_key="problem" \
    label_key="answer" \
    add_prompt_suffix="" \
    wandb_project="ReTool-MA" \
    wandb_run_name="${EXP}" \
    use_wandb="${WANDB_KEY}" \
    use_tensorboard="${TENSORBOARD}" 2>&1 | tee "${ROOT_DIR}/logs/std/${DATE}-${EXP}.log"

echo "Model Training Finished. Shutting down..."
