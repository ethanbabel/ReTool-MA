#!/bin/bash
set -x

source /cpfs04/user/xxxx/anaconda3/etc/profile.d/conda.sh
conda activate /cpfs04/user/xxxx/anaconda3/envs/marti_review
cd /mnt/workspace/xxxx/MARTI-Dev
export WANDB_API_KEY=""

which python
which conda


mkdir -p "${ROOT_DIR}/logs"
mkdir -p "${ROOT_DIR}/logs/std"
mkdir -p "${ROOT_DIR}/logs/tensorboard"
mkdir -p "${ROOT_DIR}/outputs"




DATE=$(date +%m%d)
ADVANTAGE="reinforce"
SHORT_NAME="ReviewRL"
# TASK="DeepScaler"
TASK="REVIEW"
ALGO="review-rl-large"
PRETRAIN="/cpfs02/shared/public/models/openreview4/0627/qwen2.5-7b/epoch2"
EXP="${DATE}-${TASK}-${SHORT_NAME}-${ADVANTAGE}-${ALGO}"

ROOT_DIR="/mnt/workspace/xxxx/MARTI-Dev"
SAVE_PATH="${ROOT_DIR}/outputs/${ADVANTAGE}-${ALGO}/${DATE}/${SHORT_NAME}/model"

PROMPT_DATA="json@/mnt/workspace/xxxx/openreviewer/rl_marti/data_preprocess/rl_data_deepreview"
TENSORBOARD="${ROOT_DIR}/logs/${ADVANTAGE}-${ALGO}-${DATE}-${SHORT_NAME}"
CKPT_PATH="${ROOT_DIR}/outputs/${ADVANTAGE}-${ALGO}//${DATE}/${SHORT_NAME}/ckpt"

PROMPT_MAX_LEN=24000
GENERATE_MAX_LEN=16000



# ray start --head --port=6379 --dashboard-host=0.0.0.0 --dashboard-port=8265 --include-dashboard=true


export PYTORCH_NVML_BASED_CUDA_CHECK=1
# export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_ENABLE_V1_MULTIPROCESSING=1
export HYDRA_FULL_ERROR=1
export CUDA_LAUNCH_BLOCKING=1
export NCCL_DEBUG=WARN
export RAY_PICKLE_VERBOSE_DEBUG=1

num_gpus=16
RANK=${RANK:-0}  # 如果没有设置RANK，默认为0
PORT=6379
MASTER_ADDR=${MASTER_ADDR}

echo "Rank $RANK is running on $MASTER_ADDR"
if [ "$RANK" -eq 0 ]; then
    echo "Starting head node (RANK=${RANK}) on port $PORT..."

    ray start --head --port=$PORT --num-gpus=$num_gpus

    python -m marti.cli.commands.train --config-name "ma_reviewrl" \
    async_workflow=True \
    parallel_loading=True \
    default_agent.is_reasoning_model=False \
    default_agent.ref_num_nodes=1 \
    default_agent.ref_num_gpus_per_node=8 \
    default_agent.critic_num_nodes=1 \
    default_agent.critic_num_gpus_per_node=8 \
    default_agent.actor_num_nodes=1 \
    default_agent.actor_num_gpus_per_node=8 \
    default_agent.vllm_num_engines=8 \
    default_agent.vllm_tensor_parallel_size=1 \
    default_agent.vllm_sync_backend="nccl" \
    default_agent.colocate_all_models=True \
    default_agent.vllm_enable_sleep=True \
    default_agent.deepspeed_enable_sleep=True \
    default_agent.vllm_gpu_memory_utilization=0.8 \
    default_agent.pretrain="${PRETRAIN}" \
    default_agent.save_path="${SAVE_PATH}" \
    default_agent.micro_train_batch_size=1 \
    default_agent.train_batch_size=32 \
    default_agent.num_episodes=1 \
    default_agent.save_steps=3 \
    default_agent.eval_steps=1 \
    default_agent.logging_steps=1 \
    default_agent.max_samples=400000 \
    default_agent.micro_rollout_batch_size=2 \
    default_agent.rollout_batch_size=32 \
    default_agent.training_mode="rl" \
    default_agent.n_samples_per_prompt=8 \
    default_agent.max_epochs=1 \
    default_agent.prompt_max_len=${PROMPT_MAX_LEN} \
    default_agent.generate_max_len=${GENERATE_MAX_LEN} \
    default_agent.advantage_estimator=${ADVANTAGE} \
    default_agent.temperature=1.0 \
    default_agent.lambd=1.0 \
    default_agent.gamma=1.0 \
    default_agent.zero_stage=3 \
    default_agent.bf16=True \
    default_agent.actor_learning_rate=1e-6 \
    default_agent.critic_learning_rate=9e-6 \
    default_agent.init_kl_coef=0.00 \
    default_agent.use_kl_loss=True \
    default_agent.max_ckpt_num=3 \
    default_agent.normalize_reward=True \
    default_agent.adam_offload=True \
    default_agent.gradient_checkpointing=True \
    default_agent.ckpt_path="${CKPT_PATH}" \
    workflow_func_path="marti/worlds/workflows/judge_workflow.py" \
    processor_func_path="marti/worlds/workflows/default_processor.py" \
    reward_alloc.name="margin" \
    reward_alloc.alpha=0.5 \
    reward_alloc.beta=0.5 \
    reward_alloc.use_ttrl=False \
    eval_before_training=False \
    eval_only=False \
    eval_workers=-1 \
    mask_truncated_completions=True \
    shared_agents=False \
    packing_samples=True \
    prompt_data="${PROMPT_DATA}" \
    input_key="problem" \
    label_key="answer" \
    add_prompt_suffix=null \
    add_think_token=1 \
    wandb_project="MARTI" \
    wandb_run_name="${EXP}" \
    tools_config.num_workers=16 \
    use_tensorboard="${TENSORBOARD}" 2>&1 | tee "${ROOT_DIR}/logs/std/${DATE}-${EXP}.log"

    # use_wandb="${WANDB_KEY}" \

else
    echo "Starting worker node (RANK=${RANK}), connecting to ${MASTER_ADDR}:${PORT}..."
    ray start --address=${MASTER_ADDR}:${PORT} --num-gpus=$num_gpus

    sleep 120

    # worker保持运行状态
    while true; do
        # 获取ray status的输出
        status=$(ray status 2>&1)

        # 检查是否有 active 的 node
        if echo "$status" | grep -q "Active:"; then
            # 如果有 active 的 node，继续 sleep
            echo "Active nodes found. Sleeping for 5 min..."
            sleep 300
        else
            # 如果没有 active 的 node，退出脚本
            echo "No active nodes found. Exiting..."
            exit 0
        fi
    done
fi



echo "Model Training Finished. Shutting down..."
