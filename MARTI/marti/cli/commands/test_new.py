import asyncio
import importlib.util
from pathlib import Path
from collections import defaultdict
import ray
import hydra
from omegaconf import DictConfig, OmegaConf
from vllm import SamplingParams
import torch
import random
import numpy as np
from typing import Union, Callable, Dict, List
from tqdm import tqdm
import json
from copy import deepcopy
import srsly

from marti.helpers.common import get_strategy, blending_datasets, get_tokenizer
from marti.models.vllm.engine import create_vllm_engines
from marti.models.openai import OpenAIModel, FakeTokenizer
from marti.worlds.base_world import BaseWorld, Samples
from marti.agents.multi_agent import MAGraph, get_kwargs
from marti.verifiers.qwen.qwen_eval import qwen_reward_fn, majority_vote
from marti.dataset.prompts_loader import PromptDatasetWithLabel
from marti.helpers.distributed.distributed_sampler import DistributedSampler, ResumableRandomSampler
from marti.worlds.tools.manager import ToolManager
from marti.worlds.tools.mcp_manager import MCPManager
from marti.worlds.tool_world import register_openai_tools, register_mcp_tools, print_tools

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def _validate_config(cfg: DictConfig):
    actor_world_size = cfg.actor_num_nodes * cfg.actor_num_gpus_per_node

    assert (
        actor_world_size & (actor_world_size - 1)
    ) == 0, f"actor_world_size must be power of 2, got {actor_world_size}"

    if cfg.critic_pretrain:
        critic_world_size = cfg.critic_num_nodes * cfg.critic_num_gpus_per_node
        assert (
            critic_world_size & (critic_world_size - 1)
        ) == 0, f"critic_world_size must be power of 2, got {critic_world_size}"
        assert (
            actor_world_size % critic_world_size == 0
        ), f"actor_world_size must be divisible by critic_world_size, got {actor_world_size} and {critic_world_size}"

    assert cfg.zero_stage != 3 or cfg.vllm_num_engines > 0, f"ZeRO-3 is only supported when vLLM enabled"

def _rationalize_config(cfg: DictConfig):
    if cfg.advantage_estimator not in ["gae"]:
        cfg.critic_pretrain = None
    elif cfg.critic_pretrain is None:
        if cfg.reward_pretrain is not None:
            cfg.critic_pretrain = cfg.reward_pretrain.split(",")[0]
        else:
            cfg.critic_pretrain = cfg.pretrain

    if cfg.advantage_estimator == "rloo":
        assert cfg.n_samples_per_prompt > 1, "RLOO requires n_samples_per_prompt > 1"

    if cfg.remote_rm_url:
        cfg.remote_rm_url = cfg.remote_rm_url.split(",")

    if cfg.vllm_num_engines >= 1 and cfg.enable_prefix_caching:
        import vllm
        if vllm.__version__ < "0.7.0":
            cfg.enable_prefix_caching = False
            print("[Warning] Disable prefix cache because vLLM updates weights without updating the old KV Cache for vLLM version below 0.7.0.")

    if cfg.input_template and "{}" not in cfg.input_template:
        print("[Warning] {} not in cfg.input_template, set to None")
        cfg.input_template = None

    if cfg.input_template and "\\n" in cfg.input_template:
        print(
            "[Warning] input_template contains \\n chracters instead of newline. "
            "You likely want to pass $'\\n' in Bash or \"`n\" in PowerShell."
        )

    if cfg.packing_samples:
        if not cfg.flash_attn:
            print(
                "[Warning] Please --flash_attn to accelerate when --packing_samples is enabled.")
            cfg.flash_attn = True
        assert cfg.vllm_num_engines > 0, "Only support `--packing_samples` with vLLM."
        assert not cfg.pretrain_data, "`--pretrain_data` is not supported with `--packing_samples` yet."

    return cfg


class MultiAgentWorld(BaseWorld):
    def __init__(self, strategy, agents):
        super().__init__(strategy, agents)

    def run_collect(self, all_prompts: List[str], all_labels: List[str]):
        args = self.strategy.args
        print(args.chat_template)
        print(args.agents)

        kwargs = get_kwargs(args)

        graph = MAGraph(
            agents=self.agents,
            agent_ids=kwargs['agent_ids'],
            agent_roles=kwargs['agent_roles'],
            agent_workflow=args.agent_workflow,
            prompt=kwargs['prompt'],
            spatial_adj_mats=kwargs['spatial_adj_mats'],
            temporal_adj_mats=kwargs['temporal_adj_mats'],
            sampling_params=kwargs['sampling_params'],
            node_kwargs=kwargs['node_kwargs'] if 'node_kwargs' in kwargs else None,
        )
        history = graph.run(all_prompts, num_rounds=args.workflow_args.num_rounds)

        # eval the results
        history_per_problem = {problem: [] for problem in all_prompts}
        problem2id = {problem: all_prompts.index(problem) for problem in all_prompts}
        for problem in all_prompts:
            problem_id = problem2id[problem]
            for node_id, node_history in enumerate(history):
                temp_history = []
                for round_id, round_history in enumerate(node_history):
                    inputs = round_history['inputs']
                    outputs = round_history['outputs']
                    rewards = []
                    for output, label in zip(outputs, all_labels):
                        if isinstance(output, str):
                            reward = qwen_reward_fn(output, label)
                        elif isinstance(output, dict):
                            if isinstance(output['output'], str):
                                reward = qwen_reward_fn(output['output'], label)
                            elif isinstance(output['output'], list):
                                reward = majority_vote(output['output'], label)
                        rewards.append(reward)
                    temp_history.append({
                        'agent_id': round_history['agent_id'],
                        'agent_role': round_history['agent_role'],
                        'pretrain': round_history['pretrain'],
                        'turn_id': round_history['turn_id'],
                        'inputs': inputs[problem_id],
                        'outputs': outputs[problem_id],
                        'rewards': rewards[problem_id],
                        'spatial_predecessors': round_history['spatial_predecessors'],
                        'temporal_predecessors': round_history['temporal_predecessors'],
                    })
                if len(temp_history) == 1:
                    temp_history = temp_history[0]
                history_per_problem[problem].append(temp_history)

        return history_per_problem

    @torch.no_grad()
    def generate_samples(self, all_prompts: Union[List[str], dict], rank=0, world_size=8, **kwargs) -> List[Samples]:
        args = self.strategy.args

        all_prompts, all_labels = all_prompts["prompt"], all_prompts["label"]
        # Expand prompt list based on the number of samples per prompt
        all_prompts = sum([[prompt] * args.n_samples_per_prompt for prompt in all_prompts], [])
        all_labels = sum([[label] * args.n_samples_per_prompt for label in all_labels], [])

        if len(all_prompts):
            history = self.run_collect(all_prompts=all_prompts, all_labels=all_labels)
        else:
            history = {}

        return history


@ray.remote
def generate_samples_remote(samples_maker, chunk_prompts, rank, world_size):
    history = samples_maker.generate_samples(chunk_prompts, rank, world_size)
    return history


def generate_shared_samples(samples_maker, rand_prompts, world_size):
    any_key = next(iter(rand_prompts.keys()))
    length = len(rand_prompts[any_key])
    chunk_size = (length + world_size - 1) // world_size
    chunked = [dict() for _ in range(world_size)]
    for key, data_list in rand_prompts.items():
        for i in range(world_size):
            start_idx = i * chunk_size
            end_idx = min(start_idx + chunk_size, length)
            sub_slice = data_list[start_idx:end_idx]
            chunked[i][key] = sub_slice

    all_refs = []
    for rank in range(world_size):
        samples_ref = generate_samples_remote.remote(samples_maker, chunked[rank], rank, world_size)
        all_refs.append(samples_ref)

    all_results = {}
    for r in ray.get(all_refs):
        all_results.update(r)

    return all_results


def _load_workflow_function(path: str):
    workflow_path = Path(path)
    if not workflow_path.exists():
        raise FileNotFoundError(f"Workflow file not found: {workflow_path}")
    spec = importlib.util.spec_from_file_location("retool_workflow", str(workflow_path))
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    if not hasattr(module, "workflow"):
        raise AttributeError(f"Workflow module {workflow_path} must define `workflow`.")
    return module.workflow


def _to_container(obj):
    if isinstance(obj, DictConfig):
        return OmegaConf.to_container(obj, resolve=True)
    return obj


def _build_tool_manager(cfg: DictConfig):
    tools_cfg = getattr(cfg, "tools_config", None)
    if not tools_cfg:
        raise ValueError("tools_config must be specified when using a custom workflow.")
    tools_cfg = _to_container(tools_cfg)
    if tools_cfg.get("mcp_url"):
        manager = MCPManager(tools_cfg)
        tools = register_mcp_tools(manager)
    else:
        manager = ToolManager(tools_cfg)
        tools = register_openai_tools(tools_cfg, manager)
    manager.set_tools(tools)
    print_tools(tools)
    return manager


def _run_custom_workflow(cfg, agent_list, prompts_dataset):
    workflow_fn = _load_workflow_function(cfg.workflow_func_path)
    workflow_args = _to_container(cfg.workflow_args) if getattr(cfg, "workflow_args", None) else {}
    tool_manager = _build_tool_manager(cfg)

    runtime_agents = []
    for agent in agent_list:
        runtime_agents.append({
            "agent_id": agent["agent_id"],
            "agent_role": agent["agent_role"],
            "llm": agent["llms"][0],
            "tokenizer": agent["tokenizer"],
            "sampling_params": agent["sampling_params"],
        })

    samples = prompts_dataset.get_all_prompts()
    prompts = [sample["prompt"] for sample in samples]
    labels = [sample["label"] for sample in samples]

    metadata = []
    for sample in samples:
        sample_meta = sample.get("metadata")
        if isinstance(sample_meta, str):
            try:
                metadata.append(json.loads(sample_meta))
            except json.JSONDecodeError:
                metadata.append({})
        elif isinstance(sample_meta, dict):
            metadata.append(sample_meta)
        else:
            metadata.append({})

    async def _run_all():
        results = []
        for idx, (prompt, label, meta) in enumerate(zip(prompts, labels, metadata)):
            res = await workflow_fn(
                prompt=prompt,
                label=label,
                agents=runtime_agents,
                tool_manager=tool_manager,
                task=getattr(cfg, "verify_task", "math"),
                metadata=meta,
                workflow_args=workflow_args,
                prompt_id=idx,
                is_eval=True,
            )
            results.append(res)
        return results

    return asyncio.run(_run_all())


def _aggregate_retool_metrics(histories):
    if not histories:
        return None
    entries = [h for h in histories if isinstance(h, dict) and "metrics" in h]
    if not entries:
        return None

    total = len(entries)
    accuracy = 0
    total_attempts = total_passes = total_failures = 0
    correct_attempts = correct_passes = 0
    incorrect_attempts = incorrect_passes = 0
    total_steps = 0

    role_totals = defaultdict(float)
    role_counts = defaultdict(int)

    for entry in entries:
        metrics = entry.get("metrics", {})
        is_correct = bool(metrics.get("is_correct"))
        code_stats = metrics.get("code_execution", {})
        attempts = code_stats.get("attempts", 0)
        passes = code_stats.get("passes", 0)
        failures = code_stats.get("failures", attempts - passes)

        accuracy += 1 if is_correct else 0
        total_attempts += attempts
        total_passes += passes
        total_failures += failures
        total_steps += metrics.get("trajectory_length", 0)

        if is_correct:
            correct_attempts += attempts
            correct_passes += passes
        else:
            incorrect_attempts += attempts
            incorrect_passes += passes

        for turn in entry.get("trajectory", []):
            role = turn.get("agent_role")
            reward_val = turn.get("agent_reward")
            if role and reward_val is not None:
                role_totals[role] += reward_val
                role_counts[role] += 1

    summary = {
        "num_samples": total,
        "final_accuracy": accuracy / total if total else 0.0,
        "avg_trajectory_length": total_steps / total if total else 0.0,
        "code_metrics": {
            "total_attempts": total_attempts,
            "total_passes": total_passes,
            "total_failures": total_failures,
            "overall_pass_rate": (total_passes / total_attempts) if total_attempts else 0.0,
            "pass_rate_when_correct": (correct_passes / correct_attempts) if correct_attempts else 0.0,
            "pass_rate_when_incorrect": (incorrect_passes / incorrect_attempts) if incorrect_attempts else 0.0,
            "avg_calls_per_problem": (total_attempts / total) if total else 0.0,
        },
    }
    if role_totals:
        summary["reward_stats"] = {
            role: role_totals[role] / role_counts[role] for role in role_totals
        }
    return summary


@hydra.main(config_path="../configs", config_name="default.yaml", version_base=None)
def train(cfg: DictConfig):
    OmegaConf.set_struct(cfg, False)

    for key, value in cfg.default_agent.items():
        cfg[key] = value

    _rationalize_config(cfg)
    _validate_config(cfg)
    print(OmegaConf.to_yaml(cfg))

    # configure strategy
    strategy = get_strategy(cfg)

    agent2config = {}
    for agent_dict in cfg.agents:
        for agent_name, agent_info in agent_dict.items():
            for key, value in cfg.default_agent.items():
                if key not in agent_info:
                    agent_info[key] = value
            agent2config[agent_name] = agent_info

    agent_list = []
    llm_dict = {}
    seed = 0
    for agent_name, agent_config in agent2config.items():
        if 'gpt' in agent_config.pretrain.lower():
            agent_llms = [
                OpenAIModel.remote(api_key=cfg.api_key, base_url=cfg.api_base_url, config={"model_name": agent_config.pretrain})
            ]
            tokenizer = FakeTokenizer()
        else:
            max_len = None
            if agent_config.max_len is not None:
                max_len = agent_config.max_len
            elif agent_config.prompt_max_len is not None and agent_config.generate_max_len is not None:
                max_len = agent_config.prompt_max_len + agent_config.generate_max_len

            if agent_config.pretrain in llm_dict:
                agent_llms = llm_dict[agent_config.pretrain]
            else:
                agent_llms = create_vllm_engines(
                    agent_config.vllm_num_engines,
                    agent_config.vllm_tensor_parallel_size,
                    agent_config.pretrain,
                    seed,
                    agent_config.enable_prefix_caching,
                    agent_config.enforce_eager,
                    max_len,
                    None,  # shared_pg
                    agent_config.vllm_gpu_memory_utilization,
                    getattr(agent_config, "vllm_enable_sleep", False),
                )
                llm_dict[agent_config.pretrain] = agent_llms
                seed += agent_config.vllm_num_engines

            # Create tokenizer
            tokenizer = get_tokenizer(
                agent_config.pretrain, None, "left", strategy, use_fast=not agent_config.disable_fast_tokenizer)

        generate_kwargs = {
            "do_sample": True,
            "max_new_tokens": agent_config.generate_max_len,
            "max_length": agent_config.max_len,
            "temperature": agent_config.temperature,
            "top_p": agent_config.top_p,
            "pad_token_id": tokenizer.pad_token_id if hasattr(tokenizer, "pad_token_id") else None,
            "eos_token_id": tokenizer.eos_token_id if hasattr(tokenizer, "eos_token_id") else None,
        }

        sampling_params = SamplingParams(
            n=agent_config.n_samples_per_prompt,
            temperature=agent_config.temperature,
            top_p=agent_config.top_p,
            top_k=agent_config.get("top_k", -1),
            max_tokens=agent_config.generate_max_len,
            min_tokens=agent_config.get("min_new_tokens", 1),
            skip_special_tokens=agent_config.get("skip_special_tokens", False),
        )

        agent = {
            "agent_name": agent_name,
            "agent_id": agent_name,
            "agent_role": agent_config.role,
            "pretrain": agent_config.pretrain,
            "llms": agent_llms,
            "tokenizer": tokenizer,
            "generate_kwargs": generate_kwargs,
            "sampling_params": sampling_params,
            "is_reasoning_model": agent_config.get("is_reasoning_model", False),
            "enable_thinking": agent_config.get("enable_thinking", agent_config.get("is_reasoning_model", False)),
            "code_execution": agent_config.get("code_execution", False),
        }
        agent_list.append(agent)

    args = strategy.args

    # prepare datasets
    prompts_data, prompts_data_eval = blending_datasets(
        args.prompt_data,
        str(args.prompt_data_probs),
        strategy,
        args.seed,
        max_count=args.max_samples,
        return_eval=True,
        train_split=args.prompt_split,
    )
    prompts_data = prompts_data.select(
        range(min(args.max_samples, len(prompts_data))))
    prompts_dataset = PromptDatasetWithLabel(
        prompts_data, None, strategy, input_template=args.input_template, add_prompt_suffix=args.add_prompt_suffix
    )

    workflow_path = getattr(cfg, "workflow_func_path", None)
    use_custom_workflow = bool(workflow_path)

    if use_custom_workflow:
        final_histories = _run_custom_workflow(cfg, agent_list, prompts_dataset)
    else:
        sampler = ResumableRandomSampler(
            data_source=prompts_dataset,
            batch_size=args.rollout_batch_size,
            drop_last=False,
            shuffle=False,
            seed=args.seed
        )

        prompts_dataloader = strategy.setup_dataloader(
            prompts_dataset, args.rollout_batch_size, True, shuffle=False,
            sampler=sampler, drop_last=False
        )

        sample_maker_class = MultiAgentWorld
        samples_maker = sample_maker_class(
            strategy=strategy, agents=agent_list)

        steps = 1
        start_episode = 0
        consumed_samples = 0
        all_histories = {}
        final_histories = []
        question2idx = {}
        idx2question = []
        for episode in range(start_episode, args.num_episodes):
            if isinstance(prompts_dataloader.sampler, ResumableRandomSampler):
                prompts_dataloader.sampler.set_epoch(
                    episode, consumed_samples=0
                )
            pbar = tqdm(
                range(prompts_dataloader.__len__()),
                desc=f"Episode [{episode + 1}/{args.num_episodes}]"
            )

            for rand_prompts in prompts_dataloader:
                history = generate_shared_samples(samples_maker, rand_prompts=rand_prompts, world_size=args.vllm_num_engines)
                for idx, question in enumerate(rand_prompts["prompt"]):
                    if question not in idx2question:
                        idx2question.append(question)
                        question2idx[question] = len(question2idx)
                        all_histories[question] = []
                    all_histories[question].append(history[question])

                pbar.update()
                steps = steps + 1

        for idx, question in enumerate(idx2question):
            final_histories.append(all_histories[question][0])

    output_path = os.path.join(args.save_path)
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    srsly.write_json(os.path.join(output_path, f"results.json"), final_histories)

    summary = _aggregate_retool_metrics(final_histories)
    if summary:
        srsly.write_json(os.path.join(output_path, "summary.json"), summary)
        print(f"[ReTool-MA] Aggregated metrics: {json.dumps(summary, indent=2)}")


if __name__ == "__main__":
    train()
