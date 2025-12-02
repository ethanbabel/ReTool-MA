"""
Custom workflow for ReTool-MA:
Planner -> Executor -> Verifier with explicit code execution + metrics logging.
"""
from __future__ import annotations

import json
import math
import os
import re
from typing import Any, Dict, List, Optional

from marti.helpers.logging import init_logger
from marti.verifiers.qwen.qwen_eval_timeout import qwen_reward_fn_timeout
from marti.worlds.workflows.utils import apply_template_with_tokenizer

logger = init_logger(__name__)
logger.setLevel(os.getenv("MARTI_LOGGING_LEVEL", "WARN"))


def _extract_code_snippet(message: str) -> str:
    """Grab the content of the first ```python``` block, fallback to raw text."""
    if not isinstance(message, str):
        return ""
    pattern = re.compile(r"```python(.*?)```", re.DOTALL | re.IGNORECASE)
    match = pattern.search(message)
    if match:
        return match.group(1).strip()
    # allow bare ``` blocks
    pattern = re.compile(r"```(.*?)```", re.DOTALL)
    match = pattern.search(message)
    if match:
        return match.group(1).strip()
    return message.strip()


def _bool_from_metadata(metadata: Dict[str, Any]) -> bool:
    status = metadata.get("status") or metadata.get("run_status")
    if status:
        status = status.lower()
        if status in {"success", "finished", "ok"}:
            return True
    if metadata.get("timeout"):
        return False
    return metadata.get("return_code", 1) == 0


def _default_prompts():
    planner_prompt = (
        "You are the Planner in a multi-agent math system.\n"
        "Break down the following problem into reasoning steps and, when numerical "
        "work is required, describe exactly what code the Executor should run.\n"
        "Respond with the following JSON structure:\n"
        "{{\n"
        '  "reasoning": "...",\n'
        '  "code_plan": "High-level instructions for the executor",\n'
        '  "expected_output": "What quantity to compute."\n'
        "}}\n\n"
        "Problem:\n{problem}\n"
        "Think carefully before forming the code plan."
    )

    executor_prompt = (
        "You are the Executor. ONLY output Python code that completes the request.\n"
        "Follow the plan below exactly and include `print()` so that the answer is visible.\n"
        "If the plan requires external libraries (e.g., math, sympy), add the appropriate import "
        "statements at the top of your snippet (for SymPy, use `from sympy import symbols, Eq, solve`).\n"
        "Plan from the Planner:\n{planner_plan}\n\n"
        "Problem:\n{problem}\n"
        ".Output ONLY Python code. Do not wrap with explanations."
    )

    verifier_prompt = (
        "You are the Verifier. Review the reasoning and execution log to decide if the answer is correct.\n"
        "State any discrepancies and finish with FINAL ANSWER: \\boxed{{value}}.\n\n"
        "Problem:\n{problem}\n"
        "Planner reasoning:\n{planner_reasoning}\n\n"
        "Executor log:\n{execution_log}\n"
    )
    return planner_prompt, executor_prompt, verifier_prompt


PLANNER_KEYS = ("reasoning", "code_plan", "expected_output")


def _parse_planner_output(raw_text: str) -> Dict[str, str]:
    for parser in (json.loads,):
        try:
            parsed = parser(raw_text)
            if isinstance(parsed, dict):
                return {k: parsed.get(k, "") for k in PLANNER_KEYS}
        except Exception:
            continue

    parsed_fields: Dict[str, str] = {}
    for key in PLANNER_KEYS:
        pattern = rf'"{key}"\s*:\s*"(?P<val>.*?)"'
        match = re.search(pattern, raw_text, re.DOTALL)
        if match:
            parsed_fields[key] = match.group("val").strip()
        else:
            parsed_fields[key] = raw_text.strip()
    return parsed_fields


def _keyword_overlap(plan_text: str, code_text: str) -> float:
    plan_tokens = {
        token for token in re.findall(r"[A-Za-z_]{4,}", plan_text.lower())
        if token not in {"please", "therefore", "problem", "output"}
    }
    if not plan_tokens:
        return 0.0
    code_lower = code_text.lower()
    hits = sum(1 for token in plan_tokens if token in code_lower)
    return hits / len(plan_tokens)


def _extract_numeric_value(text: Optional[str]) -> Optional[float]:
    if text is None:
        return None
    cleaned = text
    cleaned = re.sub(r"\\boxed", "", cleaned)
    cleaned = re.sub(r"[^\d\.\-/]", " ", cleaned)
    cleaned = cleaned.strip()
    if not cleaned:
        return None
    frac_match = re.search(r"(-?\d+)\s*/\s*(\d+)", cleaned)
    if frac_match:
        numerator = int(frac_match.group(1))
        denominator = int(frac_match.group(2))
        if denominator != 0:
            return numerator / denominator
    num_match = re.search(r"-?\d*\.?\d+", cleaned)
    if num_match:
        try:
            return float(num_match.group(0))
        except ValueError:
            return None
    return None


def _normalize_answer(text: Optional[str]) -> str:
    if text is None:
        return ""
    lowered = text.strip().lower()
    lowered = re.sub(r"\\boxed", "", lowered)
    lowered = re.sub(r"\s+", " ", lowered)
    return lowered


def _values_match(a: Optional[str], b: Optional[str]) -> bool:
    if not a or not b:
        return False
    return _normalize_answer(a) == _normalize_answer(b)


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


async def _call_llm(agent_cfg: Dict[str, Any], prompt: str):
    sampling_params = agent_cfg["sampling_params"]
    llm = agent_cfg["llm"]
    try:
        ref = llm.generate_async.remote(prompt, sampling_params=sampling_params)
    except AttributeError:
        ref = llm.generate.remote(prompt, sampling_params=sampling_params)
    result = await ref
    if isinstance(result, list):
        if not result:
            raise ValueError("LLM returned an empty response list.")
        result = result[0]
    return result

async def workflow(
    prompt: str,
    label: str,
    agents: List[Dict[str, Any]],
    tool_manager,
    task: str,
    metadata: Optional[Dict] = None,
    workflow_args: Optional[Dict[str, Any]] = None,
    **kwargs,
) -> Dict[str, Any]:
    assert tool_manager is not None, "Tool manager required for ReTool-MA workflow."
    workflow_args = workflow_args or {}
    metadata = metadata or {}

    role_map = {agent["agent_role"]: agent for agent in agents}
    planner_agent = role_map.get(workflow_args.get("planner_role", "planner"))
    executor_agent = role_map.get(workflow_args.get("executor_role", "executor"))
    verifier_agent = role_map.get(workflow_args.get("verifier_role", "verifier"))
    if not all([planner_agent, executor_agent, verifier_agent]):
        raise ValueError("Planner/Executor/Verifier roles must be provided in the config.")

    planner_prompt_tpl, executor_prompt_tpl, verifier_prompt_tpl = _default_prompts()
    planner_prompt_tpl = workflow_args.get("planner_prompt", planner_prompt_tpl)
    executor_prompt_tpl = workflow_args.get("executor_prompt", executor_prompt_tpl)
    verifier_prompt_tpl = workflow_args.get("verifier_prompt", verifier_prompt_tpl)

    code_tool_name = workflow_args.get("code_tool_name", "code_interpreter")
    code_timeout = workflow_args.get("code_timeout", 20)
    reward_cfg = {
        "planner_success_bonus": workflow_args.get("planner_success_bonus", 0.4),
        "planner_spec_bonus": workflow_args.get("planner_spec_bonus", 0.4),
        "planner_failure_penalty": workflow_args.get("planner_failure_penalty", -0.2),
        "executor_success_bonus": workflow_args.get("executor_success_bonus", 0.6),
        "executor_failure_penalty": workflow_args.get("executor_failure_penalty", -0.4),
        "executor_alignment_bonus": workflow_args.get("executor_alignment_bonus", 0.4),
        "verifier_missed_penalty": workflow_args.get("verifier_missed_penalty", 0.5),
        "verifier_reject_penalty": workflow_args.get("verifier_reject_penalty", 0.3),
    }
    reward_min = workflow_args.get("reward_min", -1.0)
    reward_max = workflow_args.get("reward_max", 1.0)

    trajectory: List[Dict[str, Any]] = []
    code_runs: List[Dict[str, Any]] = []

    # Planner
    planner_input = apply_template_with_tokenizer(
        planner_agent["tokenizer"],
        planner_prompt_tpl.format(problem=prompt),
    )
    planner_response = await _call_llm(planner_agent, planner_input)
    planner_output = planner_response.outputs[0].text.strip()
    planner_json = _parse_planner_output(planner_output)

    trajectory.append(
        {
            "turn_id": 0,
            "agent_role": planner_agent["agent_role"],
            "agent_id": planner_agent["agent_id"],
            "agent_input": planner_input,
            "agent_output": planner_output,
            "metadata": {"parsed_plan": planner_json},
        }
    )

    # Executor
    executor_input = apply_template_with_tokenizer(
        executor_agent["tokenizer"],
        executor_prompt_tpl.format(
            problem=prompt,
            planner_plan=json.dumps(planner_json, ensure_ascii=False),
        ),
    )
    executor_response = await _call_llm(executor_agent, executor_input)
    executor_output = executor_response.outputs[0].text
    code_snippet = _extract_code_snippet(executor_output)

    execution_success = False
    execution_response = ""
    execution_metadata: Dict[str, Any] = {}
    if code_snippet:
        try:
            execution_response, execution_metadata = await tool_manager.execute_tool(
                code_tool_name,
                {"code": code_snippet, "timeout": code_timeout},
                metadata=metadata,
            )
            execution_success = _bool_from_metadata(execution_metadata)
        except Exception as exc:
            execution_response = f"[tool_error] {exc}"
            execution_metadata = {"status": "failed", "exception": str(exc)}
            execution_success = False
    else:
        execution_response = "[no code generated]"
        execution_metadata = {"status": "failed", "exception": "empty_code"}

    code_runs.append(
        {
            "code": code_snippet,
            "status": execution_metadata.get("status"),
            "stdout": execution_metadata.get("stdout", ""),
            "stderr": execution_metadata.get("stderr", ""),
            "passed": execution_success,
        }
    )

    trajectory.append(
        {
            "turn_id": 1,
            "agent_role": executor_agent["agent_role"],
            "agent_id": executor_agent["agent_id"],
            "agent_input": executor_input,
            "agent_output": executor_output,
            "metadata": {
                "code": code_snippet,
                "tool_metadata": execution_metadata,
                "tool_response": execution_response,
            },
        }
    )

    execution_summary = (
        f"Code:\n{code_snippet or '[none]'}\n"
        f"Status: {execution_metadata.get('status', 'unknown')}\n"
        f"Stdout: {execution_metadata.get('stdout', '')[:512]}\n"
        f"Stderr: {execution_metadata.get('stderr', '')[:512]}"
    )

    # Verifier
    verifier_input = apply_template_with_tokenizer(
        verifier_agent["tokenizer"],
        verifier_prompt_tpl.format(
            problem=prompt,
            planner_reasoning=json.dumps(planner_json, ensure_ascii=False),
            execution_log=execution_summary,
        ),
    )
    verifier_response = await _call_llm(verifier_agent, verifier_input)
    verifier_output = verifier_response.outputs[0].text
    final_reward = float(qwen_reward_fn_timeout(verifier_output, label))

    plan_alignment = _keyword_overlap(planner_json.get("code_plan", ""), code_snippet or "")
    expected_value = planner_json.get("expected_output", "")
    planner_reward = 0.0
    if execution_success:
        planner_reward += reward_cfg["planner_success_bonus"]
    else:
        planner_reward += reward_cfg["planner_failure_penalty"]
    if expected_value:
        if execution_success and _values_match(expected_value, execution_response):
            planner_reward += reward_cfg["planner_spec_bonus"]
    planner_reward = _clamp(planner_reward, reward_min, reward_max)

    executor_reward = reward_cfg["executor_failure_penalty"]
    if execution_success:
        executor_reward = reward_cfg["executor_success_bonus"]
    executor_reward += reward_cfg["executor_alignment_bonus"] * plan_alignment
    executor_reward = _clamp(executor_reward, reward_min, reward_max)

    code_value = _extract_numeric_value(execution_response)
    label_value = _extract_numeric_value(label)
    code_matches: Optional[bool] = None
    if code_value is not None and label_value is not None:
        code_matches = math.isclose(code_value, label_value, rel_tol=1e-6, abs_tol=1e-6)

    verifier_reward = final_reward
    if execution_success and code_matches is False:
        verifier_reward -= reward_cfg["verifier_missed_penalty"]
    elif execution_success and code_matches is True and final_reward < 1.0:
        verifier_reward -= reward_cfg["verifier_reject_penalty"]
    verifier_reward = _clamp(verifier_reward, reward_min, reward_max)

    trajectory.append(
        {
            "turn_id": 2,
            "agent_role": verifier_agent["agent_role"],
            "agent_id": verifier_agent["agent_id"],
            "agent_input": verifier_input,
            "agent_output": verifier_output,
            "agent_reward": verifier_reward,
            "metadata": {"reward": final_reward},
        }
    )

    code_attempts = len(code_runs)
    code_passes = sum(1 for run in code_runs if run["passed"])
    code_failures = code_attempts - code_passes
    metrics = {
        "is_correct": final_reward >= 0.5,
        "code_execution": {
            "attempts": code_attempts,
            "passes": code_passes,
            "failures": code_failures,
            "pass_rate": (code_passes / code_attempts) if code_attempts else 0.0,
            "invalid_calls": code_failures,
        },
        "trajectory_length": len(trajectory),
        "reward_breakdown": {
            "planner": planner_reward,
            "executor": executor_reward,
            "verifier": verifier_reward,
        },
    }

    trajectory[0]["agent_reward"] = planner_reward
    trajectory[1]["agent_reward"] = executor_reward

    return {
        "prompt": prompt,
        "label": label,
        "trajectory": trajectory,
        "code_runs": code_runs,
        "final_reward": final_reward,
        "metrics": metrics,
    }
