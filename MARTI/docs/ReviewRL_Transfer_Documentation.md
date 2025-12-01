# ReviewRL Transfer Documentation

## Overview
This document records the transfer of ReviewRL code from `ReviewRL-Preview` to `MARTI-Dev` framework, implementing a multi-agent review generation and evaluation system.

## Transfer Requirements
- **Source**: ReviewRL-Preview (original implementation)
- **Target**: MARTI-Dev (mature framework)
- **Task**: Multi-agent review generation with generator and judge agents
- **Rewards**: Combined judge model rewards and rule-based rewards (review_eval)
- **Pattern**: Adapt existing debate_workflow.py and debate_processor.py patterns

## Implementation Components

### 1. Auto Verification Integration (`marti/verifiers/auto_verify.py`)
**Changes made:**
- Added import for `group_review_reward_fn` from `marti.verifiers.review.review_eval`
- Added `"review_group": group_review_reward_fn` to `task2verify` dictionary
- Modified condition to include `review_group` in group processing: `if "ttt" in task or task == "review_group"`

**Purpose:** Integrates the existing review_eval.py functionality into MARTI-Dev's verification system.

### 2. Enhanced Judge Workflow (`marti/worlds/workflows/judge_workflow.py`)
**Enhancement of existing workflow** - Now **configurable and general**
**Key improvements:**
- **Configurable judge templates** via `workflow_args.judge_template` with placeholders ({prompt}, {generated_answer}, {label})
- **Multiple score parsing methods** via `workflow_args.score_parser`:
  - `"regex"`: Default pattern matching (Score: 8/10)
  - `"keywords"`: Binary judgments (REVIEW_1_BETTER, REVIEW_2_BETTER) 
  - `"normalize"`: Extract and normalize numbers to [0,1]
- **Flexible label parsing** via `workflow_args.label_separator` (e.g., "||DIV REVIEW SCORE||")
- **Weighted reward combination** via `workflow_args.judge_weight`
- **Output format options**: Legacy format or trajectory format (matches debate_workflow.py)
- **Backward compatible** - existing behavior preserved as defaults

**ReviewRL Configuration:**
- Uses `score_parser: "keywords"` for peer review comparison
- Uses original 6-criteria peer review template
- Uses `label_separator: "||DIV REVIEW SCORE||"` for structured labels

### 3. Existing Processor (`marti/worlds/workflows/debate_processor.py`)
**Reused existing component** - No changes needed
**Key features:**
- Handles trajectory format from judge workflow
- Integrates with MultiAgentRewardAllocation system
- Returns per-agent training data (prompts, outputs, labels)

### 4. Configuration (`marti/cli/configs/ma_judge.yaml`)
**Based on:** `default.yaml` with review-specific settings
**Key settings:**
- `agent_workflow: "review"`
- `workflow_version: "new"` (uses async workflow system)
- `async_workflow: true`
- `workflow_func_path: "marti/worlds/workflows/judge_workflow.py"` (enhanced general workflow)
- `processor_func_path: "marti/worlds/workflows/debate_processor.py"` (existing processor)
- **ReviewRL-specific workflow_args:**
  - `judge_weight: 0.5` (weight for combining judge vs rule rewards)
  - `score_parser: "keywords"` (for REVIEW_1_BETTER/REVIEW_2_BETTER parsing)
  - `label_separator: "||DIV REVIEW SCORE||"` (for structured label parsing)
  - `judge_template: |` (original 6-criteria peer review comparison template)
- `verify_task: "review_group"`
- Review-specific parameters (prompt_max_len: 24000, generate_max_len: 8000)
- Agent definitions for generator and judge

### 5. Training Integration (`marti/cli/commands/train.py`)
**Changes made:**
- Added `"review"` to supported workflow list for MultiAgentController
- Updated condition: `elif agent_workflow in ["multi-agents-debate", "chain-of-agents", "mixture-of-agents", "review"]:`

### 6. Training Script (`scripts/run_train_review.sh`)
**Based on:** `run_train_mad_async.sh` pattern, **Adapted** for judge workflow
**Key features:**
- Explicit async workflow configuration (`async_workflow=True`)
- Explicit workflow and processor paths for judge workflow
- Organized logging structure (std/, tensorboard/ directories)
- Enhanced Ray environment with additional packages
- **Complete default_agent configuration** for review-specific settings:
  - Resource allocation (2 GPUs per component for smaller models)
  - Review-specific batch sizes (smaller: 1/8 vs MAD's 4/128)
  - Review-specific sequence lengths (24k prompt, 8k generation)
  - REINFORCE advantage estimator for review tasks
  - Appropriate learning rates and training episodes
- Support for separate generator and judge models
- WANDB and tensorboard logging

## Key Technical Decisions

### 1. General vs Specific Implementation
- **Choice**: Enhanced existing `judge_workflow.py` to be configurable instead of creating review-specific workflow
- **Reason**: Maintains MARTI-Dev principle of generality, benefits other tasks, easier maintenance
- **Implementation**: Configurable templates, scoring methods, and output formats

### 2. Workflow Architecture
- **Choice**: New async workflow system (`workflow_version: "new"`)
- **Reason**: Better performance and modern pattern used in MARTI-Dev
- **Implementation**: Uses dynamic workflow loading via file paths

### 3. Agent Pattern
- **Choice**: Generator-Judge pattern (not multi-agent debate)
- **Reason**: Matches ReviewRL requirements and original script structure
- **Agents**: 2 agents (generator, judge) instead of N-agent debate

### 4. Reward System
- **Choice**: Combined judge model + rule-based rewards
- **Rule-based**: Uses existing `review_eval.py` through `auto_verify`
- **Judge model**: Uses configurable scoring methods
- **Integration**: `judge_weight` parameter controls weighting

### 4. Data Flow
```
Input Paper/Submission + Ground Truth Review
    ↓
Generator Agent → Generated Review
    ↓
Judge Agent → Peer Review Comparison (Generated vs Ground Truth)
    ↓     ↓
Judge Score (0.0-1.0)    Rule-based Evaluation (review_eval) → Format/Rating Consistency  
    ↓     ↓
Combined Reward = judge_score * judge_weight + rule_score * (1 - judge_weight)
    ↓
Training Samples (Generator + Judge)
```

## Configuration Parameters

### Core Settings
- `verify_task: "review_group"` - Uses review evaluation
- `input_key: "problem"` - Input paper/submission key
- `label_key: "answer"` - Ground truth review/rating key (format: "review||DIV REVIEW SCORE||score")
- `add_think_token: 1` - Enables thinking tokens
- `workflow_args.judge_weight: 0.5` - Weight for combining judge vs rule-based rewards

### Resource Allocation
- Smaller GPU allocation than math tasks (2 GPUs per component)
- `micro_train_batch_size: 1`, `train_batch_size: 8`
- `n_samples_per_prompt: 4`
- `advantage_estimator: "reinforce"`

### Model Settings
- Generator: Configurable via script parameter
- Judge: Larger model for better evaluation (e.g., 14B vs 1.5B)
- Max lengths: 24000 prompt, 8000 generation

## Usage Instructions

### 1. Data Preparation
- Place review data in `data/review_data/` directory
- Format: JSON with `problem` (paper) and `answer` (review/rating) fields

### 2. Training Execution
```bash
cd MARTI-Dev
./scripts/run_train_review.sh /path/to/models WANDB_API_KEY
```

### 3. Customization
- **Data**: Modify `PROMPT_DATA` path in script for different datasets
- **Models**: Adjust `PRETRAIN` and `JUDGE_PRETRAIN` paths in script
- **Training**: Modify `default_agent.*` parameters in script or `ma_judge.yaml` config
- **Resources**: Adjust `default_agent.ref_num_gpus_per_node`, `vllm_num_engines`, etc. in script
- **Workflow**: Modify `workflow_args.*` in `ma_judge.yaml` for judge template/scoring

## Integration with Existing Systems

### 1. Preserved Components
- **review_eval.py**: Unchanged, integrated via auto_verify
- **MultiAgentRewardAllocation**: Used for reward distribution
- **Ray/VLLM infrastructure**: Standard MARTI-Dev setup

### 2. New Components
- Enhanced general judge_workflow.py (configurable templates + scoring)
- ma_judge configuration with ReviewRL-specific settings
- Review training script

### 3. Framework Compatibility
- Uses MARTI-Dev's async multi-agent system
- Compatible with existing training infrastructure
- Integrates with WANDB, Tensorboard logging

## Differences from Original ReviewRL-Preview

### 1. Architecture
- **Original**: Custom judge world implementation
- **New**: Standard MARTI-Dev multi-agent workflow

### 2. Configuration
- **Original**: Hardcoded configurations in script
- **New**: Hydra-based configuration system

### 3. Execution
- **Original**: Direct Python execution
- **New**: Ray job submission with environment isolation

### 4. Workflow
- **Original**: Synchronous execution
- **New**: Async workflow execution for better performance

## Testing and Validation

### 1. Before Running
- Verify `review_eval.py` is in `marti/verifiers/review/`
- Check data format matches expected keys
- Ensure model paths are accessible

### 2. Validation Steps
- Test config loading: `python -m marti.cli.commands.train --config-name ma_judge --help`
- Verify workflow imports work
- Check data loading with sample data

### 3. Monitoring
- WANDB project: "MARTI-ReviewRL"
- Tensorboard logs in `logs/` directory
- Ray dashboard on port 8265

## Troubleshooting

### 1. Common Issues
- **Workflow not found**: Check `workflow_func_path` points to correct file
- **Agent not found**: Verify agent names in config match workflow expectations
- **Reward errors**: Ensure `review_group` task is properly registered

### 2. Debug Steps
- Check Ray cluster status
- Verify file paths are relative to working directory
- Monitor GPU memory usage for large models

## Future Enhancements

### 1. Potential Improvements
- Support for multi-round review refinement
- Integration with more sophisticated judge models
- Dynamic reward weighting based on review quality

### 2. Extensibility
- Easy addition of new review metrics
- Support for different review formats
- Integration with external review systems

## General Applicability

### Enhanced Judge Workflow Usage for Other Tasks

The enhanced `judge_workflow.py` can now be used for various generator-judge tasks:

#### Example: Math Problem Evaluation
```yaml
workflow_args:
  judge_weight: 0.8
  score_parser: "regex"
  judge_template: |
    <|im_start|>user
    Evaluate this solution to the math problem. Rate from 1-10.
    
    Problem: {prompt}
    Solution: {generated_answer}
    
    Score: X/10
    <|im_end|>
    <|im_start|>assistant
```

#### Example: Code Quality Assessment
```yaml
workflow_args:
  judge_weight: 0.6
  score_parser: "keywords"
  judge_template: |
    <|im_start|>user
    Compare these two code solutions. Which is better?
    
    Problem: {prompt}
    Solution 1: {generated_answer}
    Solution 2: {label}
    
    Respond: CODE_1_BETTER or CODE_2_BETTER
    <|im_end|>
    <|im_start|>assistant
```

**Benefits of General Approach:**
- **Reusability**: One workflow serves multiple tasks
- **Consistency**: Same interface across different judge-based evaluations
- **Maintainability**: Single codebase to maintain and improve
- **Flexibility**: Easy customization for new tasks without code changes

---

**Transfer Date**: [Current Date]
**Framework Version**: MARTI-Dev (mature)
**Status**: Complete and ready for testing 