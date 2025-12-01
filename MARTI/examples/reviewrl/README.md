# ReviewRL: Towards Automated Scientific Review with RL

This repository contains the RL implementation for the research paper: *ReviewRL: Towards Automated Scientific Review with RL*.

## Abstract

Peer review is essential for scientific progress but faces growing challenges due to increasing submission volumes and reviewer fatigue. Existing automated review approaches struggle with factual accuracy, rating consistency, and analytical depth, often generating superficial or generic feedback lacking the insights characteristic of high-quality human reviews. We introduce ReviewRL, a reinforcement learning framework for generating comprehensive and factually grounded scientific paper reviews. Our approach combines: (1) an ArXiv-MCP retrieval-augmented context generation pipeline that incorporates relevant scientific literature, (2) supervised fine-tuning that establishes foundational reviewing capabilities, and (3) a reinforcement learning procedure with a composite reward function that jointly enhances review quality and rating accuracy. Experiments on ICLR 2025 papers demonstrate that ReviewRL significantly outperforms existing methods across both rule-based metrics and model-based quality assessments. ReviewRL establishes a foundational framework for RL-driven automatic critique generation in scientific discovery, demonstrating promising potential for future development in this domain.


## Usage

The training process can be started using the provided bash script.

```bash
bash examples/reviewrl/run_train_reviewrl_async_multinode.sh
```

This script will initiate the RL training process for the ReviewRL task.


### Dataset Format

For Reinforcement Learning training, the dataset is provided as a `jsonl` file. Each line in the file is a JSON object with the following structure:

```json
{
  "problem": "The content of the paper and any retrieval results...",
  "answer": "reference_review||DIV REVIEW SCORE||reference_rating"
}
```

* `problem`: This key holds the prompt for the model, which includes the full text of the scientific paper, any relevant retrieved information, and instructions.
* `answer`: This key contains a string with three parts, delimited by `||DIV REVIEW SCORE||`:
    * `reference_review`: The ground truth or reference review of the paper.
    * A separator string.
    * `reference_rating`: The corresponding average rating for the paper.

