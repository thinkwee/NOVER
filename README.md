<div align="center">
  <img src="logo.png" alt="NOVER Logo" width="600">
</div>

<p align="center">
    【📄 <a href="https://www.arxiv.org/pdf/2505.16022">arXiv</a> | 🤗 <a href="https://huggingface.co/papers/2505.16022">HuggingFace Papers</a> | 🧠 <a href="https://www.alphaxiv.org/abs/2505.16022">alphaArxiv</a>】
</p>

# NOVER: Incentive Training for Language Models via Verifier-Free Reinforcement Learning

This is the official implementation of the paper ["NOVER: Incentive Training for Language Models via Verifier-Free Reinforcement Learning"](https://www.arxiv.org/pdf/2505.16022). This code repo is built based on [HuggingFace trl](https://github.com/huggingface/trl).

## Overview

NOVER is a novel reinforcement learning approach for training language models without requiring explicit verifiers. The method can perform DeepSeek R1-Zero-like training on any SFT data, extending reasoning abilities beyond math and coding to fields such as translation, scientific reasoning, creative writing, social reasoning, and more.

## Installation

1. Clone this repository
```bash
git clone https://github.com/thinkwee/NOVER.git
cd NOVER
```

2. Install dependencies
```bash
pip install -r requirements.txt
```

## Configuration

Before training, you need to modify the configuration settings in `config.py`. We recommend using [Qwen/Qwen2.5-7B](https://huggingface.co/Qwen/Qwen2.5-7B).
The key parameters to update include:

- **Project identification**:
  - `SUFFIX`: A unique identifier for your training run
  - `WANDB_PROJECT`: Your Weights & Biases project name
  - `HF_HOME`: Your huggingface root path for loading models and datasets
  - `DATASET_NAME`: Your dataset name under `HF_HOME`

- **Model settings**:
  - `MODEL_NAME_VLLM`: Model to use for VLLM server (e.g., "Qwen/Qwen2.5-7B")
  - `MODEL_NAME`: Model to use for training (usually same as VLLM model)
  - `SAVE_BASE_PATH`: Directory to save model checkpoints

## Training Process

### Step 1: Prepare Your Data
- Format your data as a standard Hugging Face Arrow dataset with at least two columns: `prompt` and `reference`, representing input and output in any standard SFT dataset. No conversational format or system prompts are required.
- Structure your input data in the `prompt` column as follows:
```markdown
Question: {input}

Answer the question and return in the following format:

<think>
...
</think>

<answer>
...
</answer>
```

#### NOVER for ANY SFT DATA
For convenience, you can use the included dataset formatter to automatically format your dataset:
```bash
# Format Hugging Face dataset
./format_dataset.sh squad --prompt-column question --reference-column answers.text

# Format CSV file
./format_dataset.sh data.csv --prompt-column question --reference-column answer

# Format JSONL file
./format_dataset.sh data.jsonl --prompt-column input --reference-column output
```

#### Using Custom Tags

NOVER now supports custom intermediate and final tags beyond the default `<think>` and `<answer>` tags. This allows you to adapt the system to different incentivived abilities and task types, for example:

```bash
# Format dataset with custom tags
./format_dataset.sh data.jsonl --intermediate-tag role_play --final-tag story

# Start training with custom tags
./run_training.sh --intermediate-tag role_play --final-tag story
```

You can also set default tags in `config.py` by modifying the `INTERMEDIATE_TAG` and `FINAL_TAG` variables. This feature enables flexibility in how models express their reasoning process and final answers.

### Step 2: Weights & Biases Setup

Set up your Weights & Biases credentials:

```bash
export WANDB_API_KEY=your_api_key
export WANDB_ENTITY=your_entity
```

### Step 3: Start the VLLM Server

First, start the VLLM server for model rollouts:

```bash
sh run_vllm_server.sh
```

This will launch a VLLM server using the model specified in `MODEL_NAME_VLLM` and the port specified in `VLLM_PORT`.

### Step 4: Start Training

After the VLLM server is running, start the training process:

```bash
sh run_training.sh
```

This will begin the training process using the configuration parameters defined in `config.py`.

## Model Generation

The model is trained with Hugging Face PEFT and saved in the standard LoRA adapter format. To use the trained model:

1. Load the adapter and merge it with the base model:

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM

base_model = AutoModelForCausalLM.from_pretrained(base_model_path)
model = PeftModel.from_pretrained(base_model, adapter_path)
merged_model = model.merge_and_unload()

model.save_pretrained(
    merge_output_dir,
    safe_serialization=True)

tokenizer = AutoTokenizer.from_pretrained(base_model_path)
tokenizer.save_pretrained(merge_output_dir)
```

2. Use the vllm to serve the merged model in ```merge_output_dir```.

## Citation

If you find this work useful, please cite our paper:

```bibtex
@article{liu2025noverincentivetraininglanguage,
      title={NOVER: Incentive Training for Language Models via Verifier-Free Reinforcement Learning}, 
      author={Wei Liu and Siya Qi and Xinyu Wang and Chen Qian and Yali Du and Yulan He},
      year={2025},
      eprint={2505.16022},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2505.16022}, 
}
```

## Todo
- [x] init for training code
- [x] support easier SFT data import
- [x] support custom tags
- [ ] upgrade to trl 0.17.0
- [x] cleaned NOVER sampled data
- [ ] cleaned NOVER full data
- [ ] support inverse incentive training
- [ ] support incentive steering
