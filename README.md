<div align="center">
  <img src="logo.png" alt="NOVER Logo" width="600">
</div>

<p align="center">
    【📖 <a href="https://www.arxiv.org/pdf/2505.16022">arXiv</a> | 🤗 <a href="https://huggingface.co/papers/2505.16022">HF Papers</a> | 🧠 <a href="https://www.alphaxiv.org/abs/2505.16022">alphaArxiv</a> | 📦 <a href="https://huggingface.co/collections/thinkwee/novereason-68937ca75331dfaddaf24016">NOVEReason Datasets</a> | 🚀 <a href="https://huggingface.co/collections/thinkwee/nover1-68a6524eac725c915abd77e3">NOVER1 Models</a>】
</p>

# NOVER: Incentive Training for Language Models via Verifier-Free Reinforcement Learning

This is the official implementation of the paper ["NOVER: Incentive Training for Language Models via Verifier-Free Reinforcement Learning"](https://www.arxiv.org/pdf/2505.16022). This code repo is built based on [HuggingFace trl](https://github.com/huggingface/trl).

## Overview

- **NOVER** (**NO-VER**ifier) is a novel reinforcement learning approach for training language models **without requiring explicit verifiers**. 
- The method can perform DeepSeek R1-Zero-like training on **ANY SFT DATA**, extending reasoning abilities beyond math and coding.
- We released the NOVEReason dataset collections, including [NOVEReason_2k](https://huggingface.co/datasets/thinkwee/NOVEReason_2k), [NOVEReason_5k](https://huggingface.co/datasets/thinkwee/NOVEReason_5k) and [NOVEReason_full](https://huggingface.co/datasets/thinkwee/NOVEReason_full).

## Updates
- [x] Initialize training code  
- [x] Simplify SFT data import  
- [x] Add custom tag support  
- [x] Upgrade to `trl==0.20.0`  
- [x] Cleaned NOVER sampled data  
- [x] Integrate Hydra config management  
- [x] Simplify logging in `CustomGRPOTrainer`  
- [x] Streamline NCCL logging  
- [x] Simplify start script  
- [x] Clean full NOVEReason data (data has been cleaned, but not all was used in the paper)  
- [x] Release NOVER1-Qwen2.5-7B and NOVER1-Qwen3-4B and config files
- [ ] Add inverse incentive training support  
- [ ] Add incentive steering support  

## QuickStart

1. Clone this repository, install dependencies, set your [Weights & Biases](https://wandb.ai/site) credentials.
```bash
git clone https://github.com/thinkwee/NOVER.git
cd NOVER
pip install -r requirements.txt
export WANDB_API_KEY=your_api_key
export WANDB_ENTITY=your_entity
```

2. NOVER uses [Hydra](https://hydra.cc/docs/intro/) for configuration management. To run an experiment, simply create a YAML file specifying only the parameters you want to override. For example, see `config/my_exp.yaml`. For the full list of configurable options, refer to `config/config.yaml`.  

- Some key parameters to customize include:
  - `project.suffix`: A unique identifier for your training run
  - `project.wandb_project`: Your Weights & Biases project name
  - `project.save_base_path`: Directory to save model checkpoints
  - `dataset.hf_home`: Your huggingface root path for loading models and datasets
  - `dataset.name`: Your dataset name under `HF_HOME`
  - `model.name_vllm`: Model to use for VLLM server (e.g., "Qwen/Qwen2.5-7B")
  - `model.name`: Model to use for training (usually same as VLLM model)

3. Prepare your data, see the [data section](#data) below for more details.

4. Start the training! This will begin the training process using the configuration parameters defined in `config/my_exp.yaml`.

```bash
# first, start the vllm server
# This will launch a VLLM server for model rollouts in Reinforcement Learning
./scripts/run_vllm_server.sh my_exp

# then, start the training
./scripts/run_training.sh my_exp
```

## Data
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

- For convenience, you can use the included dataset formatter to automatically format your dataset.

```bash
# Format Hugging Face dataset
./scripts/format_dataset.sh squad --prompt-column question --reference-column answers.text

# Format a custom CSV file
./scripts/format_dataset.sh data.csv --prompt-column question --reference-column answer

# Format a custom JSONL file
./scripts/format_dataset.sh data.jsonl --prompt-column input --reference-column output
```

## Generation

The model is trained with Hugging Face PEFT and saved in the standard LoRA adapter format. To use the trained model, you can use the following code:


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
