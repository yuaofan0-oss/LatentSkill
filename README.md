# LatentSkill: From In-Context Textual Skills to In-Weight Latent Skills for LLM Agents

<p align="center">
  <a href="https://arxiv.org/abs/2606.06087"><img src="https://img.shields.io/badge/Paper-arXiv-red" alt="Paper"></a>
  <a href="https://github.com/yuaofan0-oss/LatentSkill"><img src="https://img.shields.io/badge/Code-GitHub-blue" alt="Code"></a>
  <a href="https://huggingface.co/datasets/AofaYu71/LatentSkill"><img src="https://img.shields.io/badge/Data-HuggingFace-yellow" alt="Data"></a>
  <a href="https://huggingface.co/AofaYu71/LatentSkill"><img src="https://img.shields.io/badge/Checkpoints-HuggingFace-yellow" alt="Checkpoints"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-green" alt="License"></a>
</p>

This is the official repository for **LatentSkill**, a method that converts reusable textual agent skills into plug-and-play LoRA adapters through a hypernetwork-based skill compiler. Instead of injecting skill text into every prompt, LatentSkill stores skill knowledge in weight space, reducing prompt overhead while keeping skills modular, scalable, and composable.

## Overview

<p align="center">
  <img src="framework.png" width="95%" alt="LatentSkill framework">
</p>

LatentSkill follows a two-stage training pipeline:

1. **Skill-document pretraining** teaches the skill compiler to map reusable skill text into LoRA adapter weights.
2. **Trajectory-supervised fine-tuning** aligns generated adapters with agent behavior on downstream tasks.

At inference time, the compiler reads a skill description once, generates a task-specific adapter, and runs the frozen backbone model without including the full skill text in the prompt.

We are actively extending LatentSkill to additional backbone models. Please stay tuned for future releases.

## Installation

```bash
git clone https://github.com/yuaofan0-oss/LatentSkill.git
cd LatentSkill

conda create -n latentskill python=3.10 -y
conda activate latentskill

# Install PyTorch according to your CUDA version.
# Example for CUDA 12.1:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

pip install -r requirements.txt
pip install -U huggingface_hub
```

## Repository Layout

```text
latentskill/
  models/        Qwen3 LoRA wrapper and skill hypernetwork
  data/          dataset loaders and group-index preprocessing
  training/      pretraining, fine-tuning, and checkpoint utilities
  utils/         distributed training and helper utilities

configs/
  models/        model and hypernetwork configurations

scripts/
  train/         pretraining and fine-tuning launch scripts

evals/
  alfworld/      ALFWorld evaluation
  searchqa/      SearchQA evaluation and retrieval server
```

## Resources

Prepare the following resources under the project root, or pass custom paths through the environment variables shown below.

| Resource | Target path |
|---|---|
| Qwen3-8B backbone | `models/Qwen3-8B/` |
| LatentSkill pretraining data | `data/skill_pretrain/` from the [dataset repo](https://huggingface.co/datasets/AofaYu71/LatentSkill) |
| LatentSkill fine-tuning data | `data/skill_ift/` from the [dataset repo](https://huggingface.co/datasets/AofaYu71/LatentSkill) |
| SearchQA test data | `data/search_test/` from the [dataset repo](https://huggingface.co/datasets/AofaYu71/LatentSkill) |
| Pretrained LatentSkill checkpoint | `checkpoints/latentskill_pretrain_qwen3_8b/pretrain/` from the [model repo](https://huggingface.co/AofaYu71/LatentSkill) |
| Fine-tuned LatentSkill checkpoint | `checkpoints/latentskill_sft_qwen3_8b/train/` from the [model repo](https://huggingface.co/AofaYu71/LatentSkill) |
| E5 retriever model | `models/e5-base-v2/` |
| SearchQA wiki index and corpus | `wiki_index/` |
| ALFWorld data | `alfworld_data/alfworld/` |

Download the LatentSkill data released with this project:

```bash
hf download AofaYu71/LatentSkill \
  --repo-type dataset \
  --local-dir . \
  --include "data/skill_pretrain/train.jsonl" \
            "data/skill_pretrain/val.jsonl" \
            "data/skill_ift/train.json" \
            "data/search_test/search_test_all.jsonl"
```

Download the released LatentSkill checkpoints:

```bash
hf download AofaYu71/LatentSkill \
  --repo-type model \
  --local-dir . \
  --include "checkpoints/latentskill_pretrain_qwen3_8b/pretrain.tar.gz" \
            "checkpoints/latentskill_sft_qwen3_8b/train.tar.gz"
```

Extract the checkpoint archives:

```bash
tar -xzf checkpoints/latentskill_pretrain_qwen3_8b/pretrain.tar.gz \
  -C checkpoints/latentskill_pretrain_qwen3_8b/

tar -xzf checkpoints/latentskill_sft_qwen3_8b/train.tar.gz \
  -C checkpoints/latentskill_sft_qwen3_8b/
```

Download the backbone and retriever models:

```bash
hf download Qwen/Qwen3-8B --local-dir models/Qwen3-8B
hf download intfloat/e5-base-v2 --local-dir models/e5-base-v2
```

Please download the SearchQA wiki index and corpus, then place them at:

```text
wiki_index/e5_Flat.index
wiki_index/wiki-18.jsonl
```

Please download ALFWorld data and place it under:

```text
alfworld_data/alfworld/
```

The expected ALFWorld layout is:

```text
alfworld_data/alfworld/json_2.1.1/train/
alfworld_data/alfworld/json_2.1.1/valid_seen/
alfworld_data/alfworld/json_2.1.1/valid_unseen/
alfworld_data/alfworld/logic/alfred.pddl
alfworld_data/alfworld/logic/alfred.twl2
alfworld_data/alfworld/detectors/mrcnn.pth
```

## Training

Skill-document pretraining:

```bash
MODEL_PATH=models/Qwen3-8B \
DATA_ROOT=data \
CHECKPOINT_ROOT=checkpoints \
bash scripts/train/pretrain_qwen3_8b.sh
```

Trajectory-supervised fine-tuning:

```bash
MODEL_PATH=models/Qwen3-8B \
DATA_ROOT=data \
CHECKPOINT_ROOT=checkpoints \
PRETRAIN_CHECKPOINT_NAME=latentskill_pretrain_qwen3_8b \
bash scripts/train/sft_qwen3_8b.sh
```

The scripts support common overrides such as `NUM_GPUS`, `LEARNING_RATE`, `NUM_EPOCHS`, `TRAIN_BATCH_SIZE`, `EVAL_BATCH_SIZE`, `CONTEXT_MAX_LEN`, and `CONVERSATION_MAX_LEN`.

## Evaluation

### ALFWorld

```bash
export PROJECT_ROOT=/path/to/LatentSkill
export CHECKPOINT_DIR=/path/to/your/ift_checkpoint
export MODEL_PATH=/path/to/your/Qwen3-8B
export ALFWORLD_DATA_ROOT=/path/to/your/alfworld_data/alfworld
export ALFWORLD_CONFIG=/path/to/your/config_tw.yaml
export ALFWORLD_SKILL_DIR=/path/to/your/alfworld_skill_texts
export OUTPUT_DIR=/path/to/your/output_dir
export LOG_DIR=/path/to/your/log_dir
export GPU_ID=0

mkdir -p "$OUTPUT_DIR" "$LOG_DIR"

CUDA_VISIBLE_DEVICES="$GPU_ID" \
PYTHONPATH="$PROJECT_ROOT" \
python -m evals.alfworld.evaluate \
  --checkpoint "$CHECKPOINT_DIR" \
  --config_name models/qwen3_8b \
  --split unseen \
  --alfworld_data "$ALFWORLD_DATA_ROOT" \
  --alfworld_config "$ALFWORLD_CONFIG" \
  --skill_context_dir "$ALFWORLD_SKILL_DIR" \
  --max_steps 50 \
  --max_new_tokens 4096 \
  --context_max_length 4096 \
  --conversation_max_length 4096 \
  --output_dir "$OUTPUT_DIR" \
  --device cuda \
2>&1 | tee "$LOG_DIR/alfworld.log"
```

### SearchQA

Start a retrieval server separately or use the helper script in `evals/searchqa/run_eval.sh`. The evaluation command expects a running server at `RETRIEVAL_URL`.

```bash
export PROJECT_ROOT=/path/to/LatentSkill
export CHECKPOINT_DIR=/path/to/your/ift_checkpoint
export MODEL_PATH=/path/to/your/Qwen3-8B
export SEARCHQA_TEST_DATA=/path/to/your/search_test_all.jsonl
export SEARCHQA_SKILL_DIR=/path/to/your/searchqa_skill_texts
export RETRIEVAL_URL=http://127.0.0.1:8000/retrieve
export OUTPUT_DIR=/path/to/your/output_dir
export LOG_DIR=/path/to/your/log_dir
export GPU_ID=0

mkdir -p "$OUTPUT_DIR" "$LOG_DIR"

CUDA_VISIBLE_DEVICES="$GPU_ID" \
PYTHONPATH="$PROJECT_ROOT" \
python -m evals.searchqa.evaluate \
  --checkpoint "$CHECKPOINT_DIR" \
  --config_name models/qwen3_8b \
  --test_data "$SEARCHQA_TEST_DATA" \
  --skill_context_dir "$SEARCHQA_SKILL_DIR" \
  --retrieval_url "$RETRIEVAL_URL" \
  --retrieval_topk 3 \
  --max_steps 4 \
  --max_new_tokens 2048 \
  --context_max_length 4096 \
  --conversation_max_length 4096 \
  --output_dir "$OUTPUT_DIR" \
  --device cuda \
2>&1 | tee "$LOG_DIR/searchqa.log"
```

To start the bundled E5 retrieval server with the default paths:

```bash
E5_MODEL=models/e5-base-v2 \
python evals/searchqa/retrieval_server.py \
  --index_path wiki_index/e5_Flat.index \
  --corpus_path wiki_index/wiki-18.jsonl \
  --retriever_name e5 \
  --retriever_model models/e5-base-v2 \
  --topk 3 \
  --port 8000
```

## Citation

If you find this work useful, please cite:

```bibtex
@article{yu2026latentskillincontexttextualskills,
      title={LatentSkill: From In-Context Textual Skills to In-Weight Latent Skills for LLM Agents},
      author={Aofan Yu and Chenyu Zhou and Tianyi Xu and Zihan Guo and Rong Shan and Zhihui Fu and Jun Wang and Weiwen Liu and Yong Yu and Weinan Zhang and Jianghao Lin},
      year={2026},
      eprint={2606.06087},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2606.06087},
}
```

## License

This project is released under the MIT License.
