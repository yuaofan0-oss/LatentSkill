---
license: mit
task_categories:
- question-answering
language:
- en
pretty_name: LatentSkill Data
tags:
- agents
- large-language-models
- lora
- hypernetwork
- skill-learning
---

# LatentSkill Data

This dataset repository contains the data released for **LatentSkill: From In-Context Textual Skills to In-Weight Latent Skills for LLM Agents**.

Code: https://github.com/yuaofan0-oss/LatentSkill  
Paper: https://arxiv.org/abs/2606.06087  
Checkpoint repository: https://huggingface.co/AofaYu71/LatentSkill

## Contents

```text
data/
  skill_pretrain/
    train.jsonl
    val.jsonl
  skill_ift/
    train.json
  search_test/
    search_test_all.jsonl
```

The repository contains:

- `data/skill_pretrain/`: skill-document pretraining data.
- `data/skill_ift/`: trajectory-supervised fine-tuning data.
- `data/search_test/`: SearchQA evaluation data released with this project.

Model checkpoints are not stored in this dataset repository. Please use the LatentSkill model repository for checkpoints.

## Download

From the root of the code repository:

```bash
hf download AofaYu71/LatentSkill \
  --repo-type dataset \
  --local-dir . \
  --include "data/skill_pretrain/train.jsonl" \
            "data/skill_pretrain/val.jsonl" \
            "data/skill_ift/train.json" \
            "data/search_test/search_test_all.jsonl"
```

## Usage

The downloaded files should match the following paths in the code repository:

```text
data/skill_pretrain/train.jsonl
data/skill_pretrain/val.jsonl
data/skill_ift/train.json
data/search_test/search_test_all.jsonl
```

See the GitHub repository for training and evaluation commands.

## Citation

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
