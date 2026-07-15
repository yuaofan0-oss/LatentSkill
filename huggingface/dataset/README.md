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
skill_pretrain/
  train.jsonl
  val.jsonl
skill_ift/
  train.json
search_test/
  2wikimultihopqa_test.jsonl
  bamboogle_test.jsonl
  comparison_214.jsonl
  hotpotqa_test.jsonl
  musique_test.jsonl
  nq_test.jsonl
  popqa_test.jsonl
  search_test_all.jsonl
  triviaqa_test.jsonl
```

The repository contains:

- `skill_pretrain/`: skill-document pretraining data.
- `skill_ift/`: trajectory-supervised fine-tuning data.
- `search_test/`: SearchQA evaluation data released with this project.

## Splits

| Split group | Files |
|---|---|
| Skill pretraining | `skill_pretrain/train.jsonl`, `skill_pretrain/val.jsonl` |
| Skill fine-tuning | `skill_ift/train.json` |
| SearchQA aggregate test | `search_test/search_test_all.jsonl` |
| SearchQA source tests | `search_test/2wikimultihopqa_test.jsonl`, `search_test/bamboogle_test.jsonl`, `search_test/comparison_214.jsonl`, `search_test/hotpotqa_test.jsonl`, `search_test/musique_test.jsonl`, `search_test/nq_test.jsonl`, `search_test/popqa_test.jsonl`, `search_test/triviaqa_test.jsonl` |

Model checkpoints are not stored in this dataset repository. Please use the LatentSkill model repository for checkpoints.

## Download

From the root of the code repository:

```bash
hf download AofaYu71/LatentSkill \
  --repo-type dataset \
  --local-dir data \
  --include "skill_pretrain/train.jsonl" \
            "skill_pretrain/val.jsonl" \
            "skill_ift/train.json" \
            "search_test/2wikimultihopqa_test.jsonl" \
            "search_test/bamboogle_test.jsonl" \
            "search_test/comparison_214.jsonl" \
            "search_test/hotpotqa_test.jsonl" \
            "search_test/musique_test.jsonl" \
            "search_test/nq_test.jsonl" \
            "search_test/popqa_test.jsonl" \
            "search_test/search_test_all.jsonl" \
            "search_test/triviaqa_test.jsonl"
```

## Usage

The downloaded files should match the following paths in the code repository:

```text
data/skill_pretrain/train.jsonl
data/skill_pretrain/val.jsonl
data/skill_ift/train.json
data/search_test/2wikimultihopqa_test.jsonl
data/search_test/bamboogle_test.jsonl
data/search_test/comparison_214.jsonl
data/search_test/hotpotqa_test.jsonl
data/search_test/musique_test.jsonl
data/search_test/nq_test.jsonl
data/search_test/popqa_test.jsonl
data/search_test/search_test_all.jsonl
data/search_test/triviaqa_test.jsonl
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
