---
license: mit
base_model:
- Qwen/Qwen3-8B
library_name: transformers
datasets:
- AofaYu71/LatentSkill
language:
- en
tags:
- agents
- large-language-models
- qwen3
- lora
- hypernetwork
- skill-learning
---

# LatentSkill Checkpoints

This model repository contains the released checkpoints for **LatentSkill: From In-Context Textual Skills to In-Weight Latent Skills for LLM Agents**.

Code: https://github.com/yuaofan0-oss/LatentSkill  
Paper: https://arxiv.org/abs/2606.06087  
Dataset repository: https://huggingface.co/datasets/AofaYu71/LatentSkill

## Contents

```text
latentskill_pretrain_qwen3_8b/
  pretrain.tar.gz
latentskill_sft_qwen3_8b/
  train.tar.gz
```

The checkpoint archives are expected to be extracted under the code repository root:

```text
checkpoints/latentskill_pretrain_qwen3_8b/pretrain/
checkpoints/latentskill_sft_qwen3_8b/train/
```

## Download

From the root of the code repository:

```bash
hf download AofaYu71/LatentSkill \
  --repo-type model \
  --local-dir checkpoints \
  --include "latentskill_pretrain_qwen3_8b/pretrain.tar.gz" \
            "latentskill_sft_qwen3_8b/train.tar.gz"

tar -xzf checkpoints/latentskill_pretrain_qwen3_8b/pretrain.tar.gz \
  -C checkpoints/latentskill_pretrain_qwen3_8b/

tar -xzf checkpoints/latentskill_sft_qwen3_8b/train.tar.gz \
  -C checkpoints/latentskill_sft_qwen3_8b/
```

## Intended Use

These checkpoints are intended for reproducing the LatentSkill training and evaluation pipeline described in the paper. They are used with the LatentSkill codebase and the Qwen3-8B backbone.

The checkpoints are not standalone conversational models. Please load them through the project code and follow the paths documented in the GitHub README.

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
