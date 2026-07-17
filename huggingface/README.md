# Hugging Face Repository Cards

This directory contains the cards used by the Hugging Face repositories:

- `dataset/README.md`: upload as the README for `https://huggingface.co/datasets/AofaYu71/LatentSkill`
- `model/README.md`: upload as the README for `https://huggingface.co/AofaYu71/LatentSkill`

Recommended maintainer commands:

```bash
# Dataset card
hf upload AofaYu71/LatentSkill \
  huggingface/dataset/README.md README.md \
  --repo-type dataset \
  --commit-message "Update dataset card"

# Upload data directories without a top-level data/ prefix
hf upload AofaYu71/LatentSkill data/skill_pretrain skill_pretrain \
  --repo-type dataset
hf upload AofaYu71/LatentSkill data/skill_ift skill_ift \
  --repo-type dataset
hf upload AofaYu71/LatentSkill data/search_test search_test \
  --repo-type dataset

# Model repo, checkpoint archives, and model card
hf repos create AofaYu71/LatentSkill --repo-type model --public --exist-ok

hf upload AofaYu71/LatentSkill \
  huggingface/model/config.json config.json \
  --repo-type model \
  --commit-message "Add download tracking config"

hf upload AofaYu71/LatentSkill \
  checkpoints/latentskill_pretrain_qwen3_8b/pretrain.tar.gz \
  latentskill_pretrain_qwen3_8b/pretrain.tar.gz \
  --repo-type model \
  --commit-message "Upload LatentSkill pretraining checkpoint"

hf upload AofaYu71/LatentSkill \
  checkpoints/latentskill_sft_qwen3_8b/train.tar.gz \
  latentskill_sft_qwen3_8b/train.tar.gz \
  --repo-type model \
  --commit-message "Upload LatentSkill fine-tuned checkpoint"

hf upload AofaYu71/LatentSkill \
  huggingface/model/README.md README.md \
  --repo-type model \
  --commit-message "Add model card"
```

The dataset repository should contain only data files. The model repository should contain checkpoint archives without a top-level `checkpoints/` prefix.
