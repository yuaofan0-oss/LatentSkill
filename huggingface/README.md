# Hugging Face Repository Cards

This directory contains the cards used by the Hugging Face repositories:

- `dataset/README.md`: upload as the README for `https://huggingface.co/datasets/AofaYu71/LatentSkill`
- `model/README.md`: upload as the README for `https://huggingface.co/AofaYu71/LatentSkill`

Recommended maintainer commands:

```bash
# Dataset card. This also removes checkpoint archives from the dataset repo.
hf upload AofaYu71/LatentSkill \
  huggingface/dataset/README.md README.md \
  --repo-type dataset \
  --delete "checkpoints/**" \
  --commit-message "Add dataset card and remove checkpoints"

# Model repo, checkpoint archives, and model card
hf repos create AofaYu71/LatentSkill --repo-type model --public --exist-ok

hf upload AofaYu71/LatentSkill \
  checkpoints/latentskill_pretrain_qwen3_8b/pretrain.tar.gz \
  checkpoints/latentskill_pretrain_qwen3_8b/pretrain.tar.gz \
  --repo-type model \
  --commit-message "Upload LatentSkill pretraining checkpoint"

hf upload AofaYu71/LatentSkill \
  checkpoints/latentskill_sft_qwen3_8b/train.tar.gz \
  checkpoints/latentskill_sft_qwen3_8b/train.tar.gz \
  --repo-type model \
  --commit-message "Upload LatentSkill fine-tuned checkpoint"

hf upload AofaYu71/LatentSkill \
  huggingface/model/README.md README.md \
  --repo-type model \
  --commit-message "Add model card"
```

The dataset repository should contain only data files. Checkpoint archives should be moved to the model repository.
