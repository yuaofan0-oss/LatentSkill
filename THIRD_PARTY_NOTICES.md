# Third-party notices

LatentSkill as a whole, including its original code and LatentSkill-authored
modifications to third-party code, is licensed under the [MIT License](LICENSE),
subject to the applicable third-party terms described here.

The third-party portions of the files listed below remain subject to the Apache
License, Version 2.0. The MIT license for LatentSkill and its modifications does
not replace or remove those upstream terms. Copyright, attribution, license, and
modification notices are retained in the source files. A copy of the Apache-2.0
license is included in
[third_party/licenses/Apache-2.0.txt](third_party/licenses/Apache-2.0.txt).

Files containing both kinds of contributions use
`SPDX-License-Identifier: MIT AND Apache-2.0`. MIT applies to the LatentSkill-authored
portions and modifications; Apache-2.0 applies to the upstream third-party
portions. Distribution of a combined file must comply with both licenses for
their respective portions. This is not an offer to choose either license for
the entire file. LatentSkill modifications are Copyright (c) 2026 Aofan Yu.

## SHINE

The following files are based on and adapted from
[SHINE](https://github.com/MuLabPKU/SHINE), which is licensed under
[Apache-2.0](https://github.com/MuLabPKU/SHINE/blob/main/LICENSE).
Their headers record modifications by the LatentSkill authors in 2026. Those
LatentSkill modifications are licensed under MIT; the SHINE-derived upstream
portions retain their Apache-2.0 terms.

- `configs/models/qwen3_8b.yaml`
- `latentskill/data/datasets.py`
- `latentskill/data/group_index.py`
- `latentskill/models/hypernetwork.py`
- `latentskill/models/lora_ops.py`
- `latentskill/models/qwen_lora.py`
- `latentskill/training/checkpointing.py`
- `latentskill/training/train_compiler.py`
- `latentskill/utils/__init__.py`
- `latentskill/utils/ddp.py`
- `latentskill/utils/freeze.py`
- `latentskill/utils/init.py`
- `latentskill/utils/logging.py`
- `latentskill/utils/optimize.py`
- `latentskill/utils/seed.py`
- `scripts/train/pretrain_qwen3_8b.sh`
- `scripts/train/sft_qwen3_8b.sh`

`latentskill/models/qwen_lora.py` also retains this upstream notice:

> Copyright 2025 The Qwen team, Alibaba Group and the HuggingFace Inc. team. All rights reserved.

## Evaluation prompts

The following files retain Apache-2.0 notices from the verl-agent (GiGPO) team:

- `evals/alfworld/prompts.py`
- `evals/searchqa/prompts.py`

> Copyright 2025 Nanyang Technological University (NTU), Singapore
> and the verl-agent (GiGPO) team.

## SearchQA evaluation utilities

`evals/searchqa/skillrl_utils.py` is adapted from
[Search-R1's answer-scoring utilities](https://github.com/PeterGriffinJin/Search-R1/blob/main/verl/utils/reward_score/qa_em.py)
and retains the following Apache-2.0 copyright notices:

> Copyright 2024 Bytedance Ltd. and/or its affiliates
>
> Copyright 2023-2024 SGLang Team
>
> Copyright 2025 Search-R1 Contributors

## External dependencies and resources

Separately downloaded dependencies, backbone models, retrievers, datasets, and
other external resources remain subject to their respective licenses. The
repository's MIT license does not replace those licenses.
