#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os

import torch
from transformers import AutoTokenizer
from omegaconf import DictConfig, OmegaConf
import hydra
from datasets import load_dataset

from latentskill.data.datasets import StaticSkillGroupDataset
from latentskill.utils.seed import seed_everything
from latentskill.utils.logging import get_logger
from latentskill.utils.ddp import (
    process_rank,
    is_primary_process,
    initialize_distributed,
)

logger = get_logger("metalora")
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


@hydra.main(version_base=None, config_path="../../configs")
def main(cfg: DictConfig):
    if cfg.mode == "train":
        os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # ========= DDP init (safe for single-process) =========
    initialize_distributed()

    if is_primary_process():
        logger.info("Resolved config:")
        logger.info(f"\n\n{OmegaConf.to_yaml(cfg, resolve=True)}")

    # Seed & device
    # Make seed rank-dependent to vary shuffles but keep reproducibility per rank
    seed_everything(int(cfg.run.seed) + process_rank())
    torch.backends.cudnn.benchmark = True

    tokenizer = AutoTokenizer.from_pretrained(cfg.model.tokenizer_from, padding_side="left", use_fast=True)
    tokenizer.add_tokens(['<RECON>', '<COMP>', '<NOTHING>'])
    tokenizer.chat_template = "{%- if tools %}\n    {{- '<|im_start|>system\\n' }}\n    {%- if messages[0].role == 'system' %}\n        {{- messages[0].content + '\\n\\n' }}\n    {%- endif %}\n    {{- \"# Tools\\n\\nYou may call one or more functions to assist with the user query.\\n\\nYou are provided with function signatures within <tools></tools> XML tags:\\n<tools>\" }}\n    {%- for tool in tools %}\n        {{- \"\\n\" }}\n        {{- tool | tojson }}\n    {%- endfor %}\n    {{- \"\\n</tools>\\n\\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\\n<tool_call>\\n{\\\"name\\\": <function-name>, \\\"arguments\\\": <args-json-object>}\\n</tool_call><|im_end|>\\n\" }}\n{%- else %}\n    {%- if messages[0].role == 'system' %}\n        {{- '<|im_start|>system\\n' + messages[0].content + '<|im_end|>\\n' }}\n    {%- endif %}\n{%- endif %}\n{%- set ns = namespace(multi_step_tool=true, last_query_index=messages|length - 1) %}\n{%- for message in messages[::-1] %}\n    {%- set index = (messages|length - 1) - loop.index0 %}\n    {%- if ns.multi_step_tool and message.role == \"user\" and message.content is string and not(message.content.startswith('<tool_response>') and message.content.endswith('</tool_response>')) %}\n        {%- set ns.multi_step_tool = false %}\n        {%- set ns.last_query_index = index %}\n    {%- endif %}\n{%- endfor %}\n{%- for message in messages %}\n    {%- if message.content is string %}\n        {%- set content = message.content %}\n    {%- else %}\n        {%- set content = '' %}\n    {%- endif %}\n    {%- if (message.role == \"user\") or (message.role == \"system\" and not loop.first) %}\n        {{- '<|im_start|>' + message.role + '\\n' + content + '<|im_end|>\\n' }}\n    {%- elif message.role == \"assistant\" %}\n        {%- set reasoning_content = '' %}\n        {%- if message.reasoning_content is string %}\n            {%- set reasoning_content = message.reasoning_content %}\n        {%- else %}\n            {%- if '</think>' in content %}\n                {%- set reasoning_content = content.split('</think>')[0].rstrip('\\n').split('<think>')[-1].lstrip('\\n') %}\n                {%- set content = content.split('</think>')[-1].lstrip('\\n') %}\n            {%- endif %}\n        {%- endif %}\n        {%- if loop.index0 > ns.last_query_index %}\n            {%- if (loop.last or (not loop.last and reasoning_content)) and (enable_thinking is not defined or enable_thinking != false) %}\n                {{- '<|im_start|>' + message.role + '\\n<think>\\n' + reasoning_content.strip('\\n') + '\\n</think>\\n\\n' + content.lstrip('\\n') }}\n            {%- else %}\n                {{- '<|im_start|>' + message.role + '\\n' + content }}\n            {%- endif %}\n        {%- else %}\n            {{- '<|im_start|>' + message.role + '\\n' + content }}\n        {%- endif %}\n        {%- if message.tool_calls %}\n            {%- for tool_call in message.tool_calls %}\n                {%- if (loop.first and content) or (not loop.first) %}\n                    {{- '\\n' }}\n                {%- endif %}\n                {%- if tool_call.function %}\n                    {%- set tool_call = tool_call.function %}\n                {%- endif %}\n                {{- '<tool_call>\\n{\"name\": \"' }}\n                {{- tool_call.name }}\n                {{- '\", \"arguments\": ' }}\n                {%- if tool_call.arguments is string %}\n                    {{- tool_call.arguments }}\n                {%- else %}\n                    {{- tool_call.arguments | tojson }}\n                {%- endif %}\n                {{- '}\\n</tool_call>' }}\n            {%- endfor %}\n        {%- endif %}\n        {{- '<|im_end|>\\n' }}\n    {%- elif message.role == \"tool\" %}\n        {%- if loop.first or (messages[loop.index0 - 1].role != \"tool\") %}\n            {{- '<|im_start|>user' }}\n        {%- endif %}\n        {{- '\\n<tool_response>\\n' }}\n        {{- content }}\n        {{- '\\n</tool_response>' }}\n        {%- if loop.last or (messages[loop.index0 + 1].role != \"tool\") %}\n            {{- '<|im_end|>\\n' }}\n        {%- endif %}\n    {%- endif %}\n{%- endfor %}\n{%- if add_generation_prompt %}\n    {{- '<|im_start|>assistant\\n' }}\n    {%- if enable_thinking is not defined or enable_thinking != false %}\n        {{- '<think>\\n\\n</think>\\n\\n' }}\n    {%- endif %}\n{%- endif %}"

    # Data
    if is_primary_process():
        logger.info("Preparing data...")
    data_paths = cfg.data.paths
    skill_pretrain_dir = str(data_paths.skill_pretrain_dir)

    if cfg.data.source == "skill-pretrain":
        dataset = load_dataset(
            "json",
            data_files={
                "train": str(data_paths.skill_pretrain_train),
                "val": str(data_paths.skill_pretrain_val),
            }
        )
        train_texts = dataset["train"]
        val_texts = dataset["val"]
        # Generate group index files needed by training.
        train_ds = StaticSkillGroupDataset(
            train_texts["text"], tokenizer, cfg.data.conversation_max_length,
            skill_pretrain_dir, "train",
            preprocess_mode=True
        )
        val_ds = StaticSkillGroupDataset(
            val_texts["text"], tokenizer, cfg.data.conversation_max_length,
            skill_pretrain_dir, "val",
            preprocess_mode=True
        )
    elif cfg.data.source == "skill-pretrain-dynamic":
        if is_primary_process():
            logger.info("skill-pretrain-dynamic: no static group index needed, skipping.")
    else:
        raise ValueError(
            "group_index only supports source='skill-pretrain' or "
            f"source='skill-pretrain-dynamic', got {cfg.data.source!r}."
        )


if __name__ == "__main__":
    main()
