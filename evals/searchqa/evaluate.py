#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import json
import re
import argparse
import requests
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple

import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from omegaconf import OmegaConf

from latentskill.models.hypernetwork import SkillHypernetwork
from latentskill.training.checkpointing import restore_latentskill_checkpoint
from latentskill.utils.freeze import freeze_backbone_except_memory
from latentskill.utils.init import _import_class

from evals.searchqa.skillrl_utils import em_check, extract_solution
from evals.searchqa.prompts import (
    SEARCH_TEMPLATE_NO_HIS, SEARCH_TEMPLATE,
    SEARCH_TEMPLATE_WITH_MEMORY, SEARCH_TEMPLATE_NO_HIS_WITH_MEMORY,
    SEARCH_REACT_TEMPLATE_NO_HIS, SEARCH_REACT_TEMPLATE,
    SEARCH_COT_TEMPLATE, SEARCH_RAG_TEMPLATE,
    SEARCH_IRCOT_TEMPLATE_NO_HIS, SEARCH_IRCOT_TEMPLATE,
    ATTACK_CONDITIONS,
)

def parse_args():
    repo_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description="LatentSkill Search Benchmark Evaluation")
    parser.add_argument("--checkpoint", type=str, required=False, default=None,
                    help="IFT checkpoint directory; not required for zero-shot mode")
    parser.add_argument("--zero_shot", action="store_true",
                    help="Skip LatentSkill and run the Qwen3-8B backbone directly")
    parser.add_argument("--config_name", type=str, default="models/qwen3_8b",
                        help="Hydra config name under configs/")
    parser.add_argument("--test_data", type=str,
                        default=os.environ.get(
                            "SEARCHQA_TEST_DATA",
                            str(repo_root / "data/search_test/search_test_all.jsonl"),
                        ),
                        help="Merged test JSONL file")
    parser.add_argument("--skill_context_dir", type=str,
                        default=os.environ.get(
                            "SEARCHQA_SKILL_DIR",
                            str(repo_root / "evals/searchqa/skills"),
                        ),
                        help="Directory containing SearchQA skill text files")
    parser.add_argument("--retrieval_url", type=str,
                        default=os.environ.get("SEARCHQA_RETRIEVAL_URL", "http://127.0.0.1:8030/retrieve"),
                        help="Retrieval server HTTP endpoint")
    parser.add_argument("--retrieval_topk", type=int, default=3,
                        help="Number of retrieved documents per query")
    parser.add_argument("--max_steps", type=int, default=4,
                        help="Maximum interaction steps per question")
    parser.add_argument("--max_new_tokens", type=int, default=700,
                        help="Maximum generated tokens per step")
    parser.add_argument("--context_max_length", type=int, default=4096,
                        help="Maximum tokenized skill context length")
    parser.add_argument("--conversation_max_length", type=int, default=4096,
                        help="Maximum prompt length")
    parser.add_argument("--output_dir", type=str,
                        default=os.environ.get(
                            "SEARCHQA_OUTPUT_DIR",
                            str(repo_root / "evals/searchqa/results"),
                        ),
                        help="Output directory")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Inference device")
    parser.add_argument("--skill_incontext", action="store_true",
                    help="Put skill text in the prompt instead of using LoRA")
    parser.add_argument("--react", action="store_true",
                    help="Run the ReAct baseline with a few-shot demonstration")
    parser.add_argument("--cot", action="store_true",
                    help="Run the CoT baseline without retrieval")
    parser.add_argument("--rag", action="store_true",
                    help="Run the RAG baseline with one pre-retrieval step")
    parser.add_argument("--raw_base", action="store_true",
                    help="Load AutoModelForCausalLM directly without the LatentSkill LoRA wrapper")
    parser.add_argument("--ircot", action="store_true",
                help="Run the IRCoT baseline")
    parser.add_argument("--model_path", type=str, default=None,
                help="Optional model path for raw-base or external SFT checkpoints")
    parser.add_argument("--lora_combo", type=str, default="none",
                choices=["none", "multi_hop_only", "weight_sum", "hypernet_concat"],
                help=(
                    "Skill-LoRA composition mode: none routes by skill field; "
                    "multi_hop_only forces multi_hop_reasoning LoRA for all examples; "
                    "weight_sum combines comparison and multi_hop_reasoning LoRAs; "
                    "hypernet_concat feeds concatenated skill text to the hypernetwork"
                ))
    parser.add_argument("--scale", type=float, default=None,
                    help="Override the LoRA scale; None keeps the config value")
    parser.add_argument("--attack", type=str, default="none",
                    choices=["none", "task_explicit", "task_disguised",
                             "extract_naive", "extract_authority", "extract_indirect"],
                    help=(
                        "Adversarial attack suffix injected into the question; "
                        "one of none, task_explicit, task_disguised, "
                        "extract_naive, extract_authority, extract_indirect"
                    ))
    return parser.parse_args()


def load_model(args):
    """Load the LatentSkill skill_hypernet and return model handles."""
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    config_path = Path("configs") / f"{args.config_name}.yaml"
    cfg = OmegaConf.load(config_path)

    if args.raw_base:
        print("[model] raw_base mode: loading AutoModelForCausalLM directly")
        model_from     = args.model_path if args.model_path else cfg.model.model_from
        tokenizer_from = args.model_path if args.model_path else cfg.model.tokenizer_from

        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_from,
            padding_side="left",
            use_fast=True,
            trust_remote_code=True,
        )
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token

        raw_model = AutoModelForCausalLM.from_pretrained(
            model_from,
            torch_dtype=torch.bfloat16 if device.type == "cuda" else torch.float32,
            trust_remote_code=True,
        )
        raw_model.to(device)
        raw_model.eval()
        raw_model.config.use_cache = True
        return None, None, tokenizer, device, cfg, raw_model

    BackboneModelCls = _import_class(cfg.model.backbone_class_path)
    ConfigCls    = _import_class(cfg.model.config_class_path)

    model_config = ConfigCls.from_pretrained(cfg.model.model_from)
    model_config.num_mem_token = -1
    cfg.hidden_size = model_config.hidden_size
    cfg.num_layers  = model_config.num_hidden_layers

    tmp_model = BackboneModelCls.from_pretrained(cfg.model.model_from, config=model_config)
    adapter_numel = tmp_model.adapter_params_numel(cfg.model.lora_r)
    assert adapter_numel % (cfg.hidden_size * cfg.num_layers) == 0
    model_config.num_mem_token = (
        adapter_numel * cfg.hypernetwork.transformer_cfg.mean_pool_size
        // (cfg.hidden_size * cfg.num_layers)
    )
    cfg.num_mem_token = model_config.num_mem_token
    del tmp_model

    tokenizer = AutoTokenizer.from_pretrained(
        cfg.model.tokenizer_from, padding_side="left", use_fast=True
    )
    tokenizer.add_tokens(['<RECON>', '<COMP>', '<NOTHING>'])
    tokenizer.chat_template = (
        "{%- if tools %}\n    {{- '<|im_start|>system\\n' }}\n"
        "    {%- if messages[0].role == 'system' %}\n"
        "        {{- messages[0].content + '\\n\\n' }}\n    {%- endif %}\n"
        "    {{- \"# Tools\\n\\nYou may call one or more functions to assist with the user query."
        "\\n\\nYou are provided with function signatures within <tools></tools> XML tags:\\n<tools>\" }}\n"
        "    {%- for tool in tools %}\n        {{- \"\\n\" }}\n        {{- tool | tojson }}\n    {%- endfor %}\n"
        "    {{- \"\\n</tools>\\n\\nFor each function call, return a json object with function name and arguments "
        "within <tool_call></tool_call> XML tags:\\n<tool_call>\\n{\\\"name\\\": <function-name>, "
        "\\\"arguments\\\": <args-json-object>}\\n</tool_call><|im_end|>\\n\" }}\n{%- else %}\n"
        "    {%- if messages[0].role == 'system' %}\n"
        "        {{- '<|im_start|>system\\n' + messages[0].content + '<|im_end|>\\n' }}\n"
        "    {%- endif %}\n{%- endif %}\n"
        "{%- set ns = namespace(multi_step_tool=true, last_query_index=messages|length - 1) %}\n"
        "{%- for message in messages[::-1] %}\n"
        "    {%- set index = (messages|length - 1) - loop.index0 %}\n"
        "    {%- if ns.multi_step_tool and message.role == \"user\" and message.content is string "
        "and not(message.content.startswith('<tool_response>') and message.content.endswith('</tool_response>')) %}\n"
        "        {%- set ns.multi_step_tool = false %}\n        {%- set ns.last_query_index = index %}\n"
        "    {%- endif %}\n{%- endfor %}\n{%- for message in messages %}\n"
        "    {%- if message.content is string %}\n        {%- set content = message.content %}\n"
        "    {%- else %}\n        {%- set content = '' %}\n    {%- endif %}\n"
        "    {%- if (message.role == \"user\") or (message.role == \"system\" and not loop.first) %}\n"
        "        {{- '<|im_start|>' + message.role + '\\n' + content + '<|im_end|>\\n' }}\n"
        "    {%- elif message.role == \"assistant\" %}\n        {%- set reasoning_content = '' %}\n"
        "        {%- if message.reasoning_content is string %}\n"
        "            {%- set reasoning_content = message.reasoning_content %}\n"
        "        {%- else %}\n            {%- if '</think>' in content %}\n"
        "                {%- set reasoning_content = content.split('</think>')[0].rstrip('\\n').split('<think>')[-1].lstrip('\\n') %}\n"
        "                {%- set content = content.split('</think>')[-1].lstrip('\\n') %}\n"
        "            {%- endif %}\n        {%- endif %}\n"
        "        {%- if loop.index0 > ns.last_query_index %}\n"
        "            {%- if (loop.last or (not loop.last and reasoning_content)) and "
        "(enable_thinking is not defined or enable_thinking != false) %}\n"
        "                {{- '<|im_start|>' + message.role + '\\n<think>\\n' + "
        "reasoning_content.strip('\\n') + '\\n</think>\\n\\n' + content.lstrip('\\n') }}\n"
        "            {%- else %}\n                {{- '<|im_start|>' + message.role + '\\n' + content }}\n"
        "            {%- endif %}\n        {%- else %}\n"
        "            {{- '<|im_start|>' + message.role + '\\n' + content }}\n        {%- endif %}\n"
        "        {%- if message.tool_calls %}\n            {%- for tool_call in message.tool_calls %}\n"
        "                {%- if (loop.first and content) or (not loop.first) %}\n                    {{- '\\n' }}\n"
        "                {%- endif %}\n                {%- if tool_call.function %}\n"
        "                    {%- set tool_call = tool_call.function %}\n                {%- endif %}\n"
        "                {{- '<tool_call>\\n{\"name\": \"' }}\n                {{- tool_call.name }}\n"
        "                {{- '\", \"arguments\": ' }}\n"
        "                {%- if tool_call.arguments is string %}\n"
        "                    {{- tool_call.arguments }}\n                {%- else %}\n"
        "                    {{- tool_call.arguments | tojson }}\n                {%- endif %}\n"
        "                {{- '}\\n</tool_call>' }}\n            {%- endfor %}\n        {%- endif %}\n"
        "        {{- '<|im_end|>\\n' }}\n    {%- elif message.role == \"tool\" %}\n"
        "        {%- if loop.first or (messages[loop.index0 - 1].role != \"tool\") %}\n"
        "            {{- '<|im_start|>user' }}\n        {%- endif %}\n"
        "        {{- '\\n<tool_response>\\n' }}\n        {{- content }}\n"
        "        {{- '\\n</tool_response>' }}\n"
        "        {%- if loop.last or (messages[loop.index0 + 1].role != \"tool\") %}\n"
        "            {{- '<|im_end|>\\n' }}\n        {%- endif %}\n    {%- endif %}\n{%- endfor %}\n"
        "{%- if add_generation_prompt %}\n    {{- '<|im_start|>assistant\\n' }}\n"
        "    {%- if enable_thinking is not defined or enable_thinking != false %}\n"
        "        {{- '<think>\\n' }}\n    {%- endif %}\n{%- endif %}"
    )

    backbone = BackboneModelCls.from_pretrained(cfg.model.model_from, config=model_config)
    backbone.reset_mem_tokens()
    backbone.resize_token_embeddings(len(tokenizer))

    if args.zero_shot or args.skill_incontext or args.react or args.cot or args.rag or args.ircot:
        print("[model] zero-shot/baseline mode: loading only the Qwen3-8B backbone")
        backbone.to(device)
        backbone.eval()
        backbone.config.use_cache = True
        return None, None, tokenizer, device, cfg, backbone

    skill_hypernet = SkillHypernetwork(backbone, cfg, backbone.adapter_params_numel(cfg.model.lora_r))
    skill_hypernet.to(device)
    freeze_backbone_except_memory(backbone)

    print(f"[model] Loading IFT checkpoint from {args.checkpoint}...")
    skill_hypernet, metalora, _ = restore_latentskill_checkpoint(
        skill_hypernet, args.checkpoint, device, load_ift_additional_metalora=False
    )
    skill_hypernet.eval()
    backbone.config.use_cache = True

    print(f"[model] Loaded checkpoint. num_mem_token={model_config.num_mem_token}")
    return skill_hypernet, metalora, tokenizer, device, cfg, backbone


def load_skill_contexts(skill_context_dir: str) -> Dict[str, str]:
    """Load SearchQA skill context text."""
    skill_dir = Path(skill_context_dir)
    contexts = {}
    for skill in ["direct_retrieval", "multi_hop_reasoning", "comparison"]:
        path = skill_dir / f"{skill}.txt"
        if not path.exists():
            raise FileNotFoundError(f"Skill context file not found: {path}")
        contexts[skill] = path.read_text(encoding="utf-8").strip()
    print(f"[skill context] Loaded: {list(contexts.keys())}")
    return contexts


def retrieve(query: str, retrieval_url: str, topk: int = 3, timeout: int = 600) -> str:
    """
    Query the retrieval server and return formatted documents.
    Returns an empty string on failure so inference can continue.
    """
    try:
        resp = requests.post(
            retrieval_url,
            json={"query": query, "topk": topk, "return_scores": False},
            timeout=timeout,
        )
        resp.raise_for_status()
        print(f"[RETRIEVE] status={resp.status_code} query={query!r}", flush=True)
        data = resp.json()
        docs = data["result"][0]
        if not docs:
            return ""
        parts = []
        for i, doc in enumerate(docs, 1):
            title   = doc.get("title", "").strip()
            content = doc.get("contents", doc.get("text", "")).strip()
            parts.append(f"Doc {i}: {title}\n{content}")
        return "\n\n".join(parts)
    except Exception as e:
        print(f"[retrieval warning] retrieval failed (query={query!r}): {e}")
        return ""


def run_agent_loop(
    question:          str,
    skill_hypernet,
    metalora,
    adapter_state,
    tokenizer,
    device:            torch.device,
    retrieval_url:     str,
    retrieval_topk:    int,
    max_steps:         int,
    max_new_tokens:    int,
    conversation_max_length: int,
    zero_shot:         bool = False,
    backbone = None,
    skill_incontext:   bool = False,
    skill_text:        str = "",
    react:             bool = False,
    cot:               bool = False,
    rag:               bool = False,
    raw_base:          bool = False,
    ircot:             bool = False
) -> Tuple[str, List[dict]]:
    """
    Run the agent loop and return the final answer plus per-step details.
    """
    history_str = ""
    steps_log   = []

    final_answer = ""

    for step in range(1, max_steps + 1):
        if skill_incontext and skill_text:
            if step == 1:
                prompt_text = SEARCH_TEMPLATE_NO_HIS_WITH_MEMORY.format(
                    task_description=question,
                    retrieved_memories=skill_text,
                )
            else:
                prompt_text = SEARCH_TEMPLATE_WITH_MEMORY.format(
                    task_description=question,
                    retrieved_memories=skill_text,
                    step_count=step - 1,
                    memory_context=history_str,
                )
        elif react:
            if step == 1:
                prompt_text = SEARCH_REACT_TEMPLATE_NO_HIS.format(
                    task_description=question
                )
            else:
                prompt_text = SEARCH_REACT_TEMPLATE.format(
                    task_description=question,
                    step_count=step - 1,
                    memory_context=history_str,
                )
        elif cot:
            prompt_text = SEARCH_COT_TEMPLATE.format(
                task_description=question
            )
        elif rag:
            retrieved_docs = retrieve(
                question, retrieval_url, topk=retrieval_topk
            )
            prompt_text = SEARCH_RAG_TEMPLATE.format(
                task_description=question,
                retrieved_docs=retrieved_docs if retrieved_docs else "No documents retrieved.",
            )
        elif ircot:
            if step == 1:
                prompt_text = SEARCH_IRCOT_TEMPLATE_NO_HIS.format(
                    task_description=question
                )
            else:
                prompt_text = SEARCH_IRCOT_TEMPLATE.format(
                    task_description=question,
                    step_count=step - 1,
                    memory_context=history_str,
                )
        elif step == 1:
            prompt_text = SEARCH_TEMPLATE_NO_HIS.format(
                task_description=question
            )
        else:
            prompt_text = SEARCH_TEMPLATE.format(
                task_description=question,
                step_count=step - 1,
                memory_context=history_str,
            )
        prompt_text = prompt_text.lstrip("\n")

        messages = [{"role": "user", "content": prompt_text}]

        input_enc = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_tensors="pt",
            max_length=conversation_max_length,
            truncation=True,
            return_dict=True,
            padding=False,
            enable_thinking=False,
        )
        input_ids      = input_enc["input_ids"].to(device)
        attention_mask = input_enc["attention_mask"].to(device)

        with torch.no_grad():
            if raw_base:
                output_ids = backbone.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=max_new_tokens,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    do_sample=False,
                )
            elif zero_shot or skill_incontext or react or cot or rag or ircot:
                output_ids = backbone.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=max_new_tokens,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    do_sample=False,
                    ignore_mem_token=True,
                )
            else:
                output_ids = skill_hypernet.backbone.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=max_new_tokens,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    do_sample=False,
                    ignore_mem_token=True,
                    adapter_state=adapter_state,
                )

        new_tokens  = output_ids[0, input_ids.shape[1]:]
        output_text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

        step_record = {"step": step, "output": output_text}

        if "<answer>" in output_text and "</answer>" in output_text:
            answer = extract_solution(output_text)
            final_answer = answer if answer is not None else ""
            step_record["action"] = "answer"
            step_record["answer"] = final_answer
            steps_log.append(step_record)
            break

        elif "<search>" in output_text and "</search>" in output_text:
            match = re.search(r"<search>(.*?)</search>", output_text, re.DOTALL)
            query = match.group(1).strip() if match else ""
            step_record["action"] = "search"
            step_record["query"]  = query

            retrieved_docs = retrieve(
                query, retrieval_url, topk=retrieval_topk
            )
            step_record["retrieved_docs"] = retrieved_docs[:500]

            history_str += f"<search>{query}</search>\n\n"
            if retrieved_docs:
                history_str += f"<information>\n{retrieved_docs}\n</information>\n"
            else:
                history_str += "<information>\nNo results found.\n</information>\n"

            steps_log.append(step_record)

        else:
            step_record["action"] = "invalid"
            steps_log.append(step_record)
            break

    if not final_answer and steps_log and steps_log[-1].get("action") != "answer":
        last_output = steps_log[-1].get("output", "")
        answer = extract_solution(last_output)
        final_answer = answer if answer is not None else ""

    return final_answer, steps_log

def main():
    args = parse_args()

    output_dir = Path(args.output_dir)
    if args.lora_combo != "none":
        output_dir = output_dir / f"combo_{args.lora_combo}"
    output_dir.mkdir(parents=True, exist_ok=True)

    skill_hypernet, metalora, tokenizer, device, cfg, backbone = load_model(args)

    DEFAULT_SCALE = 0.001
    if args.scale is not None and skill_hypernet is not None:
        old_scale = skill_hypernet.scale
        skill_hypernet.scale = args.scale
        multiplier = args.scale / DEFAULT_SCALE if DEFAULT_SCALE > 0 else float('inf')
        print(f"[SCALE] previous scale: {old_scale}")
        print(f"[SCALE] current scale: {args.scale} ({multiplier:.1f}x default)")
    else:
        if skill_hypernet is not None:
            print(f"[SCALE] config scale: {skill_hypernet.scale}")
        else:
            print("[SCALE] baseline mode: skipping scale setup")

    skill_contexts = load_skill_contexts(args.skill_context_dir)

    baseline_mode = (
        args.raw_base or args.zero_shot or args.skill_incontext or
        args.react or args.cot or args.rag or args.ircot
    )

    if baseline_mode:
        print("[adapter] baseline mode: skipping Skill-LoRA generation")
        skill_adapter_states = {skill: None for skill in skill_contexts}
    else:
        print("[adapter] generating skill adapter states...")
        skill_adapter_states = {}
        for skill, context_text in skill_contexts.items():
            evidence_enc = tokenizer(
                context_text,
                max_length=args.context_max_length,
                truncation=True,
                return_tensors="pt",
                padding=False,
            )
            evidence_ids  = evidence_enc["input_ids"].to(device)
            evidence_mask = evidence_enc["attention_mask"].to(device)
            with torch.no_grad():
                adapter_state = skill_hypernet.build_adapter_state(
                    evidence_ids, evidence_mask, metalora
                )
            skill_adapter_states[skill] = adapter_state
            print(f"  [OK] {skill}")

    def merge_adapter_states(ld_A, ld_B, w=0.5):
        """
        Merge two adapter states by rank concatenation.
        This is equivalent to dW_merged = w*dW_A + w*dW_B.
        """
        import math
        sw = math.sqrt(w)
        merged = {}
        for i in ld_A:
            merged[i] = {"attention": {}, "mlp": {}}
            for proj in ["q", "k", "v", "o"]:
                a = ld_A[i]["attention"][proj]
                b = ld_B[i]["attention"][proj]
                merged[i]["attention"][proj] = {
                    "A": torch.cat([a["A"] * sw, b["A"] * sw], dim=-1),  # [1, in, 2r]
                    "B": torch.cat([a["B"] * sw, b["B"] * sw], dim=-2),  # [1, 2r, out]
                    "C": None,
                }
            for proj in ["gate", "up", "down"]:
                a = ld_A[i]["mlp"][proj]
                b = ld_B[i]["mlp"][proj]
                merged[i]["mlp"][proj] = {
                    "A": torch.cat([a["A"] * sw, b["A"] * sw], dim=-1),
                    "B": torch.cat([a["B"] * sw, b["B"] * sw], dim=-2),
                    "C": None,
                }
        return merged

    combo_adapter_state = None
    if not baseline_mode:

        if args.lora_combo == "multi_hop_only":
            combo_adapter_state = skill_adapter_states["multi_hop_reasoning"]
            print("[lora_combo] multi_hop_only: forcing multi_hop_reasoning LoRA")

        elif args.lora_combo == "weight_sum":
            combo_adapter_state = merge_adapter_states(
                skill_adapter_states["comparison"],
                skill_adapter_states["multi_hop_reasoning"],
                w=0.5
            )
            print("[lora_combo] weight_sum: rank expands from 8 to 16")

        elif args.lora_combo == "hypernet_concat":
            concat_text = (
                skill_contexts["multi_hop_reasoning"]
                + "\n\n"
                + skill_contexts["comparison"]
            )
            concat_enc = tokenizer(
                concat_text,
                max_length=args.context_max_length,
                truncation=True,
                return_tensors="pt",
                padding=False,
            )
            concat_ids  = concat_enc["input_ids"].to(device)
            concat_mask = concat_enc["attention_mask"].to(device)
            with torch.no_grad():
                combo_adapter_state = skill_hypernet.build_adapter_state(
                    concat_ids, concat_mask, metalora
                )
            print(f"[lora_combo] hypernet_concat: concatenated length={concat_ids.shape[1]} tokens")

        else:
            print("[lora_combo] none: routing by skill field")

    print(f"[data] Loading {args.test_data}...")
    test_records = []
    with open(args.test_data, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                test_records.append(json.loads(line))
    print(f"[data] Loaded {len(test_records)} test records")

    detail_path = output_dir / "results_detail.jsonl"
    dataset_em  = defaultdict(list)

    with open(detail_path, "w", encoding="utf-8") as detail_f:
        for record in tqdm(test_records, desc="inference"):
            qid            = record["id"]
            dataset        = record["dataset"]
            skill          = record["skill"]
            question       = record["question"]
            golden_answers = record["golden_answers"]

            _attack_suffix = ATTACK_CONDITIONS.get(args.attack, "")
            if _attack_suffix:
                question = question + _attack_suffix

            if args.lora_combo != "none" and combo_adapter_state is not None:
                adapter_state = combo_adapter_state
            else:
                adapter_state = skill_adapter_states.get(skill, skill_adapter_states["direct_retrieval"])

            pred_answer, steps_log = run_agent_loop(
                question=question,
                skill_hypernet=skill_hypernet,
                metalora=metalora,
                adapter_state=adapter_state,
                tokenizer=tokenizer,
                device=device,
                retrieval_url=args.retrieval_url,
                retrieval_topk=args.retrieval_topk,
                max_steps=args.max_steps,
                max_new_tokens=args.max_new_tokens,
                conversation_max_length=args.conversation_max_length,
                zero_shot=args.zero_shot,
                backbone=backbone,
                skill_incontext=args.skill_incontext,
                skill_text=skill_contexts.get(skill, ""),
                react=args.react,
                cot=args.cot,
                rag=args.rag,
                raw_base=args.raw_base,
                ircot=args.ircot,
            )

            em = em_check(pred_answer, golden_answers)
            dataset_em[dataset].append(em)

            detail_record = {
                "id":             qid,
                "dataset":        dataset,
                "skill":          skill,
                "question":       question,
                "golden_answers": golden_answers,
                "pred_answer":    pred_answer,
                "em":             em,
                "steps":          steps_log,
                "attack_condition": args.attack,
            }
            detail_f.write(json.dumps(detail_record, ensure_ascii=False) + "\n")

    DATASET_ORDER = [
        "nq", "hotpotqa", "triviaqa", "popqa",
        "2wikimultihopqa", "musique", "bamboogle"
    ]

    summary = {}
    all_ems = []
    print("\n" + "=" * 50)
    print(f"{'Dataset':<20} {'EM':>8} {'Count':>8}")
    print("=" * 50)

    for ds in DATASET_ORDER:
        ems = dataset_em.get(ds, [])
        if ems:
            avg_em = sum(ems) / len(ems)
            summary[ds] = {"em": round(avg_em, 4), "count": len(ems)}
            all_ems.extend(ems)
            print(f"{ds:<20} {avg_em:>8.4f} {len(ems):>8}")
        else:
            summary[ds] = {"em": None, "count": 0}
            print(f"{ds:<20} {'N/A':>8} {0:>8}")

    for ds, ems in dataset_em.items():
        if ds not in DATASET_ORDER:
            avg_em = sum(ems) / len(ems)
            summary[ds] = {"em": round(avg_em, 4), "count": len(ems)}
            all_ems.extend(ems)
            print(f"{ds:<20} {avg_em:>8.4f} {len(ems):>8}")

    overall_em = sum(all_ems) / len(all_ems) if all_ems else 0.0
    summary["average"] = {"em": round(overall_em, 4), "count": len(all_ems)}
    print("-" * 50)
    print(f"{'average':<20} {overall_em:>8.4f} {len(all_ems):>8}")
    print("=" * 50)

    summary_path = output_dir / "results_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\nResults saved:")
    print(f"  detail:  {detail_path}")
    print(f"  summary: {summary_path}")


if __name__ == "__main__":
    main()
