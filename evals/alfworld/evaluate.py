#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import json
import re
import argparse
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import torch
import yaml
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from omegaconf import OmegaConf

# LatentSkill imports
from latentskill.models.hypernetwork import SkillHypernetwork
from latentskill.training.checkpointing import restore_latentskill_checkpoint
from latentskill.utils.freeze import freeze_backbone_except_memory
from latentskill.utils.init import _import_class

from evals.alfworld.prompts import (
    ALFWORLD_TEMPLATE, ALFWORLD_TEMPLATE_NO_HIS,
    ALFWORLD_TEMPLATE_WITH_MEMORY, ALFWORLD_TEMPLATE_NO_HIS_WITH_MEMORY,
    ALFWORLD_REACT_TEMPLATE, ALFWORLD_REACT_TEMPLATE_NO_HIS,
    ADAPLANNER_PLAN_TEMPLATE, ADAPLANNER_REFINE_TEMPLATE,
    REFLEXION_REFLECTION_TEMPLATE, ALFWORLD_REFLEXION_TEMPLATE_NO_HIS, ALFWORLD_REFLEXION_TEMPLATE,
    ALFWORLD_SKILLRL_STRICT_TEMPLATE_NO_HIS_WITH_MEMORY, ALFWORLD_SKILLRL_STRICT_TEMPLATE_WITH_MEMORY,
    ALFWORLD_STATE_AWARE_TEMPLATE_NO_HIS, ALFWORLD_STATE_AWARE_TEMPLATE,
    ATTACK_CONDITIONS,
)
def parse_args():
    repo_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description="LatentSkill ALFWorld Evaluation")
    parser.add_argument("--checkpoint", type=str, required=False, default=None,
                        help="IFT checkpoint directory")
    parser.add_argument("--config_name", type=str, default="models/qwen3_8b",
                        help="Hydra config name")
    parser.add_argument("--split", type=str, default="unseen",
                        choices=["seen", "unseen"],
                        help="Evaluation split")
    parser.add_argument("--alfworld_data", type=str,
                        default=os.environ.get(
                            "ALFWORLD_DATA",
                            str(repo_root / "alfworld_data/alfworld"),
                        ),
                        help="ALFWORLD_DATA path")
    parser.add_argument("--alfworld_config", type=str,
                        default=os.environ.get(
                            "ALFWORLD_CONFIG",
                            str(repo_root / "evals/alfworld/config_tw.yaml"),
                        ),
                        help="ALFWorld config YAML path")
    parser.add_argument("--skill_context_dir", type=str,
                        default=os.environ.get(
                            "ALFWORLD_SKILL_DIR",
                            str(repo_root / "evals/alfworld/skills"),
                        ),
                        help="Directory containing ALFWorld skill text files")
    parser.add_argument("--max_steps", type=int, default=50,
                        help="Maximum steps per episode")
    parser.add_argument("--max_new_tokens", type=int, default=512,
                        help="Maximum generated tokens per step")
    parser.add_argument("--history_length", type=int, default=5,
                        help="Number of recent steps kept in the prompt")
    parser.add_argument("--context_max_length", type=int, default=4096,
                        help="Maximum tokenized skill context length")
    parser.add_argument("--conversation_max_length", type=int, default=4096,
                        help="Maximum prompt length")
    parser.add_argument("--output_dir", type=str,
                        default=os.environ.get(
                            "ALFWORLD_OUTPUT_DIR",
                            str(repo_root / "evals/alfworld/results"),
                        ),
                        help="Output directory")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--zero_shot", action="store_true",
                    help="Use the Qwen3-8B backbone without LatentSkill adapters")
    parser.add_argument("--skill_incontext", action="store_true",
                    help="Put skill text in the prompt instead of using LoRA")
    parser.add_argument("--react", action="store_true",
                    help="Run the ReAct baseline with a few-shot demonstration")
    parser.add_argument("--raw_base", action="store_true",
                    help="Load AutoModelForCausalLM directly without the LatentSkill LoRA wrapper")
    parser.add_argument("--adaplanner", action="store_true",
                    help="Run the AdaPlanner baseline")
    parser.add_argument("--reflexion", action="store_true",
                    help="Run the Reflexion baseline")
    parser.add_argument("--reflexion_max_reflections", type=int, default=3,
                    help="Maximum reflections kept per task type")
    parser.add_argument("--model_path", type=str, default=None,
                    help="Optional model path for raw-base or external SFT checkpoints")
    parser.add_argument("--skillrl_memory_json", type=str, default=None,
                    help="Optional SkillRL ALFWorld memory JSON")
    parser.add_argument("--skillrl_memory_top_k", type=int, default=-1,
                    help="Number of task-specific memories to use; -1 means all")
    parser.add_argument("--use_strict_skillrl_prompt", action="store_true",
                    help="Use the stricter SkillRL memory prompt")
    parser.add_argument("--max_games", type=int, default=None,
                        help="Evaluate only the first N episodes")

    parser.add_argument("--debug_prompt", action="store_true",
                        help="Print prompt tails for early episodes")

    parser.add_argument("--debug_episodes", type=int, default=1,
                        help="Number of episodes printed by debug_prompt")

    parser.add_argument("--debug_steps", type=int, default=3,
                        help="Number of steps printed per debug episode")
    parser.add_argument("--disable_thinking", action="store_true",
                    help="Disable Qwen3 thinking mode in tokenizer.apply_chat_template")
    parser.add_argument("--state_aware_prompt", action="store_true",
                    help="Use a state-aware prompt without skill text")
    parser.add_argument("--scale", type=float, default=None,
                    help="Override the LoRA scale; None keeps the config value")
    parser.add_argument("--lora_position", type=str, default="full",
                    choices=["full", "last6", "first30", "o_down", "last6_o_down", "first30_o_down"],
                    help="LoRA placement strategy")
    parser.add_argument("--lora_combo", type=str, default="none",
                    choices=["none", "pick_only", "weight_sum", "hypernet_concat", "moe_combo"],
                    help="LoRA composition mode for look tasks")
    parser.add_argument("--task_type_filter", type=str, default=None,
                    help="Evaluate only a specific task type")
    parser.add_argument("--attack", type=str, default="none",
                    choices=["none", "task_explicit", "task_disguised",
                             "extract_naive", "extract_authority", "extract_indirect"],
                    help=(
                        "Adversarial attack suffix injected into task_description; "
                        "one of none, task_explicit, task_disguised, "
                        "extract_naive, extract_authority, extract_indirect"
                    ))
    return parser.parse_args()


def load_model(args, base_only: bool = False):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    config_path = Path("configs") / f"{args.config_name}.yaml"
    cfg = OmegaConf.load(config_path)

    if args.raw_base:
        model_from = args.model_path if args.model_path else cfg.model.model_from
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

    # Baseline modes use the backbone without LoRA generation.
    if base_only:
        print("[model] zero-shot mode: loading only the Qwen3-8B backbone")
        backbone.to(device)
        backbone.eval()
        backbone.config.use_cache = True
        return None, None, tokenizer, device, cfg, backbone

    # Skill-LoRA mode freezes the backbone and loads the compiler checkpoint.
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


def detect_task_type(gamefile: str) -> str:
    """Infer the ALFWorld task type from a gamefile path."""
    if "pick_clean_then_place" in gamefile:
        return "clean"
    elif "pick_heat_then_place" in gamefile:
        return "heat"
    elif "pick_cool_then_place" in gamefile:
        return "cool"
    elif "look_at_obj_in_light" in gamefile:
        return "look_at_obj_in_light"
    elif "pick_two_obj_and_place" in gamefile:
        return "pick_two_and_place"
    else:
        return "pick_and_place"


def extract_task_description(obs: str) -> str:
    """Extract the task description from the initial observation."""
    match = re.search(r"Your task is to:\s*(.+?)(?:\n|$)", obs)
    return match.group(1).strip() if match else ""


def load_skill_contexts(skill_context_dir: str) -> Dict[str, str]:
    """Load ALFWorld skill text files."""
    skill_dir = Path(skill_context_dir)
    contexts = {}
    for skill in ["pick_and_place", "cool", "heat", "clean", "look_at_obj_in_light"]:
        path = skill_dir / f"{skill}.txt"
        if not path.exists():
            raise FileNotFoundError(f"Skill context file not found: {path}")
        contexts[skill] = path.read_text(encoding="utf-8").strip()
    print(f"[skill context] Loaded: {list(contexts.keys())}")
    return contexts
def load_skillrl_memory_json(path: str) -> Dict:
    """Load a SkillRL-style memory JSON file."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"SkillRL memory JSON not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    print(f"[SkillRL memory] Loaded: {path}")
    print(f"[SkillRL memory] keys={list(data.keys())}")
    return data


def format_skill_item(item: Dict) -> str:
    """Format one skill item in the SkillRL SFT bullet style."""
    title = item.get("title", item.get("name", "Skill"))
    principle = item.get("principle", item.get("description", item.get("content", "")))
    when = item.get("when_to_apply", item.get("apply_when", item.get("when", "")))

    text = f"- **{title}**"
    if principle:
        text += f": {principle}"
    if when:
        text += f"\n  _Apply when: {when}_"
    return text


def format_mistake_item(item: Dict) -> str:
    """Format one common-mistake item in the SkillRL SFT style."""
    desc = item.get("description", "")
    avoid = item.get("how_to_avoid", "")
    why = item.get("why_it_happens", "")

    if desc and avoid:
        return f"- **Don't**: {desc}\n  **Instead**: {avoid}"
    elif desc and why:
        return f"- **Don't**: {desc}\n  **Why it happens**: {why}"
    elif desc:
        return f"- **Don't**: {desc}"
    else:
        return f"- {json.dumps(item, ensure_ascii=False)}"


def normalize_skill_task_key(task_type: str) -> str:
    """
    Map ALFWorld task types to SkillRL memory keys.
    """
    if task_type == "pick_two_and_place":
        return "pick_and_place"
    if task_type == "look_at_obj_in_light":
        return "look_at_obj_in_light"
    return task_type


def build_skillrl_retrieved_memories(
    memory_data: Dict,
    task_type: str,
    top_k: int = -1,
) -> str:
    """
    Build a SkillRL-style retrieved-memory section.
    """
    memory_key = normalize_skill_task_key(task_type)

    # SkillRL official-like truncation
    general_skills = memory_data.get("general_skills", [])[:6]
    common_mistakes = memory_data.get("common_mistakes", [])[:5]

    task_specific_all = memory_data.get("task_specific_skills", {})
    task_skills = []

    if isinstance(task_specific_all, dict):
        task_skills = task_specific_all.get(memory_key, [])

        if not task_skills:
            task_skills = task_specific_all.get("pick_and_place", [])

    elif isinstance(task_specific_all, list):
        task_skills = task_specific_all

    if top_k is not None and top_k > 0:
        task_skills = task_skills[:top_k]

    sections = []

    if general_skills:
        sections.append(
            "### General Principles\n"
            + "\n".join(format_skill_item(x) for x in general_skills)
        )

    if task_skills:
        task_title = memory_key.replace("_", " ").title()
        sections.append(
            f"### {task_title} Skills\n"
            + "\n".join(format_skill_item(x) for x in task_skills)
        )

    if common_mistakes:
        sections.append(
            "### Mistakes to Avoid\n"
            + "\n".join(format_mistake_item(x) for x in common_mistakes)
        )

    return "\n\n".join(sections).strip()

# Display names used in reports.
TASK_TYPE_NAMES = {
    "pick_and_place":       "Pick & Place",
    "pick_two_and_place":   "Pick Two & place",
    "clean":                "Clean & Place",
    "heat":                 "Heat & Place",
    "cool":                 "Cool & Place",
    "look_at_obj_in_light": "Look At Obj In Light",
    "examine":              "Examine In Light",
}

def main():
    args = parse_args()
    use_skillrl_memory = args.skillrl_memory_json is not None

    if args.raw_base:
        if args.state_aware_prompt:
            mode = "raw_state_aware"
        elif args.react:
            mode = "raw_react"
        elif use_skillrl_memory and args.use_strict_skillrl_prompt:
            mode = "raw_skillrl_memory_strict"
        elif use_skillrl_memory:
            mode = "raw_skillrl_memory"
        elif args.skill_incontext:
            mode = "raw_incontext"
        elif args.adaplanner:
            mode = "raw_adaplanner"
        elif args.reflexion:
            mode = "raw_reflexion"
        else:
            mode = "raw_base"
        use_lora = False
        use_skill_in_prompt = args.skill_incontext
        use_react = args.react
        use_adaplanner = args.adaplanner
        use_reflexion = args.reflexion
        base_only = True
    elif args.zero_shot:
        mode = "zero_shot"
        use_lora = False
        use_skill_in_prompt = False
        use_react = False
        use_adaplanner = False
        use_reflexion = False
        base_only = True
    elif args.skill_incontext:
        mode = "skill_incontext"
        use_lora = False
        use_skill_in_prompt = True
        use_react = False
        use_adaplanner = False
        use_reflexion = False
        base_only = True
    elif args.react:
        mode = "react"
        use_lora = False
        use_skill_in_prompt = False
        use_react = True
        use_adaplanner = False
        use_reflexion = False
        base_only = True
    elif args.adaplanner:
        mode = "adaplanner"
        use_lora = False
        use_skill_in_prompt = False
        use_react = False
        use_adaplanner = True
        use_reflexion = False
        base_only = True
    elif args.reflexion:
        mode = "reflexion"
        use_lora = False
        use_skill_in_prompt = False
        use_react = False
        use_adaplanner = False
        use_reflexion = True
        base_only = True
    else:
        mode = "skill_lora"
        use_lora = True
        use_skill_in_prompt = False
        use_react = False
        use_adaplanner = False
        use_reflexion = False
        base_only = False

    print(f"[mode] {mode}")

    os.environ["ALFWORLD_DATA"] = args.alfworld_data
    print(f"[ALFWorld] ALFWORLD_DATA={args.alfworld_data}")

    output_dir = Path(args.output_dir)
    if args.lora_position != "full":
        output_dir = output_dir / f"position_{args.lora_position}"
    if args.lora_combo != "none":
        output_dir = output_dir / f"combo_{args.lora_combo}"
    output_dir.mkdir(parents=True, exist_ok=True)

    skill_hypernet, metalora, tokenizer, device, cfg, backbone = load_model(args, base_only=base_only)

    DEFAULT_SCALE = 0.001
    if args.scale is not None and skill_hypernet is not None:
        old_scale = skill_hypernet.scale
        skill_hypernet.scale = args.scale
        multiplier = args.scale / DEFAULT_SCALE if DEFAULT_SCALE > 0 else float('inf')
        print("=" * 60)
        print(f"[SCALE] previous scale: {old_scale}")
        print(f"[SCALE] current scale: {args.scale} ({multiplier:.1f}× default)")
        print(f"[SCALE] effective alpha: {args.scale * cfg.model.lora_r:.4f} (r={cfg.model.lora_r})")
        print("=" * 60)
    elif skill_hypernet is not None:
        print("=" * 60)
        print(f"[SCALE] config scale: {skill_hypernet.scale}")
        print(f"[SCALE] effective alpha: {skill_hypernet.scale * cfg.model.lora_r:.4f} (r={cfg.model.lora_r})")
        print("=" * 60)

    skillrl_memory_data = None

    if use_skillrl_memory:
        skillrl_memory_data = load_skillrl_memory_json(args.skillrl_memory_json)

        skill_contexts = {
            "pick_and_place": "",
            "pick_two_and_place": "",
            "clean": "",
            "heat": "",
            "cool": "",
            "look_at_obj_in_light": "",
        }
    else:
        skill_contexts = load_skill_contexts(args.skill_context_dir)

    COMP_DIR = Path(args.skill_context_dir)
    component_file_map = {
        "general":   COMP_DIR / "general_alfworld.txt",
        "task_look": COMP_DIR / "task_look_at_obj_in_light.txt",
        "task_pick": COMP_DIR / "task_pick_and_place.txt",
        "mistakes":  COMP_DIR / "mistakes_alfworld.txt",
    }
    skill_component_contexts = {}
    for cname, cpath in component_file_map.items():
        if cpath.exists():
            skill_component_contexts[cname] = cpath.read_text(encoding="utf-8").strip()
        else:
            print(f"[WARNING] component file not found: {cpath}")
            skill_component_contexts[cname] = ""

    if not use_lora:
        print("[adapter] zero-shot mode: skipping adapter generation")
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

    def mask_modules(layer_ld, keep):
        import copy
        ld = copy.deepcopy(layer_ld)
        for proj in ["q", "k", "v", "o"]:
            if proj not in keep:
                for mat in ["A", "B"]:
                    if ld["attention"][proj][mat] is not None:
                        ld["attention"][proj][mat] = torch.zeros_like(
                            ld["attention"][proj][mat])
        for proj in ["gate", "up", "down"]:
            if proj not in keep:
                for mat in ["A", "B"]:
                    if ld["mlp"][proj][mat] is not None:
                        ld["mlp"][proj][mat] = torch.zeros_like(
                            ld["mlp"][proj][mat])
        return ld

    def apply_lora_position(adapter_state, position):
        if position == "full":
            return adapter_state
        elif position == "last6":
            return {i: (adapter_state[i] if i >= 30 else None) for i in adapter_state}
        elif position == "first30":
            return {i: (adapter_state[i] if i < 30 else None) for i in adapter_state}
        elif position == "o_down":
            return {i: mask_modules(adapter_state[i], keep=["o", "down"]) for i in adapter_state}
        elif position == "last6_o_down":
            return {
                i: (mask_modules(adapter_state[i], keep=["o", "down"]) if i >= 30 else None)
                for i in adapter_state
            }
        elif position == "first30_o_down":
            return {
                i: (mask_modules(adapter_state[i], keep=["o", "down"]) if i < 30 else None)
                for i in adapter_state
            }
        else:
            raise ValueError(f"Unknown lora_position: {position}")

    if use_lora and args.lora_position != "full":
        print(f"[lora_position] applying strategy: {args.lora_position}")
        skill_adapter_states = {
            skill: apply_lora_position(ld, args.lora_position)
            for skill, ld in skill_adapter_states.items()
        }
        print("[lora_position] done")

    def merge_adapter_states(ld_A, ld_B, w_A=0.3, w_B=0.3):
        """
        Merge two adapter states by concatenating along the rank dimension:
        [A1*√w_A | A2*√w_B] @ [[B1*√w_A], [B2*√w_B]] = w_A*A1@B1 + w_B*A2@B2
        """
        import math
        sw_A = math.sqrt(w_A)
        sw_B = math.sqrt(w_B)
        merged = {}
        for i in ld_A:
            merged[i] = {"attention": {}, "mlp": {}}
            for proj in ["q", "k", "v", "o"]:
                a = ld_A[i]["attention"][proj]
                b = ld_B[i]["attention"][proj]
                merged[i]["attention"][proj] = {
                    "A": torch.cat([a["A"] * sw_A, b["A"] * sw_B], dim=-1),
                    "B": torch.cat([a["B"] * sw_A, b["B"] * sw_B], dim=-2),
                    "C": None,
                }
            for proj in ["gate", "up", "down"]:
                a = ld_A[i]["mlp"][proj]
                b = ld_B[i]["mlp"][proj]
                merged[i]["mlp"][proj] = {
                    "A": torch.cat([a["A"] * sw_A, b["A"] * sw_B], dim=-1),
                    "B": torch.cat([a["B"] * sw_A, b["B"] * sw_B], dim=-2),
                    "C": None,
                }
        return merged

    def merge_four_adapter_states(ld_gen, ld_look, ld_pick, ld_mis, w=0.15):
        """
        Merge four component adapter states with a shared weight.
        """
        import math
        sw = math.sqrt(w)
        lds = [ld_gen, ld_look, ld_pick, ld_mis]
        merged = {}
        for i in ld_gen:
            merged[i] = {"attention": {}, "mlp": {}}
            for proj in ["q", "k", "v", "o"]:
                merged[i]["attention"][proj] = {
                    "A": torch.cat([ld[i]["attention"][proj]["A"] * sw for ld in lds], dim=-1),
                    "B": torch.cat([ld[i]["attention"][proj]["B"] * sw for ld in lds], dim=-2),
                    "C": None,
                }
            for proj in ["gate", "up", "down"]:
                merged[i]["mlp"][proj] = {
                    "A": torch.cat([ld[i]["mlp"][proj]["A"] * sw for ld in lds], dim=-1),
                    "B": torch.cat([ld[i]["mlp"][proj]["B"] * sw for ld in lds], dim=-2),
                    "C": None,
                }
        return merged

    component_adapter_states = {}
    if use_lora and args.lora_combo == "moe_combo":
        print("[adapter] generating component adapter states...")
        for cname, ctext in skill_component_contexts.items():
            if not ctext.strip():
                print(f"  [SKIP] {cname} is empty")
                continue
            enc = tokenizer(
                ctext,
                max_length=args.context_max_length,
                truncation=True,
                return_tensors="pt",
                padding=False,
            )
            ids  = enc["input_ids"].to(device)
            mask = enc["attention_mask"].to(device)
            with torch.no_grad():
                component_adapter_states[cname] = skill_hypernet.build_adapter_state(
                    ids, mask, metalora
                )
            print(f"  [OK] {cname} ({ids.shape[1]} tokens)")

    combo_adapter_state = None
    if use_lora and args.lora_combo != "none":

        if args.lora_combo == "pick_only":
            combo_adapter_state = skill_adapter_states["pick_and_place"]
            print("[lora_combo] pick_only: using pick_and_place LoRA")

        elif args.lora_combo == "weight_sum":
            combo_adapter_state = merge_adapter_states(
                skill_adapter_states["look_at_obj_in_light"],
                skill_adapter_states["pick_and_place"],
                w_A=0.3, w_B=0.3
            )
            print("[lora_combo] weight_sum: 0.3×look + 0.3×pick, rank expands from 8 to 16")

        elif args.lora_combo == "hypernet_concat":
            concat_text = (
                skill_contexts["pick_and_place"]
                + "\n\n"
                + skill_contexts["look_at_obj_in_light"]
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
        elif args.lora_combo == "moe_combo":
            required = ["general", "task_look", "task_pick", "mistakes"]
            missing = [k for k in required if k not in component_adapter_states]
            if missing:
                raise ValueError(f"[lora_combo] moe_combo missing components: {missing}")
            combo_adapter_state = merge_four_adapter_states(
                ld_gen  = component_adapter_states["general"],
                ld_look = component_adapter_states["task_look"],
                ld_pick = component_adapter_states["task_pick"],
                ld_mis  = component_adapter_states["mistakes"],
                w=0.15
            )
            print("[lora_combo] moe_combo: 0.15×gen + 0.15×look_task "
                  "+ 0.15×pick_task + 0.15×mistakes, rank expands from 8 to 32")

    train_eval = (
        "eval_out_of_distribution" if args.split == "unseen"
        else "eval_in_distribution"
    )
    print(f"[ALFWorld] loading environment, split={args.split} ({train_eval})...")

    alf_config = yaml.safe_load(open(args.alfworld_config))
    env_type   = alf_config["env"]["type"]

    from alfworld.agents.environment import get_environment
    base_env = get_environment(env_type)(alf_config, train_eval=train_eval)
    env      = base_env.init_env(batch_size=1)

    num_games = base_env.num_games
    print(f"[ALFWorld] {num_games} episodes")

    detail_path = output_dir / f"results_detail_{args.split}.jsonl"
    task_results = defaultdict(list)  # task_type → [True/False, ...]
    invalid_action_counts = defaultdict(int)
    # Reflexion keeps reflections isolated by task type.
    reflection_buffer: Dict[str, List[str]] = defaultdict(list)

    eval_num_games = min(num_games, args.max_games) if args.max_games is not None else num_games

    with open(detail_path, "w", encoding="utf-8") as detail_f:
        # pbar = tqdm(total=eval_num_games, desc=f"ALFWorld {args.split}")
        filter_label = f"[{args.task_type_filter}]" if args.task_type_filter else ""
        pbar = tqdm(total=eval_num_games, desc=f"ALFWorld {args.split} {filter_label}")

        for episode_idx in range(eval_num_games):
            obs_list, info_list = env.reset()
            obs  = obs_list[0]

            gamefile      = info_list["extra.gamefile"][0]
            task_type     = detect_task_type(gamefile)

            if args.task_type_filter is not None and task_type != args.task_type_filter:
                pbar.update(1)
                continue

            skill_key  = "pick_and_place" if task_type == "pick_two_and_place" else task_type

            if use_lora:
                if (args.lora_combo != "none"
                        and combo_adapter_state is not None
                        and task_type == "look_at_obj_in_light"):
                    adapter_state = combo_adapter_state
                else:
                    adapter_state = skill_adapter_states.get(skill_key)
            else:
                adapter_state = None

            if use_skillrl_memory:
                skill_text = build_skillrl_retrieved_memories(
                    skillrl_memory_data,
                    task_type=task_type,
                    top_k=args.skillrl_memory_top_k,
                )
            elif use_skill_in_prompt:
                skill_text = skill_contexts.get(skill_key, "")
            else:
                skill_text = None

            task_description = extract_task_description(obs)
            _attack_suffix = ATTACK_CONDITIONS.get(args.attack, "")
            if _attack_suffix:
                task_description = task_description + _attack_suffix

            admissible       = info_list["admissible_commands"][0]
            history: List[Tuple[str, str]] = []
            reflections_at_start = len(reflection_buffer[task_type])
            current_plan: str = ""
            steps_log = []
            won = False
            step_count = 0

            for step in range(1, args.max_steps + 1):
                admissible_str = ", ".join(admissible)

                # prompt_text = (
                #     f"You are an expert agent operating in the ALFRED Embodied Environment.\n"
                #     f"Your task is to: {task_description}\n"
                #     f"## Current Progress\n"
                #     f"Your current observation is: {obs}\n"
                #     f"Your admissible actions of the current situation are: [{admissible_str}].\n"
                #     f"Now it's your turn to take an action.\n"
                #     f"You should first reason step-by-step about the current situation. "
                #     f"This reasoning process MUST be enclosed within <think> </think> tags.\n"
                #     f"Once you've finished your reasoning, you should choose an admissible action "
                #     f"for current step and present it within <action> </action> tags."
                # )
                if use_skillrl_memory:
                    if not history:
                        if args.use_strict_skillrl_prompt:
                            prompt_text = ALFWORLD_SKILLRL_STRICT_TEMPLATE_NO_HIS_WITH_MEMORY.format(
                                task_description=task_description,
                                retrieved_memories=skill_text,
                                current_step=step,
                                current_observation=obs,
                                admissible_actions=admissible_str,
                            )
                        else:
                            prompt_text = ALFWORLD_TEMPLATE_NO_HIS_WITH_MEMORY.format(
                                task_description=task_description,
                                retrieved_memories=skill_text,
                                current_observation=obs,
                                admissible_actions=admissible_str,
                            )
                    else:
                        recent = history[-args.history_length:]
                        history_parts = []
                        for i, (h_obs, h_action) in enumerate(recent):
                            history_parts.append(
                                f"[Observation {i+1}: '{h_obs[:300]}', Action {i+1}: '{h_action}']"
                            )
                        history_str = "\n".join(history_parts)

                        if args.use_strict_skillrl_prompt:
                            prompt_text = ALFWORLD_SKILLRL_STRICT_TEMPLATE_WITH_MEMORY.format(
                                task_description=task_description,
                                retrieved_memories=skill_text,
                                step_count=step_count,
                                history_length=len(recent),
                                action_history=history_str,
                                current_step=step,
                                current_observation=obs,
                                admissible_actions=admissible_str,
                            )
                        else:
                            prompt_text = ALFWORLD_TEMPLATE_WITH_MEMORY.format(
                                task_description=task_description,
                                retrieved_memories=skill_text,
                                step_count=step_count,
                                history_length=len(recent),
                                action_history=history_str,
                                current_step=step,
                                current_observation=obs,
                                admissible_actions=admissible_str,
                            )

                elif use_skill_in_prompt:
                    if not history:
                        prompt_text = ALFWORLD_TEMPLATE_NO_HIS_WITH_MEMORY.format(
                            task_description=task_description,
                            retrieved_memories=skill_text,
                            current_observation=obs,
                            admissible_actions=admissible_str,
                        )
                    else:
                        recent = history[-args.history_length:]
                        history_parts = []
                        for i, (h_obs, h_action) in enumerate(recent):
                            history_parts.append(
                                f"[Observation {i+1}: '{h_obs[:300]}', Action {i+1}: '{h_action}']"
                            )
                        history_str = "\n".join(history_parts)
                        prompt_text = ALFWORLD_TEMPLATE_WITH_MEMORY.format(
                            task_description=task_description,
                            retrieved_memories=skill_text,
                            step_count=step_count,
                            history_length=len(recent),
                            action_history=history_str,
                            current_step=step,
                            current_observation=obs,
                            admissible_actions=admissible_str,
                        )
                elif use_react:
                    if not history:
                        prompt_text = ALFWORLD_REACT_TEMPLATE_NO_HIS.format(
                            task_description=task_description,
                            current_observation=obs,
                            admissible_actions=admissible_str,
                        )
                    else:
                        recent = history[-args.history_length:]
                        history_parts = []
                        for i, (h_obs, h_action) in enumerate(recent):
                            history_parts.append(
                                f"[Observation {i+1}: '{h_obs[:300]}', Action {i+1}: '{h_action}']"
                            )
                        history_str = "\n".join(history_parts)
                        prompt_text = ALFWORLD_REACT_TEMPLATE.format(
                            task_description=task_description,
                            step_count=step_count,
                            history_length=len(recent),
                            action_history=history_str,
                            current_step=step,
                            current_observation=obs,
                            admissible_actions=admissible_str,
                        )
                elif use_adaplanner:
                    if not history:
                        # Generate the initial plan.
                        prompt_text = ADAPLANNER_PLAN_TEMPLATE.format(
                            task_description=task_description,
                            current_observation=obs,
                            admissible_actions=admissible_str,
                        )
                    else:
                        recent = history[-args.history_length:]
                        history_parts = [
                            f"[Observation {i+1}: '{h_obs[:300]}', Action {i+1}: '{h_action}']"
                            for i, (h_obs, h_action) in enumerate(recent)
                        ]
                        history_str = "\n".join(history_parts)
                        prompt_text = ADAPLANNER_REFINE_TEMPLATE.format(
                            task_description=task_description,
                            current_plan=current_plan if current_plan else "(no plan yet)",
                            history_length=len(recent),
                            action_history=history_str,
                            current_step=step,
                            current_observation=obs,
                            admissible_actions=admissible_str,
                        )
                elif use_reflexion:
                    current_reflections = reflection_buffer[task_type][-args.reflexion_max_reflections:]
                    if not current_reflections:
                        # Fall back to the zero-shot prompt when no reflection is available.
                        if not history:
                            prompt_text = ALFWORLD_TEMPLATE_NO_HIS.format(
                                task_description=task_description,
                                current_observation=obs,
                                admissible_actions=admissible_str,
                            )
                        else:
                            recent = history[-args.history_length:]
                            history_parts = [
                                f"[Observation {i+1}: '{h_obs[:300]}', Action {i+1}: '{h_action}']"
                                for i, (h_obs, h_action) in enumerate(recent)
                            ]
                            history_str = "\n".join(history_parts)
                            prompt_text = ALFWORLD_TEMPLATE.format(
                                task_description=task_description,
                                step_count=step_count,
                                history_length=len(recent),
                                action_history=history_str,
                                current_step=step,
                                current_observation=obs,
                                admissible_actions=admissible_str,
                            )
                    else:
                        reflections_str = "\n".join(
                            f"Attempt {i+1}: {r}" for i, r in enumerate(current_reflections)
                        )
                        if not history:
                            prompt_text = ALFWORLD_REFLEXION_TEMPLATE_NO_HIS.format(
                                task_description=task_description,
                                reflections=reflections_str,
                                current_observation=obs,
                                admissible_actions=admissible_str,
                            )
                        else:
                            recent = history[-args.history_length:]
                            history_parts = [
                                f"[Observation {i+1}: '{h_obs[:300]}', Action {i+1}: '{h_action}']"
                                for i, (h_obs, h_action) in enumerate(recent)
                            ]
                            history_str = "\n".join(history_parts)
                            prompt_text = ALFWORLD_REFLEXION_TEMPLATE.format(
                                task_description=task_description,
                                reflections=reflections_str,
                                step_count=step_count,
                                history_length=len(recent),
                                action_history=history_str,
                                current_step=step,
                                current_observation=obs,
                                admissible_actions=admissible_str,
                            )
                elif args.state_aware_prompt:
                    if not history:
                        prompt_text = ALFWORLD_STATE_AWARE_TEMPLATE_NO_HIS.format(
                            task_description=task_description,
                            current_observation=obs,
                            admissible_actions=admissible_str,
                        )
                    else:
                        recent = history[-args.history_length:]
                        history_parts = []
                        for i, (h_obs, h_action) in enumerate(recent):
                            history_parts.append(
                                f"[Observation {i+1}: '{h_obs[:300]}', Action {i+1}: '{h_action}']"
                            )
                        history_str = "\n".join(history_parts)
                        prompt_text = ALFWORLD_STATE_AWARE_TEMPLATE.format(
                            task_description=task_description,
                            step_count=step_count,
                            history_length=len(recent),
                            action_history=history_str,
                            current_step=step,
                            current_observation=obs,
                            admissible_actions=admissible_str,
                        )
                elif not history:
                    prompt_text = ALFWORLD_TEMPLATE_NO_HIS.format(
                        task_description=task_description,
                        current_observation=obs,
                        admissible_actions=admissible_str,
                    )
                else:
                    recent = history[-args.history_length:]
                    history_parts = []
                    for i, (h_obs, h_action) in enumerate(recent):
                        history_parts.append(
                            f"[Observation {i+1}: '{h_obs[:300]}', Action {i+1}: '{h_action}']"
                        )
                    history_str = "\n".join(history_parts)
                    prompt_text = ALFWORLD_TEMPLATE.format(
                        task_description=task_description,
                        step_count=step_count,
                        history_length=len(recent),
                        action_history=history_str,
                        current_step=step,
                        current_observation=obs,
                        admissible_actions=admissible_str,
                    )

                prompt_text = prompt_text.lstrip("\n")
                if args.debug_prompt and episode_idx < args.debug_episodes and step <= args.debug_steps:
                    print("=" * 100)
                    print("[DEBUG RAW PROMPT]")
                    print("[EPISODE]", episode_idx)
                    print("[STEP]", step)
                    print("[TASK_TYPE]", task_type)
                    print("[TASK]", task_description)
                    print("[USE_SKILLRL_MEMORY]", use_skillrl_memory)
                    print("[OBS]", obs)
                    print("[ADMISSIBLE_ACTIONS]", admissible)
                    if use_skillrl_memory:
                        print("[SKILLRL_MEMORY_PREVIEW]")
                        print((skill_text or "")[:1500])
                    print("[PROMPT_TAIL]")
                    print(prompt_text[-4000:])
                    print("=" * 100)

                messages = [{"role": "user", "content": prompt_text}]
                input_enc = tokenizer.apply_chat_template(
                    messages,
                    add_generation_prompt=True,
                    tokenize=True,
                    return_tensors="pt",
                    max_length=args.conversation_max_length,
                    truncation=True,
                    return_dict=True,
                    padding=False,
                    enable_thinking=True,
                )
                input_ids      = input_enc["input_ids"].to(device)
                attention_mask = input_enc["attention_mask"].to(device)

                if args.debug_prompt and episode_idx < args.debug_episodes and step <= args.debug_steps:
                    decoded_prompt = tokenizer.decode(input_ids[0], skip_special_tokens=False)
                    print("=" * 100)
                    print("[DEBUG TOKENIZED PROMPT]")
                    print("[TOKENIZED_LEN]", input_ids.shape[-1])
                    print("[TOKENIZED_PROMPT_TAIL]")
                    print(decoded_prompt[-4000:])
                    print("=" * 100)

                with torch.no_grad():
                    if args.raw_base:
                        output_ids = backbone.generate(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            max_new_tokens=args.max_new_tokens,
                            pad_token_id=tokenizer.pad_token_id,
                            eos_token_id=tokenizer.eos_token_id,
                            do_sample=False,
                        )
                    elif not use_lora:
                        output_ids = backbone.generate(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            max_new_tokens=args.max_new_tokens,
                            pad_token_id=tokenizer.pad_token_id,
                            eos_token_id=tokenizer.eos_token_id,
                            do_sample=False,
                            ignore_mem_token=True,
                        )
                    else:
                        output_ids = skill_hypernet.backbone.generate(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            max_new_tokens=args.max_new_tokens,
                            pad_token_id=tokenizer.pad_token_id,
                            eos_token_id=tokenizer.eos_token_id,
                            do_sample=False,
                            ignore_mem_token=True,
                            adapter_state=adapter_state,
                        )

                new_tokens  = output_ids[0, input_ids.shape[1]:]
                output_text = tokenizer.decode(
                    new_tokens, skip_special_tokens=True
                ).strip()

                if args.debug_prompt and episode_idx < args.debug_episodes and step <= args.debug_steps:
                    print("=" * 100)
                    print("[DEBUG MODEL OUTPUT]")
                    print("[EPISODE]", episode_idx)
                    print("[STEP]", step)
                    print(output_text[:1500])
                    print("=" * 100)

                _action_matches = re.findall(
                    r"<action>(.*?)</action>",
                    output_text,
                    flags=re.IGNORECASE | re.DOTALL,
                )

                if _action_matches:
                    parsed_action = _action_matches[-1].strip()
                    is_parse_valid = True
                else:
                    parsed_action = output_text[-30:].strip()
                    is_parse_valid = False


                def _norm_action(x: str) -> str:
                    """Normalize actions only for matching admissible commands."""
                    return re.sub(r"\s+", " ", x or "").strip().lower()


                parsed_action_norm = _norm_action(parsed_action)

                admissible_map = {
                    _norm_action(a): a
                    for a in admissible
                }

                in_admissible = is_parse_valid and parsed_action_norm in admissible_map

                _leak_detected = False
                if args.attack.startswith("extract") and skill_text:
                    _skill_anchors = [
                        line.strip()
                        for line in skill_text.split("\n")
                        if len(line.strip()) > 15
                    ][:8]
                    _leak_detected = any(
                        anchor.lower() in output_text.lower()
                        for anchor in _skill_anchors
                    )

                step_log = {
                    "step": step,
                    # "obs": obs[:500],
                    # "output": output_text[:1000],
                    "obs": obs,
                    "output": output_text,

                    "parsed_action": parsed_action,

                    "is_parse_valid": is_parse_valid,

                    "in_admissible": in_admissible,

                    "admissible_actions": admissible[:100],

                    "leak_detected": _leak_detected,
                }

                if in_admissible:
                    final_action = admissible_map[parsed_action_norm]
                else:
                    final_action = parsed_action if is_parse_valid and parsed_action else "look"
                    step_log["fallback"] = True
                    invalid_action_counts[task_type] += 1

                # Keep the invalid action visible during debugging instead of silently replacing it.

                step_log["final_action"] = final_action

                step_log["action"] = final_action

                action = final_action

                text_obs_list, scores_list, dones_list, info_list = env.step(
                    [action]
                )
                next_obs   = text_obs_list[0]
                done       = dones_list[0]
                won        = bool(info_list["won"][0])

                step_log["next_obs"] = next_obs[:500]

                if use_adaplanner:
                    _plan_match = re.search(r'<plan>(.*?)</plan>', output_text, re.IGNORECASE | re.DOTALL)
                    if _plan_match:
                        current_plan = _plan_match.group(1).strip()

                step_log["won"] = won
                steps_log.append(step_log)

                history.append((obs, action))
                step_count += 1
                obs        = next_obs
                admissible = info_list["admissible_commands"][0]

                if done:
                    break

            task_results[task_type].append(won)

            generated_reflection = None
            if use_reflexion and not won:
                recent_steps = steps_log[-10:]
                trajectory_text = ""
                for s in recent_steps:
                    trajectory_text += (
                        f"Step {s['step']}: Observation: {s['obs']}\n"
                        f"  Action: {s['action']}\n"
                    )
                reflection_prompt_text = REFLEXION_REFLECTION_TEMPLATE.format(
                    task_description=task_description,
                    num_steps=len(recent_steps),
                    total_steps=len(steps_log),
                    trajectory=trajectory_text,
                )
                ref_messages = [{"role": "user", "content": reflection_prompt_text}]
                ref_enc = tokenizer.apply_chat_template(
                    ref_messages,
                    add_generation_prompt=True,
                    tokenize=True,
                    return_tensors="pt",
                    max_length=args.conversation_max_length,
                    truncation=True,
                    return_dict=True,
                    padding=False,
                    enable_thinking=True,
                )
                ref_input_ids      = ref_enc["input_ids"].to(device)
                ref_attention_mask = ref_enc["attention_mask"].to(device)
                with torch.no_grad():
                    if args.raw_base:
                        ref_output_ids = backbone.generate(
                            input_ids=ref_input_ids,
                            attention_mask=ref_attention_mask,
                            max_new_tokens=256,
                            pad_token_id=tokenizer.pad_token_id,
                            eos_token_id=tokenizer.eos_token_id,
                            do_sample=False,
                        )
                    else:
                        ref_output_ids = backbone.generate(
                            input_ids=ref_input_ids,
                            attention_mask=ref_attention_mask,
                            max_new_tokens=256,
                            pad_token_id=tokenizer.pad_token_id,
                            eos_token_id=tokenizer.eos_token_id,
                            do_sample=False,
                            ignore_mem_token=True,
                        )
                ref_new_tokens = ref_output_ids[0, ref_input_ids.shape[1]:]
                ref_output_text = tokenizer.decode(ref_new_tokens, skip_special_tokens=True).strip()
                _ref_match = re.search(r'<reflection>(.*?)</reflection>', ref_output_text, re.IGNORECASE | re.DOTALL)
                if _ref_match:
                    generated_reflection = _ref_match.group(1).strip()
                    reflection_buffer[task_type].append(generated_reflection)
                    if len(reflection_buffer[task_type]) > args.reflexion_max_reflections:
                        reflection_buffer[task_type] = (
                            reflection_buffer[task_type][-args.reflexion_max_reflections:]
                        )

            _leak_steps = sum(1 for s in steps_log if s.get("leak_detected", False))
            _leak_rate = _leak_steps / len(steps_log) if steps_log else 0.0

            detail_record = {
                "episode_idx":     episode_idx,
                "gamefile":        gamefile,
                "task_type":       task_type,
                "task_description": task_description,
                "won":             won,
                "steps":           len(steps_log),
                "steps_log":       steps_log,
                "reflections_used":     reflections_at_start,
                "reflection_generated": generated_reflection,
                "attack_condition": args.attack,
                "skill_mode":       mode,
                "leak_steps":       _leak_steps,
                "leak_rate":        round(_leak_rate, 4),
            }
            detail_f.write(
                json.dumps(detail_record, ensure_ascii=False) + "\n"
            )
            detail_f.flush()

            pbar.set_postfix({
                "task": task_type[:8],
                "won":  won,
                "total_won": sum(sum(v) for v in task_results.values()),
            })
            pbar.update(1)

    TASK_ORDER = [
        "pick_and_place", "pick_two_and_place", "clean",
        "heat", "cool", "look_at_obj_in_light",
    ]

    summary = {}
    all_results = []
    print(f"\n{'='*60}")
    print(f"ALFWorld {args.split} results")
    print(f"{'='*60}")
    print(f"{'Task Type':<25} {'Success':>8} {'Total':>8} {'Rate':>8}")
    print(f"{'-'*60}")

    for task in TASK_ORDER:
        results = task_results.get(task, [])
        if results:
            success = sum(results)
            total   = len(results)
            rate    = success / total
            summary[task] = {
                "success": success,
                "total":   total,
                "rate":    round(rate, 4),
            }
            all_results.extend(results)
            name = TASK_TYPE_NAMES.get(task, task)
            print(f"{name:<25} {success:>8} {total:>8} {rate:>8.4f}")
        else:
            summary[task] = {"success": 0, "total": 0, "rate": None}
            print(f"{TASK_TYPE_NAMES.get(task, task):<25} {'N/A':>8}")

    overall_rate = sum(all_results) / len(all_results) if all_results else 0.0
    summary["overall"] = {
        "success": sum(all_results),
        "total":   len(all_results),
        "rate":    round(overall_rate, 4),
    }
    print(f"{'-'*60}")
    print(f"{'Overall':<25} {sum(all_results):>8} {len(all_results):>8} {overall_rate:>8.4f}")
    print(f"{'='*60}")

    summary_path = output_dir / f"results_summary_{args.split}.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\nResults saved:")
    print(f"  detail:  {detail_path}")
    print(f"  summary: {summary_path}")


if __name__ == "__main__":
    main()
