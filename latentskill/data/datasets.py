from dataclasses import dataclass
import json
import os
import random
from collections import defaultdict
from typing import Any, Dict, List

import numpy as np
import torch
from datasets import Column
from datasets import Dataset as HFDataset
from torch.utils.data import Dataset

from latentskill.utils.ddp import is_primary_process
from latentskill.utils.logging import get_logger


logger = get_logger("data")


class StaticSkillGroupDataset(Dataset):
    """
    Dataset for static skill pretraining groups.

    It estimates token lengths, packs multiple skill texts into groups close to
    the conversation length budget, and returns {"textlist": [[skill], ...]}.
    """

    def __init__(
        self,
        texts,
        tokenizer,
        conversation_max_len: int,
        cache_dir: str,
        cache_name: str,
        map_num_proc: int = 16,
        map_batch_size: int = 2048,
        num_cache: int = 100,
        preprocess_mode: bool = False,
        overwrite: bool = False,
    ):
        self.texts = texts
        self.tokenizer = tokenizer
        self.conversation_max_len = conversation_max_len
        self.cache_dir = cache_dir
        self.cache_name = cache_name
        self.map_num_proc = map_num_proc
        self.map_batch_size = map_batch_size
        self.num_cache = num_cache
        self.base_len = 0
        self.chat_len = 11

        self.cache_path = os.path.join(
            cache_dir,
            f"{cache_name}_group_idx_{conversation_max_len}.json",
        )

        if preprocess_mode:
            self.build_static_groups(overwrite=overwrite)

        if not os.path.exists(self.cache_path):
            raise FileNotFoundError(
                f"Cache not found: {self.cache_path}\n"
                "Create it first with `python -m latentskill.data.group_index`."
            )

        with open(self.cache_path, "r", encoding="utf-8") as f:
            self.group_idx = json.load(f)

        if is_primary_process():
            logger.info(
                f"[StaticSkillGroupDataset] Loaded {len(self.group_idx)} groups "
                f"for {len(self.texts)} texts. max_len={conversation_max_len}"
            )

    def build_static_groups(self, overwrite: bool = False) -> List[List[int]]:
        """Build group indices from self.texts and save them to cache."""
        os.makedirs(self.cache_dir, exist_ok=True)

        if os.path.exists(self.cache_path) and not overwrite:
            with open(self.cache_path, "r", encoding="utf-8") as f:
                self.group_idx = json.load(f)
            if is_primary_process():
                logger.info(f"[StaticSkillGroupDataset] Cache exists, loaded: {self.cache_path}")
            return self.group_idx

        if is_primary_process():
            logger.info("[StaticSkillGroupDataset] Creating group index...")
            logger.info("[StaticSkillGroupDataset] Computing token lengths with Hugging Face Dataset.map...")
        token_lens = self._compute_token_lengths_with_hf_dataset()

        self.group_idx = _pack_text_indices_by_length(
            token_lens=token_lens,
            conversation_max_len=self.conversation_max_len,
            base_len=self.base_len,
            chat_len=self.chat_len,
            num_cache=self.num_cache,
        )

        with open(self.cache_path, "w", encoding="utf-8") as f:
            json.dump(self.group_idx, f)

        if is_primary_process():
            logger.info(f"[StaticSkillGroupDataset] Saved group index to {self.cache_path}")
            logger.info(
                f"[StaticSkillGroupDataset] Created {len(self.group_idx)} groups "
                f"from {len(self.texts)} texts for max_len={self.conversation_max_len}."
            )
        return self.group_idx

    def _compute_token_lengths_with_hf_dataset(self) -> np.ndarray:
        hf_dataset = HFDataset.from_dict({"text": [str(t) for t in self.texts]})

        def tokenized_length(batch):
            enc = self.tokenizer(
                batch["text"],
                add_special_tokens=False,
                truncation=False,
                return_attention_mask=False,
                return_token_type_ids=False,
            )
            return {"tok_len": [len(ids) for ids in enc["input_ids"]]}

        hf_dataset = hf_dataset.map(
            tokenized_length,
            batched=True,
            batch_size=self.map_batch_size,
            num_proc=self.map_num_proc,
            desc="Computing token lengths",
            writer_batch_size=10,
        )

        return np.array(hf_dataset["tok_len"], dtype=np.int32)

    def __len__(self):
        return len(self.group_idx)

    def __getitem__(self, idx) -> Dict[str, Any]:
        return {"textlist": [[str(self.texts[i])] for i in self.group_idx[idx]]}


def _compute_token_lengths_for_texts(
    texts,
    tokenizer,
    batch_size: int = 256,
) -> np.ndarray:
    all_lens = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start:start + batch_size]
        enc = tokenizer(
            batch,
            add_special_tokens=False,
            truncation=False,
            return_attention_mask=False,
            return_token_type_ids=False,
        )
        all_lens.extend([len(ids) for ids in enc["input_ids"]])

    return np.array(all_lens, dtype=np.int32)


def _pack_text_indices_by_length(
    token_lens: np.ndarray,
    conversation_max_len: int,
    base_len: int = 0,
    chat_len: int = 11,
    num_cache: int = 100,
) -> List[List[int]]:
    max_body_len = conversation_max_len - base_len
    packed_groups = []
    cache_group_idx = [[] for _ in range(num_cache)]
    cache_left_len = [max_body_len for _ in range(num_cache)]

    for i, tok_len in enumerate(token_lens):
        sample_len = int(tok_len) + chat_len

        if sample_len > max_body_len:
            packed_groups.append([i])
            continue

        placed = False
        for j, left_len in enumerate(cache_left_len):
            if sample_len <= left_len:
                cache_group_idx[j].append(i)
                cache_left_len[j] -= sample_len
                placed = True
                break

        if not placed:
            cache_id = int(np.argmin(cache_left_len))
            if cache_group_idx[cache_id]:
                packed_groups.append(cache_group_idx[cache_id])
            cache_group_idx[cache_id] = [i]
            cache_left_len[cache_id] = max_body_len - sample_len

    for group in cache_group_idx:
        if group:
            packed_groups.append(group)

    return packed_groups


class SkillInstructionDataset(Dataset):
    """Dataset for one-context-one-answer skill instruction tuning."""

    def __init__(
        self,
        data_path: str,
        max_context_len: int = 3000,
        max_conversation_len: int = 256,
        use_exceed: bool = False,
    ):
        with open(data_path, "r", encoding="utf-8") as f:
            self.item_list = json.load(f)
        if not use_exceed:
            self.item_list = [
                item
                for item in self.item_list
                if item["contextlen"] <= max_context_len
                and item["conversationlen"] <= max_conversation_len
            ]
        if is_primary_process():
            logger.info(
                f"[SkillInstructionDataset] Loaded {len(self.item_list)} items "
                f"from {data_path}, use_exceed={use_exceed}"
            )

    def __len__(self):
        return len(self.item_list)

    def __getitem__(self, idx) -> Dict[str, Any]:
        item = self.item_list[idx]
        return {
            "evidence": item["context"],
            "conversations": item["conversations"],
            "contextlen": item["contextlen"],
            "conversationlen": item["conversationlen"],
        }


class DynamicSkillPretrainDataset(Dataset):
    """
    Dynamic curriculum dataset for skill pretraining.

    Each epoch samples skill composites according to curriculum ratios, packs
    the composites by length, and returns {"textlist": [[skillA], [skillB, skillC], ...]}.
    """

    def __init__(
        self,
        texts,
        tokenizer,
        cfg,
        split: str = "train",
        force_single_skill: bool = False,
    ):
        self.texts = [str(t) for t in texts]
        self.tokenizer = tokenizer
        self.cfg = cfg
        self.split = split
        self.force_single_skill = force_single_skill
        self.base_seed = int(cfg.run.seed)

        if is_primary_process():
            logger.info(f"[DynamicSkillPretrainDataset] Precomputing token lengths for {len(self.texts)} skills...")
        self.token_lens = _compute_token_lengths_for_texts(
            self.texts,
            self.tokenizer,
            batch_size=256,
        )
        if is_primary_process():
            logger.info(f"[DynamicSkillPretrainDataset] Done. avg_len={float(self.token_lens.mean()):.1f}")

        self.active_groups = []
        self.current_epoch = 1
        self.set_epoch(1)
        self.fixed_len = len(self.active_groups)

    def get_ratios_for_epoch(self, epoch: int) -> Dict[int, float]:
        if self.force_single_skill:
            return {1: 1.0}

        last_ratios = None
        for stage in self.cfg.curriculum.stages:
            last_ratios = {int(k): float(v) for k, v in dict(stage.ratios).items()}
            if stage.start_epoch <= epoch <= stage.end_epoch:
                return last_ratios

        if last_ratios is not None:
            return last_ratios

        raise ValueError("No curriculum stages configured.")

    def build_groups_for_epoch(self, epoch: int) -> List[List[List[int]]]:
        split_offset = 0 if self.split == "train" else 10_000_000
        rng = random.Random(self.base_seed + split_offset + epoch)

        ratios = self.get_ratios_for_epoch(epoch)
        n = len(self.texts)
        num_composites = n

        raw_counts = {}
        total_assigned = 0
        for k, ratio in ratios.items():
            count = int(num_composites * float(ratio))
            raw_counts[int(k)] = count
            total_assigned += count

        remain = num_composites - total_assigned
        if remain > 0:
            max_k = max(ratios, key=lambda x: float(ratios[x]))
            raw_counts[int(max_k)] += remain

        composites = []
        for k, count in raw_counts.items():
            for _ in range(count):
                if k <= n:
                    idxs = rng.sample(range(n), k)
                else:
                    idxs = [rng.randrange(n) for _ in range(k)]
                composites.append(idxs)

        rng.shuffle(composites)

        chat_len_per_skill = 11
        composite_lens = np.array(
            [
                sum(int(self.token_lens[i]) for i in group) + chat_len_per_skill * len(group)
                for group in composites
            ],
            dtype=np.int32,
        )

        packed = _pack_text_indices_by_length(
            token_lens=composite_lens,
            conversation_max_len=self.cfg.data.conversation_max_length,
            base_len=0,
            chat_len=0,
            num_cache=100,
        )

        return [[composites[j] for j in pack] for pack in packed]

    def set_epoch(self, epoch: int):
        self.current_epoch = epoch
        self.active_groups = self.build_groups_for_epoch(epoch)

        if is_primary_process():
            all_composites = [c for group in self.active_groups for c in group]
            size_count = defaultdict(int)
            for composite in all_composites:
                size_count[len(composite)] += 1
            logger.info(
                f"[DynamicSkillPretrainDataset] epoch={epoch}, "
                f"num_packed_groups={len(self.active_groups)}, "
                f"num_composites={len(all_composites)}, "
                f"composite_size_dist={dict(sorted(size_count.items()))}"
            )

    def __len__(self):
        return self.fixed_len

    def __getitem__(self, idx):
        real_idx = idx % len(self.active_groups)
        return {
            "textlist": [
                [self.texts[i] for i in composite]
                for composite in self.active_groups[real_idx]
            ]
        }


@dataclass
class SkillCollatorBase:
    tokenizer: Any
    cfg: Any = None
    context_max_length: int = 1024
    conversation_max_length: int = 1024

    def __post_init__(self):
        if self.cfg is not None and "pretrain" in self.cfg:
            self.completion_freq = self.cfg.pretrain.completion_freq
            self.max_completion_ratio = self.cfg.pretrain.max_completion_ratio
            self.min_completion_ratio = self.cfg.pretrain.min_completion_ratio
        self.eot = "<|endoftext|>"
        self.assistant_token_id = self.tokenizer.convert_tokens_to_ids("assistant")
        self.imstart_token_id = self.tokenizer.convert_tokens_to_ids("<|im_start|>")
        self.imend_token_id = self.tokenizer.convert_tokens_to_ids("<|im_end|>")

    def mask_assistant_labels(self, labels):
        masks = torch.zeros_like(labels)
        for i, token_ids in enumerate(labels):
            last_imend = self.conversation_max_length
            for j in range(len(token_ids) - 1, 0, -1):
                if token_ids[j].item() == self.imend_token_id:
                    last_imend = j
                elif token_ids[j].item() == self.assistant_token_id and token_ids[j - 1] == self.imstart_token_id:
                    masks[i, j + 2:last_imend + 2] = 1
        return labels.masked_fill(masks == 0, -100)


@dataclass
class SkillPretrainCollator(SkillCollatorBase):
    """Collator for grouped skill pretraining examples."""

    def split_completion_source(self, text):
        tokens = text.split()
        if len(tokens) < 2:
            return text, "Nothing to complete."

        ratio = 1.0 - random.uniform(self.min_completion_ratio, self.max_completion_ratio)
        split_index = round(len(tokens) * ratio)

        left = tokens[:split_index]
        right = tokens[split_index:]
        if not right:
            left, right = tokens[:-1], tokens[-1:]
        elif not left:
            left, right = tokens[:1], tokens[1:]

        return " ".join(left), " ".join(right)

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        textlists = [ex["textlist"] for ex in batch]
        user_texts_list = []
        evidence_texts_list = []
        answer_texts_list = []
        for composites in textlists:
            flat_skills = [skill for composite in composites for skill in composite]
            evidence_texts = []
            answer_texts = []
            user_texts = []
            for skill_text in flat_skills:
                if random.random() < self.completion_freq:
                    left_text, _ = self.split_completion_source(skill_text)
                    evidence_texts.append(left_text)
                    answer_texts.append(skill_text)
                    user_texts.append("<COMP>")
                else:
                    evidence_texts.append(skill_text)
                    answer_texts.append(skill_text)
                    user_texts.append("<RECON>")
            evidence_texts_list.append(evidence_texts)
            answer_texts_list.append(answer_texts)
            user_texts_list.append(user_texts)

        evidence_texts_all = [
            self.eot.join(random.sample(evidence_texts, len(evidence_texts)))
            for evidence_texts in evidence_texts_list
        ]
        evidence_enc = self.tokenizer(
            evidence_texts_all,
            max_length=self.context_max_length,
            truncation=True,
            return_tensors="pt",
            padding="max_length",
        )

        messages = []
        for i in range(len(textlists)):
            indices = list(range(len(answer_texts_list[i])))
            random.shuffle(indices)
            msg = []
            for idx in indices:
                msg.append({"role": "user", "content": user_texts_list[i][idx]})
                msg.append({"role": "assistant", "content": answer_texts_list[i][idx]})
            messages.append(msg)

        input_enc = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=False,
            tokenize=True,
            return_tensors="pt",
            max_length=self.conversation_max_length,
            truncation=True,
            return_dict=True,
            padding="max_length",
            enable_thinking=False,
        )
        input_ids = input_enc["input_ids"]
        labels = self.mask_assistant_labels(input_ids.clone())

        return {
            "evidence": evidence_texts_all,
            "evidence_ids": evidence_enc["input_ids"],
            "evidence_attention_mask": evidence_enc["attention_mask"],
            "input_ids": input_ids,
            "labels": labels,
            "input_attention_mask": input_enc["attention_mask"],
            "questions": user_texts_list,
        }


@dataclass
class SkillInstructionCollator(SkillCollatorBase):
    """Collator for skill instruction tuning examples."""

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        evidence_texts = [item["evidence"] for item in batch]
        conversation_texts = [item["conversations"] for item in batch]
        if isinstance(conversation_texts[0], Column):
            conversation_texts = list(conversation_texts[0])

        evidence_enc = self.tokenizer(
            evidence_texts,
            max_length=self.context_max_length,
            truncation=True,
            return_tensors="pt",
            padding="max_length",
        )

        input_enc = self.tokenizer.apply_chat_template(
            conversation_texts,
            add_generation_prompt=False,
            tokenize=True,
            return_tensors="pt",
            max_length=self.conversation_max_length,
            truncation=True,
            return_dict=True,
            padding="max_length",
            enable_thinking=False,
        )
        input_ids = input_enc["input_ids"]
        labels = self.mask_assistant_labels(input_ids.clone())

        return {
            "evidence": evidence_texts,
            "evidence_ids": evidence_enc["input_ids"],
            "evidence_attention_mask": evidence_enc["attention_mask"],
            "input_ids": input_ids,
            "labels": labels,
            "input_attention_mask": input_enc["attention_mask"],
        }
