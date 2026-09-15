#!/usr/bin/env python3
"""Run auditable QLoRA SFT, with full-epoch checkpoints and validation."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import random
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from transformers import get_cosine_schedule_with_warmup


@dataclass
class EncodingStats:
    examples: int = 0
    prompt_tokens: int = 0
    answer_tokens: int = 0
    kept_prompt_tokens: int = 0
    kept_answer_tokens: int = 0
    truncated_prompts: int = 0
    truncated_answers: int = 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Local model path or HF model id")
    parser.add_argument("--train-file", type=Path, required=True)
    parser.add_argument("--validation-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-length", type=int, default=4096)
    duration = parser.add_mutually_exclusive_group()
    duration.add_argument("--max-steps", type=int)
    duration.add_argument("--epochs", type=int)
    parser.add_argument("--require-untruncated", action="store_true")
    parser.add_argument("--allow-prompt-overlap", action="store_true", help="Legacy pipeline smoke only")
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=20260914)
    parser.add_argument("--generation-samples", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=768)
    parser.add_argument("--train-limit", type=int, default=None)
    parser.add_argument("--validation-limit", type=int, default=None)
    args = parser.parse_args()
    if args.epochs is None and args.max_steps is None:
        args.max_steps = 4
    for name in ("max_length", "gradient_accumulation_steps", "max_new_tokens"):
        if getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.max_length < 256 or args.max_new_tokens >= args.max_length:
        parser.error("max-length must be >=256 and exceed max-new-tokens")
    for name in ("epochs", "max_steps", "train_limit", "validation_limit"):
        if getattr(args, name) is not None and getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    return args


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        messages = row.get("messages")
        if not isinstance(messages, list) or not messages or messages[-1].get("role") != "assistant":
            raise ValueError(f"{path}:{line_number} must end with an assistant message")
        rows.append(row)
    if not rows:
        raise ValueError(f"{path} is empty")
    return rows


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def preserve_prompt_ends(ids: list[int], budget: int, prefix_tokens: int = 256) -> list[int]:
    if len(ids) <= budget:
        return ids
    prefix = min(prefix_tokens, budget // 3)
    return ids[:prefix] + ids[-(budget - prefix) :]


class ChatSFTDataset(Dataset):
    def __init__(self, rows: list[dict[str, Any]], tokenizer: Any, max_length: int):
        self.examples: list[dict[str, list[int]]] = []
        self.stats = EncodingStats()
        eos_id = tokenizer.eos_token_id
        if eos_id is None:
            raise ValueError("Tokenizer must define eos_token_id")

        for row in rows:
            messages = row["messages"]
            prompt_text = tokenizer.apply_chat_template(
                messages[:-1], tokenize=False, add_generation_prompt=True, enable_thinking=False
            )
            prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
            answer_ids = tokenizer.encode(messages[-1]["content"], add_special_tokens=False) + [eos_id]

            original_prompt = len(prompt_ids)
            original_answer = len(answer_ids)
            answer_budget = max(128, max_length - 128)
            if len(answer_ids) > answer_budget:
                answer_ids = answer_ids[: answer_budget - 1] + [eos_id]
                self.stats.truncated_answers += 1
            prompt_budget = max_length - len(answer_ids)
            if len(prompt_ids) > prompt_budget:
                prompt_ids = preserve_prompt_ends(prompt_ids, prompt_budget)
                self.stats.truncated_prompts += 1

            input_ids = prompt_ids + answer_ids
            self.examples.append(
                {
                    "input_ids": input_ids,
                    "attention_mask": [1] * len(input_ids),
                    "labels": [-100] * len(prompt_ids) + answer_ids,
                }
            )
            self.stats.examples += 1
            self.stats.prompt_tokens += original_prompt
            self.stats.answer_tokens += original_answer
            self.stats.kept_prompt_tokens += len(prompt_ids)
            self.stats.kept_answer_tokens += len(answer_ids)

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict[str, list[int]]:
        return self.examples[index]


class CompletionCollator:
    def __init__(self, pad_token_id: int):
        self.pad_token_id = pad_token_id

    def __call__(self, rows: list[dict[str, list[int]]]) -> dict[str, torch.Tensor]:
        width = max(len(row["input_ids"]) for row in rows)
        batch = {"input_ids": [], "attention_mask": [], "labels": []}
        for row in rows:
            padding = width - len(row["input_ids"])
            batch["input_ids"].append(row["input_ids"] + [self.pad_token_id] * padding)
            batch["attention_mask"].append(row["attention_mask"] + [0] * padding)
            batch["labels"].append(row["labels"] + [-100] * padding)
        return {name: torch.tensor(values, dtype=torch.long) for name, values in batch.items()}


def evaluate_loss(model: Any, loader: DataLoader) -> float:
    model.eval()
    weighted_loss = 0.0
    token_count = 0
    with torch.no_grad():
        for batch in loader:
            batch = {key: value.to(model.device) for key, value in batch.items()}
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                loss = model(**batch).loss
            active = int((batch["labels"] != -100).sum().item())
            weighted_loss += float(loss.item()) * active
            token_count += active
    return weighted_loss / max(1, token_count)


def extract_json(text: str) -> tuple[bool, str | None]:
    cleaned = text.strip()
    if cleaned.startswith("```json"):
        cleaned = cleaned[7:]
    elif cleaned.startswith("```"):
        cleaned = cleaned[3:]
    if cleaned.endswith("```"):
        cleaned = cleaned[:-3]
    try:
        value = json.loads(cleaned.strip())
        return isinstance(value, dict), None
    except json.JSONDecodeError as exc:
        return False, str(exc)


def generate_samples(
    model: Any,
    tokenizer: Any,
    rows: list[dict[str, Any]],
    max_length: int,
    max_new_tokens: int,
    count: int,
) -> list[dict[str, Any]]:
    model.eval()
    results = []
    prompt_budget = max(128, max_length - max_new_tokens)
    for row in rows[:count]:
        prompt_text = tokenizer.apply_chat_template(
            row["messages"][:-1], tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
        prompt_ids = preserve_prompt_ends(prompt_ids, prompt_budget)
        input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=model.device)
        with torch.no_grad():
            output = model.generate(
                input_ids=input_ids,
                attention_mask=torch.ones_like(input_ids),
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=None,
                top_p=None,
                top_k=None,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        text = tokenizer.decode(output[0, len(prompt_ids) :], skip_special_tokens=True)
        valid, error = extract_json(text)
        results.append({"metadata": row.get("metadata", {}), "valid_json_object": valid, "error": error, "text": text})
    return results


def git_revision() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for 4-bit QLoRA")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=False)

    all_train_rows = read_jsonl(args.train_file)
    all_validation_rows = read_jsonl(args.validation_file)
    train_rows = all_train_rows[: args.train_limit]
    validation_rows = all_validation_rows[: args.validation_limit]
    prompt_hash = lambda row: hashlib.sha256(json.dumps(row['messages'][:-1], sort_keys=True).encode()).hexdigest()
    prompt_overlap = {prompt_hash(r) for r in train_rows} & {prompt_hash(r) for r in validation_rows}
    if prompt_overlap and not args.allow_prompt_overlap:
        raise ValueError("Train/validation prompts overlap; rebuild a prompt-group-disjoint split")
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_dataset = ChatSFTDataset(train_rows, tokenizer, args.max_length)
    validation_dataset = ChatSFTDataset(validation_rows, tokenizer, args.max_length)
    if args.require_untruncated and any(
        d.stats.truncated_prompts or d.stats.truncated_answers
        for d in (train_dataset, validation_dataset)
    ):
        raise ValueError("Full-context run refuses truncated training or validation examples")
    collator = CompletionCollator(tokenizer.pad_token_id)
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, collate_fn=collator, generator=generator)
    validation_loader = DataLoader(validation_dataset, batch_size=1, shuffle=False, collate_fn=collator)

    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        quantization_config=quantization,
        torch_dtype=torch.bfloat16,
        device_map={"": 0},
        local_files_only=True,
    )
    model = prepare_model_for_kbit_training(
        model,
        use_gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )
    model.config.use_cache = False
    model = get_peft_model(
        model,
        LoraConfig(
            r=args.lora_rank,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        ),
    )
    trainable, total = model.get_nb_trainable_parameters()
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=args.learning_rate)
    target_steps = args.max_steps or args.epochs * math.ceil(len(train_loader) / args.gradient_accumulation_steps)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=max(1, int(target_steps * .05)), num_training_steps=target_steps
    )

    started = time.time()
    initial_validation_loss = evaluate_loss(model, validation_loader)
    log_path = args.output_dir / "training_log.jsonl"
    optimizer.zero_grad(set_to_none=True)
    optimizer_step = 0
    micro_step = 0
    epoch = 0
    validation_curve = [{"epoch": 0, "optimizer_step": 0, "loss": initial_validation_loss}]
    best_loss = float("inf")
    best_checkpoint = None
    window_losses = []
    with log_path.open("w", encoding="utf-8") as log_handle:
        while optimizer_step < target_steps:
            epoch += 1
            model.train()
            for batch_index, batch in enumerate(train_loader, 1):
                batch = {key: value.to(model.device) for key, value in batch.items()}
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    loss = model(**batch).loss
                window_start = ((batch_index - 1) // args.gradient_accumulation_steps) * args.gradient_accumulation_steps
                window_size = min(args.gradient_accumulation_steps, len(train_loader) - window_start)
                (loss / window_size).backward()
                window_losses.append(float(loss.detach().item()))
                micro_step += 1
                should_step = batch_index % args.gradient_accumulation_steps == 0 or batch_index == len(train_loader)
                if not should_step:
                    continue
                torch.nn.utils.clip_grad_norm_((p for p in model.parameters() if p.requires_grad), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                optimizer_step += 1
                record = {
                    "epoch": epoch,
                    "optimizer_step": optimizer_step,
                    "micro_step": micro_step,
                    "loss": sum(window_losses) / len(window_losses),
                    "learning_rate": scheduler.get_last_lr()[0],
                    "peak_gpu_memory_gib": round(torch.cuda.max_memory_allocated() / 2**30, 3),
                }
                print(json.dumps(record), flush=True)
                log_handle.write(json.dumps(record) + "\n")
                log_handle.flush()
                window_losses = []
                if optimizer_step >= target_steps:
                    break
            validation_loss = evaluate_loss(model, validation_loader)
            checkpoint = args.output_dir / f"checkpoint-epoch-{epoch}"
            model.save_pretrained(checkpoint / "adapter", safe_serialization=True)
            tokenizer.save_pretrained(checkpoint / "adapter")
            torch.save({
                "epoch": epoch, "optimizer_step": optimizer_step, "micro_step": micro_step,
                "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                "torch_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state_all(),
                "loader_rng": generator.get_state(), "python_rng": random.getstate(),
            }, checkpoint / "training_state.pt")
            validation_curve.append({
                "epoch": epoch, "optimizer_step": optimizer_step, "loss": validation_loss,
                "complete_epoch": batch_index == len(train_loader),
                "checkpoint": str(checkpoint),
            })
            if validation_loss < best_loss:
                best_loss, best_checkpoint = validation_loss, checkpoint
            (args.output_dir / "validation_curve.json").write_text(json.dumps(validation_curve, indent=2) + "\n")
            print(json.dumps({"epoch_validation": validation_curve[-1]}), flush=True)

    final_validation_loss = validation_curve[-1]["loss"]
    adapter_dir = args.output_dir / "adapter"
    model.save_pretrained(adapter_dir, safe_serialization=True)
    tokenizer.save_pretrained(adapter_dir)
    (args.output_dir / "best_adapter").symlink_to(best_checkpoint.name + "/adapter", target_is_directory=True)
    model.config.use_cache = True
    samples = generate_samples(
        model, tokenizer, validation_rows, args.max_length, args.max_new_tokens, args.generation_samples
    )
    (args.output_dir / "generated_samples.json").write_text(
        json.dumps(samples, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    metrics = {
        "status": "full_training_complete" if args.epochs else "smoke_test_complete",
        "claim_boundary": "Full available-data fitting; same-task validation is not task-disjoint efficacy or generalization evidence.",
        "arguments": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "dataset": {
            "train_examples": len(train_rows),
            "validation_examples": len(validation_rows),
            "available_train_examples": len(all_train_rows),
            "available_validation_examples": len(all_validation_rows),
            "train_validation_prompt_overlap": len(prompt_overlap),
            "train_sha256": sha256_file(args.train_file),
            "validation_sha256": sha256_file(args.validation_file),
            "train_encoding": asdict(train_dataset.stats),
            "validation_encoding": asdict(validation_dataset.stats),
        },
        "model": {
            "trainable_parameters": trainable,
            "total_parameters": total,
            "trainable_fraction": trainable / total,
        },
        "result": {
            "initial_validation_loss": initial_validation_loss,
            "final_validation_loss": final_validation_loss,
            "validation_loss_delta": final_validation_loss - initial_validation_loss,
            "best_validation_loss": best_loss,
            "best_checkpoint": str(best_checkpoint),
            "completed_epochs": sum(bool(x.get("complete_epoch")) for x in validation_curve[1:]),
            "examples_seen": micro_step,
            "validation_curve": validation_curve,
            "optimizer_steps": optimizer_step,
            "micro_steps": micro_step,
            "generated_samples": len(samples),
            "valid_json_samples": sum(bool(item["valid_json_object"]) for item in samples),
            "peak_gpu_memory_gib": round(torch.cuda.max_memory_allocated() / 2**30, 3),
            "elapsed_seconds": round(time.time() - started, 3),
        },
        "environment": {
            "git_revision": git_revision(),
            "training_script_sha256": sha256_file(Path(__file__)),
            "hostname": platform.node(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": __import__("transformers").__version__,
            "peft": __import__("peft").__version__,
            "bitsandbytes": __import__("bitsandbytes").__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        },
    }
    (args.output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(metrics["result"], indent=2), flush=True)


if __name__ == "__main__":
    main()
