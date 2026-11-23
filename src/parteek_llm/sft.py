from __future__ import annotations

import argparse
import math
import time
from dataclasses import replace
from pathlib import Path

import torch
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader

from .checkpoint import load_model_from_checkpoint, save_checkpoint
from .config import load_config
from .dataset import InstructionDataset, optimizer_steps_per_epoch
from .tokenizer import OwnTokenizer
from .training import (
    autocast_context,
    choose_device,
    create_grad_scaler,
    create_optimizer,
    infinite_batches,
    learning_rate_for_step,
    set_optimizer_learning_rate,
    set_seed,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Instruction-tune a pretrained checkpoint")
    parser.add_argument("--config", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--base-checkpoint", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.epochs < 1 or args.max_steps < 0 or args.learning_rate <= 0:
        raise ValueError("epochs and learning rate must be positive; max_steps cannot be negative")
    project = load_config(args.config)
    tokenizer = OwnTokenizer(args.tokenizer)
    device = choose_device()
    model, base_payload = load_model_from_checkpoint(args.base_checkpoint, device)
    if model.config.vocab_size != tokenizer.vocab_size:
        raise ValueError(
            f"Checkpoint vocabulary ({model.config.vocab_size}) does not match tokenizer "
            f"({tokenizer.vocab_size})"
        )

    training_config = replace(
        project.training,
        learning_rate=args.learning_rate,
        min_learning_rate=args.learning_rate * 0.1,
    )
    set_seed(training_config.seed)
    dataset = InstructionDataset(args.data, tokenizer, model.config.block_size)
    loader = DataLoader(
        dataset,
        batch_size=training_config.micro_batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=training_config.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=training_config.num_workers > 0,
    )
    estimated_steps = optimizer_steps_per_epoch(
        len(dataset),
        training_config.micro_batch_size,
        training_config.gradient_accumulation_steps,
    ) * args.epochs
    total_steps = args.max_steps or estimated_steps
    training_config.max_steps = total_steps
    training_config.warmup_steps = min(
        training_config.warmup_steps, max(1, total_steps // 20)
    )
    use_amp = training_config.amp and device.type == "cuda"
    optimizer = create_optimizer(model, training_config)
    scaler = create_grad_scaler(use_amp)
    train_model = model
    if training_config.compile and hasattr(torch, "compile"):
        train_model = torch.compile(model)

    print(f"Device: {device}")
    print(f"Examples: {len(dataset):,}")
    print(f"Parameters: {model.parameter_count():,}")
    print(f"Optimizer steps: {total_steps:,}")
    batches = infinite_batches(loader)
    optimizer.zero_grad(set_to_none=True)
    started = time.perf_counter()

    try:
        for step in range(total_steps):
            learning_rate = learning_rate_for_step(step, training_config, total_steps)
            set_optimizer_learning_rate(optimizer, learning_rate)
            accumulated_loss = 0.0
            for _ in range(training_config.gradient_accumulation_steps):
                input_ids, targets = next(batches)
                input_ids = input_ids.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)
                with autocast_context(device, use_amp):
                    _, loss = train_model(input_ids, targets)
                    if loss is None or not torch.isfinite(loss):
                        raise RuntimeError(f"Non-finite SFT loss at step {step + 1}: {loss}")
                    scaled_loss = loss / training_config.gradient_accumulation_steps
                scaler.scale(scaled_loss).backward()
                accumulated_loss += float(loss.detach().cpu())

            scaler.unscale_(optimizer)
            grad_norm = clip_grad_norm_(model.parameters(), training_config.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

            completed_step = step + 1
            if completed_step % training_config.log_interval == 0 or completed_step == 1:
                mean_loss = accumulated_loss / training_config.gradient_accumulation_steps
                print(
                    f"step {completed_step:,}/{total_steps:,} | loss {mean_loss:.4f} | "
                    f"ppl {math.exp(min(mean_loss, 20.0)):.2f} | lr {learning_rate:.2e} | "
                    f"grad {float(grad_norm):.2f}"
                )
            if completed_step % training_config.save_interval == 0:
                path = save_checkpoint(
                    args.output_dir,
                    model,
                    optimizer,
                    scaler,
                    completed_step,
                    training_config,
                    metadata={
                        "stage": "instruction_tuning",
                        "base_checkpoint": str(Path(args.base_checkpoint).resolve()),
                        "tokenizer": str(Path(args.tokenizer).resolve()),
                        "examples": len(dataset),
                    },
                )
                print(f"Saved checkpoint: {path}")
    except KeyboardInterrupt:
        print("Instruction tuning interrupted; saving checkpoint...")
    finally:
        final_step = min(total_steps, step + 1 if "step" in locals() else 0)
        path = save_checkpoint(
            args.output_dir,
            model,
            optimizer,
            scaler,
            final_step,
            training_config,
            name=f"final_{final_step:08d}.pt",
            metadata={
                "stage": "instruction_tuning",
                "base_checkpoint": str(Path(args.base_checkpoint).resolve()),
                "base_step": int(base_payload.get("step", 0)),
                "tokenizer": str(Path(args.tokenizer).resolve()),
                "examples": len(dataset),
            },
        )
        print(f"Final chat checkpoint: {path}")
        print(f"Elapsed: {(time.perf_counter() - started) / 60:.1f} minutes")


if __name__ == "__main__":
    main()
