from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import torch
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader

from .checkpoint import load_checkpoint, save_checkpoint
from .config import load_config
from .dataset import TokenBlockDataset, build_token_cache
from .model import ParteekLLM
from .tokenizer import OwnTokenizer
from .training import (
    autocast_context,
    choose_device,
    create_grad_scaler,
    create_optimizer,
    evaluate,
    infinite_batches,
    learning_rate_for_step,
    set_optimizer_learning_rate,
    set_seed,
)


def create_loader(
    dataset: TokenBlockDataset,
    batch_size: int,
    num_workers: int,
    device: torch.device,
    shuffle: bool,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=len(dataset) >= batch_size,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=num_workers > 0,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Pretrain Parteek Own LLM from random weights")
    parser.add_argument("--config", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--train-corpus", nargs="+", required=True)
    parser.add_argument("--val-corpus", nargs="+")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--resume")
    parser.add_argument("--rebuild-cache", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    project = load_config(args.config)
    training_config = project.training
    tokenizer = OwnTokenizer(args.tokenizer)
    project.model.vocab_size = tokenizer.vocab_size
    project.model.validate()
    set_seed(training_config.seed)
    device = choose_device()
    use_amp = training_config.amp and device.type == "cuda"
    output_dir = Path(args.output_dir)
    cache_dir = output_dir / "cache"

    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("Warning: CUDA was not detected; pretraining on CPU will be slow.")

    train_cache = build_token_cache(
        args.train_corpus,
        tokenizer,
        cache_dir / "train.bin",
        force=args.rebuild_cache,
    )
    train_dataset = TokenBlockDataset(train_cache, project.model.block_size)
    train_loader = create_loader(
        train_dataset,
        training_config.micro_batch_size,
        training_config.num_workers,
        device,
        shuffle=True,
    )
    val_loader = None
    if args.val_corpus:
        val_cache = build_token_cache(
            args.val_corpus,
            tokenizer,
            cache_dir / "validation.bin",
            force=args.rebuild_cache,
        )
        val_dataset = TokenBlockDataset(val_cache, project.model.block_size)
        val_loader = create_loader(
            val_dataset,
            training_config.micro_batch_size,
            training_config.num_workers,
            device,
            shuffle=False,
        )

    model = ParteekLLM(project.model).to(device)
    optimizer = create_optimizer(model, training_config)
    scaler = create_grad_scaler(use_amp)
    start_step = 0
    if args.resume:
        payload = load_checkpoint(args.resume, device)
        checkpoint_config = payload["model_config"]
        if checkpoint_config != model.config_dict():
            raise ValueError(
                "Resume checkpoint model configuration differs from the requested configuration"
            )
        model.load_state_dict(payload["model_state"])
        if payload.get("optimizer_state"):
            optimizer.load_state_dict(payload["optimizer_state"])
        if payload.get("scaler_state"):
            scaler.load_state_dict(payload["scaler_state"])
        start_step = int(payload["step"])
        print(f"Resumed from optimizer step {start_step:,}")

    train_model = model
    if training_config.compile and hasattr(torch, "compile"):
        print("Compiling model; the first step may take several minutes.")
        train_model = torch.compile(model)

    print(f"Model parameters: {model.parameter_count():,}")
    print(f"Training blocks: {len(train_dataset):,}")
    effective_tokens_per_step = (
        training_config.micro_batch_size
        * training_config.gradient_accumulation_steps
        * project.model.block_size
    )
    print(f"Effective tokens/step: {effective_tokens_per_step:,}")

    batches = infinite_batches(train_loader)
    optimizer.zero_grad(set_to_none=True)
    total_tokens = start_step * (
        training_config.micro_batch_size
        * training_config.gradient_accumulation_steps
        * project.model.block_size
    )
    started = time.perf_counter()
    last_log = started
    running_loss = 0.0

    try:
        for step in range(start_step, training_config.max_steps):
            learning_rate = learning_rate_for_step(step, training_config)
            set_optimizer_learning_rate(optimizer, learning_rate)
            step_loss = 0.0
            for _ in range(training_config.gradient_accumulation_steps):
                input_ids, targets = next(batches)
                input_ids = input_ids.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)
                with autocast_context(device, use_amp):
                    _, loss = train_model(input_ids, targets)
                    if loss is None:
                        raise RuntimeError("Model did not return a training loss")
                    scaled_loss = loss / training_config.gradient_accumulation_steps
                scaler.scale(scaled_loss).backward()
                step_loss += float(loss.detach().cpu())

            scaler.unscale_(optimizer)
            grad_norm = clip_grad_norm_(model.parameters(), training_config.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

            completed_step = step + 1
            average_step_loss = step_loss / training_config.gradient_accumulation_steps
            running_loss += average_step_loss
            total_tokens += (
                training_config.micro_batch_size
                * training_config.gradient_accumulation_steps
                * project.model.block_size
            )

            if completed_step % training_config.log_interval == 0:
                now = time.perf_counter()
                interval_seconds = max(now - last_log, 1e-9)
                interval_tokens = (
                    training_config.log_interval
                    * training_config.micro_batch_size
                    * training_config.gradient_accumulation_steps
                    * project.model.block_size
                )
                mean_loss = running_loss / training_config.log_interval
                perplexity = math.exp(min(mean_loss, 20.0))
                print(
                    f"step {completed_step:,}/{training_config.max_steps:,} | "
                    f"loss {mean_loss:.4f} | ppl {perplexity:.2f} | "
                    f"lr {learning_rate:.2e} | grad {float(grad_norm):.2f} | "
                    f"{interval_tokens / interval_seconds:,.0f} tok/s"
                )
                running_loss = 0.0
                last_log = now

            if val_loader is not None and completed_step % training_config.eval_interval == 0:
                validation_loss = evaluate(
                    train_model,
                    val_loader,
                    device,
                    training_config.eval_batches,
                    use_amp,
                )
                print(
                    f"validation | loss {validation_loss:.4f} | "
                    f"ppl {math.exp(min(validation_loss, 20.0)):.2f}"
                )

            if completed_step % training_config.save_interval == 0:
                path = save_checkpoint(
                    output_dir,
                    model,
                    optimizer,
                    scaler,
                    completed_step,
                    training_config,
                    metadata={
                        "stage": "pretraining",
                        "tokenizer": str(Path(args.tokenizer).resolve()),
                        "tokens_seen": total_tokens,
                    },
                )
                print(f"Saved checkpoint: {path}")
    except KeyboardInterrupt:
        print("Training interrupted; saving recoverable checkpoint...")
    finally:
        final_step = min(training_config.max_steps, step + 1 if "step" in locals() else start_step)
        path = save_checkpoint(
            output_dir,
            model,
            optimizer,
            scaler,
            final_step,
            training_config,
            name=f"final_{final_step:08d}.pt",
            metadata={
                "stage": "pretraining",
                "tokenizer": str(Path(args.tokenizer).resolve()),
                "tokens_seen": total_tokens,
            },
        )
        elapsed = time.perf_counter() - started
        print(f"Final checkpoint: {path}")
        print(f"Elapsed: {elapsed / 3600:.2f} hours | tokens seen: {total_tokens:,}")


if __name__ == "__main__":
    main()
