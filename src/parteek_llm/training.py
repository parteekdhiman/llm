from __future__ import annotations

import math
import random
from contextlib import nullcontext
from typing import Iterable

import numpy as np
import torch
from torch.utils.data import DataLoader

from .config import TrainingConfig
from .model import ParteekLLM


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def choose_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def create_grad_scaler(use_amp: bool):
    try:
        return torch.amp.GradScaler("cuda", enabled=use_amp)
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=use_amp)


def autocast_context(device: torch.device, use_amp: bool):
    if not use_amp:
        return nullcontext()
    return torch.autocast(device_type=device.type, dtype=torch.float16)


def learning_rate_for_step(
    step: int, config: TrainingConfig, total_steps: int | None = None
) -> float:
    maximum = total_steps or config.max_steps
    if config.warmup_steps > 0 and step < config.warmup_steps:
        return config.learning_rate * (step + 1) / config.warmup_steps
    if maximum <= config.warmup_steps:
        return config.min_learning_rate
    progress = min(1.0, (step - config.warmup_steps) / (maximum - config.warmup_steps))
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return config.min_learning_rate + cosine * (
        config.learning_rate - config.min_learning_rate
    )


def set_optimizer_learning_rate(optimizer: torch.optim.Optimizer, value: float) -> None:
    for parameter_group in optimizer.param_groups:
        parameter_group["lr"] = value


def create_optimizer(model: ParteekLLM, config: TrainingConfig) -> torch.optim.Optimizer:
    decay: list[torch.nn.Parameter] = []
    no_decay: list[torch.nn.Parameter] = []
    for parameter in model.parameters():
        if not parameter.requires_grad:
            continue
        (decay if parameter.ndim >= 2 else no_decay).append(parameter)
    groups = [
        {"params": decay, "weight_decay": config.weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    return torch.optim.AdamW(groups, lr=config.learning_rate, betas=(0.9, 0.95), eps=1e-8)


def infinite_batches(loader: DataLoader) -> Iterable[tuple[torch.Tensor, torch.Tensor]]:
    while True:
        yield from loader


@torch.no_grad()
def evaluate(
    model: ParteekLLM,
    loader: DataLoader,
    device: torch.device,
    batches: int,
    use_amp: bool,
) -> float:
    was_training = model.training
    model.eval()
    losses: list[float] = []
    iterator = iter(loader)
    for _ in range(batches):
        try:
            input_ids, targets = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            input_ids, targets = next(iterator)
        input_ids = input_ids.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        with autocast_context(device, use_amp):
            _, loss = model(input_ids, targets)
        if loss is not None:
            losses.append(float(loss.detach().cpu()))
    model.train(was_training)
    return float(np.mean(losses)) if losses else math.nan
