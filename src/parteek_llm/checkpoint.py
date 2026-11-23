from __future__ import annotations

import shutil
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch

from .config import ModelConfig, TrainingConfig
from .model import ParteekLLM


def save_checkpoint(
    output_dir: str | Path,
    model: ParteekLLM,
    optimizer: torch.optim.Optimizer | None,
    scaler: Any | None,
    step: int,
    training_config: TrainingConfig,
    *,
    name: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> Path:
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    checkpoint_name = name or f"step_{step:08d}.pt"
    checkpoint_path = destination / checkpoint_name
    temporary_path = checkpoint_path.with_suffix(checkpoint_path.suffix + ".tmp")
    payload: dict[str, Any] = {
        "format_version": 1,
        "step": step,
        "model_config": asdict(model.config),
        "training_config": asdict(training_config),
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict() if optimizer is not None else None,
        "scaler_state": scaler.state_dict() if scaler is not None else None,
        "metadata": metadata or {},
    }
    torch.save(payload, temporary_path)
    temporary_path.replace(checkpoint_path)
    latest_path = destination / "latest.pt"
    if checkpoint_path.resolve() != latest_path.resolve():
        latest_temporary = destination / "latest.pt.tmp"
        shutil.copy2(checkpoint_path, latest_temporary)
        latest_temporary.replace(latest_path)
    return checkpoint_path


def load_checkpoint(path: str | Path, device: torch.device | str = "cpu") -> dict[str, Any]:
    checkpoint_path = Path(path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    try:
        payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    except TypeError:  # Compatibility with older supported PyTorch releases.
        payload = torch.load(checkpoint_path, map_location=device)
    required = {"model_config", "model_state", "step"}
    missing = required.difference(payload)
    if missing:
        raise ValueError(f"Invalid checkpoint; missing keys: {sorted(missing)}")
    return payload


def load_model_from_checkpoint(
    path: str | Path, device: torch.device | str = "cpu"
) -> tuple[ParteekLLM, dict[str, Any]]:
    payload = load_checkpoint(path, device)
    config = ModelConfig(**payload["model_config"])
    model = ParteekLLM(config)
    model.load_state_dict(payload["model_state"])
    model.to(device)
    return model, payload

