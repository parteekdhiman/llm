from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass
class ModelConfig:
    vocab_size: int = 16_000
    block_size: int = 512
    n_layers: int = 6
    n_heads: int = 5
    d_model: int = 320
    ffn_dim: int = 960
    dropout: float = 0.05
    rope_base: float = 10_000.0

    def validate(self) -> None:
        if self.vocab_size < 16:
            raise ValueError("vocab_size must be at least 16")
        if self.block_size < 8:
            raise ValueError("block_size must be at least 8")
        if self.d_model % self.n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        if (self.d_model // self.n_heads) % 2 != 0:
            raise ValueError("attention head dimension must be even for rotary embeddings")
        if self.n_layers < 1 or self.n_heads < 1 or self.ffn_dim < 1:
            raise ValueError("layer, head, and FFN sizes must be positive")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")


@dataclass
class TrainingConfig:
    seed: int = 42
    micro_batch_size: int = 2
    gradient_accumulation_steps: int = 16
    learning_rate: float = 3e-4
    min_learning_rate: float = 3e-5
    weight_decay: float = 0.1
    warmup_steps: int = 1_000
    max_steps: int = 50_000
    grad_clip: float = 1.0
    log_interval: int = 10
    eval_interval: int = 500
    eval_batches: int = 20
    save_interval: int = 500
    num_workers: int = 0
    amp: bool = True
    compile: bool = False

    def validate(self) -> None:
        integer_fields = {
            "micro_batch_size": self.micro_batch_size,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "max_steps": self.max_steps,
            "log_interval": self.log_interval,
            "eval_interval": self.eval_interval,
            "eval_batches": self.eval_batches,
            "save_interval": self.save_interval,
        }
        for name, value in integer_fields.items():
            if value < 1:
                raise ValueError(f"{name} must be at least 1")
        if self.learning_rate <= 0 or self.min_learning_rate < 0:
            raise ValueError("learning rates must be non-negative and learning_rate > 0")
        if self.warmup_steps < 0 or self.num_workers < 0:
            raise ValueError("warmup_steps and num_workers cannot be negative")
        if self.grad_clip <= 0:
            raise ValueError("grad_clip must be positive")


@dataclass
class ProjectConfig:
    model: ModelConfig
    training: TrainingConfig

    def validate(self) -> None:
        self.model.validate()
        self.training.validate()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_config(path: str | Path) -> ProjectConfig:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)
    try:
        config = ProjectConfig(
            model=ModelConfig(**raw["model"]),
            training=TrainingConfig(**raw["training"]),
        )
    except (KeyError, TypeError) as exc:
        raise ValueError(f"Invalid configuration in {config_path}: {exc}") from exc
    config.validate()
    return config


def project_config_from_dict(raw: dict[str, Any]) -> ProjectConfig:
    config = ProjectConfig(
        model=ModelConfig(**raw["model"]),
        training=TrainingConfig(**raw["training"]),
    )
    config.validate()
    return config

