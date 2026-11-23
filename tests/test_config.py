from __future__ import annotations

import pytest

from parteek_llm.config import ModelConfig


def test_valid_model_config() -> None:
    config = ModelConfig(
        vocab_size=128,
        block_size=16,
        n_layers=2,
        n_heads=4,
        d_model=32,
        ffn_dim=64,
        dropout=0.0,
    )
    config.validate()


def test_invalid_head_dimension() -> None:
    config = ModelConfig(d_model=30, n_heads=4)
    with pytest.raises(ValueError, match="divisible"):
        config.validate()

