from __future__ import annotations

import torch

from parteek_llm.config import ModelConfig
from parteek_llm.model import ParteekLLM


def tiny_model() -> ParteekLLM:
    torch.manual_seed(7)
    return ParteekLLM(
        ModelConfig(
            vocab_size=96,
            block_size=16,
            n_layers=2,
            n_heads=4,
            d_model=32,
            ffn_dim=64,
            dropout=0.0,
        )
    )


def test_forward_backward_and_weight_tying() -> None:
    model = tiny_model()
    input_ids = torch.randint(0, 96, (2, 12))
    targets = torch.randint(0, 96, (2, 12))
    logits, loss = model(input_ids, targets)
    assert logits.shape == (2, 12, 96)
    assert loss is not None and torch.isfinite(loss)
    loss.backward()
    assert model.token_embedding.weight.grad is not None
    assert model.token_embedding.weight.data_ptr() == model.lm_head.weight.data_ptr()


def test_causal_mask_prevents_future_leakage() -> None:
    model = tiny_model().eval()
    first = torch.tensor([[1, 2, 3, 4, 5]])
    second = torch.tensor([[1, 2, 3, 4, 9]])
    with torch.no_grad():
        first_logits, _ = model(first)
        second_logits, _ = model(second)
    assert torch.allclose(first_logits[:, :4], second_logits[:, :4], atol=1e-6)


def test_generation_adds_tokens() -> None:
    model = tiny_model().eval()
    prompt = torch.tensor([[2, 10, 11]])
    output = model.generate(prompt, max_new_tokens=4, temperature=0)
    assert output.shape == (1, 7)

