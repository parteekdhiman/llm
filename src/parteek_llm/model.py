from __future__ import annotations

import math
from dataclasses import asdict

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ModelConfig


class RMSNorm(nn.Module):
    def __init__(self, dimension: int, epsilon: float = 1e-5):
        super().__init__()
        self.epsilon = epsilon
        self.weight = nn.Parameter(torch.ones(dimension))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        normalized = x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.epsilon)
        return normalized * self.weight


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x_even = x[..., ::2]
    x_odd = x[..., 1::2]
    return torch.stack((-x_odd, x_even), dim=-1).flatten(-2)


class RotaryEmbedding(nn.Module):
    def __init__(self, head_dim: int, max_sequence_length: int, base: float):
        super().__init__()
        inverse_frequency = 1.0 / (
            base ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
        )
        positions = torch.arange(max_sequence_length, dtype=torch.float32)
        frequencies = torch.outer(positions, inverse_frequency)
        angles = torch.repeat_interleave(frequencies, 2, dim=-1)
        self.register_buffer("cosine", angles.cos()[None, None, :, :], persistent=False)
        self.register_buffer("sine", angles.sin()[None, None, :, :], persistent=False)

    def forward(
        self, query: torch.Tensor, key: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        sequence_length = query.size(-2)
        cosine = self.cosine[:, :, :sequence_length].to(dtype=query.dtype)
        sine = self.sine[:, :, :sequence_length].to(dtype=query.dtype)
        return (
            query * cosine + rotate_half(query) * sine,
            key * cosine + rotate_half(key) * sine,
        )


class CausalSelfAttention(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.n_heads = config.n_heads
        self.head_dim = config.d_model // config.n_heads
        self.dropout = config.dropout
        self.qkv = nn.Linear(config.d_model, 3 * config.d_model, bias=False)
        self.output = nn.Linear(config.d_model, config.d_model, bias=False)
        self.residual_dropout = nn.Dropout(config.dropout)
        self.rotary = RotaryEmbedding(self.head_dim, config.block_size, config.rope_base)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length, model_dimension = x.shape
        query, key, value = self.qkv(x).chunk(3, dim=-1)

        def split_heads(tensor: torch.Tensor) -> torch.Tensor:
            return tensor.view(
                batch_size, sequence_length, self.n_heads, self.head_dim
            ).transpose(1, 2)

        query, key, value = map(split_heads, (query, key, value))
        query, key = self.rotary(query, key)
        attended = F.scaled_dot_product_attention(
            query,
            key,
            value,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=True,
        )
        attended = attended.transpose(1, 2).contiguous().view(
            batch_size, sequence_length, model_dimension
        )
        return self.residual_dropout(self.output(attended))


class SwiGLU(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.gate = nn.Linear(config.d_model, config.ffn_dim, bias=False)
        self.up = nn.Linear(config.d_model, config.ffn_dim, bias=False)
        self.down = nn.Linear(config.ffn_dim, config.d_model, bias=False)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.down(F.silu(self.gate(x)) * self.up(x)))


class TransformerBlock(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.attention_norm = RMSNorm(config.d_model)
        self.attention = CausalSelfAttention(config)
        self.ffn_norm = RMSNorm(config.d_model)
        self.feed_forward = SwiGLU(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attention(self.attention_norm(x))
        return x + self.feed_forward(self.ffn_norm(x))


class ParteekLLM(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        config.validate()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.dropout = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList(
            TransformerBlock(config) for _ in range(config.n_layers)
        )
        self.final_norm = RMSNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding.weight
        self.apply(self._initialize_weights)

        residual_scale = 1.0 / math.sqrt(2 * config.n_layers)
        for name, parameter in self.named_parameters():
            if name.endswith("attention.output.weight") or name.endswith(
                "feed_forward.down.weight"
            ):
                nn.init.normal_(parameter, mean=0.0, std=0.02 * residual_scale)

    @staticmethod
    def _initialize_weights(module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        input_ids: torch.Tensor,
        targets: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [batch, sequence]")
        if input_ids.size(1) > self.config.block_size:
            raise ValueError(
                f"Sequence length {input_ids.size(1)} exceeds block_size "
                f"{self.config.block_size}"
            )
        hidden = self.dropout(self.token_embedding(input_ids))
        for block in self.blocks:
            hidden = block(hidden)
        logits = self.lm_head(self.final_norm(hidden))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1),
                ignore_index=-100,
            )
        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 128,
        temperature: float = 0.8,
        top_k: int = 50,
        top_p: float = 0.95,
        repetition_penalty: float = 1.1,
        eos_id: int | None = None,
    ) -> torch.Tensor:
        if max_new_tokens < 1:
            return input_ids
        was_training = self.training
        self.eval()
        generated = input_ids
        try:
            for _ in range(max_new_tokens):
                context = generated[:, -self.config.block_size :]
                logits, _ = self(context)
                next_logits = logits[:, -1, :].float()

                if repetition_penalty != 1.0:
                    recent = generated[:, -64:]
                    for batch_index in range(next_logits.size(0)):
                        token_ids = torch.unique(recent[batch_index])
                        values = next_logits[batch_index, token_ids]
                        next_logits[batch_index, token_ids] = torch.where(
                            values < 0,
                            values * repetition_penalty,
                            values / repetition_penalty,
                        )

                if temperature <= 0:
                    next_token = torch.argmax(next_logits, dim=-1, keepdim=True)
                else:
                    next_logits = next_logits / temperature
                    if top_k > 0:
                        cutoff = min(top_k, next_logits.size(-1))
                        threshold = torch.topk(next_logits, cutoff, dim=-1).values[:, -1:]
                        next_logits = next_logits.masked_fill(next_logits < threshold, -math.inf)
                    if 0.0 < top_p < 1.0:
                        sorted_logits, sorted_indices = torch.sort(
                            next_logits, descending=True, dim=-1
                        )
                        cumulative = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                        remove = cumulative > top_p
                        remove[:, 1:] = remove[:, :-1].clone()
                        remove[:, 0] = False
                        sorted_logits = sorted_logits.masked_fill(remove, -math.inf)
                        filtered = torch.full_like(next_logits, -math.inf)
                        next_logits = filtered.scatter(1, sorted_indices, sorted_logits)
                    probabilities = F.softmax(next_logits, dim=-1)
                    next_token = torch.multinomial(probabilities, num_samples=1)

                generated = torch.cat((generated, next_token), dim=1)
                if eos_id is not None and torch.all(next_token.squeeze(-1) == eos_id):
                    break
        finally:
            self.train(was_training)
        return generated

    def parameter_count(self, trainable_only: bool = False) -> int:
        parameters = self.parameters()
        if trainable_only:
            parameters = (parameter for parameter in parameters if parameter.requires_grad)
        return sum(parameter.numel() for parameter in parameters)

    def config_dict(self) -> dict[str, object]:
        return asdict(self.config)
