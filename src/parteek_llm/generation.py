from __future__ import annotations

from dataclasses import dataclass

import torch

from .model import ParteekLLM
from .tokenizer import OwnTokenizer


DEFAULT_SYSTEM_PROMPT = (
    "You are Parteek's helpful bilingual assistant. Reply in the language used by the user. "
    "Be honest when you do not know the answer."
)


@dataclass
class Message:
    role: str
    content: str


def format_conversation(
    messages: list[Message], system_prompt: str = DEFAULT_SYSTEM_PROMPT
) -> str:
    parts = [f"<system>\n{system_prompt.strip()}\n"]
    for message in messages:
        if message.role not in {"user", "assistant"}:
            raise ValueError(f"Unsupported message role: {message.role}")
        parts.append(f"<{message.role}>\n{message.content.strip()}\n")
    parts.append("<assistant>\n")
    return "".join(parts)


def remove_generated_markers(text: str) -> str:
    for marker in ("<user>", "<assistant>", "<system>", "<context>"):
        if marker in text:
            text = text.split(marker, 1)[0]
    return text.strip()


@torch.no_grad()
def generate_reply(
    model: ParteekLLM,
    tokenizer: OwnTokenizer,
    messages: list[Message],
    *,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    max_new_tokens: int = 128,
    temperature: float = 0.8,
    top_k: int = 50,
    top_p: float = 0.95,
    repetition_penalty: float = 1.1,
) -> str:
    if max_new_tokens >= model.config.block_size:
        raise ValueError("max_new_tokens must be smaller than the model block_size")
    prompt = format_conversation(messages, system_prompt)
    prompt_ids = tokenizer.encode(prompt, add_bos=True)
    maximum_prompt = model.config.block_size - max_new_tokens
    if len(prompt_ids) > maximum_prompt:
        prompt_ids = [tokenizer.bos_id] + prompt_ids[-(maximum_prompt - 1) :]
    device = next(model.parameters()).device
    input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    generated = model.generate(
        input_ids,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        eos_id=tokenizer.eos_id,
    )
    new_ids = generated[0, input_ids.size(1) :].tolist()
    if tokenizer.eos_id in new_ids:
        new_ids = new_ids[: new_ids.index(tokenizer.eos_id)]
    return remove_generated_markers(tokenizer.decode(new_ids))

